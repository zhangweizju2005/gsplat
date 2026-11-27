"""COLMAP dataset parser: model + images + depth -> :class:`Split`.

The pipeline is intentionally a sequence of explicit, checkable steps rather
than one function, because each step has a failure mode that is invisible
downstream:

1. read the model (``colmap_io``);
2. find the image directory and decide the render grid
   (``resolution``) -- the step whose absence cost 8-12 dB in this project;
3. build cameras, attaching per-frame depth with its **own** intrinsics;
4. split train/val;
5. apply the gauge (poses + points + depth together);
6. derive the scene statistics that the trainer and strategy consume.

Step 6 deserves emphasis: ``scene_scale`` for the densification strategy and
``depth_scale`` for the disparity loss are *different numbers derived from
different distributions*. Conflating them is what the earlier session's analysis
called out as "the pristine gauge variable that decides whether a run is sane".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config.schema import Config
from .colmap_io import ColmapModel, ColmapCamera, read_colmap_model
from .depth import (
    DepthCalibration,
    find_depth_file,
    load_depth_scales,
    median_scale,
    read_depth,
)
from .gauge import (
    GaugeApplication,
    GaugeSource,
    apply_gauge,
    camera_extent,
    depth_quantiles,
    derive_scale,
    derive_scene_scale,
    Sim3,
    estimate_sim3,
    sim3_residual,
    trajectory_span,
)
from .resolution import (
    ResolutionPolicy,
    probe_image_size,
    resolve_image_dir,
)
from .scene import Camera, Scene, Split, scene_subset


class DatasetError(RuntimeError):
    """A dataset cannot be used as configured, for a stated reason."""


@dataclass
class ParseReport:
    """What the parser decided, for the run manifest."""

    image_dir: str
    #: Distinct render grids as ``(width, height)`` -- image convention, like
    #: ``Camera.width``/``height`` and COLMAP's ``cameras.txt``. The bundle path
    #: once built this as ``(height, width)``, so the same field meant different
    #: things depending on which parser produced it and a manifest could not be
    #: read without knowing that. Printers use :meth:`render_size_labels`, which
    #: states the unit, so a reader never has to guess.
    render_sizes: List[Tuple[int, int]]
    n_images: int
    n_train: int
    n_val: int
    n_points: int
    depth_maps: int
    depth_remapped: int
    gauge: Optional[GaugeApplication]
    notes: List[str]

    def describe(self) -> str:
        lines = [
            f"image dir    : {self.image_dir}",
            f"render sizes : {self.render_size_labels()}",
            f"images       : {self.n_images} (train {self.n_train} / val {self.n_val})",
            f"init points  : {self.n_points}",
            f"depth maps   : {self.depth_maps}/{self.n_images}"
            + (
                f"  ({self.depth_remapped} ray-remapped to the render grid)"
                if self.depth_remapped
                else ""
            ),
        ]
        if self.gauge is not None:
            lines.append(self.gauge.describe())
        lines.extend(f"note         : {n}" for n in self.notes)
        return "\n".join(lines)

    def render_size_labels(self) -> str:
        """Render grids as ``WxH`` strings, so the unit is never ambiguous."""
        if not self.render_sizes:
            return "?"
        return ", ".join(f"{w}x{h}" for w, h in self.render_sizes)


# --------------------------------------------------------------------------
# Camera construction
# --------------------------------------------------------------------------
def _build_cameras(
    model: ColmapModel,
    image_dir: Path,
    cfg: Config,
    *,
    depth_scales: Optional[dict] = None,
    reference_scale: float = 1.0,
    notes: Optional[List[str]] = None,
) -> Tuple[List[Camera], List[Tuple[int, int]]]:
    """One :class:`Camera` per model image, with depth attached."""
    notes = notes if notes is not None else []
    policy = ResolutionPolicy(
        factor=cfg.data.factor,
        target_long_edge=cfg.data.target_long_edge,
        min_long_edge=cfg.data.min_long_edge,
    )
    depth_cfg = cfg.data.depth
    depth_dir = None
    if depth_cfg.enabled and depth_cfg.dir:
        depth_dir = Path(depth_cfg.dir)
        if not depth_dir.is_absolute():
            depth_dir = Path(cfg.data.path) / depth_dir
        if not depth_dir.is_dir():
            raise DatasetError(
                f"depth.enabled=True but depth.dir does not exist: {depth_dir}"
            )

    cameras: List[Camera] = []
    sizes: List[Tuple[int, int]] = []
    missing_images: List[str] = []
    missing_depth: int = 0

    for idx, im in enumerate(model.sorted_images()):
        colmap_cam = model.cameras[im.camera_id]
        model_size = (colmap_cam.width, colmap_cam.height)

        image_path = image_dir / im.name
        if not image_path.exists():
            # COLMAP names can carry subdirectories relative to images/.
            alt = image_dir / Path(im.name).name
            if alt.exists():
                image_path = alt
            else:
                missing_images.append(im.name)
                continue

        file_size = probe_image_size(image_path)
        resolved = policy.resolve(model_size, file_size)
        K = resolved.scale_K(colmap_cam.K)
        sizes.append((resolved.width, resolved.height))

        cam = Camera(
            name=im.name,
            image_path=image_path,
            K=K,
            width=resolved.width,
            height=resolved.height,
            cam_to_world=im.cam_to_world,
            index=idx,
        )

        if depth_dir is not None:
            cam.depth_path = find_depth_file(depth_dir, im.name, depth_cfg.suffix)
            if cam.depth_path is None:
                missing_depth += 1
            else:
                raw = read_depth(cam.depth_path)
                if raw is None:
                    cam.depth_path = None
                    missing_depth += 1
                else:
                    cam.depth_hw = (int(raw.shape[0]), int(raw.shape[1]))

                    # The depth grid's intrinsics. When the model already
                    # carries intrinsics for exactly this grid there is nothing
                    # to infer; otherwise the exporter must have told us, and
                    # assuming "same as the render camera" is the mistake this
                    # field exists to prevent.
                    if cam.depth_hw == (colmap_cam.height, colmap_cam.width):
                        cam.depth_K = colmap_cam.K.copy()
                    elif (
                        cam.depth_hw
                        == (resolved.height, resolved.width)
                    ):
                        cam.depth_K = K.copy()
                    else:
                        # Derive from the model intrinsics by the resolution
                        # ratio. This is exactly right for a depth map produced
                        # by resizing the same pinhole image (the common
                        # feed-forward export), and it is *checked* rather than
                        # assumed at load time by ``depth_grid_differs``.
                        sy = cam.depth_hw[0] / float(colmap_cam.height)
                        sx = cam.depth_hw[1] / float(colmap_cam.width)
                        cam.depth_K = colmap_cam.K.copy()
                        cam.depth_K[0, :] *= sx
                        cam.depth_K[1, :] *= sy

                    if depth_scales is not None and depth_cfg.rescale:
                        calib = depth_scales.get(
                            Path(im.name).name
                        ) or depth_scales.get(Path(im.name).stem)
                        if calib is not None:
                            cam.depth_scale = calib.scale
                            cam.depth_offset = calib.offset

        cameras.append(cam)

    if missing_images:
        raise DatasetError(
            f"{len(missing_images)} image(s) referenced by the COLMAP model were "
            f"not found under {image_dir}, e.g. {missing_images[:3]}. "
            "Check data.image_dir, or that the images are where the model says."
        )
    if depth_cfg.enabled and missing_depth:
        frac = missing_depth / max(1, len(cameras))
        msg = (
            f"{missing_depth}/{len(cameras)} frames have no depth map under "
            f"{depth_dir}"
        )
        if frac > 0.5:
            raise DatasetError(
                msg + " -- more than half are missing, so the depth directory is "
                "probably wrong or uses a different naming scheme. "
                "Depth files are matched by frame stem; check one by hand."
            )
        notes.append(msg + " (those frames are trained photometrically only)")

    return cameras, sizes


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def _split_indices(
    n: int, cfg: Config
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(train_idx, val_idx)`` in :meth:`ColmapModel.sorted_images` order."""
    mode = cfg.data.split_mode
    if mode == "reconstruction":
        # Every image is trained on. Held-out evaluation is done on the same
        # set, so the reported PSNR measures fit. This is what the archived
        # MatrixCity baselines did; changing it makes their numbers and ours
        # incomparable, which is why it is an explicit choice rather than a
        # silent default.
        return np.arange(n), None

    all_idx = np.arange(n)
    if cfg.data.test_every and cfg.data.test_every > 0:
        val_mask = all_idx % cfg.data.test_every == 0
    else:
        ratio = float(np.clip(cfg.data.eval_ratio, 0.0, 1.0))
        if ratio <= 0.0 or ratio >= 1.0:
            raise DatasetError(
                f"split_mode='experiment' needs 0 < eval_ratio < 1 "
                f"(got {cfg.data.eval_ratio}) or a positive test_every"
            )
        n_val = max(1, int(round(n * ratio)))
        # Evenly spaced rather than a random draw: reproducible, and it spreads
        # held-out views across the whole flight line instead of clustering
        # them at one end.
        val_mask = np.zeros(n, dtype=bool)
        val_mask[np.linspace(0, n - 1, n_val).round().astype(int)] = True

    return all_idx[~val_mask], all_idx[val_mask]


# --------------------------------------------------------------------------
# Initial points
# --------------------------------------------------------------------------
def _initial_points(
    model: ColmapModel,
    cfg: Config,
    cameras: Sequence[Camera],
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Initial point cloud and colours; also returns provenance notes."""
    notes: List[str] = []
    mode = cfg.data.init_points

    if mode == "sparse":
        pts = model.points_xyz
        rgb = model.points_rgb.astype(np.float32) / 255.0
        notes.append(f"init cloud: COLMAP sparse points ({pts.shape[0]})")
        if pts.shape[0] < 100:
            notes.append(
                "WARNING: fewer than 100 sparse points; densification will have "
                "almost nothing to grow from"
            )
    elif mode == "depth":
        pts, rgb, notes = _dgi_points(cameras, cfg, seed=seed)
    elif mode == "none":
        pts = np.zeros((0, 3), np.float64)
        rgb = np.zeros((0, 3), np.float32)
        notes.append("init cloud: empty (model starts from noise)")
    else:  # pragma: no cover - validated earlier
        raise DatasetError(f"unknown init_points mode {mode!r}")

    cap = cfg.data.max_init_points
    if cap and pts.shape[0] > cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(pts.shape[0], cap, replace=False)
        keep.sort()
        pts, rgb = pts[keep], rgb[keep]
        notes.append(f"init cloud capped to {cap} points")

    return pts, rgb, notes


def _dgi_points(
    cameras: Sequence[Camera], cfg: Config, *, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Depth-guided initialisation: unproject every valid depth pixel.

    Each frame is unprojected with **its own depth intrinsics**. Using the
    render intrinsics instead rotates every ray, which is invisible in the
    point count and only shows up as a cloud that does not quite fit the
    cameras. Measured ray error at a 10% focal discrepancy grows linearly with
    distance from the principal point.

    Colours come from the render image at the *normalised* position of each
    depth sample, because the depth and image grids need not coincide.
    """
    from .dgi import DgiConfig, build_dgi_cloud

    dgi = DgiConfig(
        stride=cfg.data.dgi_stride,
        border_trim=cfg.data.dgi_border_trim,
        min_depth=cfg.data.dgi_min_depth,
        max_depth=cfg.data.dgi_max_depth,
    )
    depths, Ks, c2ws, images, scales = [], [], [], [], []
    used = 0
    for cam in cameras:
        if cam.depth_path is None:
            continue
        raw = read_depth(cam.depth_path)
        if raw is None:
            continue
        d = raw * cam.depth_scale + cam.depth_offset
        valid = np.isfinite(d) & (d > 0)
        if valid.mean() < cfg.data.depth.min_valid_frac:
            continue
        depths.append(d)
        Ks.append(cam.depth_K if cam.depth_K is not None else cam.K)
        c2ws.append(cam.cam_to_world)
        scales.append(1.0)  # already applied above
        try:
            import imageio.v2 as imageio

            images.append(imageio.imread(str(cam.image_path)))
        except Exception:
            images.append(None)
        used += 1

    if not depths:
        raise DatasetError(
            "init_points='depth' needs depth maps, but none were usable. "
            "Enable data.depth and check data.depth.dir."
        )

    xyz, rgb = build_dgi_cloud(
        depths,
        Ks,
        c2ws,
        None if any(im is None for im in images) else images,
        cfg=dgi,
        max_points=cfg.data.max_init_points,
        seed=seed,
    )
    notes = [
        f"init cloud: DGI from {used} depth maps ({xyz.shape[0]} points)",
        "  each frame unprojected with its own depth-grid intrinsics",
    ]
    if any(im is None for im in images):
        notes.append("  colours defaulted to grey (image read failed)")
    return xyz, rgb, notes


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_colmap(cfg: Config) -> Tuple[Split, ParseReport]:
    """Parse a COLMAP dataset into a train/val :class:`Split`."""
    root = Path(cfg.data.path)
    model = read_colmap_model(root)

    image_dir, nominal = resolve_image_dir(
        root, cfg.data.image_dirs, cfg.data.image_dir
    )
    notes: List[str] = []
    if nominal:
        notes.append(
            f"image dir named like a {nominal}x downsample; the real calibration "
            "is taken from the decoded file size, not the name"
        )

    depth_scales = None
    reference_scale = 1.0
    if cfg.data.depth.enabled and cfg.data.depth.rescale:
        if not cfg.data.depth.scale_file:
            raise DatasetError(
                "depth.rescale=True requires depth.scale_file (a per-frame "
                "{scale, offset} JSON). Without it there is no calibration to "
                "apply and the depth would silently stay in arbitrary units."
            )
        sf = Path(cfg.data.depth.scale_file)
        if not sf.is_absolute():
            sf = root / sf
        depth_scales = load_depth_scales(sf)
        reference_scale = median_scale(depth_scales)
        notes.append(
            f"depth calibrated from {sf.name}: {len(depth_scales)} frames, "
            f"median scale {reference_scale:.6g}"
        )

    cameras, sizes = _build_cameras(
        model,
        image_dir,
        cfg,
        depth_scales=depth_scales,
        reference_scale=reference_scale,
        notes=notes,
    )

    # Model images may have been dropped (missing files raise, but depth
    # filtering happens later), so re-align the point indices to the cameras.
    kept_names = {c.name for c in cameras}
    if len(cameras) != len(model.images):
        model = ColmapModel(
            cameras=model.cameras,
            images={k: v for k, v in model.images.items() if v.name in kept_names},
            points_xyz=model.points_xyz,
            points_rgb=model.points_rgb,
        )

    points, point_rgb, pt_notes = _initial_points(model, cfg, cameras, seed=cfg.train.seed)
    notes.extend(pt_notes)

    # Scene stats *before* the gauge, so the scale can be derived from the
    # source units; then again after, for the trainer.
    c2ws = np.stack([c.cam_to_world for c in cameras], axis=0)

    depths_for_scale = None
    if cfg.gauge.scale_source == "depth_p99":
        depths_for_scale = _sample_depths(cameras)

    sim = None
    gauge_notes: List[str] = []
    if cfg.gauge.mode == "normalize":
        s, s_source = derive_scale(
            c2ws[:, :3, 3],
            depths_for_scale,
            cfg.gauge.scale_source,
            manual=cfg.gauge.scale,
            trim=cfg.gauge.trim_quantile,
        )
        sim = Sim3(R=np.eye(3), s=s, t=np.zeros(3))
        gauge_notes.append(f"normalisation scale {s:.6g} from {s_source}")
    elif cfg.gauge.mode != "none":  # pragma: no cover
        raise DatasetError(f"unknown gauge.mode {cfg.gauge.mode!r}")

    c2ws, points, _, gauge_app = apply_gauge(
        cam_to_worlds=c2ws,
        points=points,
        depths=None,  # per-frame scaling is carried on Camera.depth_scale
        sim3=sim,
        apply_to_depth=cfg.gauge.apply_to_depth,
        source=cfg.gauge.mode,
        notes=gauge_notes,
    )
    if sim is not None and cfg.gauge.apply_to_depth and not cfg.data.depth.per_frame_median_norm:
        for cam in cameras:
            cam.depth_scale *= sim.s
        gauge_app.applied_to_depth = True

    for cam, c2w in zip(cameras, c2ws):
        cam.cam_to_world = c2w

    q = depth_quantiles(c2ws, points)
    cam_ext = camera_extent(c2ws)
    scene_scale, ss_source = derive_scene_scale(q, cam_ext)
    notes.append(f"scene_scale {scene_scale:.6g} from {ss_source}")

    scene = Scene(
        cameras=cameras,
        points=points,
        point_rgb=point_rgb,
        applied_sim3=sim.matrix() if sim is not None else None,
        camera_extent=cam_ext,
        scene_scale=scene_scale,
        depth_p01=q["p01"],
        depth_p50=q["p50"],
        depth_p99=q["p99"],
        notes=notes,
    )

    train_idx, val_idx = _split_indices(len(cameras), cfg)
    train = scene_subset(scene, train_idx)
    val = scene_subset(scene, val_idx) if val_idx is not None and len(val_idx) else None

    if val is not None:
        # A validation view must not share a camera with training.
        train_set = {c.name for c in train.cameras}
        val_set = {c.name for c in val.cameras}
        assert not (train_set & val_set), "train/val split overlap"

    report = ParseReport(
        image_dir=str(image_dir),
        render_sizes=sorted(set(sizes)),  # per-frame list -> distinct grids
        n_images=len(cameras),
        n_train=len(train.cameras),
        n_val=len(val.cameras) if val is not None else 0,
        n_points=scene.points.shape[0],
        depth_maps=sum(1 for c in cameras if c.depth_path is not None),
        depth_remapped=sum(1 for c in cameras if c.depth_grid_differs),
        gauge=gauge_app,
        notes=notes,
    )
    return Split(train=train, val=val), report


def _sample_depths(cameras: Sequence[Camera], max_frames: int = 32) -> Optional[np.ndarray]:
    """Read a subsample of depth maps, for deriving the normalisation scale."""
    if not cameras:
        return None
    step = max(1, len(cameras) // max_frames)
    out = []
    for cam in cameras[::step]:
        if cam.depth_path is None:
            continue
        raw = read_depth(cam.depth_path)
        if raw is None:
            continue
        d = raw * cam.depth_scale + cam.depth_offset
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            out.append(d)
    if not out:
        return None
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Rigid multi-source alignment (used by the feed-forward ingest path)
# --------------------------------------------------------------------------
def align_to_reference(
    source_c2ws: np.ndarray,
    reference_c2ws: np.ndarray,
    *,
    max_residual: float = 0.05,
    source: str = "scal3r",
    target: str = "colmap",
) -> Tuple[Sim3, dict]:
    """Fit and validate the Sim(3) taking ``source`` poses into ``reference``.

    A similarity can always be fitted; the *residual* is what says whether the
    two trajectories are actually the same scene. Measured on one real pair the
    residual was 8.6e-8 when the sources shared a gauge and 11.1% when they did
    not -- six orders of magnitude apart, so the threshold is not delicate.

    ``max_residual`` is normally supplied from ``cfg.gauge.sim3_max_residual``.
    ``source``/``target`` are recorded on the returned transform so its guarded
    apply helpers can reject a wrong-side application.
    """
    if source_c2ws.shape[0] != reference_c2ws.shape[0]:
        raise DatasetError(
            f"align requires matched frames: {source_c2ws.shape[0]} vs "
            f"{reference_c2ws.shape[0]}"
        )
    src_c = source_c2ws[:, :3, 3]
    ref_c = reference_c2ws[:, :3, 3]
    sim = estimate_sim3(src_c, ref_c, source=source, target=target)
    rms, mx = sim3_residual(src_c, ref_c, sim)
    span = trajectory_span(ref_c)
    frac = rms / span if span > 0 else float("inf")
    report = {
        "scale": sim.s,
        "rms": rms,
        "max": mx,
        "span": span,
        "residual_fraction": frac,
        "ok": frac <= max_residual,
    }
    if not report["ok"]:
        raise DatasetError(
            f"Sim(3) residual {frac:.3%} exceeds the {max_residual:.1%} threshold "
            f"(rms {rms:.4g} over a trajectory span of {span:.4g}). The two "
            "trajectories are not the same scene in two gauges -- most often "
            "because one side is already aligned and the other is raw, or "
            "because the frame correspondence is wrong."
        )
    return sim, report


# --------------------------------------------------------------------------
# Feed-forward bundle parser
# --------------------------------------------------------------------------
def parse_feedforward(cfg: Config) -> Tuple[Split, ParseReport]:
    """Parse a ``surfel-feedforward/1`` bundle into a train/val :class:`Split`.

    A bundle already carries poses, per-frame intrinsics and (optionally) depth,
    so this path skips SfM entirely: there is no COLMAP model, no triangulated
    cloud, and the initial points come from DGI over the bundle's own depth.

    Two invariants are enforced here rather than downstream:

    * **K belongs to the depth grid.** The bundle states ``depth_grid``
      separately from ``image_grid``; each :class:`Camera` therefore carries
      ``depth_K``/``depth_hw`` so DGI unprojects with the right intrinsics and
      depth supervision knows the grids differ.
    * **Nothing assumes the two grids match.** Colour lookup during DGI goes
      through normalised coordinates.

    ``data.init_points`` still gates the cloud: ``"depth"`` uses DGI,
    ``"none"`` starts from an empty cloud, and ``"sparse"`` is refused because a
    bundle has no sparse points.
    """
    from .feedforward import read_bundle

    bundle = read_bundle(cfg.data.path)
    # Both of these are COLMAP-path concepts that a bundle cannot honour, and
    # silently ignoring either produces a run whose config does not describe it:
    # `image_dirs` is a search order for locating files named by the SfM model,
    # while a bundle pins each frame's image path in `frames.json`; and
    # `init_points="sparse"` needs a points3D cloud this format does not carry.
    # The project's rule is to refuse rather than guess, so they raise here.
    if cfg.data.image_dir is not None:
        raise DatasetError(
            "data.image_dir is not used by a feed-forward bundle: each frame's "
            "image path comes from the bundle's frames.json. Edit the manifest, "
            "or re-export the bundle with the images in place."
        )
    if not bundle.has_depth and cfg.data.init_points == "depth":
        raise DatasetError(
            "init_points='depth' needs depth maps, but the bundle has none. "
            "Re-export the bundle with depths. (init_points='none' is not an "
            "alternative: gsplat cannot grow a model from zero Gaussians, so an "
            "empty start would train nothing.)"
        )
    if cfg.data.init_points == "sparse":
        raise DatasetError(
            "init_points='sparse' needs a COLMAP points3D cloud; a feed-forward "
            "bundle has none. Use init_points='depth' (DGI)."
        )

    notes: List[str] = [
        f"bundle: {bundle.root} (format {bundle.manifest.get('format')})",
        f"  source={bundle.manifest.get('source')} "
        f"images={bundle.image_grid[0]}x{bundle.image_grid[1]}"
        + (
            f" depth={bundle.depth_grid[0]}x{bundle.depth_grid[1]}"
            if bundle.depth_grid
            else " depth=none"
        ),
        "  poses are camera-to-world, OpenCV convention; no SfM model was read",
    ]

    # Render resolution. A bundle's images are whatever the front end was run on,
    # so the same three knobs the COLMAP path honours must be honoured here too --
    # otherwise `data.factor` is silently ignored on bundle input and a
    # bundle-based run cannot be compared with a COLMAP run of the same scene
    # (the failure mode this project treats as its most expensive bug class).
    #
    # ``ResolutionPolicy`` is reused rather than re-implemented: it is where the
    # `target_long_edge` overrides `factor`, `min_long_edge` floors it, and K is
    # scaled by the ratio *actually* achieved after rounding. A second copy of
    # those rules is how the two paths drifted apart in the first place.
    policy = ResolutionPolicy(
        factor=cfg.data.factor,
        target_long_edge=cfg.data.target_long_edge,
        min_long_edge=cfg.data.min_long_edge,
    )
    resolved = policy.resolve(
        (int(bundle.image_grid[0]), int(bundle.image_grid[1]))
    )
    render_w, render_h = resolved.width, resolved.height
    if (render_w, render_h) != (bundle.image_grid[0], bundle.image_grid[1]):
        notes.append(
            f"  render grid {bundle.image_grid[0]}x{bundle.image_grid[1]}"
            f" -> {render_w}x{render_h} (data.factor={cfg.data.factor}, "
            f"target_long_edge={cfg.data.target_long_edge})"
        )
    if cfg.data.min_long_edge and max(render_w, render_h) < cfg.data.min_long_edge:
        raise DatasetError(
            f"render long edge {max(render_w, render_h)} is below "
            f"data.min_long_edge={cfg.data.min_long_edge}"
        )

    def _render_K(K_depth: np.ndarray) -> np.ndarray:
        """Depth-grid K -> render-grid K, in one isotropic step when possible."""
        if bundle.depth_grid is None:
            # No depth: K is authored on the image grid, so scale image->render.
            sx = render_w / float(bundle.image_grid[0])
            sy = render_h / float(bundle.image_grid[1])
            out = np.array(K_depth, dtype=np.float64, copy=True)
            out[0, :] *= sx
            out[1, :] *= sy
            return out
        return _scale_K_to_grid(
            K_depth,
            bundle.depth_grid,
            (render_w, render_h),
            notes=notes,
            name="render",
            resample=bundle.depth_resample,
            crop_offset=bundle.depth_crop_offset,
        )

    cameras: List[Camera] = []
    for i, name in enumerate(bundle.names):
        K = bundle.Ks[i]
        dgrid = bundle.depth_grid
        depth_path = bundle.depth_paths[i]
        cameras.append(
            Camera(
                name=name,
                image_path=bundle.image_paths[i],
                # Render grid may be smaller than the file on disk; the dataset
                # resizes on read, and K must describe the grid it renders.
                K=_render_K(K),
                width=render_w,
                height=render_h,
                cam_to_world=np.asarray(bundle.cam_to_world[i], dtype=np.float64),
                depth_K=np.asarray(K, dtype=np.float64),
                depth_hw=None if dgrid is None else (int(dgrid[1]), int(dgrid[0])),
                depth_path=None if depth_path is None else Path(depth_path),
                depth_scale=float(bundle.depth_scales[i]),
                index=i,
            )
        )

    sizes = {(c.width, c.height) for c in cameras}
    if len(sizes) != 1:
        raise DatasetError(
            f"bundle has {len(sizes)} distinct image sizes {sorted(sizes)}; "
            "re-export at a single resolution"
        )

    c2ws = np.stack([c.cam_to_world for c in cameras], axis=0)

    # Gauge: a bundle is defined up to a similarity. Whether to normalise it, and
    # from which statistic, is the config's decision -- not this function's.
    depth_sample = None
    if cfg.gauge.scale_source == "depth_p99" and bundle.has_depth:
        chunks = []
        step = max(1, len(cameras) // 32)
        for i in range(0, len(cameras), step):
            d = bundle.load_depth(i)
            if d is None:
                continue
            v = d.ravel() * bundle.depth_scales[i]
            chunks.append(v[np.isfinite(v) & (v > 0)])
        if chunks:
            depth_sample = np.concatenate(chunks)

    sim = None
    gauge_notes: List[str] = []
    if cfg.gauge.mode == "normalize":
        s, s_source = derive_scale(
            c2ws[:, :3, 3],
            depth_sample,
            cfg.gauge.scale_source,
            manual=cfg.gauge.scale,
            trim=cfg.gauge.trim_quantile,
        )
        sim = Sim3(R=np.eye(3), s=s, t=np.zeros(3))
        gauge_notes.append(f"normalisation scale {s:.6g} from {s_source}")
    elif cfg.gauge.mode != "none":
        raise DatasetError(f"unknown gauge.mode {cfg.gauge.mode!r}")

    # Poses-only gauge step: points do not exist yet (DGI runs below, after the
    # similarity is known, so the cloud is built directly in the final gauge).
    gauge_app = None
    if sim is not None:
        c2ws_g, _, _, gauge_app = apply_gauge(
            cam_to_worlds=c2ws,
            points=np.zeros((0, 3), np.float32),
            depths=None,
            sim3=sim,
            apply_to_depth=cfg.gauge.apply_to_depth,
            source=cfg.gauge.mode,
            notes=gauge_notes,
        )
        for cam, c2w in zip(cameras, c2ws_g):
            cam.cam_to_world = c2w
        if cfg.gauge.apply_to_depth:
            # The per-frame depth_scale is where depth carries the gauge.
            for cam in cameras:
                cam.depth_scale *= sim.s
            gauge_app.applied_to_depth = True
    notes.extend(gauge_notes)

    # Initial cloud. DGI runs *after* the gauge so the cloud inherits the same
    # similarity as the poses and the per-frame depth_scale; doing it before
    # would leave points and cameras in two different frames.
    mode = cfg.data.init_points
    if mode == "depth":
        pts, rgb, dgi_notes = _dgi_points(cameras, cfg, seed=cfg.train.seed)
        notes.extend(dgi_notes)
    elif mode == "none":
        pts = np.zeros((0, 3), np.float32)
        rgb = np.zeros((0, 3), np.float32)
        notes.append("init cloud: empty (no points; densification has no seed)")
    else:  # pragma: no cover - refused above
        raise DatasetError(f"unknown init_points mode {mode!r}")

    final_c2ws = np.stack([c.cam_to_world for c in cameras], axis=0)
    q = depth_quantiles(final_c2ws, pts)
    cam_ext = camera_extent(final_c2ws)
    scene_scale, ss_source = derive_scene_scale(q, cam_ext)
    notes.append(f"scene_scale {scene_scale:.6g} from {ss_source}")

    scene = Scene(
        cameras=cameras,
        points=pts,
        point_rgb=rgb,
        applied_sim3=sim.matrix() if sim is not None else None,
        camera_extent=cam_ext,
        scene_scale=scene_scale,
        depth_p01=q["p01"],
        depth_p50=q["p50"],
        depth_p99=q["p99"],
        notes=notes,
    )

    train_idx, val_idx = _split_indices(len(cameras), cfg)
    train = scene_subset(scene, train_idx)
    val = scene_subset(scene, val_idx) if val_idx is not None and len(val_idx) else None
    if val is not None:
        assert not ({c.name for c in train.cameras} & {c.name for c in val.cameras})

    report = ParseReport(
        image_dir=str(bundle.root / "images"),
        render_sizes=sorted(set(sizes)),  # per-frame list -> distinct grids
        n_images=len(cameras),
        n_train=len(train.cameras),
        n_val=len(val.cameras) if val is not None else 0,
        n_points=scene.points.shape[0],
        depth_maps=sum(1 for c in cameras if c.depth_path is not None),
        depth_remapped=0,
        # ``gauge.mode="none"`` leaves the bundle in its own arbitrary gauge.
        # That is a valid choice -- it means "do not normalise" -- so it must be
        # reported as an identity application rather than crashing. The previous
        # `GaugeApplication(source=...)` form omitted the four required fields and
        # raised TypeError for a value the schema explicitly allows.
        gauge=gauge_app
        if gauge_app is not None
        else GaugeApplication(
            sim3=None,
            scale_only=1.0,
            source=cfg.gauge.mode,
            applied_to_depth=False,
            notes=list(gauge_notes),
        ),
        notes=notes,
    )
    return Split(train=train, val=val), report


def _scale_K_to_grid(
    K: np.ndarray,
    from_grid: Optional[Tuple[int, int]],
    to_grid: Tuple[int, int],
    *,
    notes: List[str],
    name: str,
    resample: Optional[str] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rescale intrinsics from one grid to another.

    Two models, and which one applies is a property of how the depth was made,
    not something that can be inferred from the grids:

    ``resample="cover"``
        The source grid is a uniform resize-to-cover of the destination followed
        by a centre crop (Scal3R's pipeline: ``resize_to_cover`` then
        ``crop_nhwc_image``). The scale is therefore *single* -- one number for
        both axes -- and the principal point also shifts by the crop.

    otherwise (``None`` / ``"resize"``)
        A per-axis resize of the same image. ``sx`` and ``sy`` may differ, which
        is what a caller who resampled to a different aspect actually did.

    Getting this wrong is silent: on the 1297x840 garden bundle with 518x322
    depth, treating Scal3R's cover-crop as a per-axis stretch produced
    ``fx/fy = 0.9636`` where the truth is ``1.0040`` -- a 3.6% vertical-focal
    error, with no warning and no visible artefact other than a slightly wrong
    reconstruction.
    """
    if from_grid is None or (int(from_grid[0]), int(from_grid[1])) == (
        int(to_grid[0]),
        int(to_grid[1]),
    ):
        return np.array(K, dtype=np.float64, copy=True)

    fw, fh = int(from_grid[0]), int(from_grid[1])
    tw, th = int(to_grid[0]), int(to_grid[1])

    if resample == "cover":
        # Geometry, stated once: the depth grid is a *crop* of a uniformly
        # resized image. A depth pixel ``(u, v)`` therefore sits at render pixel
        # ``((u + left) * W / resized_w, (v + top) * H / resized_h)`` -- the crop
        # offset is *added* before scaling, because the crop removed pixels from
        # the left/top of the resized frame.
        #
        # The scale is ``render / resized``, not ``render / depth``: dividing by
        # the depth grid would fold the cropped-away rows into the scale and
        # reintroduce exactly the anisotropy this branch exists to remove.
        if crop_offset is not None and len(crop_offset) >= 4:
            left, top, resized_w, resized_h = (int(v) for v in crop_offset[:4])
            sx = float(tw) / float(resized_w)
            sy = float(th) / float(resized_h)
            out = np.array(K, dtype=np.float64, copy=True)
            out[0, :] *= sx
            out[1, :] *= sy
            out[0, 2] = (K[0, 2] + left) * sx
            out[1, 2] = (K[1, 2] + top) * sy
            notes.append(
                f"  {name}: depth {fw}x{fh} is a crop of a uniform resize of "
                f"{tw}x{th} (sx={sx:.6f} sy={sy:.6f}, crop {left},{top}); "
                "K rescaled, principal point de-cropped"
            )
            return out
        # No geometry recorded: fall back to the isotropic ratio, which is exact
        # when the crop happens to be symmetric and zero, and close otherwise.
        sx = sy = float(tw) / float(fw)
        out = np.array(K, dtype=np.float64, copy=True)
        out[0, :] *= sx
        out[1, :] *= sy
        notes.append(
            f"  {name}: depth {fw}x{fh} -> {tw}x{th} assumed isotropic; the "
            "bundle records no crop geometry, so a real crop would be missed"
        )
        return out

    sx = float(tw) / float(fw)
    sy = float(th) / float(fh)
    if abs(sx - sy) > 1e-3:
        notes.append(
            f"  {name}: depth/image aspect differs "
            f"({from_grid[0]}x{from_grid[1]} -> {to_grid[0]}x{to_grid[1]}); "
            "K scaled per-axis, depth needs remapping"
        )
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def parse_dataset(cfg: Config) -> Tuple[Split, ParseReport]:
    """Dispatch on ``data.parser``: COLMAP model or a feed-forward bundle."""
    kind = getattr(cfg.data, "parser", "colmap")
    if kind == "colmap":
        return parse_colmap(cfg)
    if kind in ("feedforward", "bundle", "scal3r", "vggt"):
        return parse_feedforward(cfg)
    raise DatasetError(f"unknown data.parser {kind!r}; use 'colmap' or 'feedforward'")


def align_scal3r_to_colmap(
    scal3r_cam_to_worlds,
    scal3r_names,
    colmap_model,
    cfg: Config,
) -> Tuple[Sim3, dict]:
    """Align a Scal3R reconstruction into a COLMAP model's gauge.

    This is the one-line glue the "mixed source" path needs: fit the Sim(3) on
    matched camera centres, gate it with ``cfg.gauge.sim3_max_residual``, and
    return the transform to apply to Scal3R poses *and* depth together.

    Frames are matched by name. That is why :mod:`surfel.data.scal3r` resolves
    Scal3R's zero-padded indices to the original filenames -- with the indices
    left in place this function matched 0 of 185 garden frames and reported
    nothing, which is the failure mode it is written to avoid.

    Returns ``(sim3, report)``. ``report`` carries ``scale``/``rms``/
    ``residual_fraction`` so the caller can log *why* the gate passed.
    """
    cmap = {im.name: im for im in colmap_model.sorted_images()}
    s_index = {n: i for i, n in enumerate(scal3r_names)}
    common = sorted(set(cmap) & set(s_index))
    if not common:
        raise DatasetError(
            "no frame name overlaps between the Scal3R run and the COLMAP "
            f"model ({len(s_index)} vs {len(cmap)} frames). Check that the "
            "Scal3R reader resolved name_map.json: its names must be the "
            "original filenames, not the zero-padded indices."
        )
    sim, report = align_to_reference(
        np.asarray(scal3r_cam_to_worlds)[[s_index[n] for n in common]],
        np.stack([cmap[n].cam_to_world for n in common]),
        max_residual=cfg.gauge.sim3_max_residual,
        source="scal3r",
        target="colmap",
    )
    report["n_matched"] = len(common)
    report["n_scal3r"] = len(s_index)
    report["n_colmap"] = len(cmap)
    report["frames"] = common
    return sim, report


__all__ = [
    "DatasetError",
    "ParseReport",
    "parse_colmap",
    "parse_feedforward",
    "parse_dataset",
    "align_to_reference",
    "align_scal3r_to_colmap",
    "GaugeSource",
]
