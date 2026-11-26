"""Scal3R reader: ``intri.yml`` / ``extri.yml`` / ``depths/*.exr`` -> :class:`Scene`.

Scal3R writes an EasyVolcap-style pair of OpenCV YAML files plus one float EXR
per frame. The conventions, verified against the run directories in this
workspace:

============================  ==================================================
``intri.yml``                 ``names[]``; per-frame ``K_*`` (3x3), ``H_*``,
                              ``W_*``, ``D_*`` (distortion, zero for pinhole)
``extri.yml``                 per-frame ``Rot_*`` (3x3, **world-to-camera**),
                              ``T_*`` (3x1 w2c translation), ``R_*`` (3x1
                              rotation vector, redundant)
``depths/NNNNNN.exr``         float32 ``H x W`` z-depth, on the grid that ``K``
                              describes
``mat.txt``                   camera-to-world as 16 floats/line, but its
                              rotation block carries a ``cbrt(det)`` factor, so
                              it is **not rigid** -- prefer ``extri.yml``
============================  ==================================================

Two consequences decide whether the reconstruction is usable:

**1. ``K`` belongs to the depth grid.** A depth sample ``d(u,v)`` and ``K``
describe the same pixels. DGI must unproject with *this* K, never with an
SfM-derived one from another resolution: Scal3R estimates its own focal, and on
the garden run it is 379.92 where the COLMAP model (5187 px wide) implies
3844.90 -- a factor-of-ten difference in the *grid*, not in the optics.

**2. Scal3R's scale is arbitrary.** Poses and depth are mutually consistent but
nothing ties them to metres. Fine for training (3DGS only needs internal
consistency), fatal for mixing with metric sources without a Sim(3).

The reader also resolves Scal3R's zero-padded frame names (``000000``) to the
original filenames (``DSC07956.JPG``) via ``name_map.json``. That mapping must
reach ``Camera.name``, not only the image path: cross-source frame matching keys
off the name, and getting it wrong results in **0 matched frames with no
error**.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .depth import read_depth
from .resolution import probe_image_size
from .scene import Camera, Scene


class Scal3RError(RuntimeError):
    """The Scal3R run directory cannot be read as configured."""


@dataclass
class Scal3RData:
    """Raw contents of a Scal3R run, before any gauge decision."""

    names: List[str]
    #: Camera-to-world, OpenCV convention, in Scal3R's own arbitrary gauge.
    cam_to_worlds: np.ndarray
    #: Intrinsics **of the depth grid**.
    Ks: np.ndarray
    #: ``(H, W)`` of the depth grid.
    depth_hw: Tuple[int, int]
    depth_paths: List[Optional[Path]]
    image_paths: List[Path]
    name_map: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _read_opencv_yaml(path: Path) -> Dict[str, object]:
    """Flatten an EasyVolcap OpenCV YAML into ``{key: {name: value}}``."""
    import cv2

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise Scal3RError(f"cannot open {path}")
    try:
        names_node = fs.getNode("names")
        if names_node.empty():
            raise Scal3RError(f"{path} has no 'names' node")
        names = [names_node.at(i).string() for i in range(names_node.size())]

        out: Dict[str, object] = {"names": names}
        for name in names:
            for key in ("K", "H", "W", "R", "Rot", "T", "D"):
                node = fs.getNode(f"{key}_{name}")
                if node.empty():
                    continue
                if key in ("H", "W"):
                    out.setdefault(key, {})[name] = float(node.real())
                else:
                    out.setdefault(key, {})[name] = np.asarray(node.mat())
        return out
    finally:
        fs.release()


def _qvec_to_rotvec_pose(Rot: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build camera-to-world from Scal3R's rigid w2c pair.

    ``Rot`` is world-to-camera and orthonormal (verified: det 1.0, orthonormality
    error 1.2e-7 on the garden run). Inverting yields c2w with a rigid rotation
    block -- scaling the rotation instead of the translation is a classic mistake
    that leaves ``R^T R != I`` and breaks every downstream ``inv``.
    """
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(Rot, dtype=np.float64)
    w2c[:3, 3] = np.asarray(T, dtype=np.float64).ravel()
    # Guard against a non-rigid block sneaking in (mat.txt has this problem).
    R = w2c[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
        raise Scal3RError(
            "rotation block is not orthonormal (det="
            f"{np.linalg.det(R):.4f}); this looks like mat.txt rather than "
            "extri.yml, whose rotation carries a cbrt(det) factor"
        )
    return np.linalg.inv(w2c)


def _load_name_map(run_dir: Path) -> Dict[str, str]:
    """``{zero_padded_index: original_filename}`` if the exporter wrote one."""
    for cand in (
        run_dir / "name_map.json",
        run_dir / "scal3r" / "name_map.json",
        run_dir / "runtime" / "name_map.json",
    ):
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return {str(k): str(v) for k, v in raw.items()}
    return {}


def _image_manifest(run_dir: Path) -> Dict[str, str]:
    """Fallback mapping from ``runtime/image_manifest.txt``.

    The manifest lists the source images in order, which is the same order as
    Scal3R's zero-padded indices.
    """
    for cand in (
        run_dir / "scal3r" / "runtime" / "image_manifest.txt",
        run_dir / "runtime" / "image_manifest.txt",
    ):
        if not cand.exists():
            continue
        lines = [
            line.strip()
            for line in cand.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {f"{i:06d}": Path(line).name for i, line in enumerate(lines)}
    return {}


def find_scal3r_root(path: str | Path) -> Path:
    """Locate the directory that directly contains ``intri.yml``."""
    p = Path(path)
    if (p / "intri.yml").exists():
        return p
    for cand in (p / "scal3r", p / "s3r"):
        if (cand / "intri.yml").exists():
            return cand
    raise Scal3RError(
        f"no intri.yml under {p} (looked in the directory itself, scal3r/ and "
        "s3r/). Point data.path at the run root or at the directory holding "
        "intri.yml."
    )


def read_scal3r(
    path: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    depth_dir: Optional[str | Path] = None,
    depth_suffix: str = ".exr",
    min_valid_frac: float = 0.02,
) -> Scal3RData:
    """Read a Scal3R run directory.

    Args:
        path: Run root, or the directory holding ``intri.yml``.
        image_dir: Directory with the original-resolution images. Defaults to
            ``<run>/images``; scal3r writes only depth, so this usually has to
            point at wherever the source images live.
        depth_dir: Defaults to ``<scal3r>/depths``.
        min_valid_frac: Frames whose depth is mostly invalid are dropped.
    """
    root = find_scal3r_root(path)
    run_root = root.parent if root.name in ("scal3r", "s3r") else root

    intri = _read_opencv_yaml(root / "intri.yml")
    extri = _read_opencv_yaml(root / "extri.yml")
    names: List[str] = list(intri["names"])  # type: ignore[arg-type]
    if list(extri["names"]) != names:  # type: ignore[arg-type]
        raise Scal3RError("intri.yml and extri.yml list different frames")

    name_map = _load_name_map(run_root) or _load_name_map(root)
    manifest = _image_manifest(run_root) or _image_manifest(root)
    if not name_map and manifest:
        name_map = manifest

    img_dir = Path(image_dir) if image_dir else _default_image_dir(run_root)
    dep_dir = Path(depth_dir) if depth_dir else (root / "depths")

    notes: List[str] = []
    if name_map:
        notes.append(
            f"frame names resolved through a mapping ({len(name_map)} entries); "
            "Camera.name uses the original filename so cross-source matching works"
        )
    else:
        notes.append(
            "no name_map.json / image_manifest.txt found; frame names stay as "
            "Scal3R's zero-padded indices. Cross-source matching will fail if the "
            "other source uses original filenames."
        )

    c2ws: List[np.ndarray] = []
    Ks: List[np.ndarray] = []
    depth_paths: List[Optional[Path]] = []
    image_paths: List[Path] = []
    kept: List[str] = []
    depth_hw: Optional[Tuple[int, int]] = None
    dropped_low_valid = 0
    missing_image = 0

    Kd: Dict[str, np.ndarray] = intri["K"]  # type: ignore[assignment]
    Rot: Dict[str, np.ndarray] = extri.get("Rot", {})  # type: ignore[assignment]
    T: Dict[str, np.ndarray] = extri.get("T", {})  # type: ignore[assignment]
    if not Rot or not T:
        raise Scal3RError(
            "extri.yml lacks Rot_/T_. Fall back to mat.txt only if you accept "
            "its non-rigid rotation block; this reader refuses to."
        )

    for name in names:
        # ---- image ------------------------------------------------
        original = name_map.get(name, name)
        img = _resolve_image(img_dir, original, name)
        if img is None:
            missing_image += 1
            continue
        # ---- pose -------------------------------------------------
        c2w = _qvec_to_rotvec_pose(Rot[name], T[name])
        # ---- depth ------------------------------------------------
        dep = _resolve_depth(dep_dir, name, depth_suffix)
        if dep is not None:
            d = read_depth(dep)
            if d is None:
                dep = None
            else:
                if depth_hw is None:
                    depth_hw = (int(d.shape[0]), int(d.shape[1]))
                elif (int(d.shape[0]), int(d.shape[1])) != depth_hw:
                    raise Scal3RError(
                        f"{dep.name}: depth grid {d.shape} differs from the first "
                        f"frame's {depth_hw}; Scal3R runs are single-resolution"
                    )
                if float((d > 0).mean()) < min_valid_frac:
                    dropped_low_valid += 1
                    dep = None
        if dep is None and depth_hw is not None:
            # Keep the frame for photometric training but skip its depth.
            pass

        # The reader keeps Scal3R's zero-padded index internally (depth files and
        # the YAML keys use it), but ``Camera.name`` must be the **original**
        # filename. Cross-source frame matching keys off the name, and using the
        # index here made a Scal3R-vs-COLMAP alignment find 0 of 185 frames with
        # no error at all -- the failure mode this note prevents.
        kept.append(original)
        c2ws.append(c2w)
        Ks.append(np.asarray(Kd[name], dtype=np.float64))
        depth_paths.append(dep)
        image_paths.append(img)

    if not kept:
        raise Scal3RError(
            f"no usable frames under {root}: {missing_image} had no image at "
            f"{img_dir}. Set image_dir to where the source images live."
        )
    if missing_image:
        notes.append(f"{missing_image}/{len(names)} frames have no image and were skipped")
    if dropped_low_valid:
        notes.append(f"{dropped_low_valid} frames had <{min_valid_frac:.0%} valid depth")

    if depth_hw is None:
        # Fall back to the intrinsics' declared size.
        H = int(next(iter(intri.get("H", {}).values()), 0))  # type: ignore[arg-type]
        W = int(next(iter(intri.get("W", {}).values()), 0))  # type: ignore[arg-type]
        if H and W:
            depth_hw = (H, W)
        else:
            raise Scal3RError("cannot determine the depth grid size")

    return Scal3RData(
        names=kept,
        cam_to_worlds=np.stack(c2ws, axis=0),
        Ks=np.stack(Ks, axis=0),
        depth_hw=depth_hw,
        depth_paths=depth_paths,
        image_paths=image_paths,
        name_map=name_map,
        notes=notes,
    )


def _default_image_dir(run_root: Path) -> Path:
    for cand in ("images", "rgb", "input"):
        p = run_root / cand
        if p.is_dir():
            return p
    return run_root / "images"


def _resolve_image(img_dir: Path, original: str, name: str) -> Optional[Path]:
    """Find an image by original filename, then by Scal3R's index."""
    for cand in (img_dir / original, img_dir / name, img_dir / f"{name}.png"):
        if cand.exists():
            return cand
    # The mapping may include a subdirectory.
    p = img_dir / Path(original)
    if p.exists():
        return p
    return None


def _resolve_depth(dep_dir: Path, name: str, suffix: str) -> Optional[Path]:
    for cand in (dep_dir / f"{name}{suffix}", dep_dir / f"{name}.exr", dep_dir / f"{name}.npy"):
        if cand.exists():
            return cand
    return None


def build_scal3r_scene(
    data: Scal3RData,
    *,
    resolution: Optional[Tuple[int, int]] = None,
    depth_scale: float = 1.0,
    max_init_points: int = 1_000_000,
    target_footprint_px: float = 1.5,
) -> Scene:
    """Turn raw Scal3R output into a :class:`Scene` in Scal3R's own gauge.

    ``resolution`` is the ``(width, height)`` to render at. When it differs from
    the depth grid, each camera keeps ``K`` for the render grid **and** a separate
    ``depth_K`` for the depth grid, which is what tells the dataset to ray-remap
    rather than plain-resize.

    The render ``K`` is derived from the depth ``K`` by the resolution ratio,
    per axis. Scal3R resizes its input to a fixed short side, which changes the
    aspect ratio slightly (1920x1080 -> 518x280 gives sx=3.707, sy=3.857 for the
    aerial case), so a single scalar would tilt every ray.

    ``target_footprint_px`` is the desired projected radius of the initial
    splats, in render pixels. The DGI stride is solved from it as
    ``stride = target_footprint_px * f_depth / f_render`` (the camera-space
    depth cancels out, so this is purely a grid property). A stride that is too
    large yields splats covering many pixels: on garden a stride of 5 gave a
    measured 10.95 px median radius, which also inflates the kNN-derived 3D
    scale and interacts badly with ``grow_scale3d``. 1.5 px is a good default;
    the point budget is applied afterwards as an upper bound on the stride.
    """
    cameras: List[Camera] = []
    sizes = set()
    for i, name in enumerate(data.names):
        img_path = data.image_paths[i]
        file_size = probe_image_size(img_path)
        if file_size is None:
            raise Scal3RError(f"cannot read image size for {img_path}")
        fw, fh = file_size

        if resolution is None:
            render_w, render_h = fw, fh
        else:
            render_w, render_h = int(resolution[0]), int(resolution[1])
        if render_w > fw or render_h > fh:
            # Never upsample: that would invent detail and, more importantly,
            # reintroduce a grid/intrinsics mismatch.
            render_w, render_h = fw, fh
        sizes.add((render_w, render_h))

        depth_K = data.Ks[i]
        dh, dw = data.depth_hw
        # Derive the render K from the *depth* K, then scale by the ratio
        # between the render grid and the depth grid.
        #
        # Not by the image-to-render ratio: the depth K already describes a
        # different grid from the image (518x322 vs 1297x840 here), so scaling it
        # by 324/1297 produced fx=94.91 where the correct value is 237.63 -- a
        # factor of 2.5. The ray remap then mapped most destination pixels
        # outside the source image, and only 15.5% of depth samples survived.
        #
        # Per axis, because Scal3R resizes its input to a fixed short side and
        # changes the aspect ratio (sx != sy).
        sx_depth = render_w / float(dw)
        sy_depth = render_h / float(dh)
        K_render = np.array(depth_K, dtype=np.float64, copy=True)
        K_render[0, :] *= sx_depth
        K_render[1, :] *= sy_depth

        cameras.append(
            Camera(
                name=data.names[i],
                image_path=img_path,
                K=K_render,
                width=render_w,
                height=render_h,
                cam_to_world=data.cam_to_worlds[i],
                depth_K=depth_K,
                depth_hw=(dh, dw),
                depth_path=(
                    data.depth_paths[i] if depth_scale == 1.0 else data.depth_paths[i]
                ),
                depth_scale=depth_scale,
                index=i,
            )
        )

    if len(sizes) > 1:
        raise Scal3RError(
            f"frames resolve to {len(sizes)} different render sizes {sorted(sizes)}; "
            "pass an explicit resolution="
        )

    # Initial cloud from depth (DGI), since Scal3R supplies no sparse points.
    from .dgi import DgiConfig, build_dgi_cloud

    depths, Ks, c2ws, imgs = [], [], [], []
    for cam in cameras:
        if cam.depth_path is None:
            continue
        d = read_depth(cam.depth_path)
        if d is None:
            continue
        depths.append(d * cam.depth_scale)
        Ks.append(cam.depth_K)
        c2ws.append(cam.cam_to_world)
        import imageio.v2 as imageio

        imgs.append(imageio.imread(str(cam.image_path)))

    # Choose the pixel stride from the *projected footprint*, then cap by budget.
    #
    # The footprint is what decides whether the initial cloud can render at all.
    # For a depth pixel at camera-space depth z, the 3D spacing between adjacent
    # samples is ``stride * z / f_depth``, and projecting that back through the
    # render camera gives
    #
    #     radius_px = f_render * spacing / z = stride * f_render / f_depth
    #
    # Note the z cancels: the radius depends only on the focal ratio. That makes
    # it a pure *grid* property, which is why it is the right quantity to
    # control here rather than the point count.
    #
    # Measured on garden (f_render 961.4 at 1297 wide, f_depth 379.9):
    #   stride=1 -> 2.5 px,  stride=5 -> 12.6 px  (the 10.95 px observed)
    # A 12.6 px initial splat is far too coarse; it also inflates the kNN-derived
    # 3D scale, which then interacts badly with `grow_scale3d`. A stride giving
    # ~1.5-2 px is the target.
    stride = 1
    if depths:
        depth_hw = depths[0].shape
        # Representative focal of the depth grid (mean of fx/fy is enough: the
        # per-axis difference is a few percent and this only picks a stride).
        f_depth = float(np.mean([Ks[0][0, 0], Ks[0][1, 1]]))
        # Render focal from the first camera, which the caller has already
        # scaled to the render grid.
        f_render = float(np.mean([cameras[0].K[0, 0], cameras[0].K[1, 1]]))
        if f_depth > 0 and f_render > 0:
            # target_footprint_px is the desired projected radius of the initial
            # splats.
            stride = max(1, int(round(target_footprint_px * f_depth / f_render)))
        # Budget cap: whatever the footprint says, do not exceed max_init_points.
        total_px = sum(int(d.size) for d in depths)
        budget_stride = (
            max(1, int(np.ceil(np.sqrt(total_px / float(max_init_points)))))
            if total_px > max_init_points
            else 1
        )
        stride = max(stride, budget_stride)

    xyz, rgb = (
        build_dgi_cloud(
            depths, Ks, c2ws, imgs,
            cfg=DgiConfig(stride=stride),
            max_points=max_init_points,
        )
        if depths
        else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    )

    scene = Scene(
        cameras=cameras,
        points=xyz,
        point_rgb=rgb,
        notes=list(data.notes)
        + [
            f"init cloud: DGI from {len(depths)} depth maps "
            f"({xyz.shape[0]} points, stride {stride})"
        ],
    )
    return scene


__all__ = [
    "Scal3RData",
    "Scal3RError",
    "read_scal3r",
    "find_scal3r_root",
    "build_scal3r_scene",
]
