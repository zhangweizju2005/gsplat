"""End-to-end tests on a synthetic COLMAP dataset.

A synthetic scene with a *known* answer catches wiring errors that a smoke test
on real data cannot, because every intermediate quantity is checkable. The
fixture is an analytic ground plane seen by a ring of cameras, so:

* the image and its z-depth agree by construction;
* uncovered pixels (rays that miss the finite plane patch) exist on purpose,
  which is what exercises the two loss bugs found during development;
* a "do nothing" PSNR baseline is computable, so a training score can be judged
  against it rather than against zero.

The tests are grouped by the failure they guard. Each one failed before its fix
and the numbers in the docstrings are the measured pre-fix values.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("imageio.v2")
pytest.importorskip("cv2")

from surfel.config import build_config  # noqa: E402
from surfel.data import parse_colmap, read_colmap_model  # noqa: E402
from surfel.data.dataset import SceneDataset, collate  # noqa: E402
from surfel.model import (  # noqa: E402
    Renderer2DGS,
    create_splats_2dgs,
    derive_depth_range,
)
from surfel.tests.fixtures import make_synthetic_colmap  # noqa: E402
from surfel.train import losses as L  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2DGS rasteriser needs CUDA"
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    return make_synthetic_colmap(root, n_frames=10, width=160, height=120, with_depth=True)


def _cfg(root: Path, tmp_path: Path, **overrides: str):
    base = [
        f"data.path={root}",
        "data.init_points=sparse",
        "data.split_mode=experiment",
        "data.eval_ratio=0.2",
        "data.depth.enabled=true",
        "data.depth.dir=depths",
        "data.depth.weight=0.05",
        "model.sh_degree=0",
        f"train.out_dir={tmp_path / 'out'}",
    ]
    base.extend(f"{k}={v}" for k, v in overrides.items())
    return build_config(overrides=base)


def _scene(cfg):
    split, report = parse_colmap(cfg)
    return split, report


def _render_setup(cfg):
    split, _ = _scene(cfg)
    scene = split.train
    init = create_splats_2dgs(
        scene.points,
        scene.point_rgb,
        sh_degree=cfg.model.sh_degree,
        scene_scale=scene.scene_scale,
        device="cuda",
    )
    params = init.to_parameter_dict()
    near, far = derive_depth_range(
        torch.from_numpy(scene.points).float().cuda(),
        torch.from_numpy(scene.cam_to_worlds()).float().cuda(),
    )
    renderer = Renderer2DGS(near_plane=near, far_plane=far)
    ds = SceneDataset(scene, cache=True, load_depth=True)
    batch = collate([ds[0]]).to("cuda")
    return params, renderer, batch, scene


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------
def test_fixture_covers_enough_of_the_frame(dataset: Path):
    """A half-empty fixture cannot discriminate: its black baseline is ~9 dB."""
    import imageio.v2 as imageio

    img = imageio.imread(dataset / "images" / "000000.png").astype(np.float32) / 255.0
    coverage = float((img.sum(-1) > 0).mean())
    assert coverage > 0.5, f"fixture covers only {coverage:.1%} of the frame"
    black_psnr = -10 * math.log10(float((img**2).mean()))
    assert black_psnr < 12.0, f"black baseline is {black_psnr:.1f} dB; too easy"


def test_fixture_depth_range_is_bounded(dataset: Path):
    """Grazing rays must not set the depth range via horizon intersections."""
    from surfel.data import read_depth

    # Look the file up rather than hard-coding a suffix: the fixture writes .npy
    # by default because some OpenCV wheels cannot write EXR.
    dep_file = next((dataset / "depths").glob("000000.*"))
    dep = read_depth(dep_file)
    assert dep is not None, f"{dep_file} did not read as depth"
    pos = dep[dep > 0]
    assert pos.size > 0
    # Measured 51.8 max before the finite-patch clip was added.
    assert pos.max() < 10.0, f"depth max {pos.max():.2f} suggests unbounded grazing hits"


def test_fixture_poses_are_rigid(dataset: Path):
    model = read_colmap_model(dataset)
    R = model.cam_to_worlds()[:, :3, :3]
    dets = np.linalg.det(R)
    assert np.allclose(dets, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def test_parser_splits_and_attaches_depth(dataset: Path, tmp_path: Path):
    cfg = _cfg(dataset, tmp_path)
    split, report = _scene(cfg)
    assert len(split.train.cameras) == 8
    assert split.val is not None and len(split.val.cameras) == 2
    assert report.depth_maps == 10
    # Depth shares the render grid here, so no ray remap should be needed.
    assert report.depth_remapped == 0
    train_names = {c.name for c in split.train.cameras}
    val_names = {c.name for c in split.val.cameras}
    assert not (train_names & val_names), "train/val views overlap"


def test_parser_normalises_gauge_and_scales_depth(dataset: Path, tmp_path: Path):
    cfg = _cfg(dataset, tmp_path)
    split, report = _scene(cfg)
    assert report.gauge is not None
    assert report.gauge.sim3 is not None
    # apply_to_depth must be recorded, not merely assumed.
    assert report.gauge.applied_to_depth is True
    # scale must have moved the cameras to roughly unit extent
    assert 0.1 < split.train.camera_extent < 10.0


# ---------------------------------------------------------------------------
# Bug #24: unmasked depth term explodes on uncovered pixels
# ---------------------------------------------------------------------------
def test_depth_mask_prevents_disparity_explosion(dataset: Path, tmp_path: Path):
    """With dense GT depth, an unmasked term reaches ~2.6e5 without protection.

    This is the same failure that killed a real MatrixCity run
    (depth_loss 0.001 -> 81,124 at step 3300).

    Two independent guards now cap that: the alpha mask (excludes pixels the
    model never covered) and ``depth_outlier_clip`` (caps any single pixel's
    contribution). The test therefore checks both, and checks that with *both*
    disabled the raw explosion is still reproducible -- otherwise it would be
    silently asserting nothing.
    """
    cfg = _cfg(dataset, tmp_path)
    params, renderer, batch, scene = _render_setup(cfg)
    out = renderer(
        params, batch.cam_to_worlds, batch.Ks, 160, 120,
        sh_degree=0, render_depth=True, distloss=False,
    )
    # Real feed-forward depth is dense: fill the holes the sparse fixture left.
    dense_gt = batch.depths.clone()
    hole = dense_gt <= 0
    dense_gt[hole] = float(dense_gt[dense_gt > 0].median())

    masked = L.depth_term(
        out.depth, dense_gt, scale=1.0, eps=1e-6, alpha=out.alphas,
        alpha_threshold=0.01, outlier_clip=None,
    )
    unmasked_uncapped = L.depth_term(
        out.depth, dense_gt, scale=1.0, eps=1e-6, alpha=out.alphas,
        alpha_threshold=0.0, outlier_clip=None,
    )
    unmasked_capped = L.depth_term(
        out.depth, dense_gt, scale=1.0, eps=1e-6, alpha=out.alphas,
        alpha_threshold=0.0, outlier_clip=10.0,
    )
    masked_v = float(masked.detach())
    raw_v = float(unmasked_uncapped.detach())
    capped_v = float(unmasked_capped.detach())

    assert masked_v < 1.0, f"masked depth term unexpectedly large: {masked_v:.3g}"
    assert raw_v > 1000 * masked_v, (
        "the unmasked, uncapped term should be far larger; if not, the fixture "
        "no longer leaves pixels uncovered and this test checks nothing"
    )
    # The clip must reduce the pathological case by orders of magnitude. It does
    # not reach the masked value, and should not be expected to: with ~27% of
    # pixels uncovered, every one of them still contributes close to the clip
    # bound, so the capped-but-unmasked term is legitimately larger than the
    # masked one. Measured: masked 0.139, uncapped 2.7e5, capped 2.78.
    assert capped_v < raw_v / 1000.0, (
        f"outlier_clip failed to tame the explosion: {capped_v:.3g} "
        f"(uncapped was {raw_v:.3g})"
    )


# ---------------------------------------------------------------------------
# Footprint compensation of the densification thresholds
# ---------------------------------------------------------------------------
def test_footprint_compensation_scales_thresholds_up_with_resolution():
    """A larger splat footprint must raise the thresholds, never lower them.

    The over-threshold fraction grows with the footprint (measured: 0.65% at
    0.634 px, 2.9% at 2.538 px, i.e. ``F ~ footprint^1.08``). A *lower* threshold
    admits *more* candidates, so scaling the threshold down would make the
    measured runaway worse rather than better -- which is exactly the sign error
    that was written into the doc before a 5e-5 arm refuted it (that arm drove
    the over-threshold fraction to 8.6% and 134k duplicates at the first refine
    event, against 12.6k at the original 2e-4).
    """
    from surfel.config import build_config
    from surfel.train.density import build_strategy, footprint_threshold_scale

    cfg = build_config(overrides=[
        "data.path=/tmp/does-not-need-to-exist",
        "density.footprint_compensate=true",
        "density.footprint_ref=0.634",
        "density.footprint_exponent=1.08",
        "density.grow_grad2d=2e-4",
    ])
    lo = footprint_threshold_scale(0.634, cfg)
    hi = footprint_threshold_scale(2.538, cfg)
    assert abs(lo - 1.0) < 1e-9, "at the reference footprint the factor must be 1"
    assert hi > 1.0, "a larger footprint must raise the threshold"
    assert abs(hi - (2.538 / 0.634) ** 1.08) < 1e-9

    # The strategy must actually receive the scaled values.
    strat, _ = build_strategy(cfg, scene_scale=1.0, measured_footprint=2.538)
    assert abs(strat.grow_grad2d - 2e-4 * hi) < 1e-12
    assert abs(strat.grow_scale3d - cfg.density.grow_scale3d * hi) < 1e-12
    assert abs(strat.prune_scale3d - cfg.density.prune_scale3d * hi) < 1e-12


def test_footprint_compensation_is_off_by_default_and_requires_a_measurement():
    """Existing runs must keep their exact numbers, and a missing measurement
    must be an error rather than a silent default.

    Both halves protect reproducibility: compensation changes every threshold, so
    a default-on switch would invalidate every previously recorded result, while
    a guessed footprint would silently reintroduce the resolution dependence the
    option exists to remove.
    """
    import pytest as _pytest

    from surfel.config import build_config
    from surfel.train.density import build_strategy

    off = build_config(overrides=["data.path=/tmp/x", "density.grow_grad2d=2e-4"])
    assert off.density.footprint_compensate is False
    strat, _ = build_strategy(off, scene_scale=1.0)
    assert strat.grow_grad2d == off.density.grow_grad2d
    assert strat.prune_scale3d == off.density.prune_scale3d

    on = build_config(overrides=["data.path=/tmp/x", "density.footprint_compensate=true"])
    with _pytest.raises(ValueError, match="requires measured_footprint"):
        build_strategy(on, scene_scale=1.0)

    # A degenerate measurement must be rejected, not propagate as 0 or NaN.
    for bad in (0.0, -1.0, float("nan")):
        with _pytest.raises(ValueError, match="measured footprint"):
            build_strategy(on, scene_scale=1.0, measured_footprint=bad)


def test_prune_compensation_can_be_disabled_independently():
    """Raising the growth threshold alone was measured to starve densification.

    At the footprint-compensated value (9e-4 for 2.538 px) only 0.3-0.4% of
    Gaussians cleared the bar while prune kept removing 19-33k per refine event,
    so net growth stayed negative for five consecutive events and the
    dead-densifier guard fired. The prune side is the other half of the fix --
    but it stays independently switchable, because a run that *wants* a shrinking
    model is a legitimate tuning target.
    """
    from surfel.config import build_config
    from surfel.train.density import build_strategy

    cfg = build_config(overrides=[
        "data.path=/tmp/x",
        "density.footprint_compensate=true",
        "density.footprint_ref=0.634",
        "density.footprint_compensate_prune=false",
    ])
    strat, _ = build_strategy(cfg, scene_scale=1.0, measured_footprint=2.538)
    assert strat.grow_grad2d > cfg.density.grow_grad2d, "growth side still scaled"
    assert strat.prune_scale3d == cfg.density.prune_scale3d, "prune left alone"


def test_depth_weight_uses_geometric_interpolation():
    """GSL uses ``init * final_factor ** t``, not a linear blend.

    Every archived config sets ``final_factor: 1.0``, so those runs hold the
    weight **constant** at 0.05. A linear implementation would report the same
    endpoints and a badly different middle (0.525 vs 0.2236 at t=0.5).
    """
    from surfel.config import build_config as _bc
    from surfel.train.losses import resolve_weights as _rw

    ramp = _bc(overrides=[
        "data.path=/tmp/x", "data.depth.enabled=true", "data.depth.dir=d",
        "data.depth.weight=0.05", "data.depth.weight_final=1.0",
        "data.depth.ramp_steps=7000",
    ])
    midpoint = _rw(ramp, 3500).depth
    assert abs(midpoint - 0.2236) < 1e-3, (
        f"midpoint {midpoint:.4f} is not the geometric value 0.2236; a linear "
        "blend would give 0.525"
    )
    assert abs(_rw(ramp, 0).depth - 0.05) < 1e-9
    assert abs(_rw(ramp, 7000).depth - 1.0) < 1e-9

    # The archived recipes' actual behaviour: constant.
    const = _bc(overrides=[
        "data.path=/tmp/x", "data.depth.enabled=true", "data.depth.dir=d",
        "data.depth.weight=0.05", "data.depth.weight_final=0.05",
        "data.depth.ramp_steps=7000",
    ])
    for step in (0, 3500, 7000):
        assert abs(_rw(const, step).depth - 0.05) < 1e-9


def test_depth_alpha_mask_default_is_nonzero():
    """The default has to be non-zero, or every dense-depth run explodes."""
    from surfel.config.schema import LossConfig

    assert LossConfig().depth_alpha_mask > 0.0


# ---------------------------------------------------------------------------
# Bug #26: surf_normals must be detached or backward yields NaN
# ---------------------------------------------------------------------------
def test_normal_term_backward_is_finite(dataset: Path, tmp_path: Path):
    """Undetached surf_normals gives inf gradients via clamped sqrt backward."""
    cfg = _cfg(dataset, tmp_path)
    params, renderer, batch, scene = _render_setup(cfg)
    out = renderer(
        params, batch.cam_to_worlds, batch.Ks, 160, 120,
        sh_degree=0, render_depth=True, distloss=True,
    )
    assert not torch.isnan(out.surf_normals).any(), "forward already has NaN"

    term = L.normal_term(out.normals, out.surf_normals, out.alphas)
    term.backward()
    for name in ("means", "scales", "quats", "opacities"):
        g = params[name].grad
        assert g is not None, f"{name} received no gradient"
        assert not torch.isnan(g).any(), f"{name}.grad contains NaN"
        assert not torch.isinf(g).any(), f"{name}.grad contains inf"


# ---------------------------------------------------------------------------
# Bug #27: the normal term must be alpha-weighted, not merely alpha-masked
# ---------------------------------------------------------------------------
def test_normal_term_is_alpha_normalised(dataset: Path, tmp_path: Path):
    """A mostly-empty frame must not read as a mostly-inconsistent surface."""
    cfg = _cfg(dataset, tmp_path)
    params, renderer, batch, scene = _render_setup(cfg)
    out = renderer(
        params, batch.cam_to_worlds, batch.Ks, 160, 120,
        sh_degree=0, render_depth=True, distloss=True,
    )
    value = float(L.normal_term(out.normals, out.surf_normals, out.alphas).detach())
    uncovered = float((out.alphas <= 1e-3).float().mean())
    assert uncovered > 0.05, "fixture no longer leaves pixels uncovered"
    # Pre-fix this sat at 0.958, i.e. approximately `uncovered + small`.
    assert value < 0.95, (
        f"normal term {value:.3f} still looks dominated by the {uncovered:.1%} "
        "of uncovered pixels"
    )


# ---------------------------------------------------------------------------
# Bug #25: the dead-densifier guard must count refine events, not steps
# ---------------------------------------------------------------------------
def test_dead_densifier_counts_refine_events_only(dataset: Path, tmp_path: Path):
    from surfel.train.density import DeadDensifierError, InstrumentedStrategy, build_strategy

    cfg = _cfg(dataset, tmp_path, **{
        "density.refine_start_iter": "10",
        "density.refine_stop_iter": "100",
        "density.refine_every": "25",
        "density.dead_check_events": "2",
    })
    split, _ = _scene(cfg)
    params, renderer, batch, scene = _render_setup(cfg)
    from surfel.model import build_optimizers

    init = create_splats_2dgs(
        scene.points, scene.point_rgb, sh_degree=0,
        scene_scale=scene.scene_scale, device="cuda",
    )
    params = init.to_parameter_dict()
    opts = build_optimizers(init, batch_size=1)

    strategy, kw = build_strategy(cfg, scene_scale=scene.scene_scale)
    inst = InstrumentedStrategy(
        strategy,
        threshold=cfg.density.grow_grad2d,
        dead_check_events=cfg.density.dead_check_events,
    )
    inst.check_sanity(params, opts)
    state = inst.initialize_state(**kw)

    # Walk steps 1..60. Only steps 25, 50 are refine events; if the guard counted
    # every step it would fire at step 2 with dead_check_events=2.
    fired = None
    for step in range(1, 61):
        out = renderer(
            params, batch.cam_to_worlds, batch.Ks, 160, 120,
            sh_degree=0, render_depth=True, distloss=False,
        )
        inst.step_pre_backward(params, opts, state, step, out.meta)
        weights = L.LossWeights(
            ssim_lambda=0.2, depth=0.0, distortion=0.0, normal=0.0,
            opacity_reg=0.0, scale_reg=0.0,
        )
        bd = L.total_loss(
            render=out, pixels=batch.images, gt_depth=None, weights=weights,
            cfg=cfg, params=params, scene_scale=scene.scene_scale,
        )
        bd.total.backward()
        try:
            inst.step_post_backward(params, opts, state, step, out.meta, packed=cfg.model.packed)
        except DeadDensifierError as exc:
            fired = (step, str(exc))
            break
        for opt in opts.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

    if fired is not None:
        step, msg = fired
        # Legitimately firing requires >= dead_check_events refine events to have
        # happened, i.e. step >= 50 with refine_every=25.
        assert step >= 50, (
            f"guard fired at step {step}, before two refine events occurred; "
            f"it is counting steps rather than events. Message: {msg}"
        )
    assert inst.stats.events >= 1, "no refine events were observed at all"


# ---------------------------------------------------------------------------
# Gauge direction guard (a) -- the real defence against double-transform errors
# ---------------------------------------------------------------------------
def test_sim3_direction_guard_rejects_wrong_side():
    """A Sim(3) must refuse to move data with a transform fitted from that side.

    This is the API-level fix for a failure that hit this project three times
    and is unobservable numerically: applying Scal3R->COLMAP to COLMAP poses
    scales the scene by s, which normalisation then divides straight back out.
    On garden that pushed the cameras 5.13x too far, shrank every splat below
    one pixel and left 41% of the frame uncovered (PSNR 12.3 instead of ~24).

    The two helpers check *different* ends of the mapping, and getting that
    backwards makes the correct call crash while the wrong one is accepted:
    ``apply_to_scal3r_poses`` checks ``self.source``, ``apply_to_reference_poses``
    checks ``self.target``.
    """
    import numpy as np

    from surfel.data.gauge import Sim3, estimate_sim3

    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    sim = estimate_sim3(src, 3.0 * src + np.array([1.0, 0.0, 0.0]),
                        source="scal3r", target="colmap")

    pose = np.eye(4)[None]

    # -- the normal case: depth needs gauge conversion, poses do not ---------
    # Source-side helper moves the array...
    out = sim.apply_to_scal3r_poses(pose)
    assert out.shape == pose.shape
    assert abs(np.linalg.det(out[0, :3, :3]) - 1.0) < 1e-9, "rotation must stay rigid"
    assert np.allclose(out[0, :3, 3], [1.0, 0.0, 0.0], atol=1e-9), "walked to target"

    # ...while the reference-side helper returns its input untouched. This is
    # the call that must NOT raise: its target is colmap, which is exactly the
    # gauge those poses are already in.
    same = sim.apply_to_reference_poses(pose)
    assert np.array_equal(same, pose), "reference poses must pass through unchanged"

    # -- reversed fit: the realistic, silent failure -------------------------
    # Computing the mapping the other way round and then using it as if it went
    # scal3r -> colmap scales the scene by 1/s. Nothing about the result is
    # malformed, so only the direction check can catch it.
    rev = estimate_sim3(3.0 * src + np.array([1.0, 0.0, 0.0]), src,
                        source="colmap", target="scal3r")
    with pytest.raises(ValueError, match="source is 'colmap'"):
        rev.apply_to_scal3r_poses(pose)
    with pytest.raises(ValueError, match="target is not 'colmap'"):
        rev.apply_to_reference_poses(pose)

    # Its inverse is the usable form, and both helpers accept it.
    back = rev.inverse()
    assert back.source == "scal3r" and back.target == "colmap"
    assert np.allclose(back.apply_to_scal3r_poses(pose)[0, :3, 3], [1.0, 0.0, 0.0],
                       atol=1e-9)
    assert np.array_equal(back.apply_to_reference_poses(pose), pose)

    # -- undirected transforms are refused by both ---------------------------
    plain = Sim3(R=np.eye(3), s=2.0, t=np.zeros(3))
    for fn in (plain.apply_to_scal3r_poses, plain.apply_to_reference_poses):
        with pytest.raises(ValueError, match="no recorded direction"):
            fn(pose)

    # The unguarded escape hatch stays available, with no check to trip.
    assert np.allclose(plain.apply_cam_to_world(pose)[0, :3, 3], np.zeros(3))


def test_sim3_inverse_swaps_direction():
    import numpy as np

    from surfel.data.gauge import estimate_sim3

    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    dst = 2.5 * src + np.array([0.3, -1.0, 2.0])
    sim = estimate_sim3(src, dst, source="scal3r", target="colmap")
    inv = sim.inverse()
    assert inv.source == "colmap" and inv.target == "scal3r"
    assert np.allclose(inv.apply_points(dst), src, atol=1e-9)


# ---------------------------------------------------------------------------
# Footprint-driven DGI stride (c)
# ---------------------------------------------------------------------------
def test_projected_footprint_matches_focal_ratio():
    """DGI stride must be chosen from the focal ratio, not the point budget.

    Adjacent depth pixels are ``stride * z / f_depth`` apart in 3D; projecting
    that through the render camera gives ``stride * f_render / f_depth``
    (the z cancels). Measured on garden: stride 5 -> 12.6 px predicted, 10.95 px
    observed -- a splat covering ten pixels, which also inflates the kNN-derived
    3D scale that `grow_scale3d` thresholds against.
    """
    import numpy as np

    from surfel.data.gauge import projected_footprint_px

    f_depth, f_render = 380.0, 960.0
    z = 4.0
    # Lay points on a regular grid in the z=const plane with an exact spacing,
    # so the footprint statistic has a known answer: adjacent samples are
    # `spacing` apart, and the camera sees them from depth z.
    n = 40
    for stride in (1, 3):
        spacing = stride * z / f_depth
        xs = (np.arange(n) - n / 2) * spacing
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=-1)

        c2ws = np.stack([np.eye(4)] * 4)
        K = np.array([[f_render, 0.0, 0.0], [0.0, f_render, 0.0], [0.0, 0.0, 1.0]])
        Ks = np.stack([K] * 4)
        stats = projected_footprint_px(c2ws, pts, Ks)
        expected = stride * f_render / f_depth
        # The statistic uses an RMS-over-4-NN spacing, which for a square grid is
        # sqrt(2)*spacing-ish, so allow a generous band but require the *trend*.
        assert 0.3 * expected < stats["radius_p50"] < 3.0 * expected, (
            f"stride {stride}: radius {stats['radius_p50']:.2f} px, "
            f"expected ~{expected:.2f}"
        )
        assert stats["visible_frac"] > 0.99


def test_projected_footprint_does_not_inflate_on_large_clouds():
    """Footprint spacing must be measured on the full cloud, not a subsample.

    Regression for a real misdiagnosis: on garden the render resolution was
    queried through ``projected_footprint_px`` and it reported 8.51 px at
    1297x840 where the analytic answer is 2.54 px. Two independent errors
    stacked:

    * the nearest-neighbour spacing was measured on the same 50k subsample used
      for the camera loop, which is 3.1x sparser than the 1.5M-point cloud
      (sqrt(1.5M/50k) = 5.5x in the ideal case);
    * it used the median nearest distance, while ``create_splats_2dgs`` uses the
      RMS over the k-1 nearest distances.

    The consequence was not cosmetic: 8.51 px reads as "splats are 8 px wide,
    densification thresholds are wrong", pointing at the seed configuration,
    when the cloud was in fact correct and the *diagnostic* was broken.

    A cloud far above the sampling limit must therefore report the same
    footprint as a small one with identical density.
    """
    import numpy as np

    from surfel.data.gauge import projected_footprint_px

    f_depth, f_render, z = 380.0, 962.0, 5.0
    spacing = z / f_depth
    K = np.array([[f_render, 0.0, 0.0], [0.0, f_render, 0.0], [0.0, 0.0, 1.0]])

    def footprint(n_per_side):
        xs = (np.arange(n_per_side) - n_per_side / 2) * spacing
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=-1)
        c2ws = np.eye(4)[None]
        return projected_footprint_px(c2ws, pts, K[None], max_points=50_000)["radius_p50"]

    small = footprint(60)          # 3.6k points: below the sampling limit
    large = footprint(700)         # 490k points: above it, must not change
    expected = f_render / f_depth  # 2.53 px for a stride-1 depth grid
    assert abs(small - large) / small < 0.02, (
        f"footprint changed with cloud size ({small:.3f} -> {large:.3f} px); "
        "the spacing is being measured on the subsample instead of the cloud"
    )
    # A square grid's RMS-over-3-NN spacing is 1.0x the grid step along an axis
    # and sqrt(2)x along the diagonal, so the RMS lands in (1.0, 1.5) x expected.
    assert 0.9 * expected < large < 1.6 * expected, (
        f"footprint {large:.3f} px outside the analytic band around {expected:.3f}"
    )


def test_projected_footprint_accepts_single_shared_intrinsic():
    """A COLMAP model with one camera shared by every frame must not crash.

    ``Ks`` was indexed with the camera loop's index whenever it was 3-D, so a
    ``[1,3,3]`` array raised ``IndexError: index 11 is out of bounds`` on the
    first frame past 0. Shared intrinsics are the normal case for a single-camera
    capture, so this is the common path, not an edge case.
    """
    import numpy as np

    from surfel.data.gauge import projected_footprint_px

    n = 200
    spacing = 0.01
    xs = (np.arange(n) - n / 2) * spacing
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 4.0)], axis=-1)

    c2ws = np.stack([np.eye(4)] * 12)          # 12 cameras ...
    K = np.array([[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 1.0]])
    stats = projected_footprint_px(c2ws, pts, K[None])   # ... one shared K
    assert np.isfinite(stats["radius_p50"]) and stats["radius_p50"] > 0.0
    assert stats["visible_frac"] > 0.99

    # Per-camera intrinsics must still work.
    stats2 = projected_footprint_px(c2ws, pts, np.stack([K] * 12))
    assert abs(stats2["radius_p50"] - stats["radius_p50"]) < 1e-9


def test_consistency_report_includes_absolute_camera_radius():
    """The absolute scale must be reported: it is the only discriminating signal.

    Every internal ratio moves by <25% between a correct and a
    double-transformed scene (measured), so a caller needs the absolute number
    to compare against a reference run.
    """
    import numpy as np

    from surfel.data.gauge import check_pose_point_consistency
    from surfel.data.scene import Camera, Scene

    cams = []
    for i in range(4):
        c2w = np.eye(4)
        c2w[:3, 3] = [2.0 * i, 0.0, 4.0]
        cams.append(
            Camera(name=f"f{i}", image_path=__file__, K=np.eye(3) * 100,
                   width=64, height=64, cam_to_world=c2w, index=i)
        )
    rng = np.random.default_rng(0)
    pts = np.concatenate([
        rng.normal(size=(3000, 3)) * 0.2 + np.array([0.0, 0.0, 4.0])
    ])
    scene = Scene(cameras=cams, points=pts, point_rgb=np.zeros_like(pts))
    stats = check_pose_point_consistency(scene)
    assert "camera_radius" in stats
    assert stats["camera_radius"] > 0
    assert "ok" in stats and "reasons" in stats


# ---------------------------------------------------------------------------
# means learning rate: the gauge-free form, and the decay that never ran
# ---------------------------------------------------------------------------
def test_means_per_spacing_overrides_means_and_is_gauge_free():
    """``train.lr.means_per_spacing`` must land on exactly rel * kNN spacing.

    ``means`` is a length, so its raw value cannot transfer between scenes: Adam's
    step is ~lr * sign/sqrt(v), dimensionless for log-space parameters but not for
    ``means``. Measured on garden, the config-derived value is 0.119 kNN
    spacings/step while GSL's independently tuned MatrixCity run sits at 0.0104,
    and the difference is worth 2 dB (23.06 vs 25.09). Stating the rate in the
    point cloud's own units makes the two comparable.

    The assertion is on the *absolute* LR the optimiser ends up with, because that
    is where two scalings compose: the config value is divided by scene_scale here
    and multiplied again inside create_splats_2dgs. Getting that double scaling
    wrong would silently shift the LR by scene_scale^2.
    """
    import tempfile

    import torch

    from surfel.config import build_config
    from surfel.config.profiles import make_profile_config
    from surfel.data import parse_colmap
    from surfel.model.splats import knn_mean_distance_rms
    from surfel.tests.fixtures import make_synthetic_colmap
    from surfel.train import Trainer

    root = tempfile.mkdtemp(prefix="lrspacing_")
    make_synthetic_colmap(root, n_frames=8, width=64, height=48)

    cfg = make_profile_config("smoke")
    cfg.data.path = str(root)
    cfg.gauge.scale_source = "extent"
    cfg.data.split_mode = "experiment"
    cfg.train.max_steps = 2
    cfg.train.eval_steps = []
    cfg.train.save_steps = []
    cfg.train.lr.decay_means = False
    cfg.train.lr.means_per_spacing = 0.02
    cfg.train.out_dir = tempfile.mkdtemp(prefix="lrspacing_out_")
    cfg.validate()
    split, _ = parse_colmap(cfg)

    tr = Trainer(cfg, split, log_fn=None)
    realised = tr.optimizers["means"].param_groups[0]["lr"]
    spacing = float(
        knn_mean_distance_rms(
            torch.as_tensor(split.train.points, dtype=torch.float32)
        ).median()
    )
    assert abs(realised - 0.02 * spacing) / (0.02 * spacing) < 1e-3, (
        f"means_per_spacing 0.02 with spacing {spacing:.5g} should give "
        f"{0.02 * spacing:.5g}, got {realised:.5g}"
    )


def test_means_decay_ratio_keeps_profiles_working():
    """``decay_means=True`` used to be a silent no-op in every profile.

    The schedule is only built when a final value exists, and ``means_final``
    defaulted to None -- so aerial/aerial_long/object all asked for decay and got
    none, for the entire life of those profiles. The ratio form fixes it without
    hard-coding 1.6e-6, which is GSL's endpoint after *its* 30000 steps and is not
    transferable to a 7000-step run.
    """
    import pytest as _pytest

    from surfel.config.profiles import make_profile_config

    for name in ("aerial", "aerial_long", "object"):
        cfg = make_profile_config(name)
        cfg.data.path = "/tmp/x"
        cfg.validate()  # must not raise: a profile with decay must be runnable
        assert cfg.train.lr.decay_means is True
        assert 0.0 < cfg.train.lr.means_final_ratio < 1.0

    bad = make_profile_config("aerial")
    bad.data.path = "/tmp/x"
    bad.train.lr.means_final_ratio = 1.5
    with _pytest.raises(ValueError, match="means_final_ratio"):
        bad.validate()


def test_means_and_means_per_spacing_together_are_rejected():
    """Two ways to state the same rate must not both be accepted silently.

    They differ by more than 10x on garden (0.010 vs 0.119 spacings/step), so
    accepting both and picking one would make a config file read as something it
    is not.
    """
    import pytest as _pytest

    from surfel.config.profiles import make_profile_config

    cfg = make_profile_config("smoke")
    cfg.data.path = "/tmp/x"
    cfg.train.lr.means_per_spacing = 0.02
    cfg.train.lr.means = 1e-5  # non-default: conflicts
    with _pytest.raises(ValueError, match="means_per_spacing wins"):
        cfg.validate()


def test_means_lr_actually_decays_during_training():
    """Measure the optimiser's LR before and after, not just the config.

    The previous test only asserts that profiles validate; it passed even with the
    original bug restored (`if decay_means and means_final:` -- always false),
    because a config can be valid while the training loop ignores it. This one
    runs the trainer and reads the parameter group, which is the only place a
    silently-skipped schedule is visible.
    """
    import tempfile

    from surfel.config.profiles import make_profile_config
    from surfel.data import parse_colmap
    from surfel.tests.fixtures import make_synthetic_colmap
    from surfel.train import Trainer

    root = tempfile.mkdtemp(prefix="decay_")
    make_synthetic_colmap(root, n_frames=8, width=64, height=48)

    cfg = make_profile_config("smoke")
    cfg.data.path = str(root)
    cfg.gauge.scale_source = "extent"
    cfg.data.split_mode = "experiment"
    cfg.train.max_steps = 30
    cfg.train.eval_steps = []
    cfg.train.save_steps = []
    cfg.train.lr.decay_means = True
    cfg.train.lr.means_final = None          # exercise the ratio path
    cfg.train.lr.means_final_ratio = 0.01
    cfg.out_dir = tempfile.mkdtemp(prefix="decay_out_")
    cfg.train.out_dir = cfg.out_dir
    cfg.validate()
    split, _ = parse_colmap(cfg)

    tr = Trainer(cfg, split, log_fn=None)
    lr_init = tr.optimizers["means"].param_groups[0]["lr"]
    tr.fit()
    lr_end = tr.optimizers["means"].param_groups[0]["lr"]

    assert lr_end < lr_init, (
        f"means LR did not decay: {lr_init:.4g} -> {lr_end:.4g}. The schedule is "
        "not being built, which is the silent no-op this test exists to catch."
    )
    # 30 of 30 steps under an exponential schedule should approach the ratio.
    ratio = lr_end / lr_init
    assert 0.005 < ratio < 0.5, (
        f"decayed to {ratio:.4g} of the initial LR over the full run; expected "
        "roughly 0.01 for means_final_ratio=0.01"
    )


def test_aerial_profile_encodes_the_measured_best_arm():
    """Pin the profile to the values a measurement actually selected.

    The 25.685 dB garden arm was assembled from explicit CLI overrides. A profile
    that does not embed those values would make "aerial" a name for something
    nobody measured -- and would silently reintroduce whichever defaults happen
    to be in the dataclasses.

    The two values that matter are recorded here with their measurements so a
    future edit has to confront the evidence:

      * means_per_spacing 0.01: 24.92 dB at this rel_step in the sweep, and the
        point where GSL's f0_gtpose also sits. The absolute default (1.6e-4) is
        0.119 spacings/step, which measured 22.52 dB.
      * grow_grad2d 2e-4 / grow_scale3d 0.003: at the corrected LR these beat the
        archived GSL values 6e-4 / 0.01 by 0.67 dB (25.685 vs 25.013). With the
        wrong LR they were indistinguishable (22.52 vs 22.744), which is why the
        earlier §14.5 conclusion read the threshold as the problem.

    footprint_compensate must stay False: it was tried and is counterproductive
    once the LR is right.
    """
    from surfel.config.profiles import make_profile_config

    cfg = make_profile_config("aerial")
    cfg.data.path = "/tmp/x"
    cfg.validate()

    assert cfg.train.lr.means_per_spacing == 0.01
    assert cfg.density.grow_grad2d == 2e-4
    assert cfg.density.grow_scale3d == 0.003
    assert cfg.density.footprint_compensate is False
    # Decay must be configured so it actually runs.
    assert cfg.train.lr.decay_means is True
    assert 0.0 < cfg.train.lr.means_final_ratio < 1.0


def test_every_real_profile_states_a_gauge_free_means_lr():
    """No production profile may inherit the dimensionful default.

    The sweep found the absolute default (1.6e-4) sits at 0.119 kNN
    spacings/step on garden and scores ~3 dB below 0.010. That fix was first
    applied only to ``aerial``, leaving ``aerial_long`` -- the profile sized for a
    30000-step production run -- with the bad value, which is the kind of
    partial rollout a per-profile test catches and a per-dataset sweep does not.

    ``smoke`` is exempt by design: it runs 200 steps as a wiring check and its
    numbers are meaningless.
    """
    from surfel.config.profiles import make_profile_config, profile_names

    for name in profile_names():
        cfg = make_profile_config(name)
        cfg.data.path = "/tmp/x"
        cfg.gauge.scale_source = "extent"
        cfg.validate()
        if name == "smoke":
            assert cfg.train.lr.means_per_spacing is None
            continue
        assert cfg.train.lr.means_per_spacing is not None, (
            f"profile {name!r} inherits train.lr.means = {cfg.train.lr.means}, "
            "which is a length and therefore not comparable across scenes. "
            "Set train.lr.means_per_spacing (0.01 was measured best on garden)."
        )
        assert 0.001 <= cfg.train.lr.means_per_spacing <= 0.05, (
            f"profile {name!r} uses rel_step "
            f"{cfg.train.lr.means_per_spacing}, outside the measured-usable band "
            "(0.003-0.030; 0.1 measured 2 dB worse)"
        )
