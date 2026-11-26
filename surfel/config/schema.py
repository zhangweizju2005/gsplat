"""Configuration schema for the large-scene 2DGS trainer.

Every knob that the project log shows to be a *scale-dependent* trap is an
explicit field here rather than an inherited constant, because the failure mode
is always the same: a value tuned for an object-centric scene silently does the
wrong thing on an aerial one, and nothing raises.

Three quantities are deliberately kept separate. gsplat's own trainer conflates
them into a single ``scene_scale``, which is fine for Mip-NeRF 360 and wrong
beyond it:

``world.normalize_scale``
    The similarity factor applied to poses, points **and depth**. Brought the
    whole world to ~unit extent so that optimiser step sizes and float32
    conditioning are scale-free.

``strategy.scene_scale``
    What ``DefaultStrategy`` divides Gaussian sizes by to decide
    duplicate-vs-split. Must describe the *world extent as it is now*, i.e.
    after normalisation.

``loss.depth_scale``
    The multiplier on the disparity L1. Wants a **depth** quantile, not an
    extent: measured on MatrixCity flight0, using the camera extent put 6.8% of
    pixels into the dead zone and left only 72.8% of the usable value window,
    while ``1/d_p99`` left 92.7%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class DepthConfig:
    """Supervised depth: where it comes from and how it is compared."""

    enabled: bool = False
    #: Directory holding per-frame depth maps. Relative paths resolve against
    #: the dataset root.
    dir: Optional[str] = None
    #: File extension used when probing ``dir``; the actual per-frame suffix is
    #: taken from the file that exists.
    suffix: Optional[str] = None
    #: Optional JSON of ``{frame_name: {"scale": s, "offset": o}}``. VGGT exports
    #: this: its raw prediction is min-max normalised inverse depth, so this
    #: factor is the only thing that makes it metric.
    scale_file: Optional[str] = None
    #: Apply the per-frame ``scale``/``offset`` above. Must be ON for VGGT,
    #: OFF for a depth source already in the pose gauge (e.g. Scal3R, whose
    #: EXR is produced in the same gauge as its poses).
    rescale: bool = False

    weight: float = 0.0
    #: Target weight at ``ramp_steps``; ``None`` keeps ``weight`` constant.
    #: The archived MatrixCity recipes ramp 0.05 -> 1.0 over 7000 steps, and
    #: that ramp is worth ~1 dB against a fixed small weight.
    weight_final: Optional[float] = None
    ramp_steps: int = 7000
    #: Iteration at which supervision starts at all.
    start_iter: int = 0

    #: Divide both sides by their own per-frame median disparity before the L1.
    #: Required for a source whose *absolute* scale is meaningless per frame
    #: (VGGT). Must be OFF when the depth shares the pose gauge, because then
    #: the absolute scale is real information and normalising it away removes
    #: the only constraint tying the geometry to the cameras.
    per_frame_median_norm: bool = False

    #: Frames whose valid-depth fraction is below this are skipped.
    min_valid_frac: float = 0.02

    def validate(self) -> None:
        if self.enabled and self.dir is None:
            raise ValueError("depth.enabled=True requires depth.dir")
        if self.enabled and self.weight <= 0.0 and not self.weight_final:
            raise ValueError(
                "depth.enabled=True but both weight and weight_final are 0; "
                "the depth branch would render depth and supervise nothing"
            )
        if self.per_frame_median_norm and self.rescale:
            raise ValueError(
                "per_frame_median_norm and rescale are mutually exclusive: "
                "median normalisation already removes any per-frame scale, so "
                "applying a fitted scale first is dead work that hides the "
                "gauge question"
            )


@dataclass
class DataConfig:
    """Where the scene comes from."""

    #: "colmap" (a COLMAP model dir, optionally with depth) or "transforms".
    parser: Literal["colmap", "transforms"] = "colmap"
    path: str = ""

    #: "reconstruction" trains on every image, "experiment" holds out a split.
    #: NOTE: with "reconstruction" the reported val PSNR measures fit, not
    #: generalisation; the archived MatrixCity baselines used it, so comparing
    #: across split modes is invalid.
    split_mode: Literal["reconstruction", "experiment"] = "reconstruction"
    #: For split_mode="experiment": fraction of images held out.
    eval_ratio: float = 0.1
    #: For split_mode="experiment": hold out every N-th image instead.
    test_every: int = 0

    #: Downsample factor applied to the *images that are actually read*.
    #:
    #: This is the single most expensive parameter in the project's history: a
    #: mismatch between the render resolution and the intrinsics cost 12 dB
    #: silently once. The resolution policy in :mod:`surfel.data.resolution`
    #: exists so that ``factor`` is always relative to the COLMAP model and the
    #: intrinsics are rescaled to whatever grid is finally read.
    factor: float = 1.0
    #: Refuse to downsample below this many pixels on the long edge.
    min_long_edge: int = 0
    #: Optional absolute target long edge; overrides ``factor`` when set.
    target_long_edge: Optional[int] = None
    #: Images filename patterns to try, in order, under the dataset root.
    image_dirs: List[str] = field(
        default_factory=lambda: ["images", "rgb", "images_2", "images_4"]
    )
    #: Optional explicit image directory (overrides image_dirs).
    image_dir: Optional[str] = None

    #: Initial points: "sparse" (COLMAP points3D), "depth" (DGI from depth
    #: maps), or "none".
    init_points: Literal["sparse", "depth", "none"] = "sparse"
    #: Cap on initial points (0 = no cap).
    max_init_points: int = 0

    # -- DGI knobs (used only when init_points="depth") ---------------------
    #: Keep every N-th pixel of each depth map. 1 uses every pixel, which for
    #: a 518x280 map over 262 frames is ~38M points before capping; 2-4 is
    #: usually enough structure at a fraction of the cost.
    dgi_stride: int = 1
    #: Drop this fraction of each axis at the borders (feed-forward depth is
    #: unreliable there).
    dgi_border_trim: float = 0.0
    #: Depth range accepted by DGI, in source units. 0 and inf disable the
    #: bound respectively.
    dgi_min_depth: float = 0.0
    dgi_max_depth: float = float("inf")

    depth: DepthConfig = field(default_factory=DepthConfig)

    def validate(self, *, require_path: bool = True) -> None:
        # ``require_path=False`` is for profiles, which describe *how* to train
        # and are validated before a dataset is chosen. Everything else is
        # checked either way.
        if require_path and not self.path:
            raise ValueError("data.path is required")
        if self.target_long_edge is not None and self.target_long_edge <= 0:
            raise ValueError("data.target_long_edge must be positive")
        if self.factor <= 0:
            raise ValueError("data.factor must be positive")
        self.depth.validate()


# --------------------------------------------------------------------------
# Gauge
# --------------------------------------------------------------------------
@dataclass
class GaugeConfig:
    """How the reconstruction's arbitrary world frame is made explicit.

    A SfM or feed-forward reconstruction is defined only up to a similarity.
    Mixing a pose source with a depth source from a different one produces
    geometry that trains without error and renders wrong, so the choice is
    stated here instead of being implied by which files happened to be read.
    """

    #: "none" keeps the source gauge (COLMAP's arbitrary units).
    #: "normalize" applies a similarity bringing the scene to ~unit extent.
    mode: Literal["none", "normalize"] = "normalize"

    #: How ``normalize`` picks the scale factor:
    #:   "depth_p99"  -> c = 1 / (p99 of positive camera-space depth)
    #:   "extent"     -> c = 1 / (camera-centre extent)
    #:   "manual"     -> uses ``scale`` below
    #:
    #: ``depth_p99`` is the default because the distortion regulariser is
    #: evaluated in inverse depth: it needs the depth *distribution* to land in
    #: the kernel's usable window, and the camera extent says nothing about it.
    scale_source: Literal["depth_p99", "extent", "manual"] = "depth_p99"
    scale: Optional[float] = None

    #: Rescale the depth supervision alongside poses and points.
    #:
    #: This must be True whenever the depth is *not* per-frame-normalised, or
    #: the rendered-vs-supervised depth ratio is a constant != 1 and the
    #: disparity loss carries an offset no amount of training can remove.
    #: Measured on one strip0 arm: ratio 0.361 when the required factor was
    #: 0.471, which cost two thirds of the depth signal before step 1.
    apply_to_depth: bool = True

    #: Points in the initial cloud outside these quantiles are dropped before
    #: computing the scale. Sky and failed depth fits otherwise stretch the
    #: extent and shift every threshold derived from it.
    trim_quantile: float = 0.005

    #: Reject a fitted Sim(3) whose residual exceeds this fraction of the
    #: reference trajectory span. A similarity can always be fitted; the
    #: residual is what says "these are not the same scene".
    sim3_max_residual: float = 0.05

    def validate(self) -> None:
        if self.scale_source == "manual" and not self.scale:
            raise ValueError("gauge.scale_source='manual' requires gauge.scale")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    type: Literal["2dgs"] = "2dgs"

    sh_degree: int = 3
    #: Turn on one more SH band every this many steps (0 = use full degree
    #: from the start).
    sh_degree_interval: int = 1000

    init_opacity: float = 0.1
    init_scale: float = 1.0
    #: Multiply the kNN-derived initial scale by this. 1.0 reproduces the
    #: official trainer.
    init_scale_mult: float = 1.0

    #: Rasterisation layout. ``packed=True`` uses less memory; both are equal
    #: numerically (measured 0.02 dB apart end-to-end).
    packed: bool = False
    #: ``DefaultStrategy`` tolerance in pixels. 0.3 is hard-coded in the 2DGS
    #: kernel, so values other than 0.3 will not match the reference.
    eps2d: float = 0.3
    #: Skip Gaussians whose projected radius is at or below this.
    radius_clip: float = 0.0

    #: Near/far planes. ``auto`` derives them from the observed depth range,
    #: which matters for aerial scenes: a fixed ``1e-2 .. 1e10`` spans 12 orders
    #: of magnitude and makes near/far clipping meaningless.
    near_plane: Optional[float] = None
    far_plane: Optional[float] = None

    def validate(self) -> None:
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("model.sh_degree must be in [0, 3]")
        if self.packed and self.type != "2dgs":  # pragma: no cover - type is fixed
            raise ValueError("packed is only supported for 2dgs here")


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
@dataclass
class LossConfig:
    #: ``lerp(l1, 1-ssim, ssim_lambda)`` weight. 0.2 is the 3DGS/2DGS default.
    ssim_lambda: float = 0.2

    #: 2DGS distortion regulariser (flattens surfels onto one surface).
    #: 0.1 is the 2DGS paper value; it is what stops aerial scenes from growing
    #: a haze of parallel sheets along each building facade.
    dist_weight: float = 0.0
    dist_start_iter: int = 3000

    #: 2DGS normal consistency. This is a *self-consistency* term: it compares
    #: the rendered surfel normal against the normal implied by the model's own
    #: rendered depth. It needs **no ground-truth depth** — enabling it must not
    #: be gated on depth supervision, which was a real silent failure in this
    #: project (the term ran at weight 0 for an entire run).
    normal_weight: float = 0.0
    normal_start_iter: int = 2000

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    #: Multiplier on the disparity L1. ``None`` derives it from the scene:
    #: the median supervised camera-space depth after normalisation. See the
    #: module docstring for why this is not the camera extent.
    depth_scale: Optional[float] = None

    #: Floor on predicted depth before the reciprocal. Without one, a pixel
    #: with zero coverage produces ``1/1e-6`` and the depth term spikes by seven
    #: orders of magnitude mid-run -- the exact failure that killed a
    #: MatrixCity f0_gtpose run at step 3150.
    depth_eps: float = 1e-6

    #: Mask pixels by rendered alpha before the depth term. Excluded pixels are
    #: ones the model does not cover at all; including them compares a real
    #: depth against nothing.
    #:
    #: The default is a small positive value, NOT 0. With no mask, an uncovered
    #: pixel renders depth 0, gets clamped to ``depth_eps``, and contributes a
    #: disparity of ``1/depth_eps`` = 1e6. Measured on a synthetic scene with
    #: 576 initial points and a 160x120 image, that turned the depth term into
    #: ~1e4 and swamped the photometric loss entirely -- the same failure that
    #: killed a real MatrixCity run (depth_loss 0.001 -> 81,124 at step 3300).
    #: Masking is the physically correct thing to do: a pixel with no rendered
    #: surface has no predicted depth to compare. Set to 0.0 only to reproduce
    #: an unmasked baseline.
    depth_alpha_mask: float = 0.01

    #: Clamp the normalised disparity difference to ``±clip`` before the mean.
    #: The third line of defence after ``depth_eps`` and the alpha mask, and the
    #: one GSL actually relies on: ``depth_outlier_clip`` defaults to 10.0 there,
    #: with a comment pointing at the run where a single pixel pushed the depth
    #: term from 0.001 to 81,124. ``None`` or ``<=0`` disables it.
    depth_outlier_clip: Optional[float] = 10.0


# --------------------------------------------------------------------------
# Densification
# --------------------------------------------------------------------------
@dataclass
class DensityConfig:
    #: "default" (3DGS/2DGS split-and-clone) or "mcmc".
    strategy: Literal["default", "mcmc"] = "default"

    # -- DefaultStrategy ---------------------------------------------------
    #: Prune Gaussians below this opacity. 2DGS reference uses 0.005.
    prune_opa: float = 0.005
    #: Image-plane gradient above which a Gaussian is duplicated or split.
    #:
    #: Units matter and were verified: ``DefaultStrategy`` multiplies the
    #: kernel's ``gradient_2dgs`` by ``(W/2, H/2)`` before comparing, which
    #: reproduces the pixel-space ``means2d`` gradient the 2DGS/hbb1 backend
    #: uses. Measured on identical input, the two distributions agree to 1%
    #: (p50 6.46e-4 vs 6.54e-4), so a threshold tuned against hbb1 -- e.g. the
    #: project's 6e-4 -- transfers directly and must NOT be re-scaled.
    grow_grad2d: float = 0.0002
    #: Size boundary between duplicate and split, as a fraction of
    #: ``scene_scale``. Below -> duplicate, above -> split.
    grow_scale3d: float = 0.01
    #: Prune Gaussians larger than this fraction of ``scene_scale``.
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    refine_every: int = 100
    reset_every: int = 3000
    #: 0 disables the screen-size split/prune branches (recommended: they were
    #: disabled by a bug in the reference implementation, so matching it means
    #: leaving them off).
    refine_scale2d_stop_iter: int = 0
    #: Opacity value every Gaussian is reset to at each ``reset_every``.
    #:
    #: gsplat hard-codes this as ``prune_opa * 2`` inside the strategy and does
    #: not expose it. GSL does expose it (``opacity_reset_value``), and its GNS
    #: config uses 0.05 rather than the 0.01 that ``prune_opa*2`` would give --
    #: i.e. the value genuinely needs independent control. ``None`` keeps
    #: gsplat's default so existing numbers stay comparable.
    opacity_reset_value: Optional[float] = None
    #: Use the AbsGS absolute gradient. NOT available for 2DGS: the kernel's
    #: densify accumulator is signed and carries no absolute variant, so
    #: ``DefaultStrategy`` would raise ``AttributeError`` reading
    #: ``.absgrad``. Enabling this raises a clear error instead.
    absgrad: bool = False

    #: Rescale the gradient/size thresholds by the measured splat footprint.
    #:
    #: ``grow_grad2d`` describes a Gaussian in *screen space*: gsplat
    #: multiplies the kernel's ``gradient_2dgs`` by ``(W/2, H/2)`` before
    #: comparing, so the same physical surface produces a larger decision
    #: statistic on a larger image. ``grow_scale3d``/``prune_scale3d`` are
    #: fractions of ``scene_scale``, which is derived from the *scene*, so they
    #: do not move when the render resolution does -- while the kNN-derived
    #: initial scales do not move either. All three thresholds are therefore
    #: only comparable between runs with the same footprint,
    #: ``f_render / f_depth``.
    #:
    #: Measured on garden with one 1.5M-point cloud and one set of
    #: hyper-parameters, changing only the render resolution moved the
    #: over-threshold fraction from 0.65% to 2.9% (footprint 0.634 -> 2.538 px)
    #: and turned a shrinking model (1.50M -> 1.22M) into one that quadrupled
    #: (1.50M -> 4.73M by step 5500) while PSNR stayed flat from step 3000.
    #:
    #: When enabled, each threshold is multiplied by
    #: ``(measured_footprint / footprint_ref) ** footprint_exponent``. Only the
    #: ratio matters, so ``footprint_ref`` defaults to 1.0 and callers pass a
    #: measured footprint: this is a relative correction, never an absolute
    #: calibration.
    #:
    #: Off by default so existing numbers stay exactly reproducible.
    footprint_compensate: bool = False
    #: The footprint, in render pixels, that the configured thresholds were
    #: tuned at. Only its ratio against the measured footprint is used.
    footprint_ref: float = 1.0
    #: Exponent of the over-threshold-fraction vs footprint relation.
    #:
    #: 1.08 was measured on garden from two points (0.65% at 0.634 px, 2.9% at
    #: 2.538 px). Two points cannot pin an exponent, and values from 1.0 to 2.0
    #: shift the derived threshold by only 2.4x, so the default is the measured
    #: value rather than a round number. It is configurable precisely because it
    #: is an empirical fit; another dataset may need a different one.
    footprint_exponent: float = 1.08
    #: Compensate ``prune_scale3d`` as well as the growth thresholds.
    #:
    #: This is not optional in practice. Raising ``grow_grad2d`` alone was
    #: measured to starve densification: at the footprint-compensated value
    #: (9e-4 at 2.538 px) the over-threshold fraction fell to 0.3-0.4%, but
    #: prune still removed a fixed 19-33k per refine event while only ~5.9k
    #: candidates were admitted, so net growth stayed negative for five
    #: consecutive events and the dead-densifier guard fired. The prune side is
    #: the other half of the correction.
    footprint_compensate_prune: bool = True

    # -- MCMCStrategy ------------------------------------------------------
    #: Hard cap on Gaussian count. The main memory lever for large scenes.
    mcmc_cap_max: int = 1_000_000
    mcmc_noise_lr: float = 5e5
    mcmc_min_opacity: float = 0.005

    #: Abort if densification produces no candidate for this many consecutive
    #: refine events after ``refine_start_iter``. Guards the failure that cost
    #: this project the most: a monkey-patched ``viewspace_points`` pointing at
    #: a tensor the 1.6.0 kernel never writes made densification silently dead
    #: for an entire run, and the only symptom was an unusually small model.
    dead_check_events: int = 5
    #: Log gradient percentiles at every refine event for the first N events.
    log_first_events: int = 3

    def validate(self) -> None:
        if self.absgrad and self.strategy == "default":
            raise ValueError(
                "density.absgrad=True is not supported for 2DGS: "
                "gsplat's 2DGS rasteriser returns meta['gradient_2dgs'] without "
                "an .absgrad attribute (the kernel accumulates a signed value), "
                "so DefaultStrategy would raise AttributeError. "
                "Use the default mean-gradient threshold instead."
            )
        if self.refine_start_iter >= self.refine_stop_iter:
            raise ValueError(
                f"refine_start_iter ({self.refine_start_iter}) must be < "
                f"refine_stop_iter ({self.refine_stop_iter})"
            )
        if self.strategy == "mcmc" and self.mcmc_cap_max <= 0:
            raise ValueError("density.mcmc_cap_max must be positive")


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
@dataclass
class LRConfig:
    """Per-parameter learning rates, before any scene scaling."""

    means: float = 1.6e-4
    means_final: Optional[float] = None
    scales: float = 5e-3
    quats: float = 1e-3
    opacities: float = 5e-2
    sh0: float = 2.5e-3
    shN: float = 2.5e-3 / 20.0

    #: ``means`` LR expressed in **kNN spacings per step**, which is the only
    #: form comparable across scenes. When set, it overrides :attr:`means`.
    #:
    #: ``means`` is a length, so its raw value is meaningless without the scene
    #: scale: Adam's step is ~``lr * sign/sqrt(v)``, which is dimensionless for
    #: log-space parameters (``scales``, ``quats``, ``opacities``, ``sh``) but has
    #: units of length for ``means``. Only ``means`` therefore needs normalising,
    #: and the natural normaliser is the point cloud's own resolution -- the kNN
    #: spacing the initialiser already measured.
    #:
    #: Measured on garden 1297x840 (aerial profile otherwise fixed, 7000 steps),
    #: sweeping only this ratio:
    #:
    #: ==========  ==================  ======  ======  ==========  ==========
    #: rel_step    means_lr            PSNR    SSIM    Gaussians   dup
    #: ==========  ==================  ======  ======  ==========  ==========
    #: 0.001       1.34e-6             24.39   0.749   1.18 M      328 k
    #: 0.003       4.03e-6             24.76   0.770   1.26 M      345 k
    #: 0.010       1.34e-5             24.92   0.783   1.35 M      389 k
    #: 0.030       4.03e-5             25.09   0.783   1.46 M      564 k
    #: 0.100       1.34e-4             23.06   0.617   1.67 M      979 k
    #: ==========  ==================  ======  ======  ==========  ==========
    #:
    #: The peak is 0.010-0.030, and 0.010 is where GSL's archived MatrixCity
    #: f0_gtpose sits (8.0e-3 m/step over a 0.772 kNN spacing = 0.0104) -- the
    #: two independently tuned configurations agree once the gauge is factored
    #: out. The default-derived value without this field is 0.119, i.e. past the
    #: edge: it lands in the 0.1 row, which loses 2 dB and 0.17 SSIM while
    #: nearly tripling duplication.
    #:
    #: A second, online signal that the value is too large: split/dup falls from
    #: 0.137 at 0.001 to 0.026 at 0.1. Fast-moving centres get pushed into
    #: high-gradient regions where their footprint shrinks, so they can only
    #: duplicate. No external metric is needed to notice this.
    #:
    #: ``None`` keeps :attr:`means` as-is, so existing configs are unaffected.
    means_per_spacing: Optional[float] = None

    #: Exponential decay of ``means`` toward ``means_final`` by ``max_steps``.
    #: The official gsplat trainer does not decay; the archived project recipes
    #: do (1.6e-4 -> 1.6e-6). Profiles decide.
    #:
    #: Note ``decay_means`` alone does nothing: the schedule only runs when a
    #: final value is available.
    decay_means: bool = False

    #: Final ``means`` LR as a **fraction** of the initial one, used when
    #: :attr:`decay_means` is set and :attr:`means_final` is not.
    #:
    #: 0.01 is the ratio every archived GSL recipe uses (1.6e-6 / 1.6e-4), and
    #: it is the same endpoint gsplat's own 2DGS example reaches via
    #: ``ExponentialLR(gamma=0.01**(1/max_steps))``. The absolute final value is
    #: NOT transferable -- 1.6e-6 is the endpoint after GSL's 30000 steps, while
    #: this project's aerial profile runs 7000 -- but the ratio is, because both
    #: anchor the same ``init * ratio`` endpoint. A ratio is also the only form
    #: that survives :attr:`means_per_spacing`, where the initial LR is derived
    #: from the scene rather than configured.
    means_final_ratio: float = 0.01


@dataclass
class TrainConfig:
    max_steps: int = 30000
    #: Multiplies all step counts; the single knob for "train longer".
    steps_scaler: float = 1.0
    batch_size: int = 1
    lr: LRConfig = field(default_factory=LRConfig)

    #: Multiply ``means`` LR by this before training. 1.0 is the design value
    #: and is what the normalised-gauge contract assumes. When the world is
    #: normalised (the default) there is no reason to deviate: the measured
    #: MatrixCity sweep found that scaling this by the raw scene extent
    #: overshot and lost 0.6 dB, while freezing it lost 0.24 dB.
    spatial_lr_scale: float = 1.0

    eval_steps: List[int] = field(default_factory=lambda: [7000])
    save_steps: List[int] = field(default_factory=lambda: [7000])

    #: Photometric pose refinement. Implemented but OFF: the project has
    #: measured it negative in five arms across two gauges (-1.2 to -2.3 dB),
    #: and every city-scale pipeline in the literature runs BA before training
    #: instead of during it. Enable only to reproduce that comparison.
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6

    #: Appearance: per-image exposure affine. Aerial blocks are flown under
    #: changing light and auto-exposure, and one global colour cannot fit both.
    app_opt: bool = False
    app_opt_lr: float = 1e-3

    #: Gradient clipping on the total loss (0 = off). Cheap insurance against
    #: the loss spikes that a single bad frame can produce.
    grad_clip: float = 0.0

    #: Decode images on the fly instead of caching. Caching all images is what
    #: makes a 1000-frame block feasible; it costs width*height*3*4 bytes each.
    cache_images: bool = False
    #: 0 = cache everything; otherwise cap the cache size (frames).
    max_cached_images: int = 0

    seed: int = 42
    device: str = "cuda"
    out_dir: str = ""

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("train.max_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1")
        if self.pose_opt and self.spatial_lr_scale <= 0:
            raise ValueError("pose_opt requires a positive spatial_lr_scale")
        if self.lr.means_per_spacing is not None:
            if self.lr.means_per_spacing <= 0:
                raise ValueError(
                    "train.lr.means_per_spacing must be positive, got "
                    f"{self.lr.means_per_spacing}"
                )
            # Both fields set is not necessarily a mistake -- a caller may set
            # means as documentation and means_per_spacing as the operative
            # value -- but it is worth warning, because the raw one is silently
            # ignored and the two give very different step sizes (0.010 vs 0.119
            # spacings/step on garden, which is 2 dB).
            if self.lr.means != LRConfig.means:
                raise ValueError(
                    f"train.lr.means ({self.lr.means}) and "
                    f"train.lr.means_per_spacing ({self.lr.means_per_spacing}) "
                    "are both set; means_per_spacing wins and means is ignored. "
                    "Remove train.lr.means to make the intent explicit."
                )
        if self.lr.decay_means and self.lr.means_final is None:
            if not (0.0 < self.lr.means_final_ratio < 1.0):
                raise ValueError(
                    "train.lr.means_final_ratio must be in (0, 1) when "
                    "train.lr.decay_means=True and train.lr.means_final is "
                    f"unset, got {self.lr.means_final_ratio}"
                )


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration. Instantiated from YAML plus dotted overrides."""

    data: DataConfig = field(default_factory=DataConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    #: Name of the profile this config was derived from (informational; the
    #: resolved values are what is dumped next to the run).
    profile: str = "aerial"

    def validate(self, *, require_path: bool = True) -> None:
        self.data.validate(require_path=require_path)
        self.gauge.validate()
        self.model.validate()
        self.density.validate()
        self.train.validate()
        if self.loss.normal_weight > 0 and self.model.type != "2dgs":
            raise ValueError("loss.normal_weight requires model.type='2dgs'")
        if self.data.depth.per_frame_median_norm and self.gauge.apply_to_depth:
            # Not an error, but it must not be silently assumed safe.
            raise ValueError(
                "depth.per_frame_median_norm=True makes gauge.apply_to_depth "
                "irrelevant; set gauge.apply_to_depth=False so the config "
                "states what actually happens"
            )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
