"""Training loop: gsplat's ordering, with large-scene guard-rails.

The protocol order is not negotiable and is reproduced exactly:

1. ``render``
2. ``strategy.step_pre_backward`` -- calls ``info[key].retain_grad()``
3. ``loss.backward()``
4. ``optimizer.step()`` / ``zero_grad()``
5. ``strategy.step_post_backward`` -- reads the retained gradients, grows/prunes

Calling ``step_pre_backward`` after ``backward`` crashes (the intermediate has
been freed); calling ``step_post_backward`` before it reads a ``None`` gradient.
A previous version of this project got the *key* wrong instead and produced a
run that looked normal: the tensor existed, ``retain_grad`` worked, and the
gradient was always zero because the kernel never wrote that tensor.

Everything scene-dependent is derived, not configured by hand:

* ``scene_scale`` for the strategy comes from the parsed scene's depth
  quantiles (see :mod:`surfel.data.gauge`);
* ``depth_scale`` for the disparity loss comes from the median supervised depth;
* near/far come from the measured camera-space depth distribution;
* ``want_depth`` is the union of "supervise depth" and "need rendered depth for
  the normal term" -- conflating those two made the normal term inert.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from ..config.schema import Config
from ..data.dataset import CameraBatch, SceneDataset, collate
from ..data.scene import Scene, Split
from ..model.renderer import Renderer2DGS, derive_depth_range
from ..model.splats import (
    SplatInit,
    build_optimizers,
    create_splats_2dgs,
    means_lr_scheduler,
)
from . import losses as L
from .density import InstrumentedStrategy, build_strategy, describe_thresholds


@dataclass
class TrainMetrics:
    step: int
    total: float
    l1: float
    ssim: float
    n_gaussians: int
    seconds: float = 0.0
    terms: Dict[str, float] = field(default_factory=dict)


class Trainer:
    """Owns the model, the strategy, the loop and the checkpoints."""

    def __init__(
        self,
        cfg: Config,
        split: Split,
        *,
        device: Optional[str] = None,
        log_fn=print,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.split = split
        self.device = torch.device(device or cfg.train.device)
        # ``log_fn=None`` means "only the file log, no console". An explicit
        # no-op is needed rather than leaving it None, because every emission
        # path assumes it is callable; passing None used to raise a TypeError
        # from inside the first log line, which is a confusing way for a
        # silent-mode request to fail.
        self.log = log_fn if log_fn is not None else (lambda _msg: None)
        self.step = 0
        self.metrics: List[TrainMetrics] = []

        # Seed everything the run touches, *before* any model construction.
        # ``train.seed`` used to be read only by the DGI/kNN samplers, so the
        # quaternion initialisation (``torch.rand``) and the densification's
        # random split/duplicate sampling were never seeded: two "identical"
        # runs differed, and a resume could not reproduce an uninterrupted one.
        _seed_everything(self.cfg.train.seed)

        self._setup_dirs()
        self._setup_data()
        self._setup_model()
        self._setup_strategy()
        self._setup_optional_modules()
        self._setup_schedule()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_dirs(self) -> None:
        out = Path(self.cfg.train.out_dir or "outputs/run")
        out.mkdir(parents=True, exist_ok=True)
        self.out_dir = out
        self.ckpt_dir = out / "ckpts"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.log_path = out / "train.log"
        self._log_fh = open(self.log_path, "a", encoding="utf-8")

    def _emit(self, text: str) -> None:
        """Write to the console callback and the run log.

        The file handle is reopened on demand: ``fit()`` closes it when training
        ends, but ``save()`` / ``evaluate()`` / a second ``fit()`` can still be
        called afterwards (resume, post-hoc eval). Assuming the handle is alive
        raises ``I/O operation on closed file`` at a point far from its cause.
        """
        self.log(text)
        if self._log_fh is None or self._log_fh.closed:
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self._log_fh.write(text + "\n")
        self._log_fh.flush()

    def close(self) -> None:
        """Close the run log. Safe to call more than once."""
        if self._log_fh is not None and not self._log_fh.closed:
            self._log_fh.close()

    def _setup_data(self) -> None:
        scene = self.split.train
        self.cache_images = self.cfg.train.cache_images
        self.trainset = SceneDataset(
            scene,
            cache=self.cfg.train.cache_images,
            max_cached=self.cfg.train.max_cached_images,
            load_depth=self.cfg.data.depth.enabled,
        )
        self.valset = (
            SceneDataset(self.split.val, cache=False, load_depth=False)
            if self.split.val is not None
            else None
        )

        # One resolution for the whole run: the batch dimension requires it,
        # and mixing sizes would make PSNR meaningless. Reported per-run so a
        # number can never be quoted without it.
        sizes = {(c.height, c.width) for c in scene.cameras}
        if len(sizes) != 1:
            raise ValueError(
                f"train split has {len(sizes)} distinct render sizes {sorted(sizes)}; "
                "set data.target_long_edge to unify them"
            )
        self.height, self.width = sizes.pop()
        self._emit(
            f"render grid  : {self.width}x{self.height} "
            f"({len(self.trainset)} train / "
            f"{len(self.valset) if self.valset else 0} val)"
        )

    def _setup_model(self) -> None:
        scene = self.split.train
        # MCMC needs a different initialisation from DefaultStrategy: gsplat's
        # own preset uses init_opacity=0.5 / init_scale=0.1 together with
        # opacity_reg=0.01 / scale_reg=0.01. Running MCMC with the split-based
        # defaults (0.1 / 1.0, no regularisers) is not "MCMC with slightly
        # different values" -- the relocation/opacity logic assumes a denser,
        # flatter starting set, so the comparison would be meaningless.
        mcmc = self.cfg.density.strategy == "mcmc"
        init_opacity = 0.5 if mcmc else self.cfg.model.init_opacity
        init_scale = 0.1 if mcmc else self.cfg.model.init_scale
        if mcmc:
            if self.cfg.loss.opacity_reg == 0.0 or self.cfg.loss.scale_reg == 0.0:
                self._emit(
                    "WARNING: density.strategy='mcmc' with opacity_reg/scale_reg "
                    "at 0. gsplat's MCMC preset uses 0.01 for both; without them "
                    "the opacity distribution drifts and relocation has no "
                    "regularising pressure."
                )
            self._emit(f"mcmc init    : opacity={init_opacity} scale={init_scale}")

        # means LR: resolve the gauge-free form if the config asked for it.
        #
        # ``means`` is a length, so a raw value is only meaningful relative to the
        # scene. ``means_per_spacing`` states the rate in the point cloud's own
        # units (kNN spacings per step), which is what transfers between scenes;
        # measured optimum on garden is 0.010-0.030 and GSL's archived MatrixCity
        # f0_gtpose sits at 0.0104. See LRConfig.means_per_spacing.
        lr_means_cfg = self.cfg.train.lr.means
        rel = self.cfg.train.lr.means_per_spacing
        if rel is not None:
            from surfel.model.splats import knn_mean_distance_rms

            spacing = float(
                knn_mean_distance_rms(
                    torch.as_tensor(scene.points, dtype=torch.float32)
                ).median()
            )
            if not (spacing > 0.0):
                raise ValueError(
                    f"measured kNN spacing is {spacing!r}, cannot convert "
                    "train.lr.means_per_spacing into an absolute learning rate"
                )
            # create_splats_2dgs multiplies lr_means by scene_scale again, so
            # divide it out here to land on exactly rel * spacing.
            lr_means_cfg = rel * spacing / scene.scene_scale
            self._emit(
                f"means lr     : {rel:.4g} spacings/step "
                f"(kNN {spacing:.4g} x rel / scene_scale "
                f"{scene.scene_scale:.4g} = {lr_means_cfg:.4g} raw, "
                f"x scene_scale = {rel * spacing:.4g} absolute)"
            )

        init: SplatInit = create_splats_2dgs(
            scene.points,
            scene.point_rgb,
            sh_degree=self.cfg.model.sh_degree,
            scene_scale=scene.scene_scale,
            init_opacity=init_opacity,
            init_scale=init_scale,
            init_scale_mult=self.cfg.model.init_scale_mult,
            lr_means=lr_means_cfg * self.cfg.train.spatial_lr_scale,
            lr_scales=self.cfg.train.lr.scales,
            lr_quats=self.cfg.train.lr.quats,
            lr_opacities=self.cfg.train.lr.opacities,
            lr_sh0=self.cfg.train.lr.sh0,
            lr_shN=self.cfg.train.lr.shN,
            device=self.device,
        )
        self.params = init.to_parameter_dict()
        # Kept so resume can rebuild optimisers at the checkpoint's parameter
        # count with the same learning rates (see load()).
        self.init = init
        self.optimizers = build_optimizers(
            init, batch_size=self.cfg.train.batch_size, device=self.device
        )
        self._emit(f"model        : {len(self.params['means'])} 2DGS surfels")

        self.renderer = Renderer2DGS(
            packed=self.cfg.model.packed,
            eps2d=self.cfg.model.eps2d,
            radius_clip=self.cfg.model.radius_clip,
            near_plane=self.cfg.model.near_plane,
            far_plane=self.cfg.model.far_plane,
        )
        if self.renderer.near_plane is None or self.renderer.far_plane is None:
            near, far = self._derive_near_far(scene)
            self.renderer.set_depth_range(near, far)
            self._emit(
                f"depth range  : near={near:.6g} far={far:.6g} "
                "(derived from measured camera-space depth)"
            )

        # Resolved once: used by the strategy and by the loss multiplier.
        self.scene_scale = scene.scene_scale
        self.depth_scale = L.resolve_depth_scale(
            scene.depth_p50, self.cfg, fallback=self.scene_scale
        )
        self._emit(f"scene_scale  : {self.scene_scale:.6g}")
        self._emit(f"depth_scale  : {self.depth_scale:.6g} (disparity L1 multiplier)")

        # Depth rendering is needed for two independent reasons. Taking the union
        # is what keeps the normal term alive on datasets with no GT depth.
        self.want_gt_depth = (
            self.cfg.data.depth.enabled
            and self.cfg.data.depth.weight > 0.0
            or self.cfg.data.depth.weight_final is not None
            and float(self.cfg.data.depth.weight_final) > 0.0
        )
        self.want_normal = self.cfg.loss.normal_weight > 0.0
        self.want_dist = self.cfg.loss.dist_weight > 0.0
        self.render_depth = self.want_gt_depth or self.want_normal or self.want_dist
        if self.render_depth:
            self._emit(
                "depth render : on"
                f" (gt={self.want_gt_depth} normal={self.want_normal} dist={self.want_dist})"
            )

        self._probe_backend_contract()
        self._report_pose_point_geometry()

    def _report_pose_point_geometry(self) -> None:
        """Log the camera/point footprint and flag gross gauge inconsistencies.

        Deliberately a *report*, not a gate. Measurement showed that a
        similarity applied to the wrong side of a transform leaves every
        internal ratio within ~25% of the correct value, so no cheap runtime
        check can detect it -- see
        :func:`surfel.data.gauge.check_pose_point_consistency` for the numbers.
        What this does catch is the severe end: sub-pixel splats (which the
        rasteriser silently drops) and points behind the cameras.

        The absolute ``camera_radius`` is logged too, because comparing it
        against a reference run is the only signal that actually discriminates.
        """
        from ..data.gauge import check_pose_point_consistency

        stats = check_pose_point_consistency(self.split.train)
        r50 = stats["radius_p50"]
        if np.isfinite(r50):
            self._emit(
                f"geometry     : median footprint {r50:.2f} px, "
                f"visible {stats['visible_frac']:.1%}, "
                f"camera radius {stats['camera_radius']:.4g}"
            )
        else:
            self._emit("geometry     : footprint unmeasurable (no point in front of any camera)")
        for reason in stats["reasons"]:
            self._emit(f"WARNING      : {reason}")

    def _probe_backend_contract(self) -> None:
        """Render one frame and verify gsplat's interface invariants.

        Cheap insurance against the class of bug this project hits repeatedly:
        gsplat's C++ layer already normalises depth by alpha, rotates normals to
        world space and clamps SH colours, so any Python-side repetition produces
        a plausible-looking wrong result rather than an error. Running the checks
        at startup means a mistake surfaces immediately instead of as a mysteriously
        low PSNR after twenty minutes. See §3.7 of the architecture doc.
        """
        from ..model.renderer import assert_backend_contract

        sample = self.trainset[0]
        from ..data.dataset import collate

        batch = collate([sample]).to(self.device)
        with torch.no_grad():
            out = self.renderer(
                self.params,
                batch.cam_to_worlds,
                batch.Ks,
                self.width,
                self.height,
                sh_degree=min(1, self.cfg.model.sh_degree),
                render_depth=self.render_depth,
                distloss=self.want_dist,
            )
        try:
            assert_backend_contract(out)
        except AssertionError as exc:
            raise RuntimeError(
                "gsplat backend contract violation detected at startup: "
                f"{exc}. This means the Python side is duplicating a transform "
                "the C++ layer already applied (depth alpha-normalisation, "
                "normal world-rotation, or the SH +0.5 bias)."
            ) from exc
        self._emit("backend probe : contract OK")

    def _derive_near_far(self, scene: Scene) -> tuple[float, float]:
        pts = torch.from_numpy(scene.points).float().to(self.device)
        c2ws = torch.from_numpy(scene.cam_to_worlds()).float().to(self.device)
        return derive_depth_range(pts, c2ws)

    def _setup_strategy(self) -> None:
        # The compensation needs the real footprint, so measure it here instead
        # of accepting a configured constant: it is a property of the scene and
        # the render grid (f_render / f_depth), not a tunable.
        measured_fp = None
        if self.cfg.density.footprint_compensate:
            from surfel.data.gauge import projected_footprint_px

            scene = self.split.train
            stats = projected_footprint_px(
                scene.cam_to_worlds(), scene.points, scene.Ks()
            )
            measured_fp = stats["radius_p50"]
            self._emit(
                f"footprint     : {measured_fp:.3f} px measured "
                f"(ref {self.cfg.density.footprint_ref:.3f}, "
                f"exponent {self.cfg.density.footprint_exponent:.3f})"
            )

        strategy, init_kwargs = build_strategy(
            self.cfg,
            scene_scale=self.scene_scale,
            n_initial=len(self.params["means"]),
            measured_footprint=measured_fp,
        )
        self.raw_strategy = strategy
        # The instrumented wrapper must report against the threshold the strategy
        # actually compares to. Reporting the configured value while the
        # strategy used a compensated one would make every logged gain look
        # "above threshold" and quietly corrupt the diagnostics.
        self.instrumented = InstrumentedStrategy(
            strategy,
            threshold=getattr(strategy, "grow_grad2d", self.cfg.density.grow_grad2d),
            dead_check_events=self.cfg.density.dead_check_events,
            log_first_events=self.cfg.density.log_first_events,
        )
        self.instrumented.check_sanity(self.params, self.optimizers)
        # ``scene_scale`` must be passed here. The official 2DGS example omits
        # it and silently gets 1.0, which inverts prune and duplicate on any
        # large scene; see surfel.train.density.
        self.strategy_state = self.instrumented.initialize_state(**init_kwargs)
        self._emit(
            describe_thresholds(
                self.scene_scale,
                self.cfg,
                threshold_scale=(
                    strategy.grow_scale3d / self.cfg.density.grow_scale3d
                    if self.cfg.density.grow_scale3d > 0.0
                    else 1.0
                ),
            )
        )

    def _setup_optional_modules(self) -> None:
        self.pose_module = None
        self.pose_optimizers: List[torch.optim.Optimizer] = []
        if self.cfg.train.pose_opt:
            from .pose import CameraOptModule

            self.pose_module = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_module.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_module.parameters(),
                    lr=self.cfg.train.pose_opt_lr,
                    weight_decay=self.cfg.train.pose_opt_reg,
                )
            ]
            # This is implemented but off by default because it has measured
            # negative in five arms across two gauges. Saying so at startup keeps
            # a later reader from assuming it was on by oversight.
            self._emit(
                "pose opt     : ENABLED (measured negative in this project's "
                f"history; lr={self.cfg.train.pose_opt_lr:.2e})"
            )

        self.app_module = None
        self.app_optimizers: List[torch.optim.Optimizer] = []
        if self.cfg.train.app_opt:
            from .appearance import ExposureAffine

            self.app_module = ExposureAffine(len(self.trainset)).to(self.device)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.parameters(), lr=self.cfg.train.app_opt_lr
                )
            ]
            self._emit(f"appearance   : per-image exposure affine ({len(self.trainset)})")

    def _setup_schedule(self) -> None:
        self.means_lr_step = None
        lr_cfg = self.cfg.train.lr
        if lr_cfg.decay_means:
            # Resolve the endpoint. ``means_final`` is an absolute alternative for
            # callers that want a specific value; otherwise use the ratio, which
            # is the form that transfers (GSL anchors init * 0.01 over its own
            # step count, and so do we).
            lr_init = self.optimizers["means"].param_groups[0]["lr"]
            lr_final = lr_cfg.means_final
            if lr_final is None:
                lr_final = lr_init * lr_cfg.means_final_ratio
            if not (lr_final > 0.0):
                raise ValueError(
                    f"resolved means LR endpoint is {lr_final!r}; the schedule "
                    "needs a positive value"
                )
            self.means_lr_step = means_lr_scheduler(
                self.optimizers,
                lr_init=lr_init,
                lr_final=lr_final,
                max_steps=self.cfg.train.max_steps,
            )
            self._emit(
                f"means decay  : {lr_init:.4g} -> {lr_final:.4g} "
                f"(ratio {lr_final / lr_init:.4g}) over {self.cfg.train.max_steps} steps"
            )

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------
    def _sh_degree_at(self, step: int) -> int:
        interval = self.cfg.model.sh_degree_interval
        if not interval:
            return self.cfg.model.sh_degree
        return min(step // interval, self.cfg.model.sh_degree)

    def train_step(self, batch: CameraBatch) -> TrainMetrics:
        t0 = time.time()
        cfg = self.cfg
        step = self.step

        batch = batch.to(self.device)
        cam_to_worlds = batch.cam_to_worlds
        if self.pose_module is not None:
            cam_to_worlds = self.pose_module(cam_to_worlds, batch.indices)

        pixels = batch.images  # [C,H,W,3] in [0,1]
        gt_depth = batch.depths if self.want_gt_depth else None

        # Forward. The render mode follows the union of the reasons depth is
        # needed, not just GT availability.
        render = self.renderer(
            self.params,
            cam_to_worlds,
            batch.Ks,
            self.width,
            self.height,
            sh_degree=self._sh_degree_at(step),
            render_depth=self.render_depth,
            distloss=self.want_dist,
        )

        # -- order matters: pre -> backward -> post --
        self.instrumented.step_pre_backward(
            self.params, self.optimizers, self.strategy_state, step, render.meta
        )

        weights = L.resolve_weights(cfg, step)
        breakdown = L.total_loss(
            render=render,
            pixels=pixels,
            gt_depth=gt_depth,
            weights=weights,
            cfg=cfg,
            params=self.params,
            scene_scale=self.scene_scale,
            depth_scale=self.depth_scale,
        )

        if cfg.train.grad_clip and cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.params.values()], cfg.train.grad_clip
            )

        breakdown.total.backward()
        # ``lr`` is only needed by MCMCStrategy (position noise); the wrapper
        # forwards it only if the strategy's signature accepts it.
        cur_lr = self.optimizers["means"].param_groups[0]["lr"]
        self.instrumented.step_post_backward(
            self.params,
            self.optimizers,
            self.strategy_state,
            step,
            render.meta,
            packed=self.cfg.model.packed,
            lr=cur_lr,
        )

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for opt in self.pose_optimizers + self.app_optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if self.means_lr_step is not None:
            self.means_lr_step(step)

        self.step += 1
        m = TrainMetrics(
            step=step,
            total=float(breakdown.total.detach()),
            l1=float(breakdown.l1.detach()),
            ssim=float(breakdown.ssim.detach()),
            n_gaussians=len(self.params["means"]),
            seconds=time.time() - t0,
            terms=breakdown.as_dict(),
        )
        self.metrics.append(m)
        return m

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.valset is None:
            return {}
        from gsplat.losses import ssim_loss

        psnrs: List[float] = []
        ssims: List[float] = []
        for i in range(len(self.valset)):
            batch = collate([self.valset[i]]).to(self.device)
            render = self.renderer(
                self.params,
                batch.cam_to_worlds,
                batch.Ks,
                self.width,
                self.height,
                sh_degree=self.cfg.model.sh_degree,
                render_depth=False,
                distloss=False,
            )
            pred = render.colors.clamp(0.0, 1.0)
            gt = batch.images.clamp(0.0, 1.0)
            mse = ((pred - gt) ** 2).mean()
            psnrs.append(float(-10.0 * torch.log10(mse.clamp_min(1e-10))))
            ssims.append(
                float(
                    1.0
                    - ssim_loss(
                        pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
                    )
                )
            )
        out = {
            "val/psnr": sum(psnrs) / len(psnrs),
            "val/ssim": sum(ssims) / len(ssims),
            "val/n": float(len(psnrs)),
        }
        self._emit(
            f"step {self.step}: val PSNR {out['val/psnr']:.3f} "
            f"SSIM {out['val/ssim']:.4f} "
            f"({self.width}x{self.height}, n={len(psnrs)})"
        )
        return out

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """Save parameters plus the resolved config.

        The config is stored as a plain dict and rebuilt through the same
        validated setter on load, so a later evaluation cannot silently run with
        defaults: a renamed key raises instead.
        """
        path = path or (self.ckpt_dir / f"step_{self.step:06d}.pt")
        payload = {
            "step": self.step,
            "params": {k: v.detach().cpu() for k, v in self.params.items()},
            # The densification strategy's running state (``grad2d`` / ``count``
            # / ``radii``) is a plain Python dict in gsplat and is not saved by
            # its own examples, so a resumed run restarts the gradient statistics
            # from zero: the first refine events after resume judge candidates on
            # a handful of frames, and ``max_radii2D`` (used by screen-space
            # pruning) is empty. Serialising it here makes resume equivalent to an
            # uninterrupted run.
            "strategy_state": _serialise_strategy_state(self.strategy_state),
            # Optimiser momenta and the *current* LRs. Without these a resume
            # restarts Adam from zero and, worse, re-seeds the means LR from its
            # initial value instead of continuing the decay -- two resumptions
            # of the same run then follow different trajectories.
            "optimizer_state": {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            "lrs": {
                name: opt.param_groups[0]["lr"]
                for name, opt in self.optimizers.items()
            },
            "config": self.cfg.to_dict(),
            "scene_scale": self.scene_scale,
            "depth_scale": self.depth_scale,
            "render_size": [self.width, self.height],
            "applied_sim3": (
                self.split.train.applied_sim3.tolist()
                if self.split.train.applied_sim3 is not None
                else None
            ),
            "protocol": {
                # Recorded so a PSNR can never be quoted without its conditions.
                "split_mode": self.cfg.data.split_mode,
                "n_train": len(self.trainset),
                "n_val": len(self.valset) if self.valset else 0,
                "resolution": f"{self.width}x{self.height}",
                "eval_on_train": self.valset is None,
            },
        }
        torch.save(payload, path)
        self._emit(f"saved {path}")
        return path

    def load(self, path: str | Path) -> Dict[str, Any]:
        """Restore parameters, step counter and strategy state.

        Every restored parameter is written *in place*, which is the whole
        point: the optimisers built in ``_setup_model`` hold references to the
        exact ``Parameter`` objects created there, and so does the strategy.
        A previous version replaced ``self.params[k]`` with a new
        ``Parameter`` whenever the checkpoint's shape differed from the freshly
        initialised model's. That is the normal case for any densified run, and
        it silently froze training: ``optimizer.step()`` kept updating the
        *orphaned* tensors, ``loss`` and every logged metric kept looking
        normal, and the live model never changed. Nothing raised, which is why
        it went unnoticed until a resume was instrumented.

        So the shape differences are absorbed by resizing the existing storage
        rather than replacing the object (see :func:`_resize_parameter`), and
        the optimiser references are verified afterwards. If a resume still
        cannot be wired up, this raises instead of training a frozen model.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        resized = False
        for k, v in payload["params"].items():
            v = v.detach().to(self.device)
            if k not in self.params:
                self.params[k] = torch.nn.Parameter(v)
                resized = True
                continue
            if tuple(self.params[k].data.shape) != tuple(v.shape):
                resized = True
            _resize_parameter(self.params[k], v)
        if resized:
            # The moment tensors are shaped by the parameter count, so they cannot
            # be loaded wholesale. They are rebuildable in prefix though: the
            # first ``n`` rows of a checkpoint's state still describe the first
            # ``n`` Gaussians, and that is exactly which rows survive a resume
            # that starts from a smaller initial cloud. Discarding them would
            # restart Adam on the majority of the model for no reason.
            #
            # The rebuild must wrap the *current* ``self.params``: passing the
            # original ``self.init`` builds Adam over the initialisation tensors,
            # which after a resize are neither the live objects nor the live
            # shape, and the next densification then fails with a shape mismatch
            # between a 6522-row gradient and a 3783-row moment.
            saved_state = payload.get("optimizer_state")
            self.optimizers = _rebuild_optimizers(
                self.params,
                self.init.lrs,
                batch_size=self.cfg.train.batch_size,
                device=self.device,
            )
            migrated = _migrate_optimizer_state_prefix(
                self.optimizers, saved_state, self.params
            )
            if saved_state is not None:
                self._emit(
                    "resume       : parameter count changed "
                    f"({len(payload['params']['means'])} saved -> "
                    f"{len(self.params['means'])} live); Adam momenta migrated for "
                    f"{migrated} row(s), fresh beyond that"
                )
        else:
            opt_state = payload.get("optimizer_state")
            if opt_state is not None:
                _restore_optimizer_state(self.optimizers, opt_state)
            lrs = payload.get("lrs")
            if lrs:
                for name, lr in lrs.items():
                    if name in self.optimizers:
                        self.optimizers[name].param_groups[0]["lr"] = float(lr)

        self.step = int(payload.get("step", 0))
        state = payload.get("strategy_state")
        if state is not None:
            _restore_strategy_state(self.strategy_state, state, self.device)
        # An optimiser that no longer points at a live parameter is a silent
        # no-op, so this is checked rather than assumed.
        _assert_optimizers_track_params(self.params, self.optimizers)
        return payload

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        cfg = self.cfg
        # ``generator`` makes the shuffling reproducible from ``train.seed``, and
        # -- because it is a *named* generator -- it can be fast-forwarded on
        # resume so the batch order continues where the previous leg stopped.
        # Without the fast-forward a resumed run draws from the start of the
        # permutation again, which is why a resumed trajectory differed from the
        # uninterrupted one even with identical parameters and LRs.
        gen = torch.Generator()
        gen.manual_seed(int(cfg.train.seed))
        skip = 0
        if self.step > 0:
            skip = _fast_forward_loader(
                gen, len(self.trainset), cfg.train.batch_size, self.step
            )
        loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
            drop_last=False,
            generator=gen,
        )
        it = iter(loader)
        # Mid-epoch resumes: the generator is positioned at the start of the
        # epoch that contains ``self.step``, so consume the batches already seen.
        for _ in range(skip):
            try:
                next(it)
            except StopIteration:  # pragma: no cover - skip is always < per_epoch
                break
        eval_steps = set(cfg.train.eval_steps)
        save_steps = set(cfg.train.save_steps)
        last_val: Dict[str, float] = {}

        self._emit(f"training     : {cfg.train.max_steps} steps, profile {cfg.profile}")
        start = time.time()
        log_every = max(1, min(100, cfg.train.max_steps // 20))
        while self.step < cfg.train.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            m = self.train_step(batch)
            if self.step % log_every == 0 or self.step == 1:
                self._emit(
                    f"step {self.step:6d}  loss {m.total:.4f}  "
                    f"l1 {m.l1:.4f}  ssim {m.ssim:.4f}  "
                    f"gs {m.n_gaussians}  {1.0 / max(m.seconds, 1e-6):.1f} it/s"
                )
                if self.instrumented.should_log() and self.instrumented.stats.events:
                    self._emit("  " + self.instrumented.stats.summary())
                    self.instrumented.maybe_warn(self._emit)

            if self.step in eval_steps:
                last_val = self.evaluate()
            if self.step in save_steps:
                self.save()

        self.save(self.ckpt_dir / "final.pt")
        elapsed = time.time() - start
        self._emit(
            f"done         : {self.step} steps in {elapsed / 60:.1f} min, "
            f"{len(self.params['means'])} Gaussians"
        )
        self._emit("  " + self.instrumented.stats.summary())
        # Do not close the handle here: callers routinely save or evaluate after
        # fit(), which would then fail. The handle is reopened lazily by _emit,
        # and close() is available for a caller that is truly done.
        return last_val


__all__ = ["Trainer", "TrainMetrics"]


# --------------------------------------------------------------------------
# Strategy-state (de)serialisation
# --------------------------------------------------------------------------
def _serialise_strategy_state(state: Any) -> Optional[Dict[str, Any]]:
    """Convert a strategy state dict into something ``torch.save``-friendly.

    The state may hold tensors (``grad2d``, ``count``, ``radii``), scalars
    (``scene_scale``) and a 51x51 binomial table. Tensors are moved to CPU and
    non-tensor scalars are kept as-is; anything not trivially serialisable is
    dropped rather than silently pickled into an unusable object.
    """
    import torch

    if not isinstance(state, dict):
        return None
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        # Anything else (closures, references) is intentionally skipped: it
        # cannot be restored meaningfully and would make the checkpoint fragile.
    return out


def _seed_everything(seed: int) -> None:
    """Make a run reproducible from ``train.seed``.

    All three RNGs matter and none were previously seeded:

    * ``torch`` -- ``create_splats_2dgs`` draws ``quats`` with ``torch.rand``,
      and gsplat's split/duplicate ops sample with ``torch.randn``;
    * ``numpy`` -- DGI subsampling, the kNN cap, and gauge quantile sampling;
    * ``random`` -- used indirectly by some library paths.

    ``PYTHONHASHSEED`` cannot be set after interpreter start, so dict ordering
    is only reproducible when the process is launched with it already set; that
    is documented rather than silently assumed.
    """
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fast_forward_loader(
    gen: torch.Generator, n_items: int, batch_size: int, steps: int
) -> int:
    """Position a shuffle generator at the epoch containing ``steps``.

    ``DataLoader(shuffle=True, generator=g)`` draws one permutation per epoch
    (``torch.randperm(n, generator=g)``) and consumes ``per_epoch =
    ceil(n / batch_size)`` batches from it before drawing the next. To make a
    resumed run see the same batch order the uninterrupted run would have seen,
    the generator must be advanced past the permutations of the *completed*
    epochs, and the caller must then skip the batches already consumed inside
    the current epoch.

    Returns the number of batches to skip.
    """
    if steps <= 0 or n_items <= 0:
        return 0
    per_epoch = max(1, math.ceil(n_items / max(1, batch_size)))
    completed_epochs = int(steps) // per_epoch
    for _ in range(completed_epochs):
        torch.randperm(n_items, generator=gen)
    return int(steps) % per_epoch


def _rebuild_optimizers(
    params: Dict[str, torch.nn.Parameter],
    lrs: Dict[str, float],
    *,
    batch_size: int,
    device,
) -> Dict[str, torch.optim.Optimizer]:
    """Build optimisers over *existing* parameter objects.

    :func:`~surfel.model.splats.build_optimizers` takes a ``SplatInit`` and wraps
    the tensors it holds. On resume the live tensors are the ones in
    ``self.params`` (possibly resized by a checkpoint), so the optimizer must be
    constructed from those identities, not from the original initialisation.
    The batch-size scaling is gsplat's convention and is reproduced exactly as
    ``build_optimizers`` does it.
    """
    import math as _math

    sqrt_b = _math.sqrt(batch_size)
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for name, param in params.items():
        optimizers[name] = torch.optim.Adam(
            [{"params": param, "lr": float(lrs[name]) * sqrt_b}],
            eps=1e-15 / sqrt_b,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
    return optimizers


def _restore_optimizer_state(
    optimizers: Dict[str, torch.optim.Optimizer],
    saved: Dict[str, Any],
) -> None:
    """Load Adam momenta, tolerating a size change in an individual group.

    ``load_state_dict`` raises when the tensor shapes disagree, and a strategy
    may have changed only *one* parameter's count. Per-optimiser guarding keeps
    the restorable groups restorable instead of discarding all of them.
    """
    for name, opt in optimizers.items():
        state = saved.get(name)
        if state is None:
            continue
        try:
            opt.load_state_dict(state)
        except (ValueError, KeyError) as exc:
            # Reported rather than swallowed: the caller decides whether a
            # restarted momentum is acceptable.
            print(
                f"WARNING: could not restore optimizer[{name!r}] state ({exc}); "
                "its momenta restart from zero"
            )


def _migrate_optimizer_state_prefix(
    optimizers: Dict[str, torch.optim.Optimizer],
    saved: Optional[Dict[str, Any]],
    params: Dict[str, torch.nn.Parameter],
) -> int:
    """Carry over the leading rows of a saved optimiser state.

    Adam's ``exp_avg`` / ``exp_avg_sq`` / ``step`` describe Gaussians, so a
    resume into a differently sized model cannot load them wholesale -- but the
    first ``min(saved_n, live_n)`` rows are still valid, because the checkpoint
    and the live model share their ``min`` prefix (the live model was
    initialised from the same cloud and then grew, or the checkpoint grew).
    Rows beyond that start at zero, which is the same state a new Gaussian has.

    Returns the number of rows migrated, for the resume log line.
    """
    if not saved:
        return 0
    migrated = 0
    for name, opt in optimizers.items():
        state = saved.get(name)
        if not state:
            continue
        live_param = opt.param_groups[0]["params"][0]
        live_n = int(live_param.shape[0])
        per_param = state.get("state", {})
        for saved_idx, saved_state in per_param.items():
            rows = {
                k: v
                for k, v in saved_state.items()
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] <= live_n
            }
            if not rows:
                continue
            n = min(int(next(iter(rows.values())).shape[0]), live_n)
            new_state: Dict[str, Any] = {}
            for k, v in saved_state.items():
                if k in rows:
                    # Pad to the full live length: Adam requires the moment to
                    # match the parameter's shape exactly, and the tail rows are
                    # Gaussians the checkpoint never saw.
                    full = torch.zeros(
                        (live_n, *v.shape[1:]), dtype=v.dtype, device=live_param.device
                    )
                    full[:n] = v[:n].to(live_param.device)
                    new_state[k] = full
                elif isinstance(v, torch.Tensor):
                    # Scalars shaped like the parameter count (none today) or
                    # genuinely scalar (``step``); keep scalars as-is.
                    new_state[k] = v.to(live_param.device).clone()
                else:
                    new_state[k] = v
            opt.state[live_param] = new_state
            migrated = max(migrated, n)
    return migrated


def _resize_parameter(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    """Write ``value`` into ``param`` while keeping the *object* alive.

    ``param.data = ...`` replaces the storage but not the ``Parameter``, so
    every holder of a reference -- the renderer's ``params`` dict, the
    densification strategy, and the optimiser's ``param_groups`` -- keeps
    pointing at a live tensor. Replacing ``self.params[k]`` with a new
    ``Parameter`` instead is what froze resumed runs: the optimiser kept
    updating an orphan.
    """
    if tuple(param.data.shape) == tuple(value.shape):
        param.data.copy_(value)
    else:
        # Densification changed the count. A fresh storage of the checkpoint's
        # shape is required; the optimiser state cannot be reused for it, which
        # is why load() rebuilds the optimisers afterwards.
        param.data = value.detach().clone()


def _assert_optimizers_track_params(
    params: Dict[str, torch.nn.Parameter],
    optimizers: Dict[str, torch.optim.Optimizer],
) -> None:
    """Fail loudly if an optimiser no longer points at the live parameters.

    This is the gate for the failure mode that made resume silent: an optimiser
    holding an orphaned tensor still runs, still reports a loss, and updates
    nothing that is rendered. Identity is the only thing that distinguishes the
    two cases, so it is compared directly.
    """
    for name, opt in optimizers.items():
        live = params.get(name)
        if live is None:
            continue
        tracked = opt.param_groups[0]["params"]
        if len(tracked) != 1 or tracked[0] is not live:
            raise RuntimeError(
                f"optimizer[{name!r}] does not track the live parameter (it holds "
                f"{len(tracked)} tensor(s), identity match failed). Training would "
                "run and log normally while never updating the model."
            )
        if tuple(opt.param_groups[0]["params"][0].shape) != tuple(live.shape):
            raise RuntimeError(
                f"optimizer[{name!r}] tracks a parameter of shape "
                f"{tuple(opt.param_groups[0]['params'][0].shape)} but the live one "
                f"is {tuple(live.shape)}"
            )


def _restore_strategy_state(state: Dict[str, Any], saved: Dict[str, Any], device) -> None:
    """Copy a serialised strategy state back in, adapting to the current size.

    The running statistics are per-Gaussian (``grad2d`` / ``count`` / ``radii``),
    and a checkpoint is normally taken *after* the model has been densified, so
    their saved length rarely matches a freshly initialised model's. Two cases:

    * **same length** -- copy the values; this is the true resume case, and
      keeping the accumulated statistics is the whole point.
    * **different length** -- rebuild a zero-filled tensor of the current size
      rather than skipping. Skipping would leave the key ``None``, and gsplat's
      ``_update_state`` does ``state["grad2d"][gs_ids] = ...`` only after
      initialising from ``len(params["means"])``; an explicit ``None`` here made
      the next ``refine`` step raise ``AttributeError: 'NoneType' has no
      attribute shape``. Rebuilding costs one refine window of statistics, which
      is the honest cost of resuming with a different size.
    """
    for k, v in saved.items():
        if isinstance(v, torch.Tensor):
            cur = state.get(k)
            if isinstance(cur, torch.Tensor) and cur.shape == v.shape:
                cur.copy_(v.to(device))
            else:
                # Rebuild at the current Gaussian count so the key is never None.
                state[k] = torch.zeros_like(cur) if isinstance(cur, torch.Tensor) else v.to(device)
        else:
            state[k] = v
