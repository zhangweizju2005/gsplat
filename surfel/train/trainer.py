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

        init: SplatInit = create_splats_2dgs(
            scene.points,
            scene.point_rgb,
            sh_degree=self.cfg.model.sh_degree,
            scene_scale=scene.scene_scale,
            init_opacity=init_opacity,
            init_scale=init_scale,
            init_scale_mult=self.cfg.model.init_scale_mult,
            lr_means=self.cfg.train.lr.means * self.cfg.train.spatial_lr_scale,
            lr_scales=self.cfg.train.lr.scales,
            lr_quats=self.cfg.train.lr.quats,
            lr_opacities=self.cfg.train.lr.opacities,
            lr_sh0=self.cfg.train.lr.sh0,
            lr_shN=self.cfg.train.lr.shN,
            device=self.device,
        )
        self.params = init.to_parameter_dict()
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
        if self.cfg.train.lr.decay_means and self.cfg.train.lr.means_final:
            self.means_lr_step = means_lr_scheduler(
                self.optimizers,
                lr_init=self.optimizers["means"].param_groups[0]["lr"],
                lr_final=self.cfg.train.lr.means_final,
                max_steps=self.cfg.train.max_steps,
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
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for k, v in payload["params"].items():
            if k in self.params and self.params[k].shape == v.shape:
                self.params[k].data.copy_(v.to(self.device))
            else:
                self.params[k] = torch.nn.Parameter(v.to(self.device))
        self.step = int(payload.get("step", 0))
        state = payload.get("strategy_state")
        if state is not None:
            _restore_strategy_state(self.strategy_state, state, self.device)
        return payload

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        cfg = self.cfg
        loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
            drop_last=False,
        )
        it = iter(loader)
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
