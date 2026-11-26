"""Densification: gsplat strategy construction, instrumentation, and calibration.

The strategy itself is gsplat's. What this module adds is (a) the wiring that
the official 2DGS example gets wrong, and (b) counters that make a silently dead
densifier impossible to miss.

Wiring bug in the official example
----------------------------------
``examples/simple_trainer_2dgs.py`` calls ``strategy.initialize_state()`` with no
arguments, so ``scene_scale`` defaults to ``1.0``, while the 3DGS example passes
it. ``DefaultStrategy`` then compares ``exp(scales).max()`` against
``grow_scale3d * 1.0 = 0.01`` and ``prune_scale3d * 1.0 = 0.1``. On an object
~1 unit across that is roughly right. On a 530 m aerial trajectory it means:

* every Gaussian larger than 10 cm is pruned as "too big" -- i.e. all structure;
* nothing is ever "small", so every candidate takes the split branch and the
  duplicate branch is unreachable.

Measured on the real scale values: with ``scene_scale=1`` the four probe sizes
{0.05, 0.2, 1.0, 5.0} m give ``prune=[False, True, True, True]`` and
``duplicate-eligible=[False, False, False, False]``; with the true 530 m they
give ``prune=[False, False, False, False]`` and
``duplicate-eligible=[True, True, True, True]``. Passing ``scene_scale`` is
therefore not a refinement, it is the difference between working and inverted.

Gradient units, verified
------------------------
``_update_state`` multiplies the kernel's ``gradient_2dgs`` by
``(width/2, height/2)``. Measured against the 2DGS/hbb1 backend on identical
input, the resulting distributions agree to ~1%:

    gsplat grad_2dgs * (W/2,H/2):  p50 6.464e-4  p90 1.958e-3  >6e-4: 53.1%
    hbb1  means2d (raw, pixels):   p50 6.542e-4  p90 1.959e-3  >6e-4: 53.3%

So a threshold calibrated against hbb1 -- including this project's 6e-4 -- is
directly valid and must **not** be rescaled. That question was left open in an
earlier analysis; it is settled here by measurement.

The dead-densifier guard
------------------------
A previous version of this project read image-plane gradients from
``proj.means2d``. gsplat 1.6.0's 2DGS kernel never writes that tensor, so the
gradient was always zero, the threshold never fired, and the only symptom was a
model that ended with half the expected Gaussian count. The counters below make
that specific failure visible within a few hundred steps instead of after a
20-minute run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class DensifyStats:
    """Counters and diagnostics accumulated across refine events."""

    events: int = 0
    total_duplicated: int = 0
    total_split: int = 0
    total_pruned: int = 0
    reset_events: int = 0
    #: Gradient percentiles observed at each refine event (visible Gaussians).
    grad_p50: List[float] = field(default_factory=list)
    grad_p90: List[float] = field(default_factory=list)
    grad_max: List[float] = field(default_factory=list)
    #: Fraction of visible Gaussians above the threshold at each event.
    over_threshold_frac: List[float] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    #: Consecutive refine events with zero candidates.
    consecutive_empty: int = 0

    def summary(self) -> str:
        if not self.events:
            return "densify: no refine events yet"
        base = (
            f"densify: {self.events} events, "
            f"+{self.total_duplicated} dup / +{self.total_split} split, "
            f"-{self.total_pruned} pruned, {self.reset_events} resets"
        )
        if self.grad_p50:
            base += (
                f"; grad p50={self.grad_p50[-1]:.3e} "
                f"over-thr={self.over_threshold_frac[-1]:.1%}"
            )
        return base


class DeadDensifierError(RuntimeError):
    """Raised when densification produces no candidates for too many events."""


class InstrumentedStrategy:
    """Wraps a gsplat strategy and records what it actually did.

    Delegates ``step_pre_backward`` / ``step_post_backward`` verbatim -- the
    ordering contract (pre -> backward -> post) is gsplat's and is not altered.
    """

    def __init__(
        self,
        strategy,
        *,
        threshold: float,
        dead_check_events: int = 5,
        log_first_events: int = 3,
    ) -> None:
        self.strategy = strategy
        self.threshold = float(threshold)
        self.dead_check_events = int(dead_check_events)
        self.log_first_events = int(log_first_events)
        self.stats = DensifyStats()
        self._n_at_last_event = 0
        self._last_grow = (0, 0)
        self._last_prune = 0
        self._install_counters()

    def _install_counters(self) -> None:
        """Record what ``_grow_gs`` / ``_prune_gs`` actually did.

        ``DefaultStrategy.step_post_backward`` computes ``(n_dupli, n_split)``
        and ``n_prune`` but only *prints* them; the values are not returned and
        the running grad stats are zeroed before it returns, so they cannot be
        recovered afterwards. Wrapping the two private helpers is the smallest
        way to observe them without re-implementing the decision logic -- which
        would be worse, since a re-implementation can drift from the real one.

        Absent helpers (MCMCStrategy has different internals) simply leave the
        counters at zero; the net Gaussian delta is still recorded.
        """
        for name, slot in (("_grow_gs", "_last_grow"), ("_prune_gs", "_last_prune")):
            original = getattr(self.strategy, name, None)
            if original is None or not callable(original):
                continue
            setattr(self.strategy, name, self._make_recorder(original, slot))

    def _make_recorder(self, fn, slot: str):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            setattr(self, slot, out)
            return out

        return wrapper

    def _accepts(self, name: str) -> bool:
        """Whether the wrapped strategy's ``step_post_backward`` takes ``name``."""
        import inspect

        try:
            sig = inspect.signature(self.strategy.step_post_backward)
        except (TypeError, ValueError):  # pragma: no cover
            return False
        if name in sig.parameters:
            return True
        # **kwargs would also accept it, but only pass what is certain.
        return False

    def _is_refine_event(self, step: int) -> bool:
        """Whether the wrapped strategy does any work at ``step``.

        ``step_post_backward`` runs every step, but the strategy only refines on
        ``step % refine_every == 0`` inside its window. Counting every call as an
        event makes the zero-growth guard fire after ``dead_check_events`` *steps*
        rather than after that many refine events, which is a false positive.

        The window bounds mirror gsplat exactly: it returns early when
        ``step >= refine_stop_iter`` and refines when ``step > refine_start_iter``
        (strict), so the interval here is ``(start, stop)``.
        """
        start = getattr(self.strategy, "refine_start_iter", 0)
        stop = getattr(self.strategy, "refine_stop_iter", float("inf"))
        every = max(1, int(getattr(self.strategy, "refine_every", 1)))
        return start < step < stop and step % every == 0

    def _is_reset_event(self, step: int) -> bool:
        every = int(getattr(self.strategy, "reset_every", 0) or 0)
        return every > 0 and step > 0 and step % every == 0

    # -- gsplat protocol ---------------------------------------------------
    def check_sanity(self, params, optimizers) -> None:
        self.strategy.check_sanity(params, optimizers)

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        # ``DefaultStrategy.initialize_state(scene_scale=...)`` and
        # ``MCMCStrategy.initialize_state()`` take different arguments, so the
        # accepted subset is filtered by signature rather than assumed.
        import inspect

        params = inspect.signature(self.strategy.initialize_state).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
        state = self.strategy.initialize_state(**accepted)
        self._n_at_last_event = 0
        return state

    def step_pre_backward(self, params, optimizers, state, step, info) -> None:
        self.strategy.step_pre_backward(
            params=params, optimizers=optimizers, state=state, step=step, info=info
        )

    def step_post_backward(
        self,
        params,
        optimizers,
        state,
        step,
        info,
        packed: bool = False,
        lr: float | None = None,
    ) -> None:
        """Delegate to the wrapped strategy, matching *its* signature.

        The two strategies in gsplat do **not** share a signature:
        ``DefaultStrategy`` takes ``packed`` while ``MCMCStrategy`` takes ``lr``
        and has no ``packed`` parameter. Passing ``packed`` unconditionally made
        MCMC fail with ``TypeError: unexpected keyword argument 'packed'``; not
        passing ``lr`` made it fail with ``missing required positional argument
        'lr'``. Both are one-line bugs that only surface the first time
        ``density.strategy="mcmc"`` is actually run, which is why the dispatch
        below inspects the signature instead of assuming.
        """
        # Snapshot the gradient distribution *before* the post step, because
        # whether a candidate exists is exactly what a dead densifier hides.
        n_before = len(params["means"])
        refine_event = self._is_refine_event(step)
        grad_stats = self._probe(info, packed) if (refine_event or self.stats.events < self.log_first_events) else None

        kwargs = dict(params=params, optimizers=optimizers, state=state,
                      step=step, info=info)
        if self._accepts("packed"):
            kwargs["packed"] = packed
        if self._accepts("lr"):
            if lr is None:
                raise ValueError(
                    "the wrapped strategy requires `lr` (MCMCStrategy uses it for "
                    "the position noise); pass lr=optimizers['means'].param_groups[0]['lr']"
                )
            kwargs["lr"] = float(lr)
        self.strategy.step_post_backward(**kwargs)

        n_after = len(params["means"])
        delta = n_after - n_before

        # Only a real refine event can be judged for growth; resets and prunes
        # legitimately shrink the model at other steps.
        if refine_event and grad_stats is not None:
            self.stats.events += 1
            self.stats.grad_p50.append(grad_stats["p50"])
            self.stats.grad_p90.append(grad_stats["p90"])
            self.stats.grad_max.append(grad_stats["max"])
            self.stats.over_threshold_frac.append(grad_stats["over_frac"])
            self.stats.counts.append(n_after)

            n_dupli, n_split = self._last_grow
            self.stats.total_duplicated += int(n_dupli)
            self.stats.total_split += int(n_split)
            self.stats.total_pruned += int(self._last_prune)
            self.stats.consecutive_empty = 0 if delta > 0 else self.stats.consecutive_empty + 1

            if (
                self.dead_check_events > 0
                and self.stats.consecutive_empty >= self.dead_check_events
            ):
                raise DeadDensifierError(
                    f"densification produced no new Gaussians for "
                    f"{self.stats.consecutive_empty} consecutive refine events "
                    f"(step {step}, {n_after} Gaussians). "
                    f"Last observed image-plane gradient: p50="
                    f"{grad_stats['p50']:.3e} p90={grad_stats['p90']:.3e} "
                    f"max={grad_stats['max']:.3e}, threshold={self.threshold:.3e}, "
                    f"{grad_stats['over_frac']:.1%} of visible Gaussians above it. "
                    "If over_frac is 0 the gradients are not reaching the "
                    "strategy (check meta['gradient_2dgs'] is the tensor the "
                    "kernel writes, and that step_pre_backward ran before "
                    "backward); if over_frac is high, the grow/prune size "
                    "thresholds are rejecting every candidate (check that "
                    "scene_scale matches the world gauge)."
                )

        if self._is_reset_event(step):
            self.stats.reset_events += 1

        self._n_at_last_event = n_after

    # -- diagnostics -------------------------------------------------------
    def _probe(
        self, info: Dict[str, Any], packed: bool
    ) -> Optional[Dict[str, float]]:
        """Gradient statistics that the strategy is about to threshold on.

        Recomputes exactly the quantity ``DefaultStrategy._update_state`` uses,
        so the reported numbers are directly comparable to the threshold.
        """
        key = getattr(self.strategy, "key_for_gradient", "gradient_2dgs")
        tensor = info.get(key)
        if tensor is None or tensor.grad is None:
            return None

        grads = tensor.grad.detach().clone()
        w = float(info.get("width", 1))
        h = float(info.get("height", 1))
        n_cams = float(info.get("n_cameras", 1))
        grads[..., 0] *= w / 2.0 * n_cams
        grads[..., 1] *= h / 2.0 * n_cams

        radii = info.get("radii")
        if packed:
            if radii is not None:
                vis = radii.reshape(-1) > 0.0
                g = grads.reshape(-1, 2)[vis]
            else:
                g = grads.reshape(-1, 2)
        else:
            if radii is not None and radii.dim() >= 2:
                vis = (radii > 0.0).all(dim=-1)
                # grads is [C,N,2]; select per-camera then flatten to [nnz,2].
                g = grads[vis]
            else:
                g = grads.reshape(-1, 2)

        if g.numel() == 0:
            return {"p50": 0.0, "p90": 0.0, "max": 0.0, "over_frac": 0.0}

        norms = g.norm(dim=-1)
        return {
            "p50": float(torch.quantile(norms, 0.5)),
            "p90": float(torch.quantile(norms, 0.9)),
            "max": float(norms.max()),
            "over_frac": float((norms > self.threshold).float().mean()),
        }

    def should_log(self) -> bool:
        return self.stats.events <= self.log_first_events

    def maybe_warn(self, log_fn=print) -> None:
        """Emit one warning if gradients are systematically below threshold."""
        if not self.stats.over_threshold_frac:
            return
        frac = self.stats.over_threshold_frac[-1]
        if frac == 0.0 and self.stats.events == self.log_first_events:
            log_fn(
                "WARNING: no visible Gaussian exceeded the densification "
                f"threshold ({self.threshold:.3e}) in the first "
                f"{self.stats.events} refine events. Measured p90 was "
                f"{self.stats.grad_p90[-1]:.3e}. Either the threshold is too "
                "high for this scene's resolution, or image-plane gradients are "
                "not reaching the strategy."
            )


def footprint_threshold_scale(measured_footprint: float, cfg) -> float:
    """Factor applied to every screen-space densification threshold.

    ``grow_grad2d`` is a screen-space quantity -- gsplat multiplies the kernel's
    ``gradient_2dgs`` by ``(W/2, H/2)`` before comparing -- so the same physical
    surface yields a larger statistic on a larger image. ``scene_scale``, by
    contrast, comes from the scene, so ``grow_scale3d``/``prune_scale3d`` do not
    respond to resolution, and the kNN-derived initial scales do not either.

    Measured on garden (one 1.5M cloud, one hyper-parameter set, only the render
    resolution changed): the over-threshold fraction went from 0.65% at 0.634 px
    to 2.9% at 2.538 px, i.e. ``F ~ footprint^1.08``. Scaling the thresholds by
    that ratio makes one configuration admit a comparable fraction at either
    resolution.

    Returns 1.0 when compensation is disabled, so callers need no branch.
    """
    d = cfg.density
    if not getattr(d, "footprint_compensate", False):
        return 1.0
    ref = float(getattr(d, "footprint_ref", 1.0))
    if ref <= 0.0:
        raise ValueError(
            f"density.footprint_ref must be positive, got {ref} (only its ratio "
            "against the measured footprint is used)"
        )
    measured = float(measured_footprint)
    if not np.isfinite(measured) or measured <= 0.0:
        raise ValueError(
            "footprint compensation is enabled but the measured footprint is "
            f"{measured!r}; a non-positive or non-finite value would scale the "
            "thresholds by 0 or NaN and silently disable densification"
        )
    exponent = float(getattr(d, "footprint_exponent", 1.0))
    return (measured / ref) ** exponent


def build_strategy(
    cfg,
    *,
    scene_scale: float,
    device: str | torch.device = "cuda",
    n_initial: int | None = None,
    measured_footprint: float | None = None,
):
    """Construct the configured gsplat strategy for 2DGS.

    ``scene_scale`` must be the world extent **in the final gauge**. It is what
    ``DefaultStrategy`` divides Gaussian sizes by; see the module docstring for
    what passing the wrong value does.

    ``n_initial`` is only used by the MCMC branch, to catch a ``cap_max`` smaller
    than the starting Gaussian count. In that case ``n_target =
    min(cap_max, 1.05*n)`` equals ``n`` forever and ``add_new_gs`` silently adds
    nothing -- a real risk here, since a DGI cloud can exceed the 1e6 default.

    ``measured_footprint`` (in render pixels) enables the resolution
    compensation described on :func:`footprint_threshold_scale`. It is required
    when ``density.footprint_compensate`` is set: guessing it would reintroduce
    the very resolution dependence the option exists to remove.
    """
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    d = cfg.density
    if d.absgrad:
        # Config validation already rejects this, but a programmatic caller can
        # bypass that; the failure it prevents is an AttributeError deep inside
        # the strategy, so it is worth restating with the reason.
        raise ValueError(
            "absgrad=True is not supported for 2DGS: meta['gradient_2dgs'] has "
            "no .absgrad attribute because the kernel accumulates a signed "
            "gradient and has no absolute variant"
        )

    threshold_scale = 1.0
    if getattr(d, "footprint_compensate", False):
        if measured_footprint is None:
            raise ValueError(
                "density.footprint_compensate=True requires measured_footprint "
                "(the median splat radius in render pixels, from "
                "surfel.data.gauge.projected_footprint_px). Defaulting it would "
                "silently reintroduce the resolution dependence."
            )
        threshold_scale = footprint_threshold_scale(measured_footprint, cfg)

    if d.strategy == "mcmc":
        if n_initial is not None and d.mcmc_cap_max <= n_initial:
            raise ValueError(
                f"density.mcmc_cap_max={d.mcmc_cap_max} is <= the initial "
                f"Gaussian count ({n_initial}), so MCMC can never add a Gaussian: "
                "gsplat computes n_target = min(cap_max, 1.05*n), which equals n "
                "for every step. Raise mcmc_cap_max above the initial count, or "
                "use a sparser init cloud."
            )
        strategy = MCMCStrategy(
            cap_max=d.mcmc_cap_max,
            noise_lr=d.mcmc_noise_lr,
            refine_start_iter=d.refine_start_iter,
            refine_stop_iter=d.refine_stop_iter,
            refine_every=d.refine_every,
            min_opacity=d.mcmc_min_opacity,
            verbose=True,
        )
        return strategy, {}

    strategy = DefaultStrategy(
        prune_opa=d.prune_opa,
        # Resolution compensation, in one place. ``grow_grad2d`` is a
        # screen-space statistic, so a larger footprint produces a larger value
        # for the same physical structure and the threshold must rise with it.
        # The size thresholds are fractions of ``scene_scale``, which does not
        # respond to render resolution, so without the same factor they would
        # stay put while every Gaussian's *screen* extent grew -- which is
        # exactly the combination measured to let a 1.5M model grow to 4.73M
        # with a flat metric. See ``footprint_threshold_scale``.
        grow_grad2d=d.grow_grad2d * threshold_scale,
        grow_scale3d=d.grow_scale3d * threshold_scale,
        prune_scale3d=(
            d.prune_scale3d * threshold_scale
            if getattr(d, "footprint_compensate_prune", True)
            else d.prune_scale3d
        ),
        refine_scale2d_stop_iter=d.refine_scale2d_stop_iter,
        refine_start_iter=d.refine_start_iter,
        refine_stop_iter=d.refine_stop_iter,
        reset_every=d.reset_every,
        refine_every=d.refine_every,
        absgrad=False,
        verbose=True,
        # Mandatory for 2DGS: the kernel writes the densification accumulator to
        # this key, not to means2d.
        key_for_gradient="gradient_2dgs",
    )
    # The wiring fix: pass the real scene scale rather than letting it default
    # to 1.0 (which is what the official 2DGS example does).
    if cfg.density.opacity_reset_value is not None:
        _apply_opacity_reset_value(strategy, cfg.density.opacity_reset_value)
    return strategy, {"scene_scale": float(scene_scale)}


def _apply_opacity_reset_value(strategy, value: float) -> None:
    """Override the opacity value that ``reset_opa`` restores to.

    ``DefaultStrategy`` calls ``reset_opa(..., value=self.prune_opa * 2.0)``
    (``default.py:219-224``): the reset target is derived from ``prune_opa`` and
    cannot be configured. GSL exposes it separately (``opacity_reset_value``) and
    its GNS config uses 0.05 where ``prune_opa*2`` would give 0.01.

    ``ops.reset_opa``'s ``value`` is the **post-sigmoid** probability (it does
    the logit internally), so it is forwarded unchanged. Implemented by patching
    the module symbol rather than editing gsplat: the call site passes it as a
    keyword, so intercepting preserves the rest of the reset path (Adam state
    zeroing and the scene hooks).
    """
    from gsplat.strategy import ops as strategy_ops
    import gsplat.strategy.default as default_mod

    target = float(value)
    if not (0.0 < target < 1.0):
        raise ValueError(
            f"opacity_reset_value must be in (0, 1) as a post-sigmoid "
            f"probability, got {target}"
        )
    original = strategy_ops.reset_opa

    def reset_opa(params, optimizers, state, value):  # noqa: ARG001
        return original(params=params, optimizers=optimizers, state=state,
                        value=target)

    strategy_ops.reset_opa = reset_opa
    if hasattr(default_mod, "reset_opa"):
        default_mod.reset_opa = reset_opa


def describe_thresholds(
    scene_scale: float, cfg, threshold_scale: float = 1.0
) -> str:
    """Human-readable statement of what the size thresholds mean in world units.

    ``threshold_scale`` is the footprint compensation actually applied (1.0 when
    disabled). Reporting it matters more than it looks: with compensation on, the
    world-unit numbers no longer match ``cfg.density.*`` and a reader comparing a
    log against a config file would otherwise conclude the values were ignored.
    """
    d = cfg.density
    grow = d.grow_scale3d * threshold_scale * scene_scale
    prune = d.prune_scale3d * threshold_scale * scene_scale
    note = (
        f", footprint-compensated x{threshold_scale:.3g}"
        if threshold_scale != 1.0
        else ""
    )
    return (
        f"densify size thresholds: duplicate below {grow:.6g}, "
        f"split above {grow:.6g}, prune above {prune:.6g} "
        f"(scene_scale {scene_scale:.6g}{note})"
    )


__all__ = [
    "DensifyStats",
    "InstrumentedStrategy",
    "DeadDensifierError",
    "build_strategy",
    "describe_thresholds",
]
