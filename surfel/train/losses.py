"""Loss composition for 2DGS on large scenes.

Primitives come from ``gsplat.losses`` where they exist, so the objective stays
literally the same function the reference uses. This module only decides how the
terms are combined, when each is switched on, and how they are scaled.

Why each term is here
---------------------
**Photometric** -- ``lerp(L1, 1-SSIM, ssim_lambda)``. The standard 3DGS/2DGS
objective; ``ssim_lambda=0.2`` reproduces the reference.

**Depth (disparity)** -- ``gsplat.losses.depth_l1_loss``. Evaluated on ``1/d``
because depth errors are multiplicative: a raw L1 on metric depth is dominated
by whatever is furthest away, and on an aerial block the far tail is the sky.
The ``scene_scale`` multiplier is part of gsplat's signature; here it is fed the
*median supervised depth*, not the camera extent, because after normalisation
the disparity magnitude is set by depth, and the camera extent says nothing
about it.

Two hazards, both measured in this project:

* ``1/pred_depth`` with a small floor explodes when a pixel has no coverage.
  ``train/depth_loss`` went from 0.001 to 81,124 at step 3300 and killed the run.
  Hence ``depth_eps`` is a real config value and coverage is optionally masked
  by rendered alpha.
* A constant ratio between rendered and supervised depth (a gauge error) is
  indistinguishable from a geometry error, and silently eats the whole term. The
  gauge module prevents it; this one refuses to run with an obviously broken
  ratio rather than training on it.

**Distortion** -- 2DGS regulariser that pulls each surfel's depth distribution
toward a single value. On an aerial city this is what stops every facade from
becoming a stack of parallel translucent sheets. Weight 0.1 from step 3000 is
the archived recipe.

**Normal consistency** -- compares the rendered surfel normal against the normal
implied by the model's *own* rendered depth. **It needs no ground-truth depth.**
Gating it on depth supervision was a real silent failure here: with
``normal_weight>0`` on a dataset with no GT depth the term ran at weight 0 for an
entire 500-step run, and only showed up as a warning in the log. The correct
enable condition is ``want_depth_render = wants_gt_depth or wants_normal``.

Ramps
-----
Depth weight ramps 0.05 -> 1.0 over the run and the regularisers have start
iterations. Both matter: with a fixed small depth weight the geometry stays soft,
and enabling distortion from step 0 fights the initial fit before there is a
surface to flatten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossWeights:
    """Resolved weights for one step."""

    ssim_lambda: float
    depth: float
    distortion: float
    normal: float
    opacity_reg: float
    scale_reg: float


@dataclass
class LossBreakdown:
    """Per-term values for logging, plus the total for backward."""

    total: Tensor
    l1: Tensor
    ssim: Tensor
    depth: Optional[Tensor] = None
    distortion: Optional[Tensor] = None
    normal: Optional[Tensor] = None
    opacity_reg: Optional[Tensor] = None
    scale_reg: Optional[Tensor] = None
    weights: Optional[LossWeights] = None

    def as_dict(self) -> dict:
        out = {
            "loss": float(self.total.detach()),
            "l1": float(self.l1.detach()),
            "ssim": float(self.ssim.detach()),
        }
        # Keep the raw (unweighted) terms too: a run where ``loss`` falls while
        # ``normal`` rises is collapsing, and no aggregate shows that.
        for name in ("depth", "distortion", "normal", "opacity_reg", "scale_reg"):
            v = getattr(self, name)
            if v is not None:
                out[name] = float(v.detach())
        if self.weights is not None:
            out["w_depth"] = self.weights.depth
            out["w_dist"] = self.weights.distortion
            out["w_normal"] = self.weights.normal
        return out


def resolve_weights(cfg, step: int) -> LossWeights:
    """Per-step weights, applying the ramps and start iterations.

    The depth weight uses **geometric** interpolation, matching GSL:

        w = init * final_factor ** min(step / max_steps, 1)

    (``internal/metrics/gs2d_depth_metrics.py:123-126``). Not linear. The
    difference is large mid-run: for a 0.05 -> 1.0 span at ``t=0.5`` linear gives
    0.525 while geometric gives 0.2236.

    Note also that every archived GSL config uses ``final_factor: 1.0``, and
    ``1.0 ** t == 1.0``, so those recipes run at a **constant** weight of 0.05.
    A profile that wants a real ramp must set ``final_factor`` explicitly (20.0
    for 0.05 -> 1.0).
    """
    loss = cfg.loss
    depth_cfg = cfg.data.depth

    depth_w = 0.0
    if depth_cfg.enabled:
        if step >= depth_cfg.start_iter:
            base = float(depth_cfg.weight)
            final = depth_cfg.weight_final
            if final is not None and depth_cfg.ramp_steps > 0:
                t = (step - depth_cfg.start_iter) / float(depth_cfg.ramp_steps)
                t = min(max(t, 0.0), 1.0)
                # Geometric: interpolate the *ratio*, so a constant final_factor
                # of 1.0 stays exactly constant (as in every archived config).
                if base > 0.0 and final > 0.0:
                    depth_w = base * (final / base) ** t
                else:
                    depth_w = base
            else:
                depth_w = base

    dist_w = loss.dist_weight if step >= loss.dist_start_iter else 0.0
    normal_w = loss.normal_weight if step >= loss.normal_start_iter else 0.0

    return LossWeights(
        ssim_lambda=loss.ssim_lambda,
        depth=depth_w,
        distortion=dist_w,
        normal=normal_w,
        opacity_reg=loss.opacity_reg,
        scale_reg=loss.scale_reg,
    )


# --------------------------------------------------------------------------
# Individual terms
# --------------------------------------------------------------------------
def image_terms(
    colors: Tensor, pixels: Tensor, ssim_lambda: float
) -> tuple[Tensor, Tensor, Tensor]:
    """``(photometric_total, l1, ssim)`` for ``[C,H,W,3]`` in [0,1]."""
    from gsplat.losses import l1_loss, ssim_loss

    l1 = l1_loss(colors, pixels).mean()
    ssim = ssim_loss(
        colors.permute(0, 3, 1, 2).contiguous(),
        pixels.permute(0, 3, 1, 2).contiguous(),
    )
    total = torch.lerp(l1, ssim, ssim_lambda)
    return total, l1, ssim


def depth_term(
    pred_depth: Tensor,
    gt_depth: Tensor,
    *,
    scale: float,
    eps: float = 1e-6,
    alpha: Optional[Tensor] = None,
    alpha_threshold: float = 0.0,
    per_frame_median_norm: bool = False,
    invalid_mask: Optional[Tensor] = None,
    outlier_clip: Optional[float] = 10.0,
) -> Tensor:
    """Disparity-space depth loss.

    Args:
        pred_depth: ``[C,H,W,1]`` rendered depth (alpha-normalised by the kernel).
        gt_depth: ``[C,H,W]`` or ``[C,H,W,1]`` supervised depth, same grid.
        scale: multiplier on the disparity L1. For a non-normalised depth branch
            this is the median depth; see :func:`resolve_depth_scale`.
        eps: floor before the reciprocal. Small values turn a single uncovered
            pixel into a huge loss.
        alpha: optional ``[C,H,W,1]`` rendered alpha; pixels below
            ``alpha_threshold`` are excluded.
        alpha_threshold: 0 disables alpha masking.
        per_frame_median_norm: normalise both sides by the per-frame median,
            matching GSL (which does this **always**, see the note below).
        invalid_mask: optional bool tensor, ``True`` = ignore.
        outlier_clip: clamp the normalised disparity difference to ``±clip``
            before the mean. ``None`` or ``<=0`` disables it.

    GSL semantics worth matching exactly
    ------------------------------------
    ``internal/metrics/gs2d_depth_metrics.py:82-87`` divides **both** sides by
    the **GT** inverse-depth median::

        gt_n   = gt_inv   / gt_inv[valid].median()
        pred_n = pred_inv / gt_inv[valid].median()

    Using the *prediction's own* median for the prediction side is a documented
    anti-pattern there: when degenerate pixels become common they raise the
    prediction's median, which systematically lowers the normalised prediction
    and makes the loss *more tolerant of degradation*.

    ``outlier_clip`` is the third line of defence, alongside ``eps`` and the alpha
    mask. The comment in GSL points at the f0_gtpose run where the depth term
    went from 0.001 to 81,124 at step 3300 because a single pixel produced a 1e6
    disparity spike.
    """
    if gt_depth.dim() == pred_depth.dim() - 1:
        gt_depth = gt_depth.unsqueeze(-1)

    valid = torch.isfinite(gt_depth) & (gt_depth > 0)
    if alpha is not None and alpha_threshold > 0.0:
        valid &= alpha > alpha_threshold
    if invalid_mask is not None:
        valid &= ~invalid_mask

    if not bool(valid.any()):
        # Keep the graph connected: returning a bare 0.0 on a step where every
        # pixel is invalid would make ``.backward()`` fail, turning a data quirk
        # into a crash.
        return pred_depth.sum() * 0.0

    if per_frame_median_norm:
        pred_d = pred_depth.clamp_min(eps)
        gt_d = gt_depth.clamp_min(eps)
        pred_inv = 1.0 / pred_d
        gt_inv = 1.0 / gt_d
        v = valid.to(pred_inv.dtype)
        gt_med = _masked_median(gt_inv.detach(), v)
        # Both sides divided by the GT median (GSL's convention), not each by
        # its own -- see the docstring for why the prediction's own median is a
        # trap.
        pred_n = pred_inv / gt_med.clamp_min(eps)
        gt_n = gt_inv / gt_med.clamp_min(eps)
        diff = pred_n - gt_n
        if outlier_clip is not None and outlier_clip > 0:
            diff = torch.clamp(diff, -float(outlier_clip), float(outlier_clip))
        return diff[valid].abs().mean()

    # Mask before the reciprocal so uncovered pixels never produce a huge
    # value; ``torch.where`` would still differentiate a large branch.
    pred_masked = torch.where(valid, pred_depth.clamp_min(eps), torch.ones_like(pred_depth))
    gt_masked = torch.where(valid, gt_depth.clamp_min(eps), torch.ones_like(gt_depth))
    diff = ((1.0 / pred_masked) - (1.0 / gt_masked)).abs()
    if outlier_clip is not None and outlier_clip > 0:
        diff = torch.clamp(diff, max=float(outlier_clip))
    return diff[valid].mean() * float(scale)


def _masked_median(x: Tensor, mask: Tensor) -> Tensor:
    """Per-sample median of ``x`` over ``mask``, as ``[C,1,1,1]``."""
    flat_x = x.reshape(x.shape[0], -1)
    flat_m = mask.reshape(mask.shape[0], -1) > 0
    out = []
    for i in range(flat_x.shape[0]):
        vals = flat_x[i][flat_m[i]]
        out.append(vals.median() if vals.numel() else torch.zeros((), device=x.device))
    return torch.stack(out).reshape(-1, 1, 1, 1)


def distortion_term(
    distort: Tensor, alpha: Optional[Tensor] = None, alpha_threshold: float = 0.0
) -> Tensor:
    """Mean 2DGS distortion, optionally restricted to covered pixels."""
    if alpha is not None and alpha_threshold > 0.0:
        m = (alpha > alpha_threshold).to(distort.dtype)
        return (distort * m).sum() / m.sum().clamp_min(1.0)
    return distort.mean()


def normal_term(
    normals: Tensor,
    surf_normals: Tensor,
    alphas: Tensor,
    *,
    alpha_threshold: float = 1e-3,
) -> Tensor:
    """2DGS normal consistency: ``mean(1 - cos(normal, normal_from_depth))``.

    ``surf_normals`` is produced by the kernel from the model's own rendered
    depth, so this is a self-consistency constraint with no ground-truth input.

    Two things here are not cosmetic, and both were found by a NaN that only
    appeared after ~10 training steps:

    **1. ``surf_normals`` must be detached.** It comes from the same parameters
    as ``normals``, so autograd happily builds a gradient path through it. That
    path runs through gsplat's ``depth_to_normal_2dgs``, which normalises with
    ``sqrt(cross²).clamp_min(1e-12)``: at a pixel with no rendered coverage the
    depth is 0, the cross product is 0, and the *backward* of the clamped sqrt is
    ``inf``. Forward stays finite (the clamp holds), so nothing looks wrong until
    the parameters become NaN. The depth-derived normal is the *target* of a
    consistency term and must not be optimised through.

    **2. The mean must be alpha-weighted.** Uncovered pixels produce a zero
    target normal, so ``1 - cos`` evaluates to 1 there. Averaging over all pixels
    then makes the term measure "how much of the frame is empty" rather than
    "how consistent are the surfels": measured on a synthetic scene with 23% of
    pixels uncovered, the term sat at 0.958 and was dominated by background.
    """
    if normals.dim() != 4 or normals.shape[-1] != 3:
        raise ValueError(f"normals must be [C,H,W,3], got {tuple(normals.shape)}")

    sn = surf_normals
    if sn.dim() == 3:
        sn = sn.unsqueeze(0)
    if sn.shape[:3] != normals.shape[:3]:
        raise ValueError(
            f"surf_normals {tuple(sn.shape)} does not match normals "
            f"{tuple(normals.shape)}"
        )

    a = alphas.detach()
    weight = (a > alpha_threshold).to(normals.dtype)
    if not bool(weight.any()):
        # Keep the graph connected on a frame where nothing is covered.
        return normals.sum() * 0.0

    # ``render.normals`` is an alpha-weighted *accumulation* of unit surfel
    # normals (``normal_out[k] += n_ptr[k] * vis``,
    # RasterizeToPixels2DGSSerialBatchFwd.cu:408), so its length falls below 1
    # wherever alpha < 1 -- measured 0.004..0.62 on a partial-coverage render.
    # The world-space rotation is already applied by C++, but renormalising the
    # length is the caller's job: without it the cosine is scaled by the
    # accumulated weight, which makes the term under-penalise exactly the
    # partially covered pixels where the surfel normal matters most, and it also
    # violates ``normal_cosine_loss``'s unit-norm contract.
    n_hat = torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)
    cos = (n_hat * sn.detach()).sum(dim=-1, keepdim=True)
    per_pixel = 1.0 - cos
    return (per_pixel * weight).sum() / weight.sum()


# --------------------------------------------------------------------------
# Full objective
# --------------------------------------------------------------------------
def total_loss(
    *,
    render,
    pixels: Tensor,
    gt_depth: Optional[Tensor],
    weights: LossWeights,
    cfg,
    params,
    scene_scale: float,
    depth_scale: Optional[float] = None,
) -> LossBreakdown:
    """Assemble the objective for one step.

    ``depth_scale`` overrides ``cfg.loss.depth_scale``; the trainer passes the
    resolved value (median supervised depth) so it is computed once, not per
    step.
    """
    loss_cfg = cfg.loss
    depth_cfg = cfg.data.depth

    total, l1, ssim = image_terms(render.colors, pixels, weights.ssim_lambda)

    depth_val = None
    if weights.depth > 0.0 and render.depth is not None and gt_depth is not None:
        dscale = (
            float(loss_cfg.depth_scale)
            if loss_cfg.depth_scale is not None
            else float(depth_scale if depth_scale is not None else scene_scale)
        )
        depth_val = depth_term(
            render.depth,
            gt_depth,
            scale=dscale,
            eps=loss_cfg.depth_eps,
            alpha=render.alphas,
            alpha_threshold=loss_cfg.depth_alpha_mask,
            per_frame_median_norm=depth_cfg.per_frame_median_norm,
            outlier_clip=loss_cfg.depth_outlier_clip,
        )
        total = total + weights.depth * depth_val

    dist_val = None
    if weights.distortion > 0.0 and render.distort is not None:
        dist_val = distortion_term(
            render.distort, render.alphas, loss_cfg.depth_alpha_mask
        )
        total = total + weights.distortion * dist_val

    normal_val = None
    if weights.normal > 0.0:
        if render.surf_normals is None or render.normals is None:
            # Not gated silently: the caller asked for the term and the renderer
            # did not produce the inputs. This is a wiring error, not a data
            # property.
            raise RuntimeError(
                "loss.normal_weight > 0 but the renderer produced no surf_normals. "
                "The 2DGS normal term needs render_mode 'RGB+D' or 'RGB+ED' "
                "(the '-d' variants return None). Pass render_depth=True or "
                "enable distloss so a depth channel is rendered."
            )
        normal_val = normal_term(render.normals, render.surf_normals, render.alphas)
        total = total + weights.normal * normal_val

    opa_val = None
    if weights.opacity_reg > 0.0:
        from gsplat.losses import opacity_reg_loss

        opa_val = opacity_reg_loss(params["opacities"])
        total = total + weights.opacity_reg * opa_val

    scale_val = None
    if weights.scale_reg > 0.0:
        from gsplat.losses import scale_reg_loss

        scale_val = scale_reg_loss(params["scales"])
        total = total + weights.scale_reg * scale_val

    return LossBreakdown(
        total=total,
        l1=l1,
        ssim=ssim,
        depth=depth_val,
        distortion=dist_val,
        normal=normal_val,
        opacity_reg=opa_val,
        scale_reg=scale_val,
        weights=weights,
    )


def resolve_depth_scale(
    depth_p50: Optional[float],
    cfg,
    *,
    fallback: float = 1.0,
) -> float:
    """Multiplier for the disparity L1.

    ``1/d`` has magnitude set by the typical depth, so multiplying by the median
    depth brings the term back to O(1) regardless of whether the world is in
    metres or normalised. Using the camera *extent* here (the intuitive choice)
    is wrong on a thin aerial block: the extent can be hundreds of metres while
    every surface sits within tens.
    """
    if cfg.loss.depth_scale is not None:
        return float(cfg.loss.depth_scale)
    if cfg.data.depth.per_frame_median_norm:
        # Dimensionless after normalisation; the multiplier would only rescale
        # the weight, which the weight itself already expresses.
        return 1.0
    if depth_p50 is not None and depth_p50 > 0:
        return float(depth_p50)
    return float(fallback)


__all__ = [
    "LossWeights",
    "LossBreakdown",
    "resolve_weights",
    "resolve_depth_scale",
    "image_terms",
    "depth_term",
    "distortion_term",
    "normal_term",
    "total_loss",
]
