"""2DGS splat parameters and initialisation.

gsplat's 2DGS kernel reads ``scales[:, :2]`` (``Projection2DGSFused.cu`` builds the
covariance with ``mat3(sx, 0, 0, 0, sy, 0, 0, 0, 1)``), so the third scale axis
never influences rendering. It is nevertheless a *parameter*, and two things
depend on it:

* ``DefaultStrategy`` classifies duplicate-vs-split with
  ``exp(scales).max(dim=-1)``, which includes the third axis;
* the split op samples children with ``rotmat @ scales @ randn``, so an
  oversized third axis throws children far outside the surfel.

If the third axis were initialised from kNN spacing like the first two, that
spacing (often ~1% of scene extent) would dominate ``max()`` and force every
candidate down the *split* branch regardless of its true footprint -- the
behaviour the project log recorded as "every high-gradient Gaussian took the
duplicate branch" inverted. It is therefore initialised to a negligible value.

The initial *size* comes from kNN spacing, matching the reference:

    dist_avg = sqrt(mean(d_j^2))  over the k-1 nearest neighbours
    scale    = log(dist_avg * init_scale)

RMS, not arithmetic mean: ``simple_knn``'s ``distCUDA2`` uses the mean of
squared distances, and using ``mean(d)`` instead shrinks every Gaussian. That
difference was measured and matters (it changes the initial model enough to
break equivalence with the reference).

The kNN search itself must not materialise an ``[N, M, 3]`` tensor. On a 1.82M
point cloud the naive ``torch.cdist`` form needs ~90 GB; on this project's
unified-memory machine that did not raise ``OutOfMemoryError``, it drove the
host into swap and rebooted it. ``scipy.cKDTree`` does the same job in O(N) memory
in about a second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> the DC spherical-harmonic coefficient.

    ``0.28209479177387814`` is ``1/(2*sqrt(pi))``, the L=0 SH basis value. The
    inverse (``+0.5``) lives in the rasteriser, not here.
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def knn_mean_distance_rms(
    points: torch.Tensor,
    k: int = 4,
    *,
    use_scipy: bool = True,
    max_points_for_exact: int = 2_000_000,
) -> torch.Tensor:
    """RMS distance to the ``k-1`` nearest neighbours, per point.

    Args:
        points: ``[N, 3]``.
        k: neighbours including self, so ``k-1`` distances are averaged.

    Uses ``scipy.cKDTree`` when available; falls back to a blocked torch
    implementation only for small clouds, because the blocked form still needs
    roughly ``N * K * 3`` floats and grows linearly with N.
    """
    pts = points.detach()
    n = pts.shape[0]
    if n < 2:
        return torch.full((n,), 1e-6, device=pts.device, dtype=pts.dtype)

    k = max(2, min(k, n))

    if use_scipy:
        try:
            from scipy.spatial import cKDTree
        except ModuleNotFoundError:  # pragma: no cover
            use_scipy = False

    if use_scipy:
        tree = cKDTree(pts.detach().cpu().double().numpy())
        # A tree query on all N returns (dist, idx) of shape [N, k]; memory is
        # ~N*k*16 bytes, which for 1.8M points and k=4 is 115 MB.
        dist, _ = tree.query(pts.detach().cpu().double().numpy(), k=k, workers=-1)
        dist = np.atleast_2d(dist)
        # Column 0 is the point itself (distance 0).
        d2 = torch.from_numpy(np.asarray(dist[:, 1:] ** 2, dtype=np.float64))
        rms = torch.sqrt(d2.mean(dim=-1))
        return rms.to(device=pts.device, dtype=pts.dtype)

    if n > max_points_for_exact:
        raise RuntimeError(
            f"kNN on {n} points without scipy would need an [N,K,3] tensor "
            f"(~{n * k * 3 * 4 / 1e9:.1f} GB) and has been refused. "
            "Install scipy, or subsample the point cloud."
        )
    # Blocked exact fallback.
    chunk = 4096
    out = torch.empty(n, device=pts.device, dtype=pts.dtype)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = pts[start:end]
        d = torch.cdist(block, pts)  # [chunk, N]
        vals, _ = torch.topk(d, k, dim=-1, largest=False)
        out[start:end] = torch.sqrt((vals[:, 1:] ** 2).mean(dim=-1))
    return out


@dataclass
class SplatInit:
    """Initial Gaussian parameters and their optimiser learning rates."""

    params: Dict[str, torch.nn.Parameter]
    #: per-parameter LR, before batch scaling
    lrs: Dict[str, float]

    def __getitem__(self, key: str) -> torch.nn.Parameter:
        return self.params[key]

    def to_parameter_dict(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict(self.params)


def create_splats_2dgs(
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor,
    *,
    sh_degree: int,
    scene_scale: float,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    init_scale_mult: float = 1.0,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_sh0: float = 2.5e-3,
    lr_shN: float = 2.5e-3 / 20.0,
    device: str | torch.device = "cuda",
    dead_axis_scale: float = 1e-6,
) -> SplatInit:
    """Build the 2DGS parameter set from an initial point cloud.

    Args:
        points: ``[N,3]`` initial centres.
        colors: ``[N,3]`` in [0,1].
        scene_scale: multiplier on the means LR and the reference the strategy
            divides Gaussian sizes by. Must be in the *final* gauge (i.e. after
            any normalisation), which is what :class:`~surfel.data.Scenes`
            reports.
    """
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {tuple(pts.shape)}")
    n = pts.shape[0]
    if n == 0:
        raise ValueError("cannot initialise from an empty point cloud")

    rgb = torch.as_tensor(np.asarray(colors), dtype=torch.float32)
    if rgb.shape != (n, 3):
        raise ValueError(f"colors must be [{n},3], got {tuple(rgb.shape)}")
    rgb = rgb.clamp(0.0, 1.0)

    spacing = knn_mean_distance_rms(pts)  # [N]
    # Floor prevents log(0) for duplicate points, which a DGI cloud can contain
    # along depth-map seams.
    spacing = spacing.clamp_min(1e-8)
    base_scale = (spacing * float(init_scale) * float(init_scale_mult)).unsqueeze(-1)

    # Two live axes share the kNN spacing; the third is inert in the kernel but
    # visible to the strategy, so keep it negligible (see the module docstring).
    scales = torch.cat(
        [torch.log(base_scale).repeat(1, 2), torch.full((n, 1), math.log(dead_axis_scale))],
        dim=-1,
    )

    quats = torch.rand((n, 4))
    opacities = torch.logit(torch.full((n,), float(init_opacity)))

    colors_sh = torch.zeros((n, (sh_degree + 1) ** 2, 3))
    colors_sh[:, 0, :] = rgb_to_sh0(rgb)

    _assert_dead_axis(scales, scene_scale, dead_axis_scale)

    params = {
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors_sh[:, :1, :]),
    }
    lrs = {
        "means": lr_means * scene_scale,
        "scales": lr_scales,
        "quats": lr_quats,
        "opacities": lr_opacities,
        "sh0": lr_sh0,
    }
    if sh_degree > 0:
        params["shN"] = torch.nn.Parameter(colors_sh[:, 1:, :])
        lrs["shN"] = lr_shN

    for v in params.values():
        v.data = v.data.to(device)
    return SplatInit(params=params, lrs=lrs)


def activated(params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict) -> Dict[str, torch.Tensor]:
    """Raw parameters -> what the rasteriser expects.

    ``means`` and ``sh`` pass through; ``scales`` and ``opacities`` are
    exponentiated / sigmoided; quaternions are normalised. Note the rasteriser
    also normalises quaternions internally, but normalising here keeps the
    parameter's gradient well scaled rather than relying on that.
    """
    p = params
    out = {
        "means": p["means"],
        "scales": torch.exp(p["scales"]),
        "opacities": torch.sigmoid(p["opacities"]),
        "quats": torch.nn.functional.normalize(p["quats"], dim=-1),
    }
    if "shN" in p and p["shN"].shape[1] > 0:
        out["sh"] = torch.cat([p["sh0"], p["shN"]], dim=1)
    else:
        out["sh"] = p["sh0"]
    return out


def build_optimizers(
    init: SplatInit,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    eps_base: float = 1e-15,
) -> Dict[str, torch.optim.Optimizer]:
    """One Adam per parameter, with batch-size-adjusted LR and betas.

    The scaling (``lr * sqrt(B)``, ``betas = (1 - B*(1-0.9), 1 - B*(1-0.999))``)
    is gsplat's own convention for its stochastic-gradient interpretation. With
    ``B=1`` it reduces to plain Adam, which is the common case here.
    """
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    sqrt_b = math.sqrt(batch_size)
    for name, param in init.params.items():
        optimizers[name] = torch.optim.Adam(
            [{"params": param, "lr": init.lrs[name] * sqrt_b}],
            eps=eps_base / sqrt_b,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
    return optimizers


def means_lr_scheduler(
    optimizers: Dict[str, torch.optim.Optimizer],
    lr_init: float,
    lr_final: float,
    max_steps: int,
    *,
    delay_steps: int = 0,
):
    """Exponential decay for ``means`` toward ``lr_final`` by ``max_steps``.

    Matches the archived project recipes (1.6e-4 -> 1.6e-6), which the upstream
    gsplat example does not do. Profiles choose which behaviour to use; this
    function is only the mechanism.
    """
    means_lr_init = lr_init
    means_lr_final = min(lr_init, lr_final)

    def step(step: int) -> None:
        if step <= delay_steps:
            lr = means_lr_init
        else:
            t = (step - delay_steps) / max(1, max_steps - delay_steps)
            t = min(max(t, 0.0), 1.0)
            # Exponential interpolation in log space: optimiser LRs are
            # multiplicative quantities, so linear interpolation spends most of
            # the schedule in the fast region.
            lr = math.exp(
                math.log(means_lr_init)
                + t * (math.log(max(means_lr_final, 1e-12)) - math.log(means_lr_init))
            )
        optimizers["means"].param_groups[0]["lr"] = lr

    return step


__all__ = [
    "SplatInit",
    "create_splats_2dgs",
    "activated",
    "build_optimizers",
    "rgb_to_sh0",
    "knn_mean_distance_rms",
    "means_lr_scheduler",
]


def _assert_dead_axis(
    log_scales: torch.Tensor, scene_scale: float, dead_axis_scale: float
) -> None:
    """Guard the 2DGS third-scale-axis invariant.

    The 2DGS kernel builds its covariance with ``mat3(sx,0,0, 0,sy,0, 0,0,1)``
    (``Projection2DGSFused.cu:166``), so the third axis never renders. It is
    still a parameter, and two things read it:

    * ``DefaultStrategy`` classifies duplicate-vs-split with
      ``exp(scales).max(dim=-1)`` -- the whole tensor;
    * the split op samples children with ``rotmat @ scales @ randn``.

    So the third axis must stay negligible. Two ways to break it: initialising
    it from kNN spacing (which makes ``max()`` dominate and forces every
    candidate down the split branch), or writing ``0`` (``log(0) = -inf``, which
    then propagates through exponentiation into every size comparison).

    GSL avoids the issue structurally (its scales are ``[N,2]``); we keep three
    axes and assert instead. The check is cheap and runs once at construction.
    """
    live = torch.exp(log_scales.detach())[:, 2]
    limit = max(1e-3 * float(scene_scale), 10.0 * float(dead_axis_scale))
    if not bool((live < limit).all()):
        raise RuntimeError(
            f"the third scale axis is not negligible: max {float(live.max()):.3e} "
            f">= limit {limit:.3e}. This axis never renders in the 2DGS kernel "
            "but DefaultStrategy uses exp(scales).max() for its duplicate/split "
            "decision, so a large third axis forces every candidate to split."
        )
    if not bool(torch.isfinite(log_scales).all()):
        raise RuntimeError(
            "non-finite initial log-scales; a zero-length kNN spacing would give "
            "log(0) = -inf"
        )
