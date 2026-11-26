"""2DGS rasterisation front-end over ``gsplat.rendering.rasterization_2dgs``.

This is a thin wrapper on purpose: every numerical semantic lives in gsplat, and
the wrapper's job is only to assemble the right arguments and to normalise the
return value. Three semantics are documented here because getting any of them
wrong produces a plausible-looking wrong image rather than an error, and all
three were verified by measurement in this project:

**Depth is already alpha-normalised.** ``Rendering.cpp`` divides the accumulated
depth channel by ``render_alphas`` for the ``E`` modes before returning. Dividing
again in Python scales ``surf_depth`` by ``1/alpha``; the measured error was 1.72
on a test render. Only the median channel (``render_median``) is *not* normalised.

**Normals are already in world space.** ``Rendering.cpp`` rotates
``render_normals`` by the camera-to-world rotation before returning. Applying the
inverse rotation again (which reads as the natural thing to do when converting a
camera-space normal) double-transforms it and poisons the normal-consistency
term.

**The SH ``+0.5`` offset and clamp live in the kernel.** ``Rendering.cpp`` does
``clamp_min(sh + 0.5, 0)``. Adding ``0.5`` again brightens every image. Measured
RGB max difference against the decomposed chain: 0.0.

`surf_normals` (normals implied by the rendered depth) is only produced for the
``RGB+D`` / ``RGB+ED`` modes -- the ``-d``/``-Ed`` variants carry depth but return
``None``. It also comes back without a batch dimension when ``C == 1`` in some
paths, so the shape is normalised here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .splats import activated


@dataclass
class RenderOutput:
    """Normalised 2DGS render result."""

    #: ``[C,H,W,3]`` in [0,1] (the kernel already clamps).
    colors: Tensor
    #: ``[C,H,W,1]`` accumulated alpha.
    alphas: Tensor
    #: ``[C,H,W,1]`` depth; ``None`` when depth was not requested.
    depth: Optional[Tensor] = None
    #: ``[C,H,W,3]`` surfel normals, world space.
    normals: Optional[Tensor] = None
    #: ``[C,H,W,3]`` normals implied by :attr:`depth`, world space.
    surf_normals: Optional[Tensor] = None
    #: ``[C,H,W,1]`` 2DGS distortion (only filled when ``distloss=True``).
    distort: Optional[Tensor] = None
    #: ``[C,H,W,1]`` median depth (not alpha-normalised).
    median_depth: Optional[Tensor] = None
    #: Raw ``meta`` dict; the densification strategy reads from it.
    meta: Dict[str, Any] = field(default_factory=dict)

    #: Which ``meta`` key the strategy uses for image-plane gradients.
    gradient_key: str = "gradient_2dgs"


class Renderer2DGS:
    """Wraps ``rasterization_2dgs`` with the project's calling conventions."""

    def __init__(
        self,
        *,
        packed: bool = False,
        eps2d: float = 0.3,
        radius_clip: float = 0.0,
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
    ) -> None:
        self.packed = packed
        self.eps2d = eps2d
        self.radius_clip = radius_clip
        # ``None`` means "derive from the scene". A fixed ``1e10`` far plane
        # makes near/far clipping meaningless on an aerial block; the derived
        # values are set by the trainer from measured depth quantiles.
        self.near_plane = near_plane
        self.far_plane = far_plane

    def set_depth_range(self, near: float, far: float) -> None:
        self.near_plane = float(near)
        self.far_plane = float(far)

    def __call__(
        self,
        params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        cam_to_worlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        *,
        sh_degree: Optional[int] = None,
        render_depth: bool = False,
        distloss: bool = False,
        backgrounds: Optional[Tensor] = None,
    ) -> RenderOutput:
        from gsplat.rendering import rasterization_2dgs

        p = activated(params)

        # The rasteriser wants world-to-camera. ``torch.linalg.inv`` on the
        # [C,4,4] stack is fine and differentiable; a manual closed form would
        # save nothing at this size.
        viewmats = torch.linalg.inv(cam_to_worlds)

        if self.near_plane is None or self.far_plane is None:
            raise RuntimeError(
                "near/far planes are unset: call set_depth_range() with the "
                "values derived from the scene (see surfel.data.gauge) rather "
                "than relying on gsplat's 1e-2/1e10 defaults, which make depth "
                "clipping a no-op on large scenes"
            )

        # Render mode choice. ``RGB+ED`` gives colour plus alpha-normalised
        # expected depth *and* the depth-derived normals that the normal
        # consistency term needs. ``distloss`` is only valid with a depth
        # channel, so the mode is upgraded if distortion is on.
        want_depth = render_depth or distloss
        render_mode = "RGB+ED" if want_depth else "RGB"

        (
            colors,
            alphas,
            normals,
            surf_normals,
            distort,
            median,
            meta,
        ) = rasterization_2dgs(
            means=p["means"],
            quats=p["quats"],
            scales=p["scales"],
            opacities=p["opacities"],
            colors=p["sh"],
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            radius_clip=self.radius_clip,
            eps2d=self.eps2d,
            sh_degree=sh_degree,
            packed=self.packed,
            backgrounds=backgrounds,
            render_mode=render_mode,
            distloss=bool(distloss),
        )

        depth = None
        if want_depth:
            # Depth channel is alpha-normalised by the kernel already: take it
            # as-is (see the module docstring).
            colors, depth = colors[..., :3], colors[..., 3:4]

        return RenderOutput(
            colors=colors,
            alphas=alphas,
            depth=depth,
            normals=normals,
            surf_normals=_normalise_batch_dim(surf_normals, len(cam_to_worlds)),
            distort=distort if distloss else None,
            median_depth=median,
            meta=meta,
            gradient_key="gradient_2dgs",
        )


def _normalise_batch_dim(x: Optional[Tensor], n_cameras: int) -> Optional[Tensor]:
    """Ensure a depth-derived map is ``[C,H,W,3]``.

    gsplat returns ``(H,W,3)`` for this tensor on some paths (observed when
    ``C == 1``), which silently broadcasts wrong against a ``[1,H,W,3]`` term.
    """
    if x is None:
        return None
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4 and x.shape[0] != n_cameras and x.shape[0] == 1:
        # Already batched with C=1 while the caller asked for C cameras: this
        # is a shape mismatch the caller must know about, not something to
        # silently expand.
        raise RuntimeError(
            f"surf_normals has batch {x.shape[0]} but {n_cameras} cameras were "
            "rendered; the render mode does not support this combination"
        )
    return x


def assert_backend_contract(
    out: "RenderOutput", *, tol: float = 1e-3, check_depth: bool = True
) -> None:
    """Verify the assumptions this wrapper makes about gsplat's backend.

    Every check here corresponds to a place where gsplat's C++ layer has already
    done something the Python side might be tempted to do again, and where doing
    it twice produces a plausible-looking wrong image instead of an error. The
    audit is summarised in ``docs/SURFEL_2DGS_LARGE_SCENE.zh.md`` §3.7.

    Intended for tests and for a startup probe, not the per-step path: each check
    costs a kernel launch.
    """
    colors = out.colors
    alphas = out.alphas

    # 1. RGB must already be in [0,1]: the SH kernel adds +0.5 and clamps.
    if colors.numel():
        lo, hi = float(colors.min()), float(colors.max())
        if lo < -tol or hi > 1.0 + tol:
            raise AssertionError(
                f"render colors outside [0,1] (min {lo:.4f}, max {hi:.4f}); the "
                "SH kernel clamps after adding the 0.5 bias, so a value outside "
                "this range means the bias was applied twice or not at all"
            )

    if out.normals is not None and out.normals.numel():
        # 2. ``render_normals`` is an **alpha-weighted accumulation** of unit
        #    surfel normals, not a normalised normal map:
        #
        #        normal_out[k] += n_ptr[k] * vis      (RasterizeToPixels2DGS...Fwd.cu:408)
        #
        #    so its length where alpha < 1 is meaningfully below 1 (measured
        #    0.004..0.62 at partial coverage). It must therefore be renormalised
        #    before any cosine comparison -- and the world-space rotation is
        #    already applied by the C++ layer, so only the length is the caller's
        #    responsibility. Asserting length == 1 would be wrong; asserting it
        #    never *exceeds* 1 catches a genuine double-rotation or a missed
        #    weighting.
        norms = out.normals.detach().norm(dim=-1)
        if float(norms.max()) > 1.0 + 1e-2:
            raise AssertionError(
                f"render_normals length exceeds 1 (max {float(norms.max()):.4f}); "
                "the kernel accumulates unit normals weighted by visibility, so "
                "a value above 1 means the accumulation is being applied twice"
            )

    if check_depth and out.depth is not None and out.depth.numel():
        # 3. Expected depth must be alpha-normalised. Test: at a covered pixel,
        #    depth should stay bounded even as alpha varies; an un-normalised
        #    accumulator would scale linearly with alpha. The robust invariant is
        #    non-negativity plus finiteness where alpha > 0 -- the actual
        #    alpha-normalisation cannot be re-derived without the raw accumulator,
        #    so this asserts the weaker property and relies on the unit tests for
        #    the exact relation.
        d = out.depth.detach()
        a = alphas.detach()
        covered = a > 1e-2
        if bool(covered.any()):
            dv = d[covered.expand_as(d)]
            if not bool(torch.isfinite(dv).all()):
                raise AssertionError(
                    "expected depth has non-finite values on covered pixels; "
                    "the kernel divides by clamp_min(alpha, 1e-10)"
                )
            if float(dv.min()) < -tol:
                raise AssertionError(
                    f"expected depth is negative (min {float(dv.min()):.4f})"
                )

    # 4. surf_normals must carry the camera dimension the caller asked for.
    if out.surf_normals is not None:
        want = colors.shape[0]
        if out.surf_normals.shape[0] != want:
            raise AssertionError(
                f"surf_normals batch {out.surf_normals.shape[0]} != {want} cameras"
            )


def derive_depth_range(
    scene_points_tensor: Tensor,
    cam_to_worlds: Tensor,
    *,
    near_quantile: float = 0.01,
    far_quantile: float = 0.99,
    near_floor_frac: float = 1e-3,
    far_headroom: float = 1.5,
    max_points: int = 100_000,
    max_per_camera: int = 2_000,
) -> tuple[float, float]:
    """Near/far planes from the measured camera-space depth distribution.

    Projecting the points rather than using a bounding-box diagonal matters on
    aerial captures: a city block spans hundreds of metres horizontally but only
    tens vertically, so the diagonal overestimates the observed depth range and
    pushes most pixels out of the distortion kernel's useful window.

    Sampling is capped on **both** axes. ``torch.quantile`` fails with
    "input tensor is too large" once the flattened input passes the CUDA kernel's
    element limit, and the natural implementation here -- every point through
    every camera -- squares the problem: garden is 138,766 points x 161 cameras
    = 22M samples, which is enough to hit it. A depth *distribution* does not
    need that many samples, and capping per camera keeps long-range views from
    dominating the quantiles.
    """
    pts = scene_points_tensor
    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts = pts[idx]

    zs = []
    for c2w in cam_to_worlds:
        w2c = torch.linalg.inv(c2w)
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        z = z[z > 0]
        if z.numel() == 0:
            continue
        if z.numel() > max_per_camera:
            z = z[torch.randperm(z.numel(), device=z.device)[:max_per_camera]]
        zs.append(z)
    if not zs:
        raise RuntimeError(
            "no scene point lies in front of any camera; the gauge or the pose "
            "convention is wrong (expected OpenCV: +z forward)"
        )

    all_z = torch.cat(zs)
    # Belt and braces: the per-camera cap above already bounds this, but a
    # caller passing a huge camera list should still not crash here.
    if all_z.numel() > max_points * 4:
        all_z = all_z[torch.randperm(all_z.numel(), device=all_z.device)[: max_points * 4]]

    near = float(torch.quantile(all_z, near_quantile))
    far = float(torch.quantile(all_z, far_quantile)) * far_headroom
    near = max(near, far * near_floor_frac, 1e-8)
    if not (near < far):
        near, far = far * near_floor_frac, far
    return near, far


__all__ = ["RenderOutput", "Renderer2DGS", "derive_depth_range"]
