"""Gauge: making a reconstruction's world frame explicit, and moving it atomically.

A reconstruction from SfM or a feed-forward model is defined only up to a
similarity transform. Nothing inside it fixes origin, orientation or absolute
scale; those come from the data source, and they are destroyed the moment two
sources are combined carelessly.

=========================  ==========================  ====================
source                     what fixes its scale       frame
=========================  ==========================  ====================
COLMAP / PixelSfM          nothing (arbitrary)         its own origin
Scal3R                     nothing (arbitrary)         its own origin
VGGT / VGGT-X              per-frame, fitted           first camera
GT / GNSS / RTK            metric (metres)             world / ECEF-derived
=========================  ==========================  ====================

Two rules follow, and violating either produces a model that trains without
error and renders wrong:

**Rule 1 -- a pose and a depth from different sources are in different gauges.**
Scal3R's depth is consistent with Scal3R's *own* estimated focal and its own
recovered scale. Pairing that depth with SfM poses gives two inconsistent
geometries: the depth unprojects to points that do not land where the cameras
look. Align first (:func:`estimate_sim3`), or take both from one source.

**Rule 2 -- a similarity applied to poses must also be applied to depth.** If
``x -> s*R*x + t`` is applied to camera centres, camera-space depths become
``s`` times larger. Depth maps and per-frame ``{scale, offset}`` must be scaled
by the same ``s``, or the depth loss and the geometry end up in different units.

Rule 2 is not theoretical. Measured on one strip0 arm: after an independent
cloud rescale the required depth factor was 0.471 while the code used 1.0, so
the rendered-vs-supervised depth ratio sat at 0.361. The disparity loss then
carried a constant offset indistinguishable from a geometry error, and two
thirds of the depth signal were gone before step 1. The other arm escaped only
because its factor happened to be 0.964.

:func:`apply_gauge` therefore performs every mutation in one place, and
deliberately has no partial-application mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class GaugeSource(str, Enum):
    """Where a reconstruction's world frame came from."""

    COLMAP = "colmap"
    SCAL3R = "scal3r"
    VGGT = "vggt"
    GT = "gt"
    PIXELSFM = "pixelsfm"


#: Sources whose distances are metres by construction.
METRIC_SOURCES = frozenset({GaugeSource.GT})


class GaugeMismatchError(ValueError):
    """Raised when a pose source and a depth source cannot be combined as-is."""


@dataclass(frozen=True)
class Gauge:
    """An explicit description of one reconstruction's world frame."""

    source: GaugeSource
    scale_is_metric: bool = False

    @property
    def needs_alignment_hint(self) -> Optional[str]:
        if self.scale_is_metric:
            return None
        return (
            "scale is arbitrary: do not mix this reconstruction's depth or "
            "points with metric poses (GT/GNSS) without a Sim(3) alignment, and "
            "do not combine depth from one non-metric source with poses from "
            "another"
        )


def needs_alignment(pose_gauge: Gauge, depth_gauge: Optional[Gauge]) -> bool:
    """Whether a Sim(3) must be fitted before these two may be combined.

    Returns ``False`` when no depth is used, when both sides share a source, or
    when the depth is metric (metric depth needs no rescaling regardless of the
    pose frame).
    """
    if depth_gauge is None:
        return False
    if depth_gauge.source == pose_gauge.source:
        return False
    if depth_gauge.scale_is_metric:
        return False
    return True


def check_compatible(pose_gauge: Gauge, depth_gauge: Optional[Gauge]) -> None:
    """Raise unless the pair is directly usable *as it stands*.

    This answers "can these be combined without extra work", not "is combining
    ever possible". A cross-source non-metric pair is resolvable by fitting a
    Sim(3) first, which is what :func:`needs_alignment` reports; this function
    is the stricter check for a code path that cannot align.
    """
    if depth_gauge is None:
        return
    if depth_gauge.source == pose_gauge.source or depth_gauge.scale_is_metric:
        return
    raise GaugeMismatchError(
        f"poses from {pose_gauge.source.value} paired with depth from "
        f"{depth_gauge.source.value}: both scales are arbitrary and independent. "
        "Align one to the other with a Sim(3) before training; mixing them raw "
        "produces geometry that is internally inconsistent even though training "
        "runs without error."
    )


# --------------------------------------------------------------------------
# Similarity fitting
# --------------------------------------------------------------------------
@dataclass
class Sim3:
    """A similarity ``x -> s * R @ x + t``, with an explicit direction.

    **A ``Sim3`` is directional and must only be applied to quantities that live
    in its source frame.** The same trap has been hit three times in this
    project, each time as a silent geometric error rather than an exception:

    * strip0 Mode B: poses were already aligned into the reference gauge and
      were transformed a second time -> PSNR 15.63 instead of 19.42;
    * a DGI run scaled depth by ``s`` while the camera-to-world matrices had
      already been transformed -> every point sat ``s`` times too close;
    * a verification script applied the Scal3R->COLMAP similarity to *COLMAP*
      poses, pushing the cameras ``s = 5.13`` times too far out, which shrank
      every Gaussian's projected radius below one pixel and left 41% of the
      frame uncovered (alpha 0.59 instead of 0.93).

    The direction is therefore recorded, and :meth:`apply_to_scal3r_poses` /
    :meth:`apply_to_reference_poses` refuse the wrong side. ``source`` defaults
    to ``None`` for transforms whose direction is not known, in which case only
    the direction-agnostic :meth:`apply_points` is available.
    """

    R: np.ndarray
    s: float
    t: np.ndarray
    #: Name of the gauge this transform maps *from* (e.g. "scal3r"). When set,
    #: the guarded helpers enforce that the input really is in that gauge.
    source: Optional[str] = None
    #: Name of the gauge it maps *to* (e.g. "colmap").
    target: Optional[str] = None

    def matrix(self) -> np.ndarray:
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """``x -> s * R @ x + t``.

        Note the translation is **added after** the scaling, not scaled with it.
        ``estimate_sim3`` computes ``t = mu_dst - s*R*mu_src``, which is the
        definition of that convention; writing ``s * (R@x + t)`` instead scales
        the translation a second time and misplaces every point by ``(s-1)*t``.
        On a real pair (garden: s=5.13) that turned a good fit into a 127%
        residual, which reads as "different scenes" rather than "algebra bug".
        """
        return (
            self.s * (np.asarray(points, dtype=np.float64) @ self.R.T) + self.t
        )

    def apply_cam_to_world(self, c2w: np.ndarray) -> np.ndarray:
        """Transform camera poses, keeping the rotation block rigid.

        The rotation of a camera-to-world matrix is *orientation*, not geometry:
        scaling it by ``s`` would produce a matrix whose ``R^T R != I``, which
        silently breaks every downstream ``inv`` used to build view matrices.

        The translation follows ``s*R@c + t`` -- same convention as
        :meth:`apply_points`, with ``t`` added *after* scaling (see that method
        for the failure this gets wrong otherwise).

        **This is the unguarded form.** Prefer :meth:`apply_to_scal3r_poses` or
        :meth:`apply_to_reference_poses`, which refuse when the caller passes
        poses from the wrong side of the mapping -- the exact mistake that made
        a garden run report 12 dB.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        out = np.empty_like(c2w)
        out[..., :3, :3] = self.R @ c2w[..., :3, :3]
        out[..., :3, 3] = self.s * (c2w[..., :3, 3] @ self.R.T) + self.t
        out[..., 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        return out

    # -- direction-guarded entry points -----------------------------------
    #
    # What these can and cannot catch, stated plainly so nobody trusts them
    # further than they go:
    #
    # They CAN catch a **reversed fit**. If the mapping was estimated the wrong
    # way round, ``apply_to_scal3r_poses`` gets a transform whose source is not
    # ``"scal3r"`` and refuses; likewise ``apply_to_reference_poses`` with a
    # transform whose target is not the reference. That is the realistic bug,
    # and it is invisible in the output because the result is still a valid
    # similarity -- it just scales the scene by s.
    #
    # They CANNOT catch applying a correctly-oriented transform to the wrong
    # array. Both sides are plain ``ndarray`` of poses; nothing in the type can
    # tell a COLMAP pose from a Scal3R pose. So the helpers only enforce that
    # the *caller's claim* about which side the data belongs to agrees with the
    # transform. Passing reference poses to ``apply_to_scal3r_poses`` still
    # scales the scene twice; the direction guard on which helper you called is
    # the whole defence, which is why ``apply_to_reference_poses`` is a no-op
    # that returns its input rather than a transform the caller has to remember
    # not to apply.

    def _require_source(self, want_source: str) -> None:
        if self.source is None or self.target is None:
            raise ValueError(
                "this Sim3 has no recorded direction, so a guarded apply is not "
                "possible. Construct it with source=/target= (estimate_sim3 does "
                "this automatically) or use apply_points()/apply_cam_to_world() "
                "if you have verified the direction yourself."
            )
        if self.source != want_source:
            raise ValueError(
                f"this Sim3 maps {self.source!r} -> {self.target!r}, so it cannot "
                f"be applied to {want_source!r}-gauge poses: its source is "
                f"{self.source!r}. Either the fit was done the wrong way round "
                "(swap source=/target= or use .inverse()), or you meant "
                "apply_to_reference_poses() -- which refuses for the same reason."
            )

    def _require_target(self, want_target: str) -> None:
        if self.source is None or self.target is None:
            raise ValueError(
                "this Sim3 has no recorded direction, so a guarded apply is not "
                "possible. Construct it with source=/target= (estimate_sim3 does "
                "this automatically) or use apply_points()/apply_cam_to_world() "
                "if you have verified the direction yourself."
            )
        if self.target != want_target:
            raise ValueError(
                f"this Sim3 maps {self.source!r} -> {self.target!r}, so its "
                f"target is not {want_target!r}. Poses in the {want_target!r} "
                "gauge are already where this transform would send you; only a "
                "transform that maps INTO that gauge may be used here. Use "
                ".inverse() if you fitted it the other way round."
            )

    def apply_to_scal3r_poses(self, c2w: np.ndarray) -> np.ndarray:
        """Move poses that are still in the ``source`` gauge into the target.

        Requires ``self.source == want_source`` (i.e. the transform was fitted
        from that side). This is the *only* case where a pose array is actually
        modified.
        """
        self._require_source("scal3r")
        return self.apply_cam_to_world(c2w)

    def apply_to_reference_poses(self, c2w: np.ndarray) -> np.ndarray:
        """Return reference-gauge poses **unchanged**, asserting the direction.

        This exists so the correct code path is also the obvious one: when the
        poses already come from the reference (COLMAP/GT) model and only the
        *depth* needs aligning, the caller writes
        ``sim.apply_to_reference_poses(c2w)`` and gets the input back, plus a
        check that this transform really does map *into* the reference gauge.

        The check is on ``self.target``, deliberately: a transform whose target
        is ``"colmap"`` is the one whose scale factor ``self.s`` converts
        Scal3R depth into COLMAP units. Applying it to the poses instead is the
        double-transform that produced a 41%-uncovered frame and 12.3 dB on
        garden. Returning the input makes "forgot to apply the gauge to the
        poses" and "applied it twice" the same line of code, so there is no
        second place to get it wrong.
        """
        self._require_target("colmap")
        return np.asarray(c2w, dtype=np.float64)

    def inverse(self) -> "Sim3":
        """Inverse under ``x -> s*R@x + t``.

        ``x = s^-1 * R^T @ (y - t)``, i.e. scale ``1/s``, rotation ``R^T`` and
        translation ``-s^-1 * R^T @ t``. Source and target swap.
        """
        s_inv = 1.0 / self.s
        R_inv = self.R.T
        return Sim3(
            R=R_inv,
            s=s_inv,
            t=-s_inv * (R_inv @ self.t),
            source=self.target,
            target=self.source,
        )


def estimate_sim3(
    src_centers: np.ndarray,
    dst_centers: np.ndarray,
    *,
    with_scale: bool = True,
    source: Optional[str] = None,
    target: Optional[str] = None,
) -> Sim3:
    """Least-squares similarity mapping ``src`` onto ``dst`` (Umeyama 1991).

    Camera **centres** are the right quantity to fit on: both sides observe
    them, no depth is needed, and they are what a pose error shows up in.

    The reflection guard is not optional. Without flipping ``R`` when
    ``det(R) < 0`` the fit can return a mirrored world, which no downstream
    check catches: the residual is small, the poses look fine, and every
    rendered image is mirrored.

    ``source`` / ``target`` name the two gauges. Supplying them makes the
    returned transform carry its direction, which enables the guarded
    :meth:`Sim3.apply_to_scal3r_poses` / :meth:`Sim3.apply_to_reference_poses`
    helpers to reject a wrong-side application instead of silently scaling the
    scene a second time.
    """
    src = np.asarray(src_centers, dtype=np.float64)
    dst = np.asarray(dst_centers, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"expected matched [N,3] arrays, got {src.shape} / {dst.shape}"
        )
    n = src.shape[0]
    if n < 3:
        raise ValueError("need at least 3 correspondences to fit a similarity")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = float((src_c**2).sum()) / n
        s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return Sim3(
        R=R, s=s, t=np.asarray(t, dtype=np.float64), source=source, target=target
    )


def sim3_residual(
    src_centers: np.ndarray, dst_centers: np.ndarray, sim3: Sim3
) -> Tuple[float, float]:
    """``(rms, max)`` centre error after applying ``sim3``, in ``dst`` units."""
    mapped = sim3.apply_points(src_centers)
    err = np.linalg.norm(mapped - np.asarray(dst_centers, dtype=np.float64), axis=-1)
    return float(np.sqrt((err**2).mean())), float(err.max())


def trajectory_span(centers: np.ndarray) -> float:
    """Diagonal of the camera-centroid-centred bounding box.

    A robust denominator for "how big is this trajectory", used to express a
    residual as a fraction so the same threshold works at 5 m and 500 m.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(c.max(axis=0) - c.min(axis=0)))


# --------------------------------------------------------------------------
# Applied gauge
# --------------------------------------------------------------------------
@dataclass
class GaugeApplication:
    """Record of what :func:`apply_gauge` did, for the run manifest."""

    sim3: Optional[Sim3]
    scale_only: float
    source: str
    applied_to_depth: bool
    notes: list

    def describe(self) -> str:
        if self.sim3 is None:
            return f"gauge: {self.source}, scale {self.scale_only:.6g}"
        return (
            f"gauge: {self.source}, s={self.sim3.s:.6g}, "
            f"R={self.sim3.R.round(6).tolist()}, t={self.sim3.t.round(6).tolist()}, "
            f"depth scaled={self.applied_to_depth}"
        )


def derive_scale(
    centers: np.ndarray,
    depths: Optional[np.ndarray],
    source: str,
    *,
    manual: Optional[float] = None,
    trim: float = 0.005,
) -> Tuple[float, str]:
    """Pick the scalar that brings a scene to ~unit extent.

    ``depth_p99`` is the default. It is chosen over the camera extent because
    the distortion regulariser is evaluated in inverse depth and needs the
    *depth distribution* inside the kernel's usable window; the trajectory
    extent is only a proxy for that and is wrong for thin aerial blocks, where
    the cameras span hundreds of metres but every surface is the same distance
    away. Measured on flight0: ``1/extent`` left 6.8% of pixels in the dead
    zone and only 72.8% of the value window usable, while ``1/d_p99`` gave
    92.7%.
    """
    if source == "manual":
        if not manual or manual <= 0:
            raise ValueError("gauge.scale_source='manual' requires a positive scale")
        return float(manual), "manual"

    if source == "depth_p99":
        if depths is None or len(depths) < 8:
            raise ValueError(
                "gauge.scale_source='depth_p99' needs supervised depth to "
                "compute the quantile; pass depths, or use 'extent'"
            )
        d = np.asarray(depths, dtype=np.float64).ravel()
        d = d[np.isfinite(d) & (d > 0)]
        if d.size < 8:
            raise ValueError("no positive depth samples to derive a scale from")
        p99 = float(np.quantile(d, 0.99))
        return 1.0 / p99, "depth_p99"

    if source == "extent":
        ext = trajectory_span(centers)
        if ext <= 0:
            raise ValueError("degenerate camera trajectory; cannot derive a scale")
        return 1.0 / ext, "extent"

    raise ValueError(f"unknown gauge.scale_source {source!r}")


def apply_gauge(
    *,
    cam_to_worlds: np.ndarray,
    points: np.ndarray,
    depths: Optional[np.ndarray] = None,
    sim3: Optional[Sim3] = None,
    apply_to_depth: bool = True,
    source: str = "normalize",
    notes: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], GaugeApplication]:
    """Apply one similarity to poses, points and depth **together**.

    Depth has to travel with the geometry: applying the transform to poses and
    points but leaving depth alone makes the rendered-vs-supervised depth ratio
    a constant other than 1, which the disparity loss then reads as a geometry
    error. There is no separate "poses only" variant here on purpose.

    Depth is a *camera-space z value*, so it scales by ``s`` and ignores ``R``
    and ``t`` -- the camera frame moves with the world.

    Returns ``(cam_to_worlds, points, depths, record)``.
    """
    rec_notes = notes if notes is not None else []

    if sim3 is None:
        s = 1.0
        new_c2w = np.asarray(cam_to_worlds, dtype=np.float64)
        new_pts = np.asarray(points, dtype=np.float64)
        new_depths = depths
        rec = GaugeApplication(
            sim3=None,
            scale_only=1.0,
            source=source,
            applied_to_depth=False,
            notes=rec_notes,
        )
        return new_c2w, new_pts, new_depths, rec

    s = float(sim3.s)
    new_c2w = sim3.apply_cam_to_world(cam_to_worlds)
    new_pts = sim3.apply_points(points)

    if depths is not None and apply_to_depth:
        new_depths = np.asarray(depths, dtype=np.float64) * s
        rec_notes.append(f"depth supervision scaled by {s:.6g} with the gauge")
    else:
        new_depths = depths
        if depths is not None:
            rec_notes.append(
                "depth supervision NOT scaled (per-frame normalisation is "
                "expected to absorb the scale)"
            )

    rec = GaugeApplication(
        sim3=sim3,
        scale_only=s,
        source=source,
        applied_to_depth=bool(depths is not None and apply_to_depth),
        notes=rec_notes,
    )
    return new_c2w, new_pts, new_depths, rec


# --------------------------------------------------------------------------
# Scene statistics of the final gauge
# --------------------------------------------------------------------------
def camera_extent(cam_to_worlds: np.ndarray) -> float:
    c = np.asarray(cam_to_worlds, dtype=np.float64)[:, :3, 3]
    return float(np.linalg.norm(c - c.mean(axis=0), axis=-1).max())


def depth_quantiles(
    cam_to_worlds: np.ndarray,
    points: np.ndarray,
    *,
    max_points: int = 200_000,
    max_per_camera: int = 2_000,
    seed: int = 0,
) -> dict:
    """Camera-space z quantiles of the points actually in front of cameras.

    Measured by projecting, not approximated from a bounding box: an aerial
    block is thin in depth while wide in extent, and a bounding-box diagonal
    overestimates the observed depth range badly (measured 14.5% of pixels into
    the distortion kernel's dead zone, versus 6.7% when measured this way).
    """
    c2w = np.asarray(cam_to_worlds, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return {"p01": None, "p50": None, "p99": None}
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]

    rng = np.random.default_rng(seed + 1)
    per_cam = []
    for c in c2w:
        try:
            w2c = np.linalg.inv(c)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        z = (pts @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        z = z[z > 0]
        if z.size == 0:
            continue
        if z.size > max_per_camera:
            z = z[rng.choice(z.size, max_per_camera, replace=False)]
        per_cam.append(z)

    if not per_cam:
        return {"p01": None, "p50": None, "p99": None}

    all_z = np.concatenate(per_cam)
    # Bounded even after the per-camera cap: with many cameras the concatenation
    # can still grow past what a quantile implementation will accept, and this
    # is a distribution estimate, not an exact statistic.
    if all_z.size > 2_000_000:
        all_z = rng.choice(all_z, 2_000_000, replace=False)
    return {
        "p01": float(np.quantile(all_z, 0.01)),
        "p50": float(np.quantile(all_z, 0.50)),
        "p99": float(np.quantile(all_z, 0.99)),
    }


def derive_scene_scale(
    q: dict, cam_extent: float, *, prefer: str = "depth_p99"
) -> Tuple[float, str]:
    """The size reference that multiplies Gaussian-size thresholds.

    ``DefaultStrategy`` compares ``exp(scales).max()`` against
    ``grow_scale3d * scene_scale`` and ``prune_scale3d * scene_scale``. Those
    constants (0.01 / 0.1) were tuned where ``scene_scale`` was a camera extent
    of order 1. Feeding them an unscaled aerial extension is not a small error:
    measured on a 530 m trajectory, ``prune_scale3d * 1.0 = 0.1`` prunes every
    Gaussian larger than 10 cm, i.e. all of the structure, while
    ``grow_scale3d * 1.0`` never marks anything as small so every candidate
    takes the split branch.

    After normalisation both interpretations collapse to ~1, so this returns
    the depth percentile when available (it describes the surfaces being
    modelled) and falls back to the trajectory extent.
    """
    if prefer == "depth_p99" and q.get("p99"):
        return float(q["p99"]), "depth_p99"
    if cam_extent > 0:
        return float(cam_extent), "camera_extent"
    return 1.0, "fallback"


__all__ = [
    "Gauge",
    "GaugeSource",
    "GaugeMismatchError",
    "GaugeApplication",
    "Sim3",
    "METRIC_SOURCES",
    "needs_alignment",
    "check_compatible",
    "estimate_sim3",
    "sim3_residual",
    "trajectory_span",
    "derive_scale",
    "apply_gauge",
    "camera_extent",
    "depth_quantiles",
    "derive_scene_scale",
]


# --------------------------------------------------------------------------
# Runtime consistency checks (defence in depth)
# --------------------------------------------------------------------------
def projected_footprint_px(
    cam_to_worlds: np.ndarray,
    points: np.ndarray,
    Ks: np.ndarray,
    *,
    max_cameras: int = 16,
    max_points: int = 50_000,
    seed: int = 0,
) -> dict:
    """Median projected radius, in pixels, of the point cloud seen by the cameras.

    This is the one quantity here that is **not** gauge-invariant in a useful
    way, because it compares a *world* length against a *camera-space* depth:

        radius_px ≈ focal * point_spacing_3d / z

    A similarity transform scales the spacing and ``z`` together, so the radius
    is invariant -- which is exactly why it cannot be fooled by normalisation,
    and also why it *does* reveal the real failure: if the poses and the points
    are not in the same gauge, ``z`` is wrong relative to the spacing and the
    radius comes out too small.

    Measured on garden at 1297x840: a correct DGI cloud gives median radius
    ~1.6 px; a run where the poses had been transformed twice gave ~0.3 px and
    left 41% of the frame uncovered.

    Returns a dict with ``radius_p50``/``radius_p10``/``visible_frac`` plus the
    inputs that produced them, so a caller can log why a check passed.
    """
    c2w = np.asarray(cam_to_worlds, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    Ks = np.asarray(Ks, dtype=np.float64)
    if pts.shape[0] < 8 or c2w.shape[0] == 0:
        return {"radius_p50": float("nan"), "radius_p10": float("nan"),
                "visible_frac": 0.0, "n_sampled": 0}

    rng = np.random.default_rng(seed)

    # Point spacing must be measured on the *full* cloud, then the camera loop
    # may work on a sample. Getting that order backwards is what made this
    # function report an 8.51 px footprint on garden where the analytic answer
    # is 2.54 px: measuring nearest-neighbour distance on a 50k subsample of a
    # 1.5M-point cloud inflates it by 3.1x, because the subsample really is
    # that much sparser. The tree is built on every point (one pass), the query
    # is limited to ``max_points`` rows.
    #
    # The statistic is the one the initialiser uses -- the RMS over the k-1
    # nearest distances, matching ``distCUDA2`` in ``create_splats_2dgs`` -- so
    # the reported radius describes the model that will actually be built, not
    # a slightly smaller one. RMS is ~1.3x the plain nearest distance on an
    # irregular cloud, which is precisely the factor that decides whether a
    # splat clears the rasteriser's ``radius_clip``.
    from scipy.spatial import cKDTree

    full = pts
    tree = cKDTree(full)
    k = min(4, full.shape[0])
    idx = rng.choice(full.shape[0], max_points, replace=False) \
        if full.shape[0] > max_points else np.arange(full.shape[0])
    d, _ = tree.query(full[idx], k=k, workers=-1)
    d = np.atleast_2d(d)
    # Column 0 is the point itself (distance 0).
    spacing = float(np.median(np.sqrt((d[:, 1:] ** 2).mean(axis=-1))))

    if full.shape[0] > max_points:
        pts = full[idx]

    step = max(1, c2w.shape[0] // max_cameras)
    radii: list[float] = []
    visible = 0
    total = 0
    for ci in range(0, c2w.shape[0], step):
        try:
            w2c = np.linalg.inv(c2w[ci])
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        # A single shared intrinsic is a legitimate input (COLMAP models often
        # have one camera shared by every frame); index it, do not walk off the
        # end of the array.
        if Ks.ndim != 3:
            K = Ks
        elif Ks.shape[0] == 1:
            K = Ks[0]
        else:
            K = Ks[ci]
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-9
        total += int(z.size)
        visible += int(ok.sum())
        if ok.any():
            f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
            radii.append(f * spacing / float(np.median(z[ok])))

    if not radii:
        return {"radius_p50": float("nan"), "radius_p10": float("nan"),
                "visible_frac": 0.0, "n_sampled": len(pts)}
    return {
        "radius_p50": float(np.median(radii)),
        "radius_p10": float(np.quantile(radii, 0.1)),
        "visible_frac": float(visible) / max(total, 1),
        "spacing": spacing,
        "n_sampled": len(pts),
    }


def check_pose_point_consistency(
    scene,
    *,
    min_radius_px: float = 0.2,
    max_radius_px: float = 400.0,
    min_visible_frac: float = 0.5,
    raise_on_fail: bool = False,
) -> dict:
    """Measure a scene's camera/point geometry and flag *gross* inconsistencies.

    **This cannot detect every gauge error, and that is a measured fact, not a
    limitation of effort.** A similarity applied to the wrong side of a
    transform is itself a similarity, so it leaves every internal ratio of the
    scene unchanged. Measured on garden, comparing a correct DGI cloud against
    one built from double-transformed poses:

    =======================  ==============  ===================
    statistic                 correct         double-transformed
    =======================  ==============  ===================
    point-extent / cam-extent  0.99            0.79   (-20%)
    z_p50 / cam_nn             9.49            7.33   (-23%)
    pt_radius / cam_radius     0.99            0.79   (-20%)
    camera radius (absolute)   4.549          22.908  (5.0x)
    camera-space z (absolute)  4.827          19.169  (4.0x)
    radius_p50 @324x210        2.71 px         0.78 px
    radius_p50 @1297x840       8.44 px         3.25 px
    =======================  ==============  ===================

    Every ratio moves by less than 25% -- comfortably inside the normal
    variation between datasets -- while the absolute quantities move 4-5x. And
    normalisation divides precisely those absolute quantities back out, which is
    why it can never fix this class of error.

    So the real defence is the direction guard on :class:`Sim3`; see
    :meth:`Sim3.apply_to_scal3r_poses` and
    :meth:`Sim3.apply_to_reference_poses`. This function is a diagnostic that
    catches only the *severe* cases (sub-pixel splats, everything behind the
    cameras) and reports the numbers otherwise, so a caller can log them and
    compare runs.

    ``raise_on_fail=False`` is the default on purpose: for an incomplete check a
    warning carrying the measurements is more useful than a hard abort.
    """
    c2w = scene.cam_to_worlds() if hasattr(scene, "cam_to_worlds") else scene[0]
    pts = scene.points if hasattr(scene, "points") else scene[1]
    Ks = scene.Ks() if hasattr(scene, "Ks") else scene[2]

    stats = projected_footprint_px(c2w, pts, Ks)
    r50 = stats["radius_p50"]
    vis = stats["visible_frac"]
    reasons: list = []

    if not np.isfinite(r50):
        reasons.append(
            "no point fell in front of any sampled camera "
            f"(visible_frac={vis:.2%}) -- a strong gauge-mismatch signal"
        )
    else:
        if r50 < min_radius_px:
            reasons.append(
                f"median projected radius {r50:.3f} px < {min_radius_px} px: "
                "splats are sub-pixel and the rasteriser will drop them"
            )
        if r50 > max_radius_px:
            reasons.append(
                f"median projected radius {r50:.1f} px > {max_radius_px} px"
            )
        if vis < min_visible_frac:
            reasons.append(f"only {vis:.1%} of points are in front of the cameras")

    stats["ok"] = not reasons
    stats["reasons"] = reasons
    # Record the absolute scale so a caller can compare it against a reference
    # run -- the only signal that actually discriminates (see the docstring).
    stats["camera_radius"] = float(
        np.linalg.norm(c2w[:, :3, 3] - np.median(c2w[:, :3, 3], axis=0), axis=1).max()
    )
    if reasons and raise_on_fail:
        raise GaugeMismatchError(
            "pose/point gauge inconsistency: "
            + "; ".join(reasons)
            + f". (spacing={stats.get('spacing', float('nan')):.6g}, "
            f"radius_p10={stats.get('radius_p10', float('nan')):.4g}, "
            f"camera_radius={stats['camera_radius']:.4g}, "
            f"n={stats.get('n_sampled', 0)}). Most likely cause: a Sim(3) applied "
            "to data already in the target gauge. Note that normalisation cannot "
            "fix this -- it divides out exactly the absolute scale that changed."
        )
    return stats
