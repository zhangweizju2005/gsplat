"""Depth: reading, per-frame calibration, and remapping onto the render grid.

Three conventions exist across the depth sources this project uses, and they are
*not* interchangeable:

``Scal3R``
    Metric z-depth as float32 EXR, produced on the same grid as its own
    intrinsics, in the same arbitrary gauge as its poses. No per-frame scale.

``VGGT / VGGT-X``
    Min-max normalised *inverse* depth. Metric meaning exists only through a
    per-frame ``{scale, offset}`` fitted against COLMAP. Training on it without
    normalising away the residual per-frame scale error makes the depth term
    dominate the photometric one (measured: 29.5 vs ~0.2 total loss).

``COLMAP sparse projection``
    Metric z-depth, consistent with the model's intrinsics, but extremely
    sparse (one sample per triangulated feature).

The invariant every function here preserves: **a depth sample and the
intrinsics used to unproject it must describe the same pixel grid.** Feed-forward
models predict depth on their own grid (518x392, 518x280, ...) with their own
estimated focal. Resampling that onto the render grid by pure image resize is
only correct when the two focals agree. Measured on a real aerial scene they
differed by **10.4%**, and the resulting ray error grows toward the image edges.
:func:`remap_depth` therefore resamples through rays, not through pixel
coordinates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

#: Depth is normalised to this dtype regardless of source.
DEPTH_DTYPE = np.float32


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_depth(path: str | Path) -> Optional[np.ndarray]:
    """Read one depth map as a 2-D float32 array, or ``None`` if absent.

    Supported: ``.npy`` (any numeric dtype), ``.exr`` (float), and 16-bit
    ``.png`` (scaled by 1/65535). A 16-bit PNG carrying metric depth has no
    declared unit, so callers that use it must supply the scale.
    """
    p = Path(path)
    if not p.exists():
        return None

    suffix = p.suffix.lower()
    if suffix == ".npy":
        depth = np.load(p)
    else:
        import cv2

        # OpenCV ships the OpenEXR codec but gates it behind this env var, and
        # it is read when imgcodecs initialises, so set it defensively each
        # call rather than relying on process startup order.
        if suffix == ".exr":
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

        depth = cv2.imread(
            str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
        )
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        if suffix == ".png" and depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 65535.0

    depth = np.asarray(depth, dtype=DEPTH_DTYPE)
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"{p}: expected 2-D depth, got {depth.shape}")
    return depth


def find_depth_file(
    depth_dir: Path, frame_name: str, suffix: Optional[str] = None
) -> Optional[Path]:
    """Locate the depth file for one frame.

    Tries the frame name with its own stem, then with each of the common depth
    extensions. Feed-forward exporters are not consistent about whether they
    keep the source extension (``train_000001.jpg.exr``), replace it
    (``train_000001.exr``) or drop to a zero-padded index (``000001.exr``).
    """
    stem = Path(frame_name).stem
    name = Path(frame_name).name
    candidates = []
    if suffix:
        candidates.extend([f"{stem}{suffix}", f"{name}{suffix}"])
    candidates.extend(
        [
            f"{stem}.exr",
            f"{name}.exr",
            f"{stem}.npy",
            f"{name}.npy",
            f"{stem}.png",
            f"{name}.png",
        ]
    )
    for cand in candidates:
        p = depth_dir / cand
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------
# Per-frame calibration
# --------------------------------------------------------------------------
@dataclass
class DepthCalibration:
    """Per-frame ``{scale, offset}`` applied to a raw depth value."""

    scale: float = 1.0
    offset: float = 0.0

    def apply(self, depth: np.ndarray) -> np.ndarray:
        if self.scale == 1.0 and self.offset == 0.0:
            return depth
        return depth * self.scale + self.offset

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.offset == 0.0


def load_depth_scales(path: str | Path) -> Dict[str, DepthCalibration]:
    """Read a ``{frame: {"scale": s, "offset": o}}`` JSON.

    Tolerates the two shapes seen in practice: a per-frame mapping, and a
    wrapper with a ``"scales"`` key holding it.
    """
    import json

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if "scales" in raw and isinstance(raw["scales"], dict):
        raw = raw["scales"]

    out: Dict[str, DepthCalibration] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict):
            out[name] = DepthCalibration(
                scale=float(entry.get("scale", 1.0)),
                offset=float(entry.get("offset", 0.0)),
            )
        else:
            out[name] = DepthCalibration(scale=float(entry))
    return out


def median_scale(scales: Dict[str, DepthCalibration]) -> float:
    vals = [c.scale for c in scales.values() if c.scale > 0]
    return float(np.median(vals)) if vals else 1.0


def filter_scale(
    scale: float,
    reference: float,
    lower: float = 0.2,
    upper: float = 5.0,
) -> bool:
    """Whether a per-frame scale is plausible relative to the dataset median.

    A frame whose fitted scale is off by more than 5x almost always means the
    fit failed (too few COLMAP points, an occluded view). Such a frame poisons
    the depth term, so dropping it is better than training on it.
    """
    if reference <= 0:
        return True
    return lower * reference <= scale <= upper * reference


# --------------------------------------------------------------------------
# Grid alignment
# --------------------------------------------------------------------------
def remap_depth(
    depth: np.ndarray,
    src_K: np.ndarray,
    dst_K: np.ndarray,
    dst_hw: Tuple[int, int],
    fill: float = 0.0,
) -> np.ndarray:
    """Resample depth from ``src_K``'s grid onto ``dst_K``'s grid.

    Correct when the two intrinsics differ (different focal, different
    principal point). Both grids are assumed to observe the same pinhole
    camera, so the mapping is a homography on normalised image coordinates:

        goal pixel -> ray via dst_K -> source pixel via src_K -> sample

    Nearest-neighbour sampling is deliberate: depth is piecewise smooth with
    step discontinuities at object boundaries, and bilinear interpolation
    invents depth values that belong to neither surface.
    """
    import cv2

    h_dst, w_dst = int(dst_hw[0]), int(dst_hw[1])
    yy, xx = np.meshgrid(
        np.arange(h_dst, dtype=np.float32),
        np.arange(w_dst, dtype=np.float32),
        indexing="ij",
    )
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # [H,W,3]

    try:
        dst_K_inv = np.linalg.inv(np.asarray(dst_K, dtype=np.float64))
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise ValueError("dst_K is singular") from exc

    cam_dirs = rays @ dst_K_inv.T
    src = cam_dirs @ np.asarray(src_K, dtype=np.float64).T
    z = src[..., 2]
    # Pixels behind the source camera have no depth sample.
    safe_z = np.where(np.abs(z) > 1e-12, z, 1e-12)
    mapx = (src[..., 0] / safe_z).astype(np.float32)
    mapy = (src[..., 1] / safe_z).astype(np.float32)

    remapped = cv2.remap(
        np.asarray(depth, dtype=np.float32),
        mapx,
        mapy,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill),
    )
    # Anything that sampled outside the source extent is not a measurement.
    h_src, w_src = depth.shape
    outside = (
        (mapx < 0)
        | (mapy < 0)
        | (mapx > w_src - 1)
        | (mapy > h_src - 1)
        | (z <= 0)
    )
    remapped[outside] = float(fill)
    return remapped


def resize_depth_nearest(depth: np.ndarray, dst_hw: Tuple[int, int]) -> np.ndarray:
    """Pure resize, correct ONLY when the two grids share intrinsics up to
    scale. Prefer :func:`remap_depth` whenever that is not known to hold."""
    import cv2

    return cv2.resize(
        np.asarray(depth, dtype=np.float32),
        (int(dst_hw[1]), int(dst_hw[0])),
        interpolation=cv2.INTER_NEAREST,
    )


def needs_ray_remap(
    depth_hw: Tuple[int, int],
    depth_K: Optional[np.ndarray],
    render_hw: Tuple[int, int],
    render_K: np.ndarray,
    *,
    focal_tolerance: float = 0.01,
) -> bool:
    """Whether the depth grid implies different intrinsics from the render grid.

    Pure resize is exact when the depth grid is the render grid scaled by a
    constant. Comparing focal ratios per axis is what detects the case where it
    is not -- e.g. a source that resized 1920x1080 to 518x280 (non-uniform:
    sx=3.707, sy=3.857) and whose own estimated focal is used for the depth.
    """
    if depth_K is None:
        return False
    dh, dw = depth_hw
    rh, rw = render_hw
    if (dh, dw) == (rh, rw):
        return False
    sx = rw / float(dw)
    sy = rh / float(dh)
    implied_fx = float(depth_K[0, 0]) * sx
    implied_fy = float(depth_K[1, 1]) * sy
    fx = float(render_K[0, 0])
    fy = float(render_K[1, 1])
    if fx <= 0 or fy <= 0:
        return True
    return (
        abs(implied_fx - fx) / fx > focal_tolerance
        or abs(implied_fy - fy) / fy > focal_tolerance
    )


def scale_K_2d(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale a 3x3 intrinsic per axis, keeping ``K[2,2] == 1``."""
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def depth_to_normal(
    depth: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Surface normals implied by a depth map, in world space.

    Used by the 2DGS normal-consistency term, which compares the rendered
    surfel normal against the normal implied by the model's *own* rendered
    depth -- no ground-truth geometry involved. Returns ``[H,W,3]`` unit
    vectors; degenerate pixels (zero or missing depth) produce zeros.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
    )
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Unproject to camera space.
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth
    pts = np.stack([x, y, depth], axis=-1)  # [H,W,3]

    # Central differences, one-sided at the border.
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dx[:, 0] = pts[:, 1] - pts[:, 0]
    dx[:, -1] = pts[:, -1] - pts[:, -2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    dy[0, :] = pts[1, :] - pts[0, :]
    dy[-1, :] = pts[-1, :] - pts[-2, :]

    n_cam = np.cross(dx, dy)
    norm = np.linalg.norm(n_cam, axis=-1, keepdims=True)
    n_cam = np.divide(
        n_cam, np.maximum(norm, eps), out=np.zeros_like(n_cam), where=norm > eps
    )

    # Camera -> world: normals rotate by the rotation block only.
    R = np.asarray(cam_to_world, dtype=np.float64)[:3, :3]
    n_world = n_cam @ R.T
    norm_w = np.linalg.norm(n_world, axis=-1, keepdims=True)
    n_world = np.divide(
        n_world, np.maximum(norm_w, eps), out=np.zeros_like(n_world), where=norm_w > eps
    )
    # COLMAP/OpenCV cameras look down +z; the cross product above points away
    # from the camera for a fronto-parallel surface, so flip to face the viewer.
    n_world *= -1.0
    return n_world.astype(np.float32)


__all__ = [
    "DEPTH_DTYPE",
    "DepthCalibration",
    "read_depth",
    "find_depth_file",
    "load_depth_scales",
    "median_scale",
    "filter_scale",
    "remap_depth",
    "resize_depth_nearest",
    "needs_ray_remap",
    "scale_K_2d",
    "depth_to_normal",
]
