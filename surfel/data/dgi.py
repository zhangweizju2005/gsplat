"""Depth-guided initialisation (DGI): build the initial cloud from depth maps.

DGI replaces SfM's sparse triangulated points with a dense cloud obtained by
unprojecting a depth map -- every valid pixel becomes a point. That is what lets
a feed-forward model seed a 2DGS run with far more structure than SfM would give,
and it is also why the initial *variance* in Gaussian size is so much smaller
than for SfM points (kNN spacing is the pixel footprint, not a triangulation
baseline). Densification thresholds calibrated for SfM therefore need attention;
see ``surfel.train.density``.

Three things decide whether the cloud is geometrically right:

**1. Unproject with the depth grid's own intrinsics.** A depth sample ``d(u,v)``
is consistent with the intrinsics of the grid it was predicted on. Unprojecting
with a different ``K`` rotates every ray: the point count is unchanged and the
cloud changes shape, most strongly away from the principal point. At the 10.4%
focal disagreement measured on one aerial scene, that is a ~10% geometric error
at the image corners -- enough to bias every subsequent step.

**2. Scale must follow the depth's gauge.** Scal3R's depth is self-consistent
but in its own arbitrary units; VGGT's is normalised inverse depth whose metric
meaning exists only through a per-frame fit. The caller supplies the per-frame
scale (already applied to the array); the points then inherit the pose gauge.

**3. Confidence and range filtering beats subsampling.** Feed-forward depth is
confident near the optical axis and unreliable at grazing angles, on textureless
regions and across depth discontinuities. Dropping those pixels at the source is
cheaper than pruning the floaters after densification has grown them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DgiConfig:
    """How many points to keep and which to trust."""

    #: Keep every ``stride``-th pixel along each axis.
    stride: int = 1
    #: Drop this fraction of each axis at every border. Feed-forward depth
    #: degrades at the edges, and those pixels project to the largest ray
    #: errors.
    border_trim: float = 0.0
    #: Keep only pixels above this confidence quantile (needs ``confs``).
    conf_quantile: float = 0.0
    #: Discard depths outside ``(min_depth, max_depth)``, in final-gauge units.
    min_depth: float = 0.0
    max_depth: float = float("inf")


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    *,
    stride: int = 1,
    border_trim: float = 0.0,
    conf: Optional[np.ndarray] = None,
    conf_quantile: float = 0.0,
    min_depth: float = 0.0,
    max_depth: float = float("inf"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject one depth map into world-space points.

    Args:
        depth: ``[H,W]`` z-depth, camera space. ``K`` must describe this grid.
        K: ``[3,3]`` intrinsics of the **depth** grid.
        cam_to_world: ``[4,4]`` OpenCV camera-to-world.

    Returns:
        ``(xyz [M,3] float32, flat_index [M] int64)``. ``flat_index`` is
        ``y*W + x`` into the source map, so the caller can fetch the matching
        colour without recomputing the sampling.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {depth.shape}")
    h, w = depth.shape
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.shape != depth.shape:
            raise ValueError(
                f"confidence {conf.shape} must match depth {depth.shape}"
            )

    y0 = int(round(h * border_trim))
    x0 = int(round(w * border_trim))
    y1 = max(y0 + 1, h - y0)
    x1 = max(x0 + 1, w - x0)
    stride = max(1, int(stride))

    ys, xs = np.mgrid[y0:y1:stride, x0:x1:stride]
    ys = ys.ravel()
    xs = xs.ravel()
    z = depth[ys, xs]

    ok = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if conf is not None and conf_quantile > 0.0:
        c = conf[ys, xs]
        finite = c[np.isfinite(c)]
        if finite.size:
            thresh = float(np.quantile(finite, conf_quantile))
            ok &= np.isfinite(c) & (c >= thresh)
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    ys, xs, z = ys[ok], xs[ok], z[ok]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z

    pts_cam = np.stack(
        [x_cam, y_cam, z, np.ones_like(z)], axis=-1
    ).astype(np.float64)
    pts_world = (np.asarray(cam_to_world, dtype=np.float64)[None] @ pts_cam[..., None])[..., 0]
    flat = ys.astype(np.int64) * w + xs.astype(np.int64)
    return pts_world[:, :3].astype(np.float32), flat


def build_dgi_cloud(
    depths: Sequence[np.ndarray],
    Ks: Sequence[np.ndarray],
    cam_to_worlds: Sequence[np.ndarray],
    images: Optional[Sequence[Optional[np.ndarray]]] = None,
    *,
    cfg: Optional[DgiConfig] = None,
    confs: Optional[Sequence[np.ndarray]] = None,
    max_points: int = -1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject many frames into one coloured cloud.

    Each frame uses **its own** intrinsics, which is the only way the rays are
    correct when frames differ in resolution or camera intrinsics.

    ``max_points > 0`` subsamples uniformly, so initialisation cost stays bounded
    regardless of frame count or depth resolution. Uniform rather than
    stride-based subsampling on purpose: stride interacts with the pixel grid
    and can alias against building edges, whereas a uniform draw cannot.
    """
    cfg = cfg or DgiConfig()
    if not (len(depths) == len(Ks) == len(cam_to_worlds)):
        raise ValueError(
            f"need one K and one pose per depth; got {len(depths)} depths, "
            f"{len(Ks)} Ks, {len(cam_to_worlds)} poses"
        )

    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    for i, (depth, K, c2w) in enumerate(zip(depths, Ks, cam_to_worlds)):
        pts, flat = unproject_depth(
            depth,
            K,
            c2w,
            stride=cfg.stride,
            border_trim=cfg.border_trim,
            conf=None if confs is None else confs[i],
            conf_quantile=cfg.conf_quantile,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
        )
        if pts.shape[0] == 0:
            continue
        all_xyz.append(pts)

        if images is not None and images[i] is not None:
            img = np.asarray(images[i])
            hh, ww = img.shape[:2]
            dh, dw = np.asarray(depth).shape
            ys = flat // dw
            xs = flat % dw
            # Look up by *normalised* position: depth and image grids need not
            # match, and assuming equality is the classic source of a cloud
            # with scrambled colours.
            iy = np.clip(np.round(ys * (hh - 1) / max(dh - 1, 1)).astype(np.int64), 0, hh - 1)
            ix = np.clip(np.round(xs * (ww - 1) / max(dw - 1, 1)).astype(np.int64), 0, ww - 1)
            rgb = img[iy, ix, :3].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            all_rgb.append(rgb)
        elif images is not None:
            all_rgb.append(np.full((pts.shape[0], 3), 0.5, np.float32))

    if not all_xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0) if all_rgb else np.full_like(xyz, 0.5)

    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], max_points, replace=False)
        idx.sort()
        xyz, rgb = xyz[idx], rgb[idx]

    return xyz.astype(np.float32), rgb.astype(np.float32)


def filter_cloud_by_extent(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    trim: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points outside the bulk of the cloud, per axis.

    Feed-forward depth occasionally emits a handful of wildly distant points
    (sky pixels, failed fits). Left in, they stretch the trajectory extent --
    and every ``DefaultStrategy`` threshold is multiplied by an extent-derived
    scale, so a few hundred stray points can retune the whole densification
    schedule. Percentile trimming is scale-free and needs no threshold.
    """
    if xyz.shape[0] < 100 or trim <= 0.0:
        return xyz, rgb
    lo = np.quantile(xyz, trim, axis=0)
    hi = np.quantile(xyz, 1.0 - trim, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=-1)
    return xyz[keep], rgb[keep]


__all__ = [
    "DgiConfig",
    "unproject_depth",
    "build_dgi_cloud",
    "filter_cloud_by_extent",
]
