"""Build a tiny synthetic COLMAP dataset for end-to-end smoke tests.

A synthetic scene with a *known* answer is worth more than a smoke test on real
data for catching wiring errors, because every quantity can be checked:
the cameras are on a ring looking at a textured ground plane, so a correct
pipeline recovers it and a wrong gauge does not.

Written as a test fixture rather than a shipped tool: it emits a COLMAP model
plus images plus EXR depth, i.e. exactly the layout the parser expects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def _qvec_from_R(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP ``(w, x, y, z)``."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]
        )
    return q if q[0] >= 0 else -q


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world for an OpenCV camera looking from ``eye`` at ``target``.

    OpenCV convention: +x right, +y **down**, +z forward. Getting the y-axis
    sign wrong here is the single most common synthetic-fixture error and it
    flips the image vertically while every projection check still passes.
    """
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-8:
        raise ValueError("degenerate look_at: forward parallel to up")
    right = right / n
    down = np.cross(fwd, right)  # +y is down in OpenCV
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = fwd
    c2w[:3, 3] = eye
    return c2w


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A checkerboard with smooth colour gradients: recoverable, not noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    cell = max(8, min(h, w) // 8)
    checker = ((yy // cell) + (xx // cell)) % 2
    base = 0.35 + 0.3 * checker
    r = (base * (0.6 + 0.4 * xx / max(w - 1, 1))).clip(0, 1)
    g = (base * (0.6 + 0.4 * yy / max(h - 1, 1))).clip(0, 1)
    b = (base * 0.9).clip(0, 1)
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)


def make_synthetic_colmap(
    root: str | Path,
    *,
    n_frames: int = 12,
    width: int = 160,
    height: int = 120,
    ring_radius: float = 2.6,
    ring_height: float = 1.1,
    focal: float = 150.0,
    depth_suffix: str = ".exr",
    with_depth: bool = True,
    plane_half_size: float = 1.6,
    seed: int = 0,
) -> Path:
    """Write a synthetic COLMAP dataset and return its root.

    Layout produced::

        root/
          images/000000.png ...
          depths/000000.exr ...
          sparse/0/{cameras,images,points3D}.txt
    """
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    if with_depth:
        (root / "depths").mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    # Ground plane at y = 0 (OpenCV +y down, so "ground" is below the cameras).
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    plane_size = plane_half_size
    names: list[str] = []
    c2ws: list[np.ndarray] = []

    # Analytic ground-plane rasterisation.
    #
    # An earlier version projected a sparse point grid and drew discs, which
    # covered only 38-53% of each frame. That is useless as a check: the
    # do-nothing "all black" PSNR on a half-empty image is ~10-11 dB, so a real
    # training score sits only ~2 dB above it and nothing can be discriminated.
    #
    # Intersecting each pixel ray with the plane y=0 instead gives a dense,
    # exactly-known image *and* an exact z-depth map, so depth supervision and
    # photometric supervision agree by construction. Pixels whose ray misses the
    # plane keep depth 0, which exercises the uncovered-pixel path deliberately.
    yy, xx = np.mgrid[0:height, 0:width]
    ray_cam = np.stack(
        [
            (xx - K[0, 2]) / K[0, 0],
            (yy - K[1, 2]) / K[1, 1],
            np.ones_like(xx, dtype=np.float64),
        ],
        axis=-1,
    )  # [H,W,3], z == 1 so t is the z-depth

    # Texture defined on the plane's (x, z) coordinates.
    def plane_texture(px: np.ndarray, pz: np.ndarray) -> np.ndarray:
        u = np.clip((px / plane_size + 1.0) * 0.5, 0.0, 1.0)
        v = np.clip((pz / plane_size + 1.0) * 0.5, 0.0, 1.0)
        grid = max(2, 8)
        checker = ((np.floor(u * grid) + np.floor(v * grid)) % 2).astype(np.float64)
        base = 0.35 + 0.30 * checker
        r = np.clip(base * (0.55 + 0.45 * u), 0, 1)
        g = np.clip(base * (0.55 + 0.45 * v), 0, 1)
        b = np.clip(base * 0.90, 0, 1)
        return np.stack([r, g, b], axis=-1)

    for i in range(n_frames):
        ang = 2 * np.pi * i / n_frames
        eye = np.array(
            [ring_radius * np.cos(ang), -ring_height, ring_radius * np.sin(ang)]
        )
        c2w = _look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        name = f"{i:06d}.png"
        names.append(name)
        c2ws.append(c2w)

        R = c2w[:3, :3]
        dirs_world = ray_cam @ R.T  # [H,W,3]
        denom = dirs_world[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -eye[1] / denom  # y=0 plane, eye[1] < 0
        hit = np.isfinite(t) & (t > 0)
        pts_world = eye[None, None, :] + t[..., None] * dirs_world

        # Confine the plane to a finite patch. Rays that graze the horizon
        # intersect y=0 at enormous distances (measured max z of 51.8 with no
        # bound), which would make the depth range -- and therefore the derived
        # near/far planes and the disparity loss scale -- depend on grazing
        # pixels instead of on the scene.
        on_patch = (
            (np.abs(pts_world[..., 0]) <= plane_size)
            & (np.abs(pts_world[..., 2]) <= plane_size)
        )
        hit = hit & on_patch

        img = np.zeros((height, width, 3), np.uint8)
        if hit.any():
            rgb = plane_texture(pts_world[..., 0], pts_world[..., 2])
            img[hit] = (np.clip(rgb[hit], 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(root / "images" / name, img)

        if with_depth:
            zbuf = np.where(hit, t, 0.0).astype(np.float32)
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2

            cv2.imwrite(
                str(root / "depths" / f"{Path(name).stem}{depth_suffix}"), zbuf
            )

    # COLMAP sparse points: a regular grid on the same plane. COLMAP ids are
    # 1-based and contiguous. Colours come from the same analytic texture
    # function, so a model initialised from these points starts in the right
    # colour space (a random or zero-colour init would make early PSNR
    # meaningless and hide a colour-path bug).
    grid_n = 32
    gx, gz = np.meshgrid(
        np.linspace(-plane_size, plane_size, grid_n),
        np.linspace(-plane_size, plane_size, grid_n),
    )
    points3d = np.stack(
        [gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=-1
    ).astype(np.float64)
    rgb = (np.clip(plane_texture(points3d[:, 0], points3d[:, 2]), 0, 1) * 255).astype(
        np.uint8
    )

    with open(root / "sparse" / "0" / "cameras.txt", "w", encoding="utf-8") as fh:
        fh.write("# Camera list\n")
        fh.write(f"1 PINHOLE {width} {height} {K[0,0]:.10g} {K[1,1]:.10g} "
                 f"{K[0,2]:.10g} {K[1,2]:.10g}\n")

    with open(root / "sparse" / "0" / "images.txt", "w", encoding="utf-8") as fh:
        fh.write("# Image list\n")
        for i, (name, c2w) in enumerate(zip(names, c2ws)):
            w2c = np.linalg.inv(c2w)
            q = _qvec_from_R(w2c[:3, :3])
            t = w2c[:3, 3]
            fh.write(
                f"{i + 1} {q[0]:.10g} {q[1]:.10g} {q[2]:.10g} {q[3]:.10g} "
                f"{t[0]:.10g} {t[1]:.10g} {t[2]:.10g} 1 {name}\n\n"
            )

    with open(root / "sparse" / "0" / "points3D.txt", "w", encoding="utf-8") as fh:
        fh.write("# 3D point list\n")
        for i in range(points3d.shape[0]):
            x, y, z = points3d[i]
            r, g, b = rgb[i]
            fh.write(
                f"{i + 1} {x:.10g} {y:.10g} {z:.10g} {int(r)} {int(g)} {int(b)} 0.0\n"
            )

    return root


def make_two_view_pair(
    root: str | Path, *, width: int = 96, height: int = 72, focal: float = 90.0
) -> Tuple[Path, np.ndarray, np.ndarray]:
    """Minimal 2-frame dataset, for tests that only need the shapes.

    Returns ``(root, K, c2w_0)``.
    """
    make_synthetic_colmap(
        root,
        n_frames=2,
        width=width,
        height=height,
        focal=focal,
        with_depth=False,
    )
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    c2w = _look_at(
        np.array([2.6, -1.1, 0.0]), np.zeros(3), np.array([0.0, -1.0, 0.0])
    )
    return Path(root), K, c2w


__all__ = ["make_synthetic_colmap", "make_two_view_pair"]
