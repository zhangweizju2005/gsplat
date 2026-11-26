"""COLMAP model I/O: read ``cameras``/``images``/``points3D``, binary or text.

Deliberately dependency-free. The official gsplat examples use ``pycolmap``,
which is fine when it is installed from source but is a real portability tax:
the PyPI package named ``pycolmap`` is an unrelated 2018 project, and the
official one has no aarch64 wheel. Reading the format directly is ~200 lines
and removes the question.

The one convenience worth preserving from pycolmap is per-camera intrinsics:
COLMAP shares one camera entry across many images, so the model is a
``camera_id -> K`` map plus a ``name -> (camera_id, c2w)`` map, not one K per
image. Collapsing that to a per-image list early keeps the rest of the code
simple, but the mapping is kept so a multi-camera rig stays expressible.

Every quaternion read here is COLMAP's ``(w, x, y, z)``; every rotation matrix
returned is world-to-camera, matching ``images.bin``. Converting to
camera-to-world is the caller's job, and :class:`ColmapModel.cam_to_worlds`
does it in one place so the convention is stated once.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# COLMAP camera model ids -> (name, num_params).
CAMERA_MODELS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_CAMERA_MODELS_BY_NAME = {v[0]: k for k, v in CAMERA_MODELS.items()}


@dataclass
class ColmapCamera:
    """One COLMAP camera entry (shared by every image that uses it)."""

    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def is_distorted(self) -> bool:
        return self.model.split("_")[0] in (
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "FULL_OPENCV",
            "FOV",
            "THIN_PRISM",
        ) or self.model in (
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "OPENCV_FISHEYE",
            "THIN_PRISM_FISHEYE",
        )

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsics (distortion terms ignored).

        COLMAP stores ``fx, fy, cx, cy`` first for every model that has them;
        the pinhole models differ only in whether ``fx == fy``.
        """
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            # Every distorted model starts with fx, fy, cx, cy.
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


@dataclass
class ColmapImage:
    """One image entry: which camera it uses and where the camera sits."""

    image_id: int
    name: str
    camera_id: int
    #: COLMAP convention: (w, x, y, z), world-to-camera.
    qvec: np.ndarray
    tvec: np.ndarray

    @property
    def world_to_camera(self) -> np.ndarray:
        R = qvec_to_rotmat(self.qvec)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = self.tvec
        return M

    @property
    def cam_to_world(self) -> np.ndarray:
        """Inverse of :attr:`world_to_camera`, rotation kept rigid."""
        R = qvec_to_rotmat(self.qvec)
        return np.array(
            [
                [R[0, 0], R[1, 0], R[2, 0], -float(R[:, 0] @ self.tvec)],
                [R[0, 1], R[1, 1], R[2, 1], -float(R[:, 1] @ self.tvec)],
                [R[0, 2], R[1, 2], R[2, 2], -float(R[:, 2] @ self.tvec)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def camera_center(self) -> np.ndarray:
        R = qvec_to_rotmat(self.qvec)
        return -R.T @ self.tvec


@dataclass
class ColmapModel:
    """A parsed COLMAP reconstruction."""

    cameras: Dict[int, ColmapCamera] = field(default_factory=dict)
    images: Dict[int, ColmapImage] = field(default_factory=dict)
    #: ``points3D``: xyz, rgb (uint8), and the tracks that produced them.
    points_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float64)
    )
    points_rgb: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.uint8)
    )

    # -- convenience views -------------------------------------------------
    def sorted_images(self) -> List[ColmapImage]:
        """Images ordered by name.

        Name order, not id order: COLMAP assigns ids by ingestion order, which
        for a re-processed model need not follow the flight path. A trajectory
        that jumps back and forth breaks every consecutive-pair statistic
        (drift, RRA) computed downstream.
        """
        return sorted(self.images.values(), key=lambda im: im.name)

    def names(self) -> List[str]:
        return [im.name for im in self.sorted_images()]

    def cam_to_worlds(self) -> np.ndarray:
        return np.stack([im.cam_to_world for im in self.sorted_images()], axis=0)

    def world_to_cameras(self) -> np.ndarray:
        return np.stack(
            [im.world_to_camera for im in self.sorted_images()], axis=0
        )

    def Ks(self) -> np.ndarray:
        return np.stack(
            [self.cameras[im.camera_id].K for im in self.sorted_images()], axis=0
        )

    def camera_centers(self) -> np.ndarray:
        return np.stack(
            [im.camera_center for im in self.sorted_images()], axis=0
        )

    def image_sizes(self) -> List[Tuple[int, int]]:
        """``(width, height)`` per image, in the same order as :meth:`names`."""
        return [
            (self.cameras[im.camera_id].width, self.cameras[im.camera_id].height)
            for im in self.sorted_images()
        ]

    def subset(self, indices: np.ndarray) -> "ColmapModel":
        """Model restricted to ``indices`` (positions in :meth:`names` order)."""
        ordered = self.sorted_images()
        keep = [ordered[i] for i in indices]
        used_cams = {im.camera_id for im in keep}
        return ColmapModel(
            cameras={cid: c for cid, c in self.cameras.items() if cid in used_cams},
            images={im.image_id: im for im in keep},
            points_xyz=self.points_xyz,
            points_rgb=self.points_rgb,
        )


# --------------------------------------------------------------------------
# Quaternion helpers
# --------------------------------------------------------------------------
def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP ``(w, x, y, z)`` quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(qvec, dtype=np.float64).ravel()
    n = w * w + x * x + y * y + z * z
    if n < 1e-16:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> ``(w, x, y, z)`` with ``w >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q


# --------------------------------------------------------------------------
# Binary readers
# --------------------------------------------------------------------------
def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian="<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(
            f"COLMAP file truncated: wanted {num_bytes} bytes, got {len(data)}"
        )
    return struct.unpack(endian + format_char_sequence, data)


def _read_cameras_binary(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id = props[0], props[1]
            width, height = int(props[2]), int(props[3])
            if model_id not in CAMERA_MODELS:
                raise ValueError(
                    f"camera {camera_id}: unsupported COLMAP model id {model_id}"
                )
            model_name, num_params = CAMERA_MODELS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.asarray(params, dtype=np.float64),
            )
    return cameras


def _read_images_binary(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            # Name is a NUL-terminated string with no length prefix.
            name_chars: List[bytes] = []
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                if char == b"":
                    raise ValueError("COLMAP images file truncated in a name")
                name_chars.append(char)
            name = b"".join(name_chars).decode("utf-8", errors="replace")
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            # x, y, point3D_id per observation; not needed here.
            fid.read(num_points2d * (8 + 8 + 8))
            images[image_id] = ColmapImage(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
            )
    return images


def _read_points3d_binary(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        xyz = np.zeros((num_points, 3), dtype=np.float64)
        rgb = np.zeros((num_points, 3), dtype=np.uint8)
        for i in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            xyz[i] = props[1:4]
            rgb[i] = props[4:7]
            track_len = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_len)
    return xyz, rgb


# --------------------------------------------------------------------------
# Text readers
# --------------------------------------------------------------------------
def _read_cameras_text(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "r", encoding="utf-8") as fid:
        next(fid, None)  # header
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = np.asarray([float(x) for x in parts[4:]], dtype=np.float64)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def _read_images_text(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with open(path, "r", encoding="utf-8") as fid:
        next(fid, None)  # header
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_id = int(parts[0])
            qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
            num_points2d = int(parts[10]) if len(parts) > 10 else 0
            for _ in range(num_points2d):
                fid.readline()
            images[image_id] = ColmapImage(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
            )
    return images


def _read_points3d_text(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xyz: List[List[float]] = []
    rgb: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as fid:
        next(fid, None)
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return (
        np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def read_colmap_model(path: str | Path) -> ColmapModel:
    """Read a COLMAP model from a directory.

    Accepts either a sparse model dir directly (``cameras.bin`` or
    ``cameras.txt`` inside) or a COLMAP project root containing ``sparse/0``.
    Binary is preferred when both exist: it is what COLMAP writes and it is
    lossless for float64 poses.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"COLMAP model path does not exist: {root}")

    candidates = [root]
    if (root / "sparse" / "0").is_dir():
        candidates.insert(0, root / "sparse" / "0")
    if (root / "sparse").is_dir():
        subdirs = sorted(p for p in (root / "sparse").iterdir() if p.is_dir())
        candidates.extend(subdirs)

    model_dir: Optional[Path] = None
    for cand in candidates:
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            model_dir = cand
            break
    if model_dir is None:
        raise FileNotFoundError(
            f"no COLMAP model (cameras.bin/.txt) under {root}; "
            "looked in the directory itself, sparse/0 and sparse/*"
        )

    if (model_dir / "cameras.bin").exists():
        cameras = _read_cameras_binary(model_dir / "cameras.bin")
        images = _read_images_binary(model_dir / "images.bin")
        pts_file = model_dir / "points3D.bin"
        if pts_file.exists():
            xyz, rgb = _read_points3d_binary(pts_file)
        else:
            xyz = np.zeros((0, 3), np.float64)
            rgb = np.zeros((0, 3), np.uint8)
    else:
        cameras = _read_cameras_text(model_dir / "cameras.txt")
        images = _read_images_text(model_dir / "images.txt")
        pts_file = model_dir / "points3D.txt"
        if pts_file.exists():
            xyz, rgb = _read_points3d_text(pts_file)
        else:
            xyz = np.zeros((0, 3), np.float64)
            rgb = np.zeros((0, 3), np.uint8)

    if not images:
        raise ValueError(f"{model_dir}: no images in the COLMAP model")

    missing = {im.camera_id for im in images.values()} - set(cameras)
    if missing:
        raise ValueError(
            f"{model_dir}: images reference camera ids {sorted(missing)} that "
            "are not present in cameras.bin/.txt"
        )

    return ColmapModel(cameras=cameras, images=images, points_xyz=xyz, points_rgb=rgb)


# --------------------------------------------------------------------------
# Writers (needed to hand a normalised / re-rigged scene back to COLMAP)
# --------------------------------------------------------------------------
def write_colmap_text(
    model: ColmapModel,
    out_dir: str | Path,
    *,
    pin_hole_only: bool = True,
) -> Path:
    """Write ``cameras.txt`` / ``images.txt`` / ``points3D.txt``.

    Text, not binary: the point of writing is to let another tool (or a human)
    inspect a transformed gauge, and text is what stays inspectable. COLMAP
    itself reads either.

    ``pin_hole_only`` drops distortion coefficients, because after a similarity
    transform and a resolution change the original distortion model is no
    longer meaningful. Use the undistorted images in that case.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "cameras.txt", "w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(f"# Number of cameras: {len(model.cameras)}\n")
        for cid in sorted(model.cameras):
            cam = model.cameras[cid]
            K = cam.K
            if pin_hole_only:
                model_name = "PINHOLE"
                params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            else:
                model_name = cam.model
                params = list(cam.params)
            joined = " ".join(f"{v:.10g}" for v in params)
            fh.write(
                f"{cid} {model_name} {cam.width} {cam.height} {joined}\n"
            )

    with open(out / "images.txt", "w", encoding="utf-8") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.write(f"# Number of images: {len(model.images)}\n")
        for im in model.sorted_images():
            q = " ".join(f"{v:.10g}" for v in im.qvec)
            t = " ".join(f"{v:.10g}" for v in im.tvec)
            fh.write(f"{im.image_id} {q} {t} {im.camera_id} {im.name}\n")
            # No 2D observations: this writer is for geometry handoff, not for
            # a bundle adjustment round trip.
            fh.write("\n")

    with open(out / "points3D.txt", "w", encoding="utf-8") as fh:
        n = model.points_xyz.shape[0]
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        fh.write(f"# Number of points: {n}\n")
        for i in range(n):
            x, y, z = model.points_xyz[i]
            r, g, b = model.points_rgb[i]
            fh.write(
                f"{i + 1} {x:.10g} {y:.10g} {z:.10g} "
                f"{int(r)} {int(g)} {int(b)} 0.0\n"
            )
    return out


__all__ = [
    "ColmapCamera",
    "ColmapImage",
    "ColmapModel",
    "CAMERA_MODELS",
    "read_colmap_model",
    "write_colmap_text",
    "qvec_to_rotmat",
    "rotmat_to_qvec",
]
