"""Scene containers: cameras, per-frame depth, and the parsed scene.

The one non-obvious design point is that a :class:`Camera` carries **two**
intrinsics: ``K`` (the grid that gets rendered) and ``depth_K`` (the grid the
supervised depth was predicted on). They differ whenever depth comes from a
feed-forward model, and conflating them is the single most damaging class of
bug in this project's history:

* unprojecting depth with the render ``K`` instead of the depth ``K`` rotates
  every ray, by ~10% on the aerial scene where the two focals disagreed;
* resizing depth by pixel ratio when the focals differ shifts samples
  systematically, worst at the image border.

Keeping both, plus the flag that says whether they agree, makes the choice
explicit at the one place where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Camera:
    """One training/eval view."""

    #: Frame identifier as it appears in the COLMAP model (used for depth and
    #: scale lookups).
    name: str
    #: Absolute path to the image actually read.
    image_path: Path
    #: Rendered grid intrinsics.
    K: np.ndarray
    #: Rendered grid size.
    width: int
    height: int
    #: Camera-to-world, OpenCV convention (+x right, +y down, +z forward).
    cam_to_world: np.ndarray

    #: Intrinsics of the depth grid, when it differs from ``K``. ``None`` means
    #: "the depth shares the render grid", which is the common case for depth
    #: exported alongside the same images.
    depth_K: Optional[np.ndarray] = None
    #: ``(height, width)`` of the depth grid.
    depth_hw: Optional[Tuple[int, int]] = None
    #: Path to this frame's depth map, when depth supervision is enabled.
    depth_path: Optional[Path] = None
    #: Per-frame multiplier on the hierarchy-inherited depth value.
    #: ``1.0`` means the depth already carries the world gauge.
    depth_scale: float = 1.0
    depth_offset: float = 0.0
    #: Index into the training split, for appearance/pose modules.
    index: int = -1

    @property
    def world_to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.cam_to_world)

    @property
    def center(self) -> np.ndarray:
        return self.cam_to_world[:3, 3]

    @property
    def depth_grid_differs(self) -> bool:
        """Whether depth needs ray-remapping rather than a plain resize."""
        if self.depth_K is None or self.depth_hw is None:
            return False
        if self.depth_hw == (self.height, self.width):
            return False
        # Compare the intrinsics that the *same* grid scaling would imply.
        sx = self.width / float(self.depth_hw[1])
        sy = self.height / float(self.depth_hw[0])
        fx_implied = float(self.depth_K[0, 0]) * sx
        fy_implied = float(self.depth_K[1, 1]) * sy
        return (
            abs(fx_implied - float(self.K[0, 0])) / max(float(self.K[0, 0]), 1e-9)
            > 0.01
            or abs(fy_implied - float(self.K[1, 1])) / max(float(self.K[1, 1]), 1e-9)
            > 0.01
        )

    def summary(self) -> str:
        d = ""
        if self.depth_path is not None:
            d = f" depth={self.depth_hw}"
            if self.depth_grid_differs:
                d += "(remap)"
            if self.depth_scale != 1.0:
                d += f" x{self.depth_scale:.4g}"
        return f"{self.name} {self.width}x{self.height}{d}"


@dataclass
class Scene:
    """Everything the trainer needs to know about one reconstruction."""

    cameras: List[Camera]
    #: Initial point cloud (world space) and colours in [0, 1].
    points: np.ndarray
    point_rgb: np.ndarray

    #: The similarity that was applied to reach this gauge, if any. Kept so a
    #: model can be mapped back to the source frame.
    applied_sim3: Optional[np.ndarray] = None
    #: Camera-trajectory extent of the *final* gauge (max distance from the
    #: trajectory centroid). Used for LR scaling.
    camera_extent: float = 1.0
    #: Gaussian-size reference for densification thresholds, in the final
    #: gauge. Derived from the *depth distribution* rather than the trajectory
    #: extent, because it multiplies Gaussian sizes.
    scene_scale: float = 1.0
    #: Depth quantiles of the final gauge, used for near/far and loss scaling.
    depth_p01: Optional[float] = None
    depth_p50: Optional[float] = None
    depth_p99: Optional[float] = None
    #: Human-readable provenance lines, for the run's manifest.
    notes: List[str] = field(default_factory=list)

    def cam_to_worlds(self) -> np.ndarray:
        return np.stack([c.cam_to_world for c in self.cameras], axis=0)

    def Ks(self) -> np.ndarray:
        return np.stack([c.K for c in self.cameras], axis=0)

    def image_sizes(self) -> List[Tuple[int, int]]:
        return [(c.width, c.height) for c in self.cameras]

    def __len__(self) -> int:
        return len(self.cameras)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def summary(self) -> str:
        with_depth = sum(1 for c in self.cameras if c.depth_path is not None)
        remapped = sum(1 for c in self.cameras if c.depth_grid_differs)
        lines = [
            f"cameras      : {len(self.cameras)}",
            f"points       : {self.points.shape[0]}",
            f"camera_extent: {self.camera_extent:.6g}",
            f"scene_scale  : {self.scene_scale:.6g}",
        ]
        if self.depth_p50 is not None:
            lines.append(
                "depth p01/50/99: "
                f"{self.depth_p01:.4g} / {self.depth_p50:.4g} / {self.depth_p99:.4g}"
            )
        if with_depth:
            lines.append(
                f"depth maps   : {with_depth}/{len(self.cameras)}"
                + (f" ({remapped} ray-remapped)" if remapped else "")
            )
        return "\n".join(lines)


@dataclass
class Split:
    """Train/val view of a :class:`Scene`."""

    train: Scene
    val: Optional[Scene]

    def __iter__(self):
        return iter((self.train, self.val))


def scene_subset(scene: Scene, indices: Sequence[int]) -> Scene:
    """Copy of ``scene`` restricted to ``indices``.

    Recomputes the trajectory extent for the subset -- a held-out strip at one
    end of a flight line has a much smaller extent than the whole block, and
    carrying the parent's extent over would set the LR to a value that
    corresponds to a scene the subset is not.
    """
    idx = list(indices)
    cameras = [scene.cameras[i] for i in idx]
    for j, cam in enumerate(cameras):
        cam.index = j

    centers = np.stack([c.center for c in cameras], axis=0) if cameras else np.zeros((0, 3))
    extent = (
        float(np.linalg.norm(centers - centers.mean(0), axis=-1).max())
        if len(cameras)
        else 1.0
    )
    return Scene(
        cameras=cameras,
        points=scene.points,
        point_rgb=scene.point_rgb,
        applied_sim3=scene.applied_sim3,
        camera_extent=extent,
        scene_scale=scene.scene_scale,
        depth_p01=scene.depth_p01,
        depth_p50=scene.depth_p50,
        depth_p99=scene.depth_p99,
        notes=list(scene.notes),
    )


__all__ = ["Camera", "Scene", "Split", "scene_subset"]
