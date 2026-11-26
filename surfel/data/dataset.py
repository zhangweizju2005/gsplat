"""Torch dataset over a :class:`Scene`.

The only non-obvious behaviour is depth: a frame's depth map is loaded at its
native grid and brought onto the render grid **once**, by the method that is
correct for that camera -- ray remap when the intrinsics differ, resize when
they do not. Doing it here (rather than inside the loss) means:

* the trainer never sees two grids, so it cannot accidentally compare them;
* the cost is paid once per frame per epoch, not once per step per loss term;
* the choice is auditable, because ``Camera.depth_grid_differs`` records it.

Image caching is optional and defaults off. A 1000-frame 1080p block is ~6 GB of
uint8; caching it is often the difference between a run that spends its time
rendering and one that spends it decoding, but it is memory the caller may not
have.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .depth import read_depth, remap_depth, resize_depth_nearest
from .scene import Camera, Scene


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as ``uint8 [H,W,3]``."""
    import imageio.v2 as imageio

    img = imageio.imread(str(path))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return np.ascontiguousarray(img[..., :3])


def resize_rgb(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resample an image to ``target_hw``, area-averaging when downsampling.

    ``INTER_AREA`` is the correct choice for downsampling: it averages the
    covered source pixels, which is the anti-aliased result. ``INTER_LINEAR``
    (and the bicubic used by gsplat's own resize helper) sample a neighbourhood
    instead and alias when the factor is large, which on a 4x reduction of a
    textured scene invents high-frequency content that is not in the data.

    Every image **must** pass through here when its file size differs from the
    render grid. Skipping this step while still rescaling ``K`` is exactly the
    mismatch that produced a 12 dB error in this project: the renderer got
    1297-wide pixels paired with intrinsics describing a 324-wide frame and
    failed with a tensor-size error, or -- for models whose shapes happened to
    broadcast -- silently rendered at the wrong field of view.
    """
    import cv2

    h, w = image.shape[:2]
    th, tw = int(target_hw[0]), int(target_hw[1])
    if (h, w) == (th, tw):
        return image
    interp = cv2.INTER_AREA if (th <= h and tw <= w) else cv2.INTER_LINEAR
    return cv2.resize(image, (tw, th), interpolation=interp)


def load_camera_arrays(cam: Camera) -> Dict[str, object]:
    """Load one frame's image and (optionally) its depth, aligned to the grid.

    Both arrays come back on ``(cam.height, cam.width)``. That guarantee is what
    lets the trainer assume a single grid for the batch, and it is why the
    resolution policy can be stated in one place and trusted everywhere.
    """
    image = resize_rgb(read_rgb(cam.image_path), (cam.height, cam.width))

    depth = None
    if cam.depth_path is not None:
        raw = read_depth(cam.depth_path)
        if raw is not None:
            if cam.depth_scale != 1.0 or cam.depth_offset != 0.0:
                raw = raw * cam.depth_scale + cam.depth_offset
            target_hw = (cam.height, cam.width)
            if raw.shape == target_hw:
                depth = raw
            elif cam.depth_grid_differs and cam.depth_K is not None:
                # Intrinsics disagree: resample through rays so each destination
                # pixel samples the direction it actually looks along.
                depth = remap_depth(raw, cam.depth_K, cam.K, target_hw)
            else:
                # Same intrinsics up to resolution: a plain resize is exact.
                depth = resize_depth_nearest(raw, target_hw)
    return {"image": image, "depth": depth}


@dataclass
class _CacheEntry:
    image: np.ndarray
    depth: Optional[np.ndarray]


class SceneDataset(Dataset):
    """Yields ``{"image", "depth", "cam_to_world", "K", "index", "name"}``."""

    def __init__(
        self,
        scene: Scene,
        *,
        cache: bool = False,
        max_cached: int = 0,
        load_depth: bool = True,
    ) -> None:
        self.scene = scene
        self.cameras: List[Camera] = list(scene.cameras)
        self.load_depth = load_depth
        self._cache_enabled = cache
        self._max_cached = max_cached
        self._cache: Dict[int, _CacheEntry] = {}
        self._lock = threading.Lock()

        # One grid for the whole split. Enforced, not assumed: the batch stacks
        # frames, so a mismatch inside a batch is unrepresentable.
        grids = {(c.height, c.width) for c in self.cameras}
        if len(grids) > 1:
            raise ValueError(
                f"scene has {len(grids)} distinct render sizes {sorted(grids)}; "
                "set data.target_long_edge to unify them"
            )
        self._grid = grids.pop() if grids else (0, 0)

        # Eagerly read a couple of frames to fail fast on a broken dataset,
        # rather than 500 steps into a run.
        if self.cameras:
            self._probe(self.cameras[0])
            self._verify_grid()

    def _probe(self, cam: Camera) -> None:
        if not Path(cam.image_path).exists():
            raise FileNotFoundError(
                f"image missing for frame {cam.name!r}: {cam.image_path}"
            )
        if self.load_depth and cam.depth_path is not None:
            if read_depth(cam.depth_path) is None:
                raise ValueError(
                    f"depth map for frame {cam.name!r} is unreadable: "
                    f"{cam.depth_path}"
                )

    @property
    def grid(self) -> tuple[int, int]:
        """The single ``(height, width)`` this dataset renders at."""
        return self._grid

    def _verify_grid(self) -> None:
        """Assert that decoding produces exactly the declared render grid.

        Reading this once up front turns the whole class of resolution bugs into
        a startup error. The failure it prevents is otherwise reported far from
        its cause -- as a broadcast error inside ``l1_loss`` thousands of lines
        later, or, worse, not reported at all because the shapes happened to
        line up while the field of view did not.
        """
        sample = load_camera_arrays(self.cameras[0])["image"]
        h, w = sample.shape[:2]
        if (h, w) != self._grid:
            raise RuntimeError(
                f"decoded image for {self.cameras[0].name!r} is {w}x{h} but the "
                f"camera declares {self._grid[1]}x{self._grid[0]}. "
                "Every image must be resampled to the render grid; if this "
                "fires, the resize step in load_camera_arrays was bypassed."
            )

    def _get(self, idx: int) -> _CacheEntry:
        if self._cache_enabled:
            with self._lock:
                hit = self._cache.get(idx)
            if hit is not None:
                return hit
        arrays = load_camera_arrays(self.cameras[idx])
        entry = _CacheEntry(image=arrays["image"], depth=arrays["depth"])
        if self._cache_enabled:
            with self._lock:
                if self._max_cached == 0 or len(self._cache) < self._max_cached:
                    self._cache[idx] = entry
        return entry

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        cam = self.cameras[idx]
        entry = self._get(idx)

        image = torch.from_numpy(entry.image)
        out: Dict[str, object] = {
            "image": image,
            "cam_to_world": torch.from_numpy(
                np.asarray(cam.cam_to_world, dtype=np.float32)
            ),
            "K": torch.from_numpy(np.asarray(cam.K, dtype=np.float32)),
            "index": idx,
            "name": cam.name,
            "width": cam.width,
            "height": cam.height,
        }
        if self.load_depth and entry.depth is not None:
            out["depth"] = torch.from_numpy(
                np.asarray(entry.depth, dtype=np.float32)
            )
        return out


class CameraBatch:
    """A batch of frames stacked into the shapes gsplat expects.

    gsplat's ``rasterization_2dgs`` takes ``viewmats [C,4,4]`` and ``Ks [C,3,3]``
    where ``C`` is the *camera* dimension. The trainer therefore stacks
    ``batch_size`` frames along ``C``. This is the multi-camera axis, which the
    2DGS kernel supports for every render mode (verified for C in {1,2,4}).
    The separate *batch* axis (leading dim on ``means``) is a different thing
    and is not used here: one set of Gaussians, several views.
    """

    def __init__(self, samples: List[Dict[str, object]]) -> None:
        if not samples:
            raise ValueError("empty batch")
        sizes = {(int(s["height"]), int(s["width"])) for s in samples}
        if len(sizes) != 1:
            raise ValueError(
                "all frames in a batch must share a render size, got "
                f"{sorted(sizes)}. Mixed-resolution blocks need batch_size=1, "
                "or a resolution policy that unifies them."
            )
        self.height, self.width = sizes.pop()
        self.names = [str(s["name"]) for s in samples]
        self.indices = torch.as_tensor(
            [int(s["index"]) for s in samples], dtype=torch.long
        )
        self.images = torch.stack(
            [s["image"] for s in samples]
        ).float() / 255.0  # [C,H,W,3]
        self.cam_to_worlds = torch.stack(
            [s["cam_to_world"] for s in samples]
        ).float()  # [C,4,4]
        self.Ks = torch.stack([s["K"] for s in samples]).float()  # [C,3,3]
        depths = [s.get("depth") for s in samples]
        self.depths = (
            torch.stack(depths).float()  # type: ignore[arg-type]
            if all(d is not None for d in depths)
            else None
        )

    def has_depth(self) -> bool:
        return self.depths is not None

    def to(self, device: torch.device | str) -> "CameraBatch":
        self.images = self.images.to(device)
        self.cam_to_worlds = self.cam_to_worlds.to(device)
        self.Ks = self.Ks.to(device)
        self.indices = self.indices.to(device)
        if self.depths is not None:
            self.depths = self.depths.to(device)
        return self

    def __len__(self) -> int:
        return self.images.shape[0]


def collate(samples: List[Dict[str, object]]) -> CameraBatch:
    return CameraBatch(samples)


__all__ = ["SceneDataset", "CameraBatch", "collate", "read_rgb", "load_camera_arrays"]
