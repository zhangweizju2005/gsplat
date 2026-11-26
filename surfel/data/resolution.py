"""Resolution policy: decide the render grid once, and rescale K to match.

This module exists because of a specific, expensive failure in this project.

A COLMAP model declares intrinsics for a specific image size. A dataset on disk
may also carry pre-downsampled copies (``images_4/``) whose *contents* are not
necessarily actually downsampled, and a trainer may be asked for yet another
factor on top. When the trainer ends up pairing image pixels from one grid with
intrinsics from another, nothing raises. Measured consequence: the same
checkpoint scored 17.47 dB at 1297x840 and 25.34 dB at 324x210 -- a 7.9 dB
swing caused entirely by this mismatch, with the evaluator computing correctly
the whole time.

So the policy here is: **``factor`` (and any target size) is always interpreted
relative to the COLMAP model's own declared image size**, and whatever grid is
finally read, ``K`` is rescaled to that grid. If the file on disk is a different
size again (a stale ``images_4/`` full-resolution copy), the scale is computed
from the *actual* decoded size, not assumed.

Downsampling never upsamples. If the resolved target is larger than the file on
disk, the file's size wins -- silently upsampling would reintroduce the exact
mismatch this module prevents, just in the other direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

#: Pre-downsampled directory names COLMAP's ``image_undistorter`` produces, in
#: the order they are tried when the plain ``images`` dir is absent.
STANDARD_FACTORS: Sequence[int] = (2, 4, 8, 16)


@dataclass
class ResolutionPolicy:
    """How to get from a COLMAP model grid to the grid actually rendered."""

    #: Requested downscale relative to the COLMAP model's declared size.
    factor: float = 1.0
    #: Optional absolute target long edge; overrides ``factor`` when set.
    target_long_edge: Optional[int] = None
    #: Refuse to go below this long edge, whatever factor says.
    min_long_edge: int = 0

    def resolve(
        self,
        model_size: Tuple[int, int],
        file_size: Optional[Tuple[int, int]] = None,
    ) -> "ResolvedResolution":
        """Compute the render grid for one camera.

        Args:
            model_size: ``(width, height)`` as declared by COLMAP.
            file_size: Optional ``(width, height)`` of the image actually on
                disk. When it differs from ``model_size``, the effective
                downscale is measured against the *file*, so ``K`` always
                describes the pixels the renderer will see.
        """
        mw, mh = int(model_size[0]), int(model_size[1])
        if mw <= 0 or mh <= 0:
            raise ValueError(f"bad COLMAP image size {model_size}")

        if self.target_long_edge is not None:
            factor = max(mw, mh) / float(self.target_long_edge)
        else:
            factor = float(self.factor)
        factor = max(1.0, factor)

        if self.target_long_edge is None and self.min_long_edge > 0:
            max_factor = max(mw, mh) / float(self.min_long_edge)
            factor = min(factor, max(1.0, max_factor))

        render_w = max(1, int(round(mw / factor)))
        render_h = max(1, int(round(mh / factor)))

        # Scale K by the ratio actually achieved, per axis. Interpreting the
        # factor by dividing the model size and then scaling K by the model
        # size ratio is what keeps the two consistent when rounding moves the
        # grid by a pixel.
        sx = render_w / float(mw)
        sy = render_h / float(mh)

        file_note = ""
        if file_size is not None and tuple(file_size) != (mw, mh):
            fw, fh = int(file_size[0]), int(file_size[1])
            # The file itself differs from the model. Never upsample: whatever
            # the file has is the most detail available, and pairing requested
            # size with the file's pixel count silently changes the FOV.
            if fw < render_w:
                render_w = fw
                render_h = fh
                sx = render_w / float(mw)
                sy = render_h / float(mh)
                file_note = (
                    f"file is {fw}x{fh}, smaller than the {int(round(mw / factor))}"
                    f"x{int(round(mh / factor))} target -> using the file grid"
                )
            else:
                file_note = (
                    f"file is {fw}x{fh} (not the model's {mw}x{mh}); "
                    "K rescaled by the achieved ratio"
                )

        return ResolvedResolution(
            width=render_w,
            height=render_h,
            scale_x=sx,
            scale_y=sy,
            factor=float(mw) / render_w,
            note=file_note,
        )


@dataclass
class ResolvedResolution:
    """Outcome of :meth:`ResolutionPolicy.resolve` for one camera."""

    width: int
    height: int
    scale_x: float
    scale_y: float
    #: Effective downscale actually applied relative to the model grid.
    factor: float
    note: str = ""

    def scale_K(self, K: np.ndarray) -> np.ndarray:
        """Rescale intrinsics from the model grid to this grid.

        ``K[2, 2]`` stays 1: an intrinsic matrix is a projective object, and
        dividing the whole matrix by the scale (a plausible-looking shortcut)
        breaks the homogeneous normalisation that every unprojection relies on.
        """
        out = np.array(K, dtype=np.float64, copy=True)
        out[0, :] *= self.scale_x
        out[1, :] *= self.scale_y
        return out

    def __str__(self) -> str:
        s = f"{self.width}x{self.height} (x{self.factor:.3g} from model grid)"
        return f"{s}: {self.note}" if self.note else s


def probe_image_size(path: str | Path) -> Optional[Tuple[int, int]]:
    """Read only the image header. Returns ``None`` if unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        from PIL import Image

        with Image.open(p) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return None


def resolve_image_dir(
    root: str | Path,
    candidates: Sequence[str],
    explicit: Optional[str] = None,
) -> Tuple[Path, Optional[int]]:
    """Locate the image directory and infer any built-in downsample factor.

    Returns ``(path, nominal_factor)``. ``nominal_factor`` is parsed from a
    ``images_<n>`` style name and is used **only** to explain the choice in logs
    -- the real calibration always comes from the decoded file size, because a
    directory named ``images_4`` is not guaranteed to contain quarter-size
    pixels (this dataset layout occurred in practice and cost 8 dB).
    """
    root = Path(root)
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if not p.is_dir():
            raise FileNotFoundError(f"data.image_dir does not exist: {p}")
        return p, _factor_from_name(p.name)

    # Prefer the plain, undownsampled directory: it is the only one whose
    # contents are guaranteed to match the COLMAP intrinsics at factor 1.
    ordered = sorted(
        candidates, key=lambda n: (1 if _factor_from_name(n) else 0, n)
    )
    for name in ordered:
        p = root / name
        if p.is_dir():
            return p, _factor_from_name(name)

    raise FileNotFoundError(
        f"no image directory under {root}; tried {list(candidates)}. "
        "Set data.image_dir explicitly if the images live elsewhere."
    )


def _factor_from_name(name: str) -> Optional[int]:
    """Parse ``images_4`` / ``images_4_png`` -> 4; anything else -> None."""
    parts = name.strip().lower().split("_")
    for part in parts[1:]:
        if part.isdigit():
            v = int(part)
            if v > 0:
                return v
    return None


def is_undistorted_dir(name: str) -> bool:
    """Whether a directory name suggests COLMAP-undistorted content."""
    low = name.lower()
    return any(tag in low for tag in ("undistort", "_png", "pinhole"))


def factor_from_standard_dir(
    file_size: Tuple[int, int], model_size: Tuple[int, int]
) -> Optional[int]:
    """If the file/model size ratio is a standard factor, return it."""
    fw, fh = file_size
    mw, mh = model_size
    if mw <= 0 or mh <= 0:
        return None
    rx, ry = mw / fw, mh / fh
    if abs(rx - ry) > 0.02:
        return None
    for f in STANDARD_FACTORS:
        if abs(rx - f) < 0.02:
            return f
    return None


__all__ = [
    "ResolutionPolicy",
    "ResolvedResolution",
    "probe_image_size",
    "resolve_image_dir",
    "is_undistorted_dir",
    "factor_from_standard_dir",
    "STANDARD_FACTORS",
]
