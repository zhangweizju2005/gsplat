"""surfel -- a large-scene oriented 2D Gaussian surfel training pipeline on gsplat.

Scope and rationale
-------------------
This package is a *training architecture*, not a new rasteriser. All rendering,
splatting and densification primitives come from ``gsplat``; what is built here
is the part gsplat deliberately leaves to the user:

* **data**: a COLMAP-model-first parser whose resolution policy makes it
  impossible to pair image pixels with mismatched intrinsics;
* **gauge**: a single place where the reconstruction's arbitrary world frame is
  measured, stated and transformed -- poses, points and depth together;
* **model**: 2DGS splats initialised correctly for depth-derived clouds;
* **densify**: gsplat's ``DefaultStrategy`` wired with the 2DGS gradient key and
  *instruments that raise when densification silently stops working*;
* **train**: the official gsplat loop ordering, with the large-scene losses
  (disparity depth, distortion, normal consistency) and schedule ramps.

Design rules that came from this project's own failure log
---------------------------------------------------------
1. **Never pair a resolution with foreign intrinsics.** The scale factor is
   always relative to the COLMAP model, and ``K`` is rescaled to whatever grid
   is actually decoded, including when a pre-downsampled directory turns out to
   contain full-resolution files. Cost of getting this wrong once: 8-12 dB,
   silently.
2. **Never mix gauges without an explicit similarity**, and when applying one,
   apply it to depth as well as poses and points.
3. **Never let a regulariser depend on ground-truth data it does not need.**
   The 2DGS normal term is self-consistency; gating it on depth supervision made
   it inert for a whole run.
4. **Never trust that densification ran.** Count refine events and candidates and
   abort loudly if the counters stay at zero.
5. **Never compare PSNR across resolutions or split modes.** Both are recorded in
   the run manifest so a number cannot be read without its protocol.

Two upstream gsplat behaviours are worked around here and documented at the
call site:

* the 2DGS example calls ``strategy.initialize_state()`` without
  ``scene_scale``, which on a large scene prunes everything as oversized;
* ``absgrad=True`` crashes on 2DGS because ``meta["gradient_2dgs"]`` carries no
  ``.absgrad`` attribute.
"""

from . import config, data

__all__ = ["config", "data"]

__version__ = "0.1.0"
