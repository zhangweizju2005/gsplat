"""Photometric pose refinement (off by default).

This exists because reproducing the project's pose-optimisation comparison needs
the same mechanism, not because it is recommended. The measured record:

* photometric-only pose refinement has been negative in every arm tried here:
  -0.27 dB on strip0, -1.20 dB on flight0, -2.30 dB on the mixed-D arm;
* a frozen-pose, pure-photometric bundle adjustment **diverged monotonically**
  (rotation error 1.15 deg -> 12.87 deg over 30k steps, 0 of 264 frames
  improved) while the *training* loss still fell -- because the global
  photometric optimum is not at the GT poses;
* the amount a pose can move is bounded by Adam's budget, ``lr * sqrt(N)``-ish,
  so on a large scene the translation update is frozen unless the world has
  been normalised first.

So: the mechanism is here, it is honest about what it does, and it is off.
Enable it to reproduce a comparison, not to improve a result.

The parameterisation is a right-multiplication in the camera frame
(``R' = R * dR``, ``t' = t + dt``), which keeps the update well-conditioned
regardless of where the camera sits in the world -- the alternative
(world-frame translation deltas) makes the effective learning rate depend on
distance from the origin, which is exactly the pathology normalisation avoids.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _six_dof_to_matrix(d6: Tensor) -> Tensor:
    """``[..., 6]`` (rotation vector, translation) -> ``[..., 4, 4]``.

    Uses a Rodrigues exponential map rather than a 6D rotation representation:
    the update magnitude, not just its direction, has to be controllable, and
    the matrix exponential gives that directly.
    """
    rot = d6[..., :3]
    trans = d6[..., 3:]
    theta = rot.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rot / theta
    zero = torch.zeros_like(theta[..., 0])
    K = torch.stack(
        [
            zero, -axis[..., 2], axis[..., 1],
            axis[..., 2], zero, -axis[..., 0],
            -axis[..., 1], axis[..., 0], zero,
        ],
        dim=-1,
    ).reshape(*rot.shape[:-1], 3, 3)
    eye = torch.eye(3, device=d6.device, dtype=d6.dtype).expand_as(K)
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    R = eye + sin * K + (1.0 - cos) * (K @ K)

    M = torch.zeros(*d6.shape[:-1], 4, 4, device=d6.device, dtype=d6.dtype)
    M[..., :3, :3] = R
    M[..., :3, 3] = trans
    M[..., 3, 3] = 1.0
    return M


class CameraOptModule(torch.nn.Module):
    """Per-image pose deltas, applied in the camera frame."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.deltas = torch.nn.Parameter(torch.zeros(n, 6))

    def zero_init(self) -> None:
        with torch.no_grad():
            self.deltas.zero_()

    def forward(self, cam_to_worlds: Tensor, indices: Tensor) -> Tensor:
        d = _six_dof_to_matrix(self.deltas[indices])
        return cam_to_worlds @ d

    def pose_drift(self, indices: Tensor) -> dict:
        """Magnitude of the accumulated correction, for monitoring."""
        with torch.no_grad():
            d6 = self.deltas[indices]
            rot = d6[:, :3].norm(dim=-1)
            trans = d6[:, 3:].norm(dim=-1)
        return {
            "rot_deg_median": float(torch.rad2deg(rot.median())) if rot.numel() else 0.0,
            "trans_median": float(trans.median()) if trans.numel() else 0.0,
        }


__all__ = ["CameraOptModule"]
