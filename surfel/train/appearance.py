"""Per-image appearance correction: an exposure affine.

Aerial blocks are flown over minutes to tens of minutes under changing light and
camera auto-exposure, so one global colour cannot fit both the first and last
strip. A single gain plus bias per image is the smallest model that absorbs that
without letting the network explain away geometry.

This is deliberately *not* a per-image full colour transform or an embedding
MLP: those can absorb view-dependent shading and hide a real geometric error. An
exposure affine has exactly two degrees of freedom per frame, and the initial
value (gain 1, bias 0) is an exact no-op, so a run with it enabled starts from the
same model as one without.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ExposureAffine(torch.nn.Module):
    """``out = gain_i * colors + bias_i``, initialised to the identity."""

    def __init__(self, n: int, *, learn_bias: bool = True) -> None:
        super().__init__()
        # Stored as raw values with gain = exp(raw) so the gain stays positive.
        self.raw_gain = torch.nn.Parameter(torch.zeros(n, 1, 1, 1))
        self.learn_bias = learn_bias
        if learn_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n, 1, 1, 1))
        else:
            self.register_parameter("bias", None)

    def forward(self, colors: Tensor, indices: Tensor) -> Tensor:
        # colors: [C,H,W,3]; one gain/bias per camera in the batch.
        gain = torch.exp(self.raw_gain[indices])
        out = colors * gain
        if self.bias is not None:
            out = out + self.bias[indices]
        return out

    def zero_init(self) -> None:
        with torch.no_grad():
            self.raw_gain.zero_()
            if self.bias is not None:
                self.bias.zero_()


__all__ = ["ExposureAffine"]
