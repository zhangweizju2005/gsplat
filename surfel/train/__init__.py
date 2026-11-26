"""Training layer: losses, densification strategy, optional modules, the loop."""

from .density import (
    DeadDensifierError,
    DensifyStats,
    InstrumentedStrategy,
    build_strategy,
    describe_thresholds,
)
from .losses import (
    LossBreakdown,
    LossWeights,
    depth_term,
    distortion_term,
    image_terms,
    normal_term,
    resolve_depth_scale,
    resolve_weights,
    total_loss,
)
from .trainer import Trainer, TrainMetrics

__all__ = [
    "Trainer",
    "TrainMetrics",
    "LossBreakdown",
    "LossWeights",
    "resolve_weights",
    "resolve_depth_scale",
    "image_terms",
    "depth_term",
    "distortion_term",
    "normal_term",
    "total_loss",
    "InstrumentedStrategy",
    "DensifyStats",
    "DeadDensifierError",
    "build_strategy",
    "describe_thresholds",
]
