"""Model layer: 2DGS splat parameters and the rasterisation front-end."""

from .renderer import (
    RenderOutput,
    Renderer2DGS,
    assert_backend_contract,
    derive_depth_range,
)
from .splats import (
    SplatInit,
    activated,
    build_optimizers,
    create_splats_2dgs,
    knn_mean_distance_rms,
    means_lr_scheduler,
    rgb_to_sh0,
)

__all__ = [
    "SplatInit",
    "create_splats_2dgs",
    "activated",
    "build_optimizers",
    "rgb_to_sh0",
    "knn_mean_distance_rms",
    "means_lr_scheduler",
    "RenderOutput",
    "Renderer2DGS",
    "derive_depth_range",
    "assert_backend_contract",
]
