"""Configuration: dataclass schema, validated loading, and scene profiles."""

from .schema import (
    Config,
    DataConfig,
    DensityConfig,
    DepthConfig,
    GaugeConfig,
    LossConfig,
    LRConfig,
    ModelConfig,
    TrainConfig,
)
from .loader import (
    apply_step_scaling,
    build_config,
    config_from_dict,
    dump_config,
    load_yaml,
)
from .profiles import (
    ProfileName,
    apply_profile,
    make_profile_config,
    profile_names,
    profile_overrides,
)

__all__ = [
    "Config",
    "DataConfig",
    "DepthConfig",
    "GaugeConfig",
    "ModelConfig",
    "LossConfig",
    "DensityConfig",
    "LRConfig",
    "TrainConfig",
    "build_config",
    "apply_step_scaling",
    "dump_config",
    "config_from_dict",
    "load_yaml",
    "ProfileName",
    "profile_names",
    "profile_overrides",
    "apply_profile",
    "make_profile_config",
]
