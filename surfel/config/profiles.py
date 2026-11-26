"""Scene profiles: named bundles of settings for a class of capture.

Rationale, from the project's measurements rather than from taste: nearly every
disaster in this project's history was a *default* that quietly assumed an
object-centric scene. A profile is one place to say "this is an aerial block"
and have all the coupled knobs move together.

What actually changes between an object-centric scene and an aerial one:

=====================  =========================  ==========================
knob                   object (garden-like)       aerial block
=====================  =========================  ==========================
world scale            ~1 unit extent            10^2..10^3 m extent
means LR               needs scene scaling        design value once normalised
distortion weight      optional                   0.1, on from step 3000
normal consistency     0.05                       0.05 from step 2000
depth supervision      absent                     present, ramped 0.05 -> 1.0
near/far               fixed 0.2 / 200            derived (up to km)
densify grad threshold 2e-4 (upstream)            6e-4 (verified transferable)
refine_stop            15k of 30k                 7k of 7k
=====================  =========================  ==========================

The last row is the one that is easy to get wrong: the archived MatrixCity runs
use the *entire* run as the densification window, because at 7000 steps a
window closing at 15000 never closes. Carrying the upstream default over would
have meant densification stopping in the last third of the run for no visible
reason.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .schema import Config

ProfileName = Literal["aerial", "aerial_long", "object", "smoke"]

#: Profiles are declared as override dicts so they compose with the same
#: validated setter the CLI uses -- no second code path to keep in sync.
_PROFILES: dict[str, dict[str, object]] = {
    # ---------------------------------------------------------------------
    # aerial: the default for a city block / flight strip.
    # ---------------------------------------------------------------------
    "aerial": {
        # 7000 steps matches every archived MatrixCity run, so numbers are
        # directly comparable with outputs/f0_gtpose etc.
        "train.max_steps": 7000,
        "train.eval_steps": "[7000]",
        "train.save_steps": "[7000]",
        "train.lr.decay_means": True,
        "train.spatial_lr_scale": 1.0,
        # Density window covers the whole run: at 7k steps the upstream 15k
        # stop never triggers anyway, but stating it keeps the semantics equal
        # if someone raises max_steps.
        "density.strategy": "default",
        "density.prune_opa": 0.005,
        "density.grow_grad2d": 0.0006,
        "density.grow_scale3d": 0.01,
        "density.prune_scale3d": 0.1,
        "density.refine_start_iter": 500,
        "density.refine_stop_iter": 7000,
        "density.refine_every": 100,
        "density.reset_every": 3000,
        # 2DGS geometry regularisers: the values the archived strip/f0 recipes
        # used. dist is what keeps building facades from becoming a stack of
        # parallel sheets.
        "loss.dist_weight": 0.1,
        "loss.dist_start_iter": 3000,
        "loss.normal_weight": 0.05,
        "loss.normal_start_iter": 2000,
        "loss.depth_scale": None,
        "loss.depth_eps": 1e-6,
        # Normalise the world so optimiser step sizes are scale-free.
        "gauge.mode": "normalize",
        "gauge.scale_source": "depth_p99",
        "model.sh_degree": 3,
        "model.near_plane": None,
        "model.far_plane": None,
    },
    # ---------------------------------------------------------------------
    # aerial_long: same physics, 30k-step budget (Mip-NeRF-360-like cadence
    # applied to a city block). Densification window opens wider and SH warms
    # up on the same schedule as the upstream reference.
    # ---------------------------------------------------------------------
    "aerial_long": {
        "train.max_steps": 30000,
        "train.eval_steps": "[7000, 30000]",
        "train.save_steps": "[7000, 30000]",
        "train.lr.decay_means": True,
        "train.spatial_lr_scale": 1.0,
        "density.strategy": "default",
        "density.prune_opa": 0.005,
        "density.grow_grad2d": 0.0006,
        "density.refine_start_iter": 500,
        "density.refine_stop_iter": 20000,
        "density.refine_every": 100,
        "density.reset_every": 3000,
        "loss.dist_weight": 0.1,
        "loss.dist_start_iter": 7000,
        "loss.normal_weight": 0.05,
        "loss.normal_start_iter": 5000,
        "gauge.mode": "normalize",
        "gauge.scale_source": "depth_p99",
        "model.sh_degree": 3,
        "model.sh_degree_interval": 1000,
    },
    # ---------------------------------------------------------------------
    # object: matches the upstream gsplat 2DGS reference so results are
    # comparable with published Mip-NeRF 360 numbers.
    # ---------------------------------------------------------------------
    "object": {
        "train.max_steps": 30000,
        "train.eval_steps": "[7000, 30000]",
        "train.save_steps": "[7000, 30000]",
        "train.lr.decay_means": True,
        "train.spatial_lr_scale": 1.0,
        "density.strategy": "default",
        "density.prune_opa": 0.005,
        "density.grow_grad2d": 0.0002,
        "density.refine_start_iter": 500,
        "density.refine_stop_iter": 15000,
        "density.reset_every": 3000,
        "loss.dist_weight": 0.01,
        "loss.dist_start_iter": 3000,
        "loss.normal_weight": 0.05,
        "loss.normal_start_iter": 7000,
        "gauge.mode": "normalize",
        "gauge.scale_source": "extent",
        "model.sh_degree": 3,
    },
    # ---------------------------------------------------------------------
    # smoke: 200 steps, everything on, for wiring checks only. Numbers from
    # this profile are meaningless by construction.
    # ---------------------------------------------------------------------
    "smoke": {
        "train.max_steps": 200,
        "train.eval_steps": "[200]",
        "train.save_steps": "[200]",
        "density.refine_start_iter": 50,
        "density.refine_stop_iter": 200,
        "density.refine_every": 25,
        "density.reset_every": 100,
        "loss.dist_start_iter": 20,
        "loss.normal_start_iter": 20,
        "data.depth.ramp_steps": 200,
        "model.sh_degree": 0,
    },
}


def profile_names() -> list[str]:
    return sorted(_PROFILES)


def profile_overrides(name: str) -> list[str]:
    """Return ``key=value`` strings for a profile, ready for ``build_config``."""
    if name not in _PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; available: {profile_names()}"
        )
    return [f"{k}={_fmt(v)}" for k, v in _PROFILES[name].items()]


def _fmt(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def apply_profile(cfg: Config, name: str) -> Config:
    """Overlay a profile onto an existing config, in place, and return it."""
    from .loader import _set_dotted

    if name not in _PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; available: {profile_names()}"
        )
    for key, value in _PROFILES[name].items():
        _set_dotted(cfg, key, _fmt(value))
    cfg.profile = name
    return cfg


def make_profile_config(name: str) -> Config:
    """A fresh config with only the profile applied (no file, no overrides)."""
    return apply_profile(Config(), name)


__all__ = [
    "ProfileName",
    "profile_names",
    "profile_overrides",
    "apply_profile",
    "make_profile_config",
]
