"""Config loading: YAML + dotted CLI overrides -> :class:`Config`.

Kept deliberately dumb. There is no schema introspection library in the path:
dotted overrides are resolved by walking ``dataclasses.fields``, so a typo like
``--model.sh_degre 2`` is an error rather than a silently ignored argument. That
matters here because this project has twice lost a run to an override that
never took effect (a config rebuilt as a bare dict, and an override applied to a
copy that was then discarded).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .schema import Config

try:  # PyYAML is optional for the library but required for file loading.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


Number = Union[int, float]


def _coerce(text: str, target_type: Any) -> Any:
    """Coerce a CLI string to the type of the field it will populate."""
    stripped = text.strip()

    # ``none`` is how an unset Optional is expressed on a command line and in
    # the profile tables; it has to be handled before any type dispatch or
    # every Optional[float] override of "null" would fail to parse.
    if stripped.lower() in ("none", "null", ""):
        origin = getattr(target_type, "__origin__", None)
        if target_type is type(None) or (
            origin is Union
            and type(None) in getattr(target_type, "__args__", ())
        ):
            return None

    # bool must be tested before int: bool is a subclass of int.
    if target_type is bool:
        low = stripped.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {text!r}")

    if target_type is int:
        return int(stripped)

    if target_type is float:
        return float(stripped)

    if target_type is str:
        return text

    # Optional[T] -> T
    origin = getattr(target_type, "__origin__", None)
    if origin is Union:
        args = [a for a in target_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return _coerce(text, args[0])
        # Union[int, float] and friends: try each in order.
        for arg in args:
            try:
                return _coerce(text, arg)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"cannot coerce {text!r} to {target_type}")

    if origin is list:
        # Accept inline JSON or comma-separated.
        if stripped.startswith("["):
            return json.loads(stripped)
        return [s.strip() for s in stripped.split(",") if s.strip()]

    # Literal[...] arrives here as typing.Literal with __args__.
    literal_args = getattr(target_type, "__args__", None)
    if literal_args:
        if text not in literal_args:
            raise ValueError(
                f"expected one of {list(literal_args)}, got {text!r}"
            )
        return text

    raise ValueError(f"unsupported override type {target_type}")


def _resolved_types(cls: type) -> Dict[str, Any]:
    """Field name -> real type object, resolving PEP 563 string annotations.

    ``from __future__ import annotations`` turns every annotation into a string,
    so ``dataclasses.fields()[...].type`` is ``"int"`` rather than ``int``.
    ``get_type_hints`` resolves them against the module namespace, which is what
    makes the coercion below able to tell ``int`` from ``float`` from
    ``Optional[str]``. Cached because this is called for every override.
    """
    cached = _TYPE_CACHE.get(cls)
    if cached is None:
        import typing

        cached = typing.get_type_hints(cls)
        _TYPE_CACHE[cls] = cached
    return cached


_TYPE_CACHE: Dict[type, Dict[str, Any]] = {}


def _field_type(node: Any, name: str) -> Any:
    """The declared type of field ``name`` on dataclass ``node``."""
    hints = _resolved_types(type(node))
    if name in hints:
        return hints[name]
    # Fall back to the raw annotation for a field that get_type_hints could not
    # resolve (a forward reference to something not importable, say).
    for f in dataclasses.fields(node):
        if f.name == name:
            return f.type
    raise KeyError(name)


def _set_dotted(root: Any, dotted: str, value_text: str) -> None:
    """Set ``a.b.c`` on a dataclass tree, validating every hop."""
    parts = dotted.split(".")
    node = root
    for i, part in enumerate(parts):
        if not dataclasses.is_dataclass(node):
            raise KeyError(
                f"cannot descend into {'.'.join(parts[:i])!r}: it is a "
                f"{type(node).__name__}, not a config section"
            )
        field_map = {f.name: f for f in dataclasses.fields(node)}
        if part not in field_map:
            raise KeyError(
                f"unknown config key {'.'.join(parts[: i + 1])!r}; "
                f"available here: {sorted(field_map)}"
            )
        target = getattr(node, part)
        is_leaf = i == len(parts) - 1
        if is_leaf:
            coerced = _coerce(value_text, _field_type(node, part))
            setattr(node, part, coerced)
            return
        if target is None:
            raise KeyError(
                f"{'.'.join(parts[: i + 1])!r} is None and cannot be traversed"
            )
        node = target


def _merge_dict(root: Any, data: Dict[str, Any], prefix: str = "") -> None:
    """Apply a nested dict onto a dataclass tree."""
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _merge_dict(root, value, path)
        else:
            _set_dotted(root, path, _scalar_to_text(value))


def _scalar_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return str(value)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyYAML is required to read config files; install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def build_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Sequence[str] = (),
    base: Optional[Config] = None,
) -> Config:
    """Build a validated :class:`Config`.

    Args:
        yaml_path: Optional YAML file. Its keys mirror the dataclass tree.
        overrides: ``key.path=value`` strings, applied after the YAML.
        base: Optional starting config (used for profile composition).

    Returns a fully resolved, validated ``Config``. Raises on any unknown key,
    so a mistyped override cannot silently leave the default in place.
    """
    cfg = base if base is not None else Config()

    if yaml_path is not None:
        data = load_yaml(yaml_path)
        # "profile" is a plain string field, handled by the same path.
        _merge_dict(cfg, data)

    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"override {item!r} is not of the form key=value"
            )
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), value.strip())

    cfg.validate()
    return cfg


def apply_step_scaling(cfg: Config) -> Config:
    """Apply ``train.steps_scaler`` to every step-count field.

    Done once, up front, and written back into the config so the checkpoint and
    the dumped YAML record the values that were actually used. The archived
    MatrixCity runs are all 7000-step; the Mip-NeRF 360 reference is 30k, so
    this factor is what makes one profile usable at either budget.
    """
    f = cfg.train.steps_scaler
    if f == 1.0:
        return cfg

    t = cfg.train
    t.max_steps = int(t.max_steps * f)
    t.eval_steps = [int(s * f) for s in t.eval_steps]
    t.save_steps = [int(s * f) for s in t.save_steps]

    m = cfg.model
    m.sh_degree_interval = (
        int(m.sh_degree_interval * f) if m.sh_degree_interval else 0
    )

    d = cfg.density
    d.refine_start_iter = int(d.refine_start_iter * f)
    d.refine_stop_iter = int(d.refine_stop_iter * f)
    d.refine_every = max(1, int(d.refine_every * f))
    d.reset_every = int(d.reset_every * f)

    cfg.loss.dist_start_iter = int(cfg.loss.dist_start_iter * f)
    cfg.loss.normal_start_iter = int(cfg.loss.normal_start_iter * f)

    dep = cfg.data.depth
    dep.start_iter = int(dep.start_iter * f)
    dep.ramp_steps = int(dep.ramp_steps * f) if dep.ramp_steps else 0
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write the resolved config next to the run, as YAML when available."""
    data = cfg.to_dict()
    p = Path(path)
    if yaml is not None:
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    else:  # pragma: no cover
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)


def config_from_dict(data: Dict[str, Any]) -> Config:
    """Rebuild a validated config from ``Config.to_dict()`` output.

    A checkpoint stores this so evaluation needs nothing but the file. Note the
    rebuild goes through the same validated setter, so a checkpoint written by
    an older revision fails loudly on a renamed key instead of silently
    evaluating with defaults -- an explicit failure mode, since a wrong eval
    config produces a plausible-looking wrong PSNR.
    """
    cfg = Config()
    _merge_dict(cfg, data)
    cfg.validate()
    return cfg


__all__ = [
    "build_config",
    "apply_step_scaling",
    "dump_config",
    "config_from_dict",
    "load_yaml",
]
