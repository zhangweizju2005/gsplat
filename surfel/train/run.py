"""Command-line entry point for the ``surfel`` 2DGS training pipeline.

Usage
-----
::

    # aerial scene, archive-comparable 7000 steps
    python -m surfel.train.run --profile aerial --data.path /data/city_block

    # a YAML file, with two dotted overrides applied on top of it
    python -m surfel.train.run --config runs/my.yaml \\
        --set train.max_steps=30000 --set model.packed=true

    # resume, and re-evaluate the final checkpoint afterwards
    python -m surfel.train.run --profile aerial --data.path /data/blk \\
        --resume outputs/blk/ckpts/step_007000.pt

    # evaluate only, using the config stored inside the checkpoint
    python -m surfel.train.run --eval-only outputs/blk/ckpts/final.pt

    # print what a run would do, without touching the GPU
    python -m surfel.train.run --profile aerial --data.path /data/blk --dry-run

Design notes
------------
**One resolution order, stated once.** Values are layered as

    dataclass defaults  ->  --config YAML  ->  --profile  ->  --set k=v

Profile *after* YAML so a profile is a scene-class preset that a run file can
start from, while `--set` stays the last word for a single experiment. Every
layer goes through the same validated setter in :mod:`surfel.config.loader`, so
a typo raises instead of being silently ignored -- the project has twice lost a
run to an override that never took effect.

**The resolved config is always written next to the run.** ``<out_dir>/config.yaml``
plus ``<out_dir>/manifest.json`` record the profile, the overrides, the library
versions and the parse report. A PSNR from a run therefore cannot be quoted
without the conditions it was produced under -- which is the whole point of the
`protocol` block the trainer already stores in its checkpoints.

**Nothing here invents a training value.** Every number comes from
:class:`surfel.config.schema.Config`; this module only picks a device, applies
``train.steps_scaler``, and decides whether to train, resume or evaluate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import (
    Config,
    apply_step_scaling,
    build_config,
    config_from_dict,
    dump_config,
    profile_names,
)
from ..config.loader import load_yaml
from ..config.profiles import apply_profile


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def _parse_set(item: str) -> str:
    """Validate one ``--set key=value`` item early, so a typo fails before setup."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f"--set expects key=value, got {item!r}"
        )
    key = item.split("=", 1)[0].strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--set expects a key before '=', got {item!r}")
    return item


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m surfel.train.run",
        description=(
            "Train 2DGS splats on a COLMAP scene. All training parameters come "
            "from the surfel config schema; this CLI only selects a profile, "
            "optional YAML, and per-run overrides."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m surfel.train.run --profile aerial --data.path /data/blk\n"
            "  python -m surfel.train.run --config run.yaml --set train.max_steps=3000\n"
            "  python -m surfel.train.run --eval-only outputs/blk/ckpts/final.pt\n"
        ),
    )

    src = p.add_argument_group("configuration sources (applied in this order)")
    src.add_argument(
        "--config",
        metavar="YAML",
        default=None,
        help="YAML file mirroring the Config dataclass tree.",
    )
    src.add_argument(
        "--profile",
        choices=profile_names(),
        default=None,
        help=(
            "Scene-class preset applied after --config. 'aerial' = city block / "
            "flight strip (7000 steps); 'aerial_long' = same physics, 30k steps; "
            "'object' = upstream 2DGS cadence; 'smoke' = 200-step wiring check."
        ),
    )
    src.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        action="append",
        type=_parse_set,
        default=[],
        help=(
            "Dotted override applied last, e.g. --set density.grow_grad2d=2e-4. "
            "Repeatable. Keys are validated against the schema, so a typo raises."
        ),
    )

    io = p.add_argument_group("paths")
    io.add_argument(
        "--data.path",
        dest="data_path",
        default=None,
        help=(
            "Dataset root (shorthand for --set data.path=...). COLMAP model "
            "directory by default; see data.parser for the alternative."
        ),
    )
    io.add_argument(
        "--out",
        dest="out_dir",
        default=None,
        help=(
            "Run directory (shorthand for --set train.out_dir=...). Defaults to "
            "outputs/<profile>_<dataset-dir-name> so two runs do not collide."
        ),
    )
    io.add_argument(
        "--resume",
        metavar="CKPT",
        default=None,
        help=(
            "Checkpoint to resume from. Restores parameters, step counter and "
            "the densification strategy's running state."
        ),
    )

    mode = p.add_argument_group("modes")
    mode.add_argument(
        "--eval-only",
        metavar="CKPT",
        default=None,
        help=(
            "Skip training; rebuild the config and the scene from the checkpoint "
            "and report validation metrics. Paths inside the checkpoint are used "
            "as recorded."
        ),
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and validate the config, parse the dataset, then print the "
            "run manifest and exit without setting up the model or training."
        ),
    )
    mode.add_argument(
        "--no-steps-scaler",
        action="store_true",
        help=(
            "Do not apply train.steps_scaler to the step-count fields. Useful "
            "when the override already states absolute steps."
        ),
    )
    mode.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output; the run log at <out_dir>/train.log is still written.",
    )

    p.add_argument(
        "--device",
        default=None,
        help="Torch device string (shorthand for --set train.device=...).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (shorthand for --set train.seed=...).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"surfel {_surfel_version()}",
    )
    return p


def _surfel_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover - only during a broken install
        return "unknown"


# --------------------------------------------------------------------------
# Config assembly
# --------------------------------------------------------------------------
def _default_out_dir(cfg: Config) -> str:
    """A run directory that encodes the profile and the dataset, so runs coexist."""
    stem = Path(cfg.data.path).resolve().name or "run"
    profile = cfg.profile or "custom"
    return str(Path("outputs") / f"{profile}_{stem}")


def assemble_config(args: argparse.Namespace) -> Config:
    """Layer defaults -> YAML -> profile -> shorthands -> ``--set``.

    The shorthands are folded into the override list rather than assigned
    directly: they must go through the same validated setter as everything else,
    and they must land *before* explicit ``--set`` items so an explicit override
    still wins.
    """
    overrides: List[str] = []

    # Shorthands first: an explicit --set of the same key must win.
    if args.data_path is not None:
        overrides.append(f"data.path={args.data_path}")
    if args.device is not None:
        overrides.append(f"train.device={args.device}")
    if args.seed is not None:
        overrides.append(f"train.seed={args.seed}")

    # --config then --profile: build_config applies YAML, then we overlay the
    # profile, then the remaining overrides -- same order the docstring states.
    cfg = build_config(yaml_path=args.config, overrides=overrides)

    if args.profile is not None:
        apply_profile(cfg, args.profile)

    for item in args.overrides:
        # Re-validate by round-tripping through the same setter the loader uses.
        from ..config.loader import _set_dotted

        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), value.strip())

    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    elif not cfg.train.out_dir:
        cfg.train.out_dir = _default_out_dir(cfg)

    cfg.validate()
    if not args.no_steps_scaler:
        apply_step_scaling(cfg)
    # validate() again: scaling writes new step values that must stay coherent
    # (e.g. refine_start <= max_steps), and it is cheap.
    cfg.validate()
    return cfg


def _config_from_checkpoint(path: str | Path) -> Config:
    """Rebuild the config stored in a checkpoint, through the validated setter."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise SystemExit(
            f"{path}: no 'config' key in the checkpoint; it predates the "
            "config-recording change and cannot be evaluated reproducibly."
        )
    return config_from_dict(payload["config"])


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def _torch_versions() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def write_manifest(
    cfg: Config,
    out_dir: Path,
    report: Any,
    *,
    args: argparse.Namespace,
    mode: str,
    path: Optional[Path] = None,
) -> Path:
    """Record everything needed to read a metric from this run honestly.

    The protocol block (split mode, resolutions, counts) is duplicated from the
    checkpoint on purpose: the manifest is what a human opens, and the trainer's
    own copy only helps a script.

    ``path`` defaults to ``<out_dir>/manifest.json``, i.e. the training record.
    Non-training modes pass their own name so that re-evaluating a checkpoint
    cannot overwrite the record of the run that produced it.
    """
    manifest = {
        "mode": mode,
        "profile": cfg.profile,
        # The *actual* arguments, not ``sys.argv``: ``main(argv)`` is callable
        # from a script or a test, and recording the interpreter's own argv there
        # would label the run with something that never described it.
        "argv": list(getattr(args, "_argv", None) or sys.argv[1:]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "versions": {
            "surfel": _surfel_version(),
            **_torch_versions(),
            **_gsplat_version(),
        },
        "config": cfg.to_dict(),
        "parse_report": _report_to_dict(report),
        "protocol": {
            "split_mode": cfg.data.split_mode,
            "n_train": getattr(report, "n_train", None),
            "n_val": getattr(report, "n_val", None),
            "render_sizes": getattr(report, "render_sizes", None),
            "eval_on_train": cfg.data.split_mode == "reconstruction",
        },
    }
    path = Path(path) if path is not None else (out_dir / "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path


def _gsplat_version() -> Dict[str, Any]:
    try:
        import gsplat

        return {
            "gsplat": gsplat.__version__,
            "has_3dgs": gsplat.has_3dgs(),
            "has_2dgs": gsplat.has_2dgs(),
        }
    except Exception as exc:  # pragma: no cover
        return {"gsplat_error": repr(exc)}


def _report_to_dict(report: Any) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    if hasattr(report, "__dict__"):
        return {k: v for k, v in vars(report).items() if not k.startswith("_")}
    return {"repr": repr(report)}


def _print_summary(cfg: Config, report: Any, manifest: Path) -> None:
    """Human-readable echo of the resolved run, before any GPU work."""
    # ``render_sizes`` used to be printed as a raw list of tuples, so nothing on
    # screen said whether ``(1297, 840)`` was width-first or height-first -- and
    # the two parsers disagreed about that. The report owns the formatting now.
    if hasattr(report, "render_size_labels"):
        sizes = report.render_size_labels()
    else:  # pragma: no cover - a report from an older/foreign caller
        raw = getattr(report, "render_sizes", None)
        sizes = str(raw) if raw else "?"
    print("=" * 72)
    print(f"profile      : {cfg.profile or '(none)'}")
    print(f"dataset      : {cfg.data.path}")
    print(f"image dir    : {getattr(report, 'image_dir', '?')}")
    print(
        "images       : "
        f"{getattr(report, 'n_images', '?')} "
        f"({getattr(report, 'n_train', '?')} train / {getattr(report, 'n_val', '?')} val)"
    )
    print(f"render sizes : {sizes}")
    print(f"points       : {getattr(report, 'n_points', '?')}")
    print(f"depth maps   : {getattr(report, 'depth_maps', 0)}")
    print(
        "gauge        : "
        f"{cfg.gauge.mode}/{cfg.gauge.scale_source} "
        f"(apply_to_depth={cfg.gauge.apply_to_depth})"
    )
    print(f"steps        : {cfg.train.max_steps} (eval {cfg.train.eval_steps})")
    print(f"strategy     : {cfg.density.strategy}")
    print(f"packed       : {cfg.model.packed}")
    print(f"out dir      : {cfg.train.out_dir}")
    print(f"manifest     : {manifest}")
    notes = getattr(report, "notes", None)
    if notes:
        print("-" * 72)
        for note in notes:
            print(f"  note: {note}")
    print("=" * 72)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def _resolve_device(requested: Optional[str]) -> Optional[str]:
    """Refuse to fall back to CPU silently: 2DGS rasterisation needs CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    if requested is None:
        return None
    return requested


def run(args: argparse.Namespace) -> int:
    from ..data import parse_dataset
    from .trainer import Trainer

    quiet = args.quiet
    log_fn = (lambda _m: None) if quiet else print

    if args.eval_only:
        cfg = _config_from_checkpoint(args.eval_only)
        mode = "eval"
    else:
        cfg = assemble_config(args)
        mode = "dry-run" if args.dry_run else ("resume" if args.resume else "train")

    device = _resolve_device(cfg.train.device)
    if device is None and mode != "dry-run":
        raise SystemExit(
            "CUDA is not available. The 2DGS rasteriser requires a GPU; "
            "re-run with --dry-run to validate the configuration only."
        )

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The dataset is parsed before the model so the manifest records what was
    # actually read, and so --dry-run can stop here.
    split, report = parse_dataset(cfg)

    # Resolved config next to the run, so the numbers can never be separated
    # from the values that produced them.
    dump_config(cfg, out_dir / "config.yaml")

    if args.eval_only:
        trainer = Trainer(cfg, split, device=device, log_fn=log_fn)
        payload = trainer.load(args.eval_only)
        metrics = trainer.evaluate()
        # An evaluation must never overwrite the training run's record. The
        # manifest is what a human opens to see how a number was produced, and
        # `--eval-only` was rewriting it with mode="eval" and its own argv,
        # destroying the training entry -- irreversibly, since nothing else kept
        # a copy. A separate file is written instead, named after the checkpoint
        # so several evaluations can coexist.
        eval_manifest = out_dir / f"manifest.eval.{Path(args.eval_only).stem}.json"
        write_manifest(
            cfg, out_dir, report, args=args, mode=mode, path=eval_manifest
        )
        trainer.close()
        if not metrics:
            print(
                json.dumps(
                    {
                        "metrics": {},
                        "step": payload.get("step"),
                        "warning": (
                            "no held-out set: data.split_mode="
                            f"{cfg.data.split_mode!r} trains on every image, so "
                            "evaluate() has nothing to score. Re-run with "
                            "data.split_mode=experiment (and data.eval_ratio>0) "
                            "to get a generalisation number."
                        ),
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps({"metrics": metrics, "step": payload.get("step")}, indent=2))
        print(f"manifest: {eval_manifest}")
        return 0

    manifest = write_manifest(cfg, out_dir, report, args=args, mode=mode)
    _print_summary(cfg, report, manifest) if not quiet else None

    if args.dry_run:
        print("dry run: configuration and dataset are valid; nothing was trained.")
        return 0

    trainer = Trainer(cfg, split, device=device, log_fn=log_fn)
    if args.resume:
        payload = trainer.load(args.resume)
        trainer._emit(
            f"resumed      : {args.resume} at step {payload.get('step', 0)}"
        )

    try:
        metrics = trainer.fit()
    finally:
        # fit() deliberately leaves the log handle open so a caller can still
        # save/evaluate; the CLI is the caller and is done here.
        trainer.close()

    if metrics:
        print(json.dumps({"val": metrics}, indent=2))
    print(f"run dir : {out_dir}")
    print(f"manifest: {manifest}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Recorded verbatim by write_manifest; see the note there.
    args._argv = list(argv) if argv is not None else sys.argv[1:]

    if args.eval_only and (args.resume or args.dry_run):
        parser.error("--eval-only cannot be combined with --resume or --dry-run")

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, KeyError) as exc:
        # Config problems are the common failure and deserve a clean message
        # rather than a traceback; anything else keeps its traceback, because a
        # CUDA fault with a swallowed stack is much harder to act on.
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
