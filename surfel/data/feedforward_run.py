"""Run a feed-forward reconstruction (Scal3R / VGGT) and export a bundle.

This module is the *outer* wrapper around the upstream predictors. It exists so
that no upstream source file has to be edited:

* The Scal3R entry point is invoked as a subprocess, so its dependency tree and
  its own argument parsing stay untouched.
* Its one environment-sensitive behaviour -- depth is written with
  ``cv2.imwrite(*.exr)``, and OpenCV only exposes the OpenEXR codec when
  ``OPENCV_IO_ENABLE_OPENEXR=1`` was set *before* ``imgcodecs`` initialised --
  is handled by exporting the variable into the child environment. That is the
  documented way to enable the codec; nothing in Scal3R needs patching.
* If the installed OpenCV has no EXR codec at all, the wrapper detects it up
  front (by probing a tiny write) and reports the fix instead of letting the
  child crash after inference has already succeeded and the depths are lost.

Depth container policy
----------------------
``depth_format`` selects what the bundle stores, independent of what the
predictor wrote:

* ``"exr"`` (default) -- keep the predictor's float32 EXR.
* ``"npy"`` -- convert to ``.npy`` at export time.

Both are lossless float32; the choice is about interoperability with other
tools, not accuracy. ``surfel.data.feedforward.read_bundle`` reads either.

Usage
-----
::

    python -m surfel.data.feedforward_run --method scal3r \\
        --images /data/garden/images_4 --out /data/garden/ff \\
        --depth-format exr

    # then train on it
    python -m surfel.train.run --profile aerial --data.path /data/garden/ff \\
        --set data.parser=feedforward --set data.init_points=depth
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

#: Depths written by Scal3R into ``<run>/depths/NNNNNN.<ext>``.
SCAL3R_DEPTH_EXTS = (".exr", ".npy")


# --------------------------------------------------------------------------
# OpenCV codec capability
# --------------------------------------------------------------------------
def exr_codec_available(python_exe: Optional[str] = None) -> bool:
    """Whether the target interpreter's OpenCV can write EXR.

    Probed by actually writing a 4x4 float array, because that is the only
    signal that reflects both "the codec is compiled in" and "the enable flag
    took effect". Build info alone reports what was compiled, not what is
    reachable at runtime.
    """
    exe = python_exe or sys.executable
    code = (
        "import os; os.environ['OPENCV_IO_ENABLE_OPENEXR']='1';"
        "import cv2, numpy as np, tempfile, os;"
        "p=os.path.join(tempfile.gettempdir(),'_ff_exr_probe.exr');"
        "ok=False\n"
        "try:\n"
        "    ok=bool(cv2.imwrite(p, np.zeros((4,4),dtype=np.float32)))\n"
        "except Exception:\n"
        "    ok=False\n"
        "finally:\n"
        "    try: os.remove(p)\n"
        "    except OSError: pass\n"
        "print('OK' if ok else 'NO')\n"
    )
    try:
        out = subprocess.run(
            [exe, "-c", code], capture_output=True, text=True, timeout=180
        )
        return out.stdout.strip().endswith("OK")
    except Exception:
        return False


def opencv_version(python_exe: Optional[str] = None) -> str:
    exe = python_exe or sys.executable
    try:
        out = subprocess.run(
            [exe, "-c", "import cv2;print(cv2.__version__)"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _child_env(enable_exr: bool) -> dict:
    """Environment for the predictor subprocess.

    The OpenEXR flag must be present *before* ``import cv2`` in the child, which
    is precisely why it is set here and not inside any upstream function.
    """
    env = dict(os.environ)
    if enable_exr:
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


# --------------------------------------------------------------------------
# Scal3R
# --------------------------------------------------------------------------
def find_scal3r_repo(explicit: Optional[str] = None) -> Path:
    """Locate the Scal3R checkout that provides ``scal3r.run``."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("SCAL3R_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / "Github" / "Scal3R")
    for c in candidates:
        if (c / "scal3r" / "run.py").exists():
            return c
    raise FileNotFoundError(
        "Scal3R checkout not found. Pass --scal3r-root, set SCAL3R_ROOT, or "
        "clone it to ~/Github/Scal3R."
    )


def run_scal3r(
    images_dir: str | Path,
    out_dir: str | Path,
    *,
    scal3r_root: Optional[str] = None,
    python_exe: Optional[str] = None,
    depth_format: str = "exr",
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> Path:
    """Run Scal3R on a folder of images, returning the run directory.

    Scal3R is invoked as a subprocess so this wrapper never imports it: its
    package expects to run as ``python -m scal3r.run`` with its own repo root on
    ``sys.path``, and importing it from another process risks pulling its
    dependency pins into the training environment.

    The only thing adjusted is the child environment's
    ``OPENCV_IO_ENABLE_OPENEXR``, which is how OpenCV's EXR codec is enabled.
    """
    root = find_scal3r_repo(scal3r_root)
    exe = python_exe or sys.executable
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = Path(images_dir)
    if not images.is_dir():
        raise FileNotFoundError(f"image directory not found: {images}")

    fmt = str(depth_format).lower().lstrip(".")
    if fmt not in ("exr", "npy"):
        raise ValueError(f"depth_format must be 'exr' or 'npy', got {depth_format!r}")

    enable_exr = fmt == "exr"
    if enable_exr and not exr_codec_available(exe):
        raise RuntimeError(
            "depth_format='exr' but this OpenCV cannot write EXR "
            f"(OpenCV {opencv_version(exe)}). OpenCV's EXR support is compiled "
            "in, so installing the system OpenEXR libraries does not add it: "
            "use an OpenCV build that has it (the 4.x wheels do; 5.1.0 does "
            "not), or re-run with --depth-format npy."
        )

    cmd = [
        exe,
        "-m",
        "scal3r.run",
        "--input_dir",
        str(images),
        "--output_dir",
        str(out / "scal3r"),
        "--tag",
        out.name,
    ]
    cmd.extend(extra_args)

    if dry_run:
        print("would run:", " ".join(cmd))
        print("env: OPENCV_IO_ENABLE_OPENEXR=1" if enable_exr else "env: (exr disabled)")
        return out / "scal3r"

    proc = subprocess.run(cmd, cwd=str(root), env=_child_env(enable_exr))
    if proc.returncode != 0:
        raise RuntimeError(
            f"Scal3R exited with {proc.returncode}. Its own diagnostics are "
            "above; this wrapper adds no flags of its own beyond --output_dir."
        )
    return out / "scal3r"


# --------------------------------------------------------------------------
# VGGT
# --------------------------------------------------------------------------
def run_vggt(
    images_dir: str | Path,
    npz_out: str | Path,
    *,
    vggt_root: Optional[str] = None,
    python_exe: Optional[str] = None,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> Path:
    """Run VGGT's demo to produce a predictions ``.npz``.

    ``demo_colmap.py`` writes COLMAP files and can optionally dump depth; the
    wrapper points it at ``--save_depth`` and leaves the rest to the caller.
    """
    root = Path(
        vggt_root or os.environ.get("VGGT_ROOT") or (Path.home() / "Github" / "VGGT-X")
    )
    demo = root / "demo_colmap.py"
    if not demo.exists():
        raise FileNotFoundError(f"VGGT demo not found: {demo}")
    exe = python_exe or sys.executable
    out = Path(npz_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        str(demo),
        "--scene_dir",
        str(images_dir),
        "--save_depth",
        *extra_args,
    ]
    if dry_run:
        print("would run:", " ".join(cmd))
        return out
    proc = subprocess.run(cmd, cwd=str(root), env=_child_env(True))
    if proc.returncode != 0:
        raise RuntimeError(f"VGGT exited with {proc.returncode}")
    return out


# --------------------------------------------------------------------------
# Depth re-containerisation (post-run, no upstream involvement)
# --------------------------------------------------------------------------
def convert_depths(
    depth_dir: str | Path,
    target: str,
    *,
    keep_source: bool = False,
) -> int:
    """Convert every depth map in ``depth_dir`` to ``target`` (``exr``/``npy``).

    This is the escape hatch when a predictor wrote one container and the bundle
    should hold another. Both directions are exact float32.
    """
    from .feedforward import load_depth_any, save_depth_npy

    tgt = str(target).lower().lstrip(".")
    if tgt not in ("exr", "npy"):
        raise ValueError(f"target must be 'exr' or 'npy', got {target!r}")
    d = Path(depth_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"no depth directory: {d}")

    sources = sorted(
        p for p in d.iterdir() if p.suffix.lower() in SCAL3R_DEPTH_EXTS
        and p.suffix.lower() != f".{tgt}"
    )
    if tgt == "exr":
        # Enabling the codec has to happen before imgcodecs initialises, so the
        # conversion refuses rather than producing a half-written set.
        if not exr_codec_available():
            raise RuntimeError(
                "converting to EXR needs an OpenCV with the OpenEXR codec; this "
                f"one (OpenCV {opencv_version()}) does not have it"
            )
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    import cv2

    n = 0
    for src in sources:
        arr = load_depth_any(src)
        dst = d / f"{src.stem}.{tgt}"
        if tgt == "npy":
            save_depth_npy(arr, dst)
        else:
            if not cv2.imwrite(str(dst), arr.astype("float32")):
                raise RuntimeError(f"cv2 could not write {dst}")
        if not keep_source and src != dst:
            src.unlink()
        n += 1
    return n


def pick_depth_suffix(depth_dir: str | Path) -> str:
    """Detect which depth container a run directory actually contains."""
    d = Path(depth_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"no depth directory: {d}")
    counts = {ext: 0 for ext in SCAL3R_DEPTH_EXTS}
    for p in d.iterdir():
        s = p.suffix.lower()
        if s in counts:
            counts[s] += 1
    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        raise FileNotFoundError(f"no depth maps in {d}")
    return best


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m surfel.data.feedforward_run",
        description=(
            "Run Scal3R or VGGT on a folder of images and export a "
            "surfel-feedforward/1 bundle. Upstream sources are never modified."
        ),
    )
    p.add_argument("--method", choices=["scal3r", "vggt", "export-only"], required=True)
    p.add_argument("--images", required=True, help="Folder of input images.")
    p.add_argument("--out", required=True, help="Output directory for the run + bundle.")
    p.add_argument(
        "--depth-format",
        choices=["exr", "npy"],
        default="exr",
        help=(
            "Container for depths in the bundle. 'exr' (default) keeps the "
            "predictor's float32 EXR; 'npy' converts at export time. Both are "
            "lossless. Requires an OpenCV with the OpenEXR codec for 'exr'."
        ),
    )
    p.add_argument("--scal3r-root", default=None, help="Scal3R checkout (else $SCAL3R_ROOT).")
    p.add_argument("--vggt-root", default=None, help="VGGT checkout (else $VGGT_ROOT).")
    p.add_argument("--python", default=None, help="Interpreter used for the subprocess.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="For --method export-only: existing predictor run to convert.",
    )
    p.add_argument(
        "--convert-depths",
        choices=["exr", "npy"],
        default=None,
        help="Convert an existing run's depths before exporting.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Everything after this is forwarded to the predictor.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run and args.method != "export-only":
        if args.method == "scal3r":
            run_scal3r(
                args.images, out, scal3r_root=args.scal3r_root, python_exe=args.python,
                depth_format=args.depth_format, extra_args=args.extra, dry_run=True,
            )
        else:
            run_vggt(
                args.images, out / "pred.npz", vggt_root=args.vggt_root,
                python_exe=args.python, extra_args=args.extra, dry_run=True,
            )
        return 0

    if args.method == "scal3r":
        run_dir = run_scal3r(
            args.images, out, scal3r_root=args.scal3r_root, python_exe=args.python,
            depth_format=args.depth_format, extra_args=args.extra,
        )
        if args.convert_depths:
            n = convert_depths(run_dir / "depths", args.convert_depths)
            print(f"converted {n} depth map(s) to .{args.convert_depths}")
        from .feedforward import export_scal3r

        suffix = pick_depth_suffix(run_dir / "depths")
        bundle = export_scal3r(
            run_dir, out / "bundle", image_dir=args.images,
            depth_suffix=suffix, depth_format=args.depth_format,
        )
        print(f"bundle: {bundle}")
        return 0

    if args.method == "vggt":
        npz = run_vggt(
            args.images, out / "pred.npz", vggt_root=args.vggt_root,
            python_exe=args.python, extra_args=args.extra,
        )
        from .feedforward import export_vggt

        bundle = export_vggt(
            npz, out / "bundle", args.images, depth_format=args.depth_format
        )
        print(f"bundle: {bundle}")
        return 0

    # export-only
    if not args.run_dir:
        raise SystemExit("--method export-only requires --run-dir")
    if args.convert_depths:
        n = convert_depths(Path(args.run_dir) / "depths", args.convert_depths)
        print(f"converted {n} depth map(s)")
    from .feedforward import export_scal3r

    suffix = pick_depth_suffix(Path(args.run_dir) / "depths")
    bundle = export_scal3r(
        args.run_dir, out / "bundle", image_dir=args.images,
        depth_suffix=suffix, depth_format=args.depth_format,
    )
    print(f"bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "exr_codec_available",
    "opencv_version",
    "find_scal3r_repo",
    "run_scal3r",
    "run_vggt",
    "convert_depths",
    "pick_depth_suffix",
    "main",
]
