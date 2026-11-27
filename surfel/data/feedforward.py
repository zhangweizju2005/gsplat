"""Standardised feed-forward bundle: write depth + poses in one portable layout.

Why this module exists
----------------------
Scal3R and VGGT both predict poses and depth, but neither writes them in a form
this project can consume on every machine:

* Scal3R's exporter hard-codes ``cv2.imwrite('depths/NNNNNN.exr')``. The OpenCV
  wheel in this environment is built without the OpenEXR codec, so depth export
  aborts *after* inference has already succeeded -- the poses are fine, the
  depths are lost. It also gates ``OPENCV_IO_ENABLE_OPENEXR`` inside
  ``save_image``, i.e. after ``imgcodecs`` has initialised, so the variable
  cannot take effect even where the codec exists.
* VGGT keeps depth in an in-memory ``np.savez`` payload whose key layout is
  internal to that repo.

Rather than patch either upstream, this module defines **one bundle format** and
two adapters that produce it. Everything downstream (DGI, the COLMAP-free
dataset path) reads the bundle, so a swap of front end changes nothing else.

Bundle layout
-------------
::

    <bundle>/
      frames.json            # manifest: frames, K, c2w, depth paths, gauge
      images/000000.jpg ...  # one file per frame, original bytes
      depths/000000.npy ...  # float32 [H,W] z-depth, compressed .npy

``frames.json``::

    {
      "format": "surfel-feedforward/1",
      "source": "scal3r",
      "image_grid": [W, H],          # resolution of `images/`
      "depth_grid": [W, H],          # resolution of `depths/` (K describes this)
      "camera_convention": "opencv",  # +z forward, x right, y down
      "pose_convention": "c2w",       # camera-to-world, 4x4 row-major
      "depth_kind": "z",              # z-depth along the optical axis
      "gauge": {"kind": "similarity", "scale": 1.234, "note": "..."},
      "frames": [
        {"index": 0, "name": "DSC07956.JPG",
         "image": "images/000000.jpg", "depth": "depths/000000.npy",
         "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],       # for the DEPTH grid
         "c2w": [[...],[...],[...],[...]]}
      ]
    }

The three facts that make a bundle self-checking, and that were each a
documented failure in this project:

1. ``depth_grid`` is stated explicitly and ``K`` is *defined* to describe it.
   Pairing a depth grid with intrinsics from another resolution is silent and
   costs 8-12 dB.
2. ``image_grid`` is separate. Depth may be lower resolution than the images
   (Scal3R: 518x322 depth against 1297x840 images). Consumers resample colour by
   *normalised* position, never by assuming the grids match.
3. ``gauge`` records where the scale came from. Feed-forward output is defined
   only up to a similarity; training tolerates that, but mixing with a metric
   source does not, and the manifest is where that has to be visible.

Dependencies: ``numpy`` and (optionally) ``cv2`` for image copying. No EXR, no
Open3D, no upstream package.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT_ID = "surfel-feedforward/1"

#: Manifest key order is fixed so a diff between two bundles is readable.
_FRAME_KEYS = ("index", "name", "image", "depth", "K", "c2w")


# --------------------------------------------------------------------------
# YAML subset reader (OpenCV FileStorage output)
# --------------------------------------------------------------------------
def read_opencv_yaml(path: str | Path) -> Dict[str, Any]:
    """Read the subset of OpenCV ``FileStorage`` YAML that Scal3R/VGGT write.

    Deliberately not PyYAML: OpenCV emits custom tags (``!!opencv-matrix``) that
    plain ``yaml.safe_load`` rejects, and pulling in a loader just for these two
    files would add a dependency for no gain. The grammar used here is the one
    OpenCV actually produces: flat ``key: value`` pairs plus ``!!opencv-matrix``
    blocks with ``rows`` / ``cols`` / ``dt`` / ``data``.
    """
    import re

    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: Dict[str, Any] = {}
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("%") or line in ("---", "..."):
            continue
        # list entry:  - "name"
        if line.startswith("- "):
            out.setdefault("__list__", []).append(line[2:].strip().strip('"'))
            continue
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest.startswith("!!opencv-matrix"):
            rows = cols = None
            data: List[float] = []
            while i < n:
                sub = lines[i].strip()
                i += 1
                if not sub or sub in ("---", "..."):
                    break
                if sub.startswith("rows:"):
                    rows = int(sub.split(":", 1)[1])
                elif sub.startswith("cols:"):
                    cols = int(sub.split(":", 1)[1])
                elif sub.startswith("data:"):
                    body = sub.split(":", 1)[1].strip()
                    if i < n and body.startswith("["):
                        while "]" not in body and i < n:
                            body += " " + lines[i].strip()
                            i += 1
                    data = [float(x) for x in re.findall(r"[-+0-9.eE]+", body)]
                elif re.match(r"^[A-Za-z_]", sub) and not sub.startswith(
                    ("rows", "cols", "dt", "data")
                ):
                    # a new top-level key: step back and let the outer loop see it
                    i -= 1
                    break
            if rows and cols:
                out[key] = np.asarray(data, dtype=np.float64).reshape(rows, cols)
            else:
                out[key] = np.asarray(data, dtype=np.float64)
            continue
        if rest.startswith("[") and rest.endswith("]"):
            out[key] = [float(x) for x in re.findall(r"[-+0-9.eE]+", rest)]
        else:
            out[key] = rest.strip('"')
    return out


def _yaml_names(parsed: Dict[str, Any]) -> List[str]:
    names = parsed.get("__list__")
    if not names:
        raise ValueError("camera YAML has no names[] list")
    return list(names)


# --------------------------------------------------------------------------
# Depth I/O (never EXR)
# --------------------------------------------------------------------------
def save_depth_npy(depth: np.ndarray, path: str | Path) -> Path:
    """Write float32 depth as a compressed ``.npy``.

    ``.npy`` rather than ``.exr`` on purpose: OpenCV's EXR codec is absent from
    the wheel used here, and PNG would quantise float depth. ``.npy`` is exact,
    needs no codec, and is what ``surfel.data.depth.read_depth`` already prefers.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(depth)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {arr.shape}")
    np.save(p, arr.astype(np.float32), allow_pickle=False)
    return p


def load_depth_any(path: str | Path) -> np.ndarray:
    """Read a depth map from ``.npy`` or a 16-bit PNG, without OpenEXR.

    Mirrors :func:`surfel.data.depth.read_depth` but raises instead of returning
    ``None``: inside the exporter an unreadable depth is an error, not a signal.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=False)
    else:
        import os

        import cv2

        # OpenCV gates the EXR codec behind this variable, and reads it when
        # imgcodecs initialises. Setting it here is defensive: a caller that set
        # it before importing cv2 is unaffected, and one that did not still gets
        # the codec when the wheel has it.
        if p.suffix.lower() == ".exr":
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            hint = ""
            if p.suffix.lower() == ".exr":
                hint = (
                    " (this OpenCV build may lack the OpenEXR codec; the 5.1.0 "
                    "wheel does, the 4.x wheels do not)"
                )
            raise ValueError(f"cannot read depth {p}{hint}")
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


# --------------------------------------------------------------------------
# Bundle writer
# --------------------------------------------------------------------------
@dataclass
class BundleFrame:
    """One frame of a feed-forward bundle."""

    name: str
    image_src: Path
    depth: Optional[np.ndarray]
    K: np.ndarray
    c2w: np.ndarray
    #: Optional per-frame light field: confidence / mask used by DGI filtering.
    confidence: Optional[np.ndarray] = None
    depth_scale: float = 1.0


@dataclass
class BundleWriter:
    """Accumulate frames and write a ``surfel-feedforward/1`` bundle."""

    root: Path
    source: str
    image_grid: Tuple[int, int]
    depth_grid: Optional[Tuple[int, int]] = None
    camera_convention: str = "opencv"
    pose_convention: str = "c2w"
    depth_kind: str = "z"
    #: Depth container written into ``depths/``: ``"exr"`` (default, float32,
    #: interoperable) or ``"npy"`` (float32, needs no codec). Both are lossless;
    #: EXR falls back to npy if this OpenCV lacks the codec, with a note.
    depth_format: str = "exr"
    gauge: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    frames: List[BundleFrame] = field(default_factory=list)

    # -- construction ---------------------------------------------------
    def add(
        self,
        *,
        name: str,
        image_src: str | Path,
        K: np.ndarray,
        c2w: np.ndarray,
        depth: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
    ) -> None:
        """Record one frame. Depth may be ``None`` (poses-only bundle)."""
        K = np.asarray(K, dtype=np.float64)
        c2w = np.asarray(c2w, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"{name}: K must be 3x3, got {K.shape}")
        if c2w.shape != (4, 4):
            raise ValueError(f"{name}: c2w must be 4x4, got {c2w.shape}")
        # A pose is only usable if its rotation block is rigid; Scal3R's
        # mat.txt carries a cbrt(det) factor and fails this, extri.yml passes.
        R = c2w[:3, :3]
        if abs(np.linalg.det(R) - 1.0) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
            raise ValueError(
                f"{name}: c2w rotation block is not orthonormal "
                f"(det={np.linalg.det(R):.5f}); this is the mat.txt signature, "
                "use extri.yml instead"
            )
        if depth is not None:
            d = np.asarray(depth)
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            if d.ndim != 2:
                raise ValueError(f"{name}: depth must be 2-D, got {d.shape}")
            grid = (int(d.shape[1]), int(d.shape[0]))  # (W, H)
            if self.depth_grid is None:
                self.depth_grid = grid
            elif self.depth_grid != grid:
                raise ValueError(
                    f"{name}: depth grid {grid} differs from {self.depth_grid}; "
                    "a bundle must have one depth resolution because K is shared"
                )
        self.frames.append(
            BundleFrame(
                name=name,
                image_src=Path(image_src),
                depth=None if depth is None else np.asarray(depth, dtype=np.float32),
                K=K,
                c2w=c2w,
                confidence=confidence,
                depth_scale=float(depth_scale),
            )
        )

    # -- writing --------------------------------------------------------
    def _write_depth(self, depth: np.ndarray, stem_without_suffix: Path) -> Path:
        """Write one depth map in ``self.depth_format``, with a safe fallback.

        EXR is the default because it is the interoperable float container. If
        this OpenCV has no OpenEXR codec, writing it would raise *after* the
        predictor had already spent its time, so the writer degrades to ``.npy``
        and records why in the manifest notes instead of failing.
        """
        fmt = str(self.depth_format).lower().lstrip(".")
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if fmt == "exr":
            import os

            import cv2

            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            out = stem_without_suffix.with_suffix(".exr")
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                ok = bool(cv2.imwrite(str(out), arr))
            except Exception:
                ok = False
            if ok:
                return out
            note = (
                "depth_format='exr' unsupported by this OpenCV build; wrote .npy"
            )
            if note not in self.notes:
                self.notes.append(note)
        return save_depth_npy(arr, stem_without_suffix.with_suffix(".npy"))

    def write(self) -> Path:
        """Materialise ``images/``, ``depths/`` and ``frames.json``."""
        root = Path(self.root)
        (root / "images").mkdir(parents=True, exist_ok=True)
        if self.depth_grid is not None:
            (root / "depths").mkdir(parents=True, exist_ok=True)

        entries = []
        for idx, fr in enumerate(self.frames):
            stem = f"{idx:06d}"
            dst_img = root / "images" / f"{stem}{fr.image_src.suffix.lower()}"
            if fr.image_src.resolve() != dst_img.resolve():
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                try:
                    # Hardlink first: bundles are often built next to their
                    # source and copying gigabytes to duplicate bytes is waste.
                    dst_img.hardlink_to(fr.image_src)
                except OSError:
                    shutil.copy2(fr.image_src, dst_img)

            entry: Dict[str, Any] = {
                "index": idx,
                "name": fr.name,
                "image": str(dst_img.relative_to(root)),
                "depth": None,
                "K": fr.K.tolist(),
                "c2w": fr.c2w.tolist(),
            }
            if fr.depth is not None:
                dep = self._write_depth(fr.depth, root / "depths" / stem)
                entry["depth"] = str(dep.relative_to(root))
                if fr.depth_scale != 1.0:
                    entry["depth_scale"] = fr.depth_scale
            if fr.confidence is not None:
                conf = root / "depths" / f"{stem}.conf.npy"
                np.save(conf, np.asarray(fr.confidence, dtype=np.float32))
                entry["confidence"] = str(conf.relative_to(root))
            entries.append({k: entry[k] for k in _FRAME_KEYS if k in entry} | {
                k: v for k, v in entry.items() if k not in _FRAME_KEYS
            })

        manifest = {
            "format": FORMAT_ID,
            "source": self.source,
            "image_grid": list(self.image_grid),
            "depth_grid": list(self.depth_grid) if self.depth_grid else None,
            "camera_convention": self.camera_convention,
            "pose_convention": self.pose_convention,
            "depth_kind": self.depth_kind,
            "gauge": self.gauge,
            "n_frames": len(entries),
            "frames": entries,
        }
        if self.notes:
            manifest["notes"] = list(self.notes)
        out = root / "frames.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return out


# --------------------------------------------------------------------------
# Bundle reader
# --------------------------------------------------------------------------
@dataclass
class LoadedBundle:
    root: Path
    manifest: Dict[str, Any]

    #: ``(W, H)`` of the images and of the depth maps.
    image_grid: Tuple[int, int]
    depth_grid: Optional[Tuple[int, int]]

    names: List[str]
    image_paths: List[Path]
    depth_paths: List[Optional[Path]]
    Ks: np.ndarray              # [N,3,3], defined on ``depth_grid``
    cam_to_world: np.ndarray    # [N,4,4]
    depth_scales: np.ndarray    # [N] per-frame multiplier (1.0 when unset)

    @property
    def has_depth(self) -> bool:
        return self.depth_grid is not None and any(p is not None for p in self.depth_paths)

    def load_depth(self, i: int) -> Optional[np.ndarray]:
        p = self.depth_paths[i]
        return None if p is None else load_depth_any(p)


def read_bundle(path: str | Path) -> LoadedBundle:
    """Parse a ``surfel-feedforward/1`` bundle and validate its invariants."""
    root = Path(path)
    if (root / "frames.json").exists():
        mpath = root / "frames.json"
    else:  # accept a bundle nested one level down (e.g. <run>/bundle/)
        cands = sorted(root.glob("*/frames.json"))
        if not cands:
            raise FileNotFoundError(f"no frames.json under {root}")
        mpath = cands[0]
        root = mpath.parent

    with open(mpath, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    fmt = manifest.get("format")
    if fmt != FORMAT_ID:
        raise ValueError(
            f"{mpath}: format is {fmt!r}, expected {FORMAT_ID!r}. Re-export the "
            "bundle with surfel.data.feedforward instead of reading it directly."
        )

    frames = manifest.get("frames") or []
    if not frames:
        raise ValueError(f"{mpath}: bundle has no frames")

    names, imgs, deps, Ks, c2ws, scales = [], [], [], [], [], []
    for fr in frames:
        names.append(str(fr["name"]))
        imgs.append(root / fr["image"])
        deps.append(root / fr["depth"] if fr.get("depth") else None)
        K = np.asarray(fr["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"{mpath}: frame {fr.get('index')} K is {K.shape}, want 3x3")
        Ks.append(K)
        c2w = np.asarray(fr["c2w"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"{mpath}: frame {fr.get('index')} c2w is {c2w.shape}, want 4x4")
        c2ws.append(c2w)
        scales.append(float(fr.get("depth_scale", 1.0)))
        if not imgs[-1].exists():
            raise FileNotFoundError(f"{mpath}: frame image missing: {imgs[-1]}")

    ig = manifest.get("image_grid")
    dg = manifest.get("depth_grid")
    return LoadedBundle(
        root=root,
        manifest=manifest,
        image_grid=(int(ig[0]), int(ig[1])) if ig else (0, 0),
        depth_grid=(int(dg[0]), int(dg[1])) if dg else None,
        names=names,
        image_paths=imgs,
        depth_paths=deps,
        Ks=np.stack(Ks, axis=0),
        cam_to_world=np.stack(c2ws, axis=0),
        depth_scales=np.asarray(scales, dtype=np.float64),
    )


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------
def export_scal3r(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    depth_dir: Optional[str | Path] = None,
    depth_suffix: Optional[str] = None,
    depth_format: str = "exr",
    notes: Optional[Sequence[str]] = None,
) -> Path:
    """Convert a Scal3R run directory into a bundle.

    Reads ``intri.yml`` / ``extri.yml`` (never ``mat.txt``: its rotation block
    carries a ``cbrt(det)`` factor, so it is not rigid) and, when present, the
    depth maps. Depth is optional: a poses-only bundle is valid and lets a run
    use Scal3R poses with SfM points.

    Scal3R's ``K`` describes the *depth* grid, which is smaller than the images
    (garden: 518x322 vs 1297x840). That is recorded, not corrected -- DGI must
    unproject with this K.

    ``depth_suffix`` is auto-detected when left as ``None``: predictors differ
    in container (Scal3R writes ``.exr``, a converted run may hold ``.npy``), and
    guessing it wrong would silently produce a poses-only bundle.
    """
    root = Path(run_dir)
    # Scal3R may nest the camera files one level down.
    if not (root / "intri.yml").exists():
        for cand in (root / "scal3r", root / "s3r"):
            if (cand / "intri.yml").exists():
                root = cand
                break
        else:
            raise FileNotFoundError(f"no intri.yml under {run_dir}")

    intri = read_opencv_yaml(root / "intri.yml")
    extri = read_opencv_yaml(root / "extri.yml")
    names = _yaml_names(intri)
    if _yaml_names(extri) != names:
        raise ValueError("intri.yml and extri.yml list different frames")

    img_dir = Path(image_dir) if image_dir else (root / "images")
    dep_dir = Path(depth_dir) if depth_dir else (root / "depths")

    # Resolve the container before walking frames. Auto-detection prefers an
    # explicit suffix, then whatever files are actually present; a wrong guess
    # would yield a poses-only bundle with no error.
    resolved_suffix = depth_suffix
    if resolved_suffix is None and dep_dir.is_dir():
        for ext in (".exr", ".npy"):
            if any(p.suffix.lower() == ext for p in dep_dir.iterdir()):
                resolved_suffix = ext
                break
    if resolved_suffix is None:
        resolved_suffix = ".exr"  # nothing to read; value is irrelevant

    # Frame-name resolution: Scal3R uses zero-padded indices, the source images
    # keep their original names. A wrong mapping gives 0 matched frames with no
    # error, so the mapping is applied to `name` and not only to the path.
    name_map: Dict[str, str] = {}
    for cand in (
        root.parent / "name_map.json",
        root / "name_map.json",
        root / "runtime" / "name_map.json",
    ):
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as fh:
                name_map = {str(k): str(v) for k, v in json.load(fh).items()}
            break
    if not name_map:
        for cand in (root / "runtime" / "image_manifest.txt", root.parent / "runtime" / "image_manifest.txt"):
            if cand.exists():
                lines = [l.strip() for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
                name_map = {f"{i:06d}": Path(l).name for i, l in enumerate(lines)}
                break

    def _img_for(tag: str, original: str) -> Optional[Path]:
        for stem in (original, tag):
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                c = img_dir / f"{Path(stem).stem}{ext}"
                if c.exists():
                    return c
        return None

    writer: Optional[BundleWriter] = None
    kept, skipped = 0, 0
    for tag in names:
        original = name_map.get(tag, tag)
        img = _img_for(tag, original)
        if img is None:
            skipped += 1
            continue

        # Build c2w from the rigid w2c pair.
        Rot = intri.get(f"Rot_{tag}", extri.get(f"Rot_{tag}"))
        T = intri.get(f"T_{tag}", extri.get(f"T_{tag}"))
        if Rot is None or T is None:
            skipped += 1
            continue
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = np.asarray(Rot, dtype=np.float64).reshape(3, 3)
        w2c[:3, 3] = np.asarray(T, dtype=np.float64).ravel()[:3]
        c2w = np.linalg.inv(w2c)

        K = np.asarray(intri[f"K_{tag}"], dtype=np.float64).reshape(3, 3)

        depth = None
        if dep_dir.exists():
            dp = dep_dir / f"{tag}{resolved_suffix}"
            if dp.exists():
                try:
                    depth = load_depth_any(dp)
                except Exception:
                    depth = None

        if writer is None:
            from PIL import Image

            with Image.open(img) as im:
                iw, ih = im.size
            writer = BundleWriter(
                root=Path(out_dir),
                source="scal3r",
                image_grid=(int(iw), int(ih)),
                depth_format=depth_format,
                gauge={
                    "kind": "similarity",
                    "scale": 1.0,
                    "note": (
                        "Scal3R's own arbitrary gauge; poses and depth are mutually "
                        "consistent but not metric. Use gauge.mode=normalize "
                        "downstream, or fit a Sim(3) against a metric reference."
                    ),
                },
                notes=list(notes or []) + [
                    "K is defined on the depth grid; DGI must unproject with it",
                    "poses come from extri.yml (rigid w2c); mat.txt was not used",
                ],
            )
        writer.add(
            name=original if original != tag else tag,
            image_src=img,
            K=K,
            c2w=c2w,
            depth=depth,
        )
        kept += 1

    if writer is None:
        raise RuntimeError(
            f"no frames could be assembled from {run_dir}: {skipped} were skipped "
            "(missing image or pose). Check image_dir."
        )
    if skipped:
        writer.notes.append(f"{skipped} frame(s) skipped for a missing image or pose")
    return writer.write()


def export_vggt(
    npz_path: str | Path,
    out_dir: str | Path,
    image_dir: str | Path,
    *,
    depth_scale: float = 1.0,
    confidence_key: str = "depth_conf",
    depth_format: str = "exr",
    notes: Optional[Sequence[str]] = None,
) -> Path:
    """Convert a VGGT ``.npz`` prediction dump into a bundle.

    Expected arrays (the layout ``demo_colmap.py`` produces)::

        extrinsic  [N,3,4]  world-to-camera, OpenCV
        intrinsic  [N,3,3]  pixels of the *prediction* grid
        depth      [N,H,W]  z-depth in VGGT's normalised gauge
        depth_conf [N,H,W]  optional

    ``depth_scale`` multiplies depth into the gauge of ``extrinsic``. VGGT's
    depth is only metric after its own per-frame alignment; passing the global
    factor here keeps that act explicit instead of hidden in a caller.
    """
    data = np.load(Path(npz_path), allow_pickle=False)
    ext = np.asarray(data["extrinsic"], dtype=np.float64)
    ixt = np.asarray(data["intrinsic"], dtype=np.float64)
    depth = np.asarray(data["depth"], dtype=np.float32) if "depth" in data else None
    conf = np.asarray(data[confidence_key]) if confidence_key in data else None
    if ext.ndim != 3 or ext.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsic must be [N,3,4] or [N,4,4], got {ext.shape}")
    if ixt.shape[1:] != (3, 3):
        raise ValueError(f"intrinsic must be [N,3,3], got {ixt.shape}")

    imgs = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    n = ext.shape[0]
    if len(imgs) < n:
        raise ValueError(f"{len(imgs)} images for {n} predictions in {npz_path}")

    from PIL import Image

    with Image.open(imgs[0]) as im:
        iw, ih = im.size

    writer = BundleWriter(
        root=Path(out_dir),
        source="vggt",
        image_grid=(int(iw), int(ih)),
        depth_format=depth_format,
        gauge={
            "kind": "normalized_depth",
            "scale": float(depth_scale),
            "note": (
                "VGGT depth is normalised inverse depth; depth_scale maps it into "
                "the extrinsic gauge. Without a per-frame fit this is a global "
                "approximation -- downstream gauge.mode=normalize absorbs it."
            ),
        },
        notes=list(notes or []),
    )

    for i in range(n):
        # extrinsic is world-to-camera: [3,4] (OpenCV) or [4,4] already.
        if ext.shape[1] == 4:
            w2c = np.array(ext[i], dtype=np.float64, copy=True)
        else:
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = ext[i]
        c2w = np.linalg.inv(w2c)
        d = None if depth is None else depth[i] * float(depth_scale)
        writer.add(
            name=imgs[i].name,
            image_src=imgs[i],
            K=ixt[i],
            c2w=c2w,
            depth=d,
            confidence=None if conf is None else conf[i],
        )
    return writer.write()


__all__ = [
    "FORMAT_ID",
    "BundleFrame",
    "BundleWriter",
    "LoadedBundle",
    "read_bundle",
    "read_opencv_yaml",
    "save_depth_npy",
    "load_depth_any",
    "export_scal3r",
    "export_vggt",
]
