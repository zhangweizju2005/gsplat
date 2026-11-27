"""Tests for the feed-forward bundle format and its adapters.

These run without CUDA and without OpenEXR: the whole point of the bundle is
that a feed-forward front end can hand off depth and poses through plain files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from surfel.data.feedforward import (
    FORMAT_ID,
    BundleWriter,
    export_scal3r,
    load_depth_any,
    read_bundle,
    read_opencv_yaml,
    save_depth_npy,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _write_opencv_yaml(path: Path, names, Ks, Rots, Ts, hw) -> None:
    """Minimal OpenCV FileStorage YAML, matching what Scal3R writes."""
    h, w = hw
    lines = ["%YAML:1.0", "---", "names:"]
    lines += [f'  - "{n}"' for n in names]
    for n, K in zip(names, Ks):
        lines += [
            f"K_{n}: !!opencv-matrix",
            "  rows: 3",
            "  cols: 3",
            "  dt: d",
            "  data: [" + ", ".join(f"{v:.10f}" for v in K.ravel()) + "]",
            f"H_{n}: {float(h):.10f}",
            f"W_{n}: {float(w):.10f}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_extri(path: Path, names, Rots, Ts) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    lines += [f'  - "{n}"' for n in names]
    for n, R, T in zip(names, Rots, Ts):
        lines += [
            f"Rot_{n}: !!opencv-matrix",
            "  rows: 3",
            "  cols: 3",
            "  dt: d",
            "  data: [" + ", ".join(f"{v:.10f}" for v in R.ravel()) + "]",
            f"T_{n}: !!opencv-matrix",
            "  rows: 3",
            "  cols: 1",
            "  dt: d",
            "  data: [" + ", ".join(f"{v:.10f}" for v in T.ravel()) + "]",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_image(path: Path, size=(160, 120)) -> None:
    """Write a tiny valid JPEG so image-size probing works."""
    from PIL import Image

    Image.new("RGB", size, (128, 128, 128)).save(path, format="JPEG")


def _ring_poses(n: int, radius: float = 2.5) -> np.ndarray:
    """Camera-to-world poses on a ring looking at the origin."""
    out = []
    for i in range(n):
        th = 2 * np.pi * i / n
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(
            [[np.cos(th), 0.0, np.sin(th)], [0.0, 1.0, 0.0], [-np.sin(th), 0.0, np.cos(th)]]
        )
        c2w[:3, 3] = [radius * np.cos(th), 0.0, radius * np.sin(th)]
        out.append(c2w)
    return np.stack(out)


@pytest.fixture
def fake_scal3r(tmp_path: Path) -> Path:
    """A Scal3R-shaped run dir with .npy depth (as the patched exporter writes)."""
    run = tmp_path / "scal3r_run"
    (run / "depths").mkdir(parents=True)
    imgs = tmp_path / "imgs"
    imgs.mkdir()

    n = 4
    names = [f"{i:06d}" for i in range(n)]
    K = np.array([[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    Ks = [K.copy() for _ in range(n)]

    c2ws = _ring_poses(n)
    Rots, Ts = [], []
    for c2w in c2ws:
        w2c = np.linalg.inv(c2w)
        Rots.append(w2c[:3, :3])
        Ts.append(w2c[:3, 3])

    _write_opencv_yaml(run / "intri.yml", names, Ks, Rots, Ts, (120, 160))
    _write_extri(run / "extri.yml", names, Rots, Ts)

    for i, name in enumerate(names):
        np.save(run / "depths" / f"{name}.npy", np.full((120, 160), 2.0, np.float32))
        _real_image(imgs / f"{name}.jpg")

    # name_map so Camera.name is the original filename, not the zero-padded tag
    (run / "name_map.json").write_text(
        json.dumps({n: f"{n}.jpg" for n in names}), encoding="utf-8"
    )
    return run


# --------------------------------------------------------------------------
# YAML reader
# --------------------------------------------------------------------------
def test_read_opencv_yaml_parses_matrix_and_scalars(tmp_path: Path):
    p = tmp_path / "intri.yml"
    _write_opencv_yaml(
        p, ["000000"], [np.eye(3)], [np.eye(3)], [np.zeros(3)], (322, 518)
    )
    d = read_opencv_yaml(p)
    assert d["__list__"] == ["000000"]
    assert d["K_000000"].shape == (3, 3)
    np.testing.assert_allclose(d["K_000000"], np.eye(3))
    assert float(d["H_000000"]) == pytest.approx(322.0)
    assert float(d["W_000000"]) == pytest.approx(518.0)


def test_read_opencv_yaml_handles_multiple_frames(tmp_path: Path):
    names = ["000000", "000001"]
    Ks = [np.eye(3), np.eye(3) * 2]
    p = tmp_path / "intri.yml"
    _write_opencv_yaml(p, names, Ks, [np.eye(3)] * 2, [np.zeros(3)] * 2, (10, 20))
    d = read_opencv_yaml(p)
    np.testing.assert_allclose(d["K_000001"], np.eye(3) * 2)


# --------------------------------------------------------------------------
# Depth I/O
# --------------------------------------------------------------------------
def test_depth_npy_roundtrip_is_exact(tmp_path: Path):
    rng = np.random.default_rng(0)
    depth = rng.random((24, 32)).astype(np.float32) * 5.0
    p = save_depth_npy(depth, tmp_path / "d.npy")
    assert p.suffix == ".npy"
    np.testing.assert_array_equal(load_depth_any(p), depth)


def test_depth_npy_squeezes_single_channel(tmp_path: Path):
    d = np.full((8, 8, 1), 3.0, np.float32)
    p = save_depth_npy(d, tmp_path / "d.npy")
    assert load_depth_any(p).shape == (8, 8)


def test_depth_rejects_non_2d(tmp_path: Path):
    with pytest.raises(ValueError):
        save_depth_npy(np.zeros((4, 4, 3), np.float32), tmp_path / "d.npy")


# --------------------------------------------------------------------------
# Bundle writer / reader
# --------------------------------------------------------------------------
def test_bundle_roundtrip_preserves_grids_and_poses(tmp_path: Path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    c2w = _ring_poses(1)[0]
    K = np.array([[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]])
    w = BundleWriter(root=tmp_path / "b", source="test", image_grid=(40, 30))
    w.add(name="a.jpg", image_src=img, K=K, c2w=c2w, depth=np.ones((20, 26), np.float32))
    w.write()

    b = read_bundle(tmp_path / "b")
    assert b.manifest["format"] == FORMAT_ID
    assert b.image_grid == (40, 30)
    assert b.depth_grid == (26, 20)  # (W, H)
    np.testing.assert_allclose(b.Ks[0], K)
    np.testing.assert_allclose(b.cam_to_world[0], c2w, atol=1e-12)
    assert b.has_depth


def test_bundle_rejects_non_rigid_pose(tmp_path: Path):
    """The mat.txt signature: rotation scaled by cbrt(det). Must be refused."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    c2w = np.eye(4)
    c2w[:3, :3] *= 2.0  # det = 8, not a rotation
    w = BundleWriter(root=tmp_path / "b", source="test", image_grid=(8, 8))
    with pytest.raises(ValueError, match="orthonormal"):
        w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=c2w)


def test_bundle_rejects_mixed_depth_resolutions(tmp_path: Path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    c2w = _ring_poses(1)[0]
    w = BundleWriter(root=tmp_path / "b", source="test", image_grid=(8, 8))
    w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=c2w, depth=np.ones((4, 4)))
    with pytest.raises(ValueError, match="depth grid"):
        w.add(name="b.jpg", image_src=img, K=np.eye(3), c2w=c2w, depth=np.ones((8, 8)))


def test_read_bundle_rejects_foreign_format(tmp_path: Path):
    (tmp_path / "frames.json").write_text(
        json.dumps({"format": "something-else/9", "frames": [{"name": "x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="format"):
        read_bundle(tmp_path)


def test_read_bundle_rejects_empty(tmp_path: Path):
    (tmp_path / "frames.json").write_text(
        json.dumps({"format": FORMAT_ID, "frames": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no frames"):
        read_bundle(tmp_path)


def test_read_bundle_finds_nested_bundle(tmp_path: Path):
    """A bundle one level down (e.g. <run>/bundle/) is still found."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    inner = tmp_path / "nested" / "bundle"
    w = BundleWriter(root=inner, source="test", image_grid=(8, 8))
    w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=_ring_poses(1)[0])
    w.write()
    b = read_bundle(tmp_path / "nested")
    assert len(b.names) == 1


# --------------------------------------------------------------------------
# Scal3R adapter
# --------------------------------------------------------------------------
def test_export_scal3r_produces_valid_bundle(tmp_path: Path, fake_scal3r: Path):
    out = export_scal3r(fake_scal3r, tmp_path / "bundle", image_dir=tmp_path / "imgs",
                        depth_suffix=".npy")
    b = read_bundle(out.parent)
    assert len(b.names) == 4
    assert b.depth_grid == (160, 120)
    assert b.image_grid == (160, 120)
    # names come from name_map, not Scal3R's zero-padded tags
    assert b.names[0] == "000000.jpg"
    # poses recovered from the rigid w2c pair
    R = b.cam_to_world[0, :3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-6
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert b.load_depth(0).shape == (120, 160)


def test_export_scal3r_without_depth_is_poses_only(tmp_path: Path, fake_scal3r: Path):
    import shutil

    shutil.rmtree(fake_scal3r / "depths")
    out = export_scal3r(fake_scal3r, tmp_path / "bundle", image_dir=tmp_path / "imgs")
    b = read_bundle(out.parent)
    assert not b.has_depth
    assert len(b.names) == 4


def test_export_scal3r_missing_images_raises(tmp_path: Path, fake_scal3r: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no frames"):
        export_scal3r(fake_scal3r, tmp_path / "bundle", image_dir=empty)


# --------------------------------------------------------------------------
# VGGT: deliberately unsupported
# --------------------------------------------------------------------------
def test_vggt_path_refuses_and_explains_why(tmp_path: Path):
    """There is no VGGT adapter, and asking for one must say so clearly.

    An adapter used to exist and was removed: it expected an ``.npz`` with raw
    ``depth``, which VGGT-X never writes. What ``demo_colmap.py`` actually
    leaves behind is ``1/(depth+eps)`` rescaled to ``[0, 1]`` per frame with the
    bounds discarded, so the absolute scale is unrecoverable. Building a bundle
    from those files would have supervised an arbitrarily-scaled quantity while
    every log line looked normal, which is the failure mode this project treats
    as its most expensive class -- worse than an outright refusal.

    The old test only ever fed a hand-built npz to the adapter, so it validated
    the adapter against its own assumption rather than against the front end.
    """
    from surfel.data import feedforward_run

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _real_image(imgs / "0.jpg", size=(16, 12))

    with pytest.raises(NotImplementedError, match="not supported"):
        feedforward_run.run_vggt(imgs, tmp_path / "pred.npz")

    # The message has to name the actual cause and the way forward.
    try:
        feedforward_run.run_vggt(imgs, tmp_path / "pred.npz")
    except NotImplementedError as exc:
        text = str(exc).lower()
        assert "inverse depth" in text
        assert "scal3r" in text


def test_no_export_vggt_symbol_is_exported():
    """A removed adapter must be gone from the public surface, not just unused."""
    import surfel.data as data_pkg

    assert not hasattr(data_pkg, "export_vggt")
    assert "export_vggt" not in getattr(data_pkg, "__all__", [])


__all__ = []


# --------------------------------------------------------------------------
# Depth container selection (exr default, npy fallback)
# --------------------------------------------------------------------------
def _exr_ok() -> bool:
    from surfel.data.feedforward_run import exr_codec_available

    return exr_codec_available()


@pytest.mark.skipif(not _exr_ok(), reason="OpenCV build has no OpenEXR codec")
def test_bundle_writes_exr_by_default(tmp_path: Path):
    img = tmp_path / "a.jpg"
    _real_image(img, size=(16, 12))
    w = BundleWriter(root=tmp_path / "b", source="test", image_grid=(16, 12))
    w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=_ring_poses(1)[0],
          depth=np.full((12, 16), 2.0, np.float32))
    w.write()
    files = list((tmp_path / "b" / "depths").iterdir())
    assert [f.suffix for f in files] == [".exr"]
    b = read_bundle(tmp_path / "b")
    np.testing.assert_allclose(b.load_depth(0), 2.0)


def test_bundle_writes_npy_when_requested(tmp_path: Path):
    img = tmp_path / "a.jpg"
    _real_image(img, size=(16, 12))
    w = BundleWriter(root=tmp_path / "b", source="test", image_grid=(16, 12),
                     depth_format="npy")
    w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=_ring_poses(1)[0],
          depth=np.full((12, 16), 3.0, np.float32))
    w.write()
    assert [f.suffix for f in (tmp_path / "b" / "depths").iterdir()] == [".npy"]
    np.testing.assert_allclose(read_bundle(tmp_path / "b").load_depth(0), 3.0)


@pytest.mark.skipif(not _exr_ok(), reason="needs OpenEXR for the comparison")
def test_exr_and_npy_bundles_are_numerically_identical(tmp_path: Path):
    """Container choice must not change values: both are lossless float32."""
    img = tmp_path / "a.jpg"
    _real_image(img, size=(16, 12))
    rng = np.random.default_rng(7)
    depth = (rng.random((12, 16)) * 5).astype(np.float32)

    for fmt, name in (("exr", "be"), ("npy", "bn")):
        w = BundleWriter(root=tmp_path / name, source="t", image_grid=(16, 12),
                         depth_format=fmt)
        w.add(name="a.jpg", image_src=img, K=np.eye(3), c2w=_ring_poses(1)[0],
              depth=depth)
        w.write()

    a = read_bundle(tmp_path / "be").load_depth(0)
    b = read_bundle(tmp_path / "bn").load_depth(0)
    np.testing.assert_array_equal(a, b)


def test_export_scal3r_autodetects_depth_suffix(tmp_path: Path, fake_scal3r: Path):
    """An npy-only run exports without the caller naming the container."""
    out = export_scal3r(fake_scal3r, tmp_path / "bundle", image_dir=tmp_path / "imgs")
    b = read_bundle(out.parent)
    assert b.has_depth
    assert len(b.names) == 4


def test_convert_depths_exr_to_npy(tmp_path: Path):
    from surfel.data.feedforward_run import convert_depths, pick_depth_suffix

    d = tmp_path / "depths"
    d.mkdir()
    for i in range(3):
        save_depth_npy(np.full((4, 4), float(i), np.float32), d / f"{i:06d}.npy")
    assert pick_depth_suffix(d) == ".npy"

    n = convert_depths(d, "npy")
    # already npy: nothing to convert, nothing destroyed
    assert n == 0
    assert pick_depth_suffix(d) == ".npy"


def test_pick_depth_suffix_rejects_empty(tmp_path: Path):
    from surfel.data.feedforward_run import pick_depth_suffix

    d = tmp_path / "depths"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        pick_depth_suffix(d)


# --------------------------------------------------------------------------
# Parser-side handling of every schema-legal init/gauge value
# --------------------------------------------------------------------------
def _minimal_bundle(root: Path, *, with_depth: bool = True) -> Path:
    """A three-frame bundle with 4x4 depth, enough for parse_feedforward."""
    import imageio.v2 as imageio

    out = root / "bundle"
    (out / "images").mkdir(parents=True, exist_ok=True)
    w = BundleWriter(
        root=out,
        source="test",
        image_grid=(16, 12),
        depth_grid=(8, 6) if with_depth else None,
        depth_format="npy",
    )
    K = np.array([[12.0, 0.0, 8.0], [0.0, 12.0, 6.0], [0.0, 0.0, 1.0]])
    for i in range(3):
        img = np.full((12, 16, 3), 30 + i, np.uint8)
        p = root / f"img_{i}.png"
        imageio.imwrite(p, img)
        c2w = np.eye(4)
        c2w[0, 3] = float(i)
        w.add(
            name=f"{i:06d}.png",
            image_src=p,
            K=K,
            c2w=c2w,
            depth=(np.full((6, 8), 2.0, np.float32) if with_depth else None),
        )
    w.write()
    return out


def _bundle_cfg(bundle: Path, out_dir: Path, **overrides: str):
    from surfel.config import build_config

    base = [
        f"data.parser=feedforward",
        f"data.path={bundle}",
        "data.init_points=depth",
        "data.split_mode=reconstruction",
        "gauge.scale_source=extent",
        "gauge.mode=normalize",
        "train.lr.means_per_spacing=0.01",
        "model.sh_degree=0",
        f"train.out_dir={out_dir}",
    ]
    base.extend(f"{k}={v}" for k, v in overrides.items())
    return build_config(overrides=base)


def test_gauge_mode_none_parses_on_the_bundle_path(tmp_path: Path):
    """``gauge.mode="none"`` must produce a report, not a TypeError.

    The schema lists "none" as legal -- it means "leave the bundle in its own
    arbitrary gauge" -- but the report builder called
    `GaugeApplication(source=...)` and raised
    `TypeError: missing 4 required positional arguments` for it. The parser test
    surface only covered the COLMAP path, where the branch is not reached.
    """
    from surfel.data import parse_dataset

    bundle = _minimal_bundle(tmp_path)
    cfg = _bundle_cfg(bundle, tmp_path / "out", **{"gauge.mode": "none"})
    cfg.validate()

    split, report = parse_dataset(cfg)
    assert report.gauge is not None
    assert report.gauge.source == "none"
    assert report.gauge.scale_only == 1.0
    assert report.gauge.applied_to_depth is False
    # No normalisation happened, so poses keep the bundle's own scale.
    assert report.n_points > 0


def test_init_points_none_is_refused_at_config_time(tmp_path: Path):
    """An empty start must be refused early, with the real reason.

    ``init_points="none"`` parses to an empty cloud, which then died deep inside
    the trainer on a kNN spacing measured over zero points
    (`measured kNN spacing is nan`) -- a message that points at the learning rate
    rather than at the impossible configuration. The deeper reason is that gsplat
    cannot grow a model from zero Gaussians at all: DefaultStrategy needs
    existing candidates to split or duplicate, and MCMC's
    `n_target = min(cap_max, 1.05 * n)` stays 0 when n = 0. It is now rejected by
    ``Config.validate`` so no data source can reach the rasteriser with it.
    """
    from surfel.config import build_config

    # ``build_config`` validates as it assembles, so the refusal happens there
    # rather than at a later explicit ``cfg.validate()``.
    with pytest.raises(ValueError, match="not trainable"):
        build_config(
            overrides=[
                f"data.path={tmp_path / 'bundle'}",
                "data.parser=feedforward",
                "data.init_points=none",
                "gauge.scale_source=extent",
            ]
        )


def test_init_points_none_message_does_not_recommend_itself():
    """The parser's own error text must not suggest an untrainable fallback.

    A missing-depth bundle used to say "Re-export with depths, or set
    data.init_points=none", i.e. it pointed at a configuration that cannot
    train. Same for the sparse-points branch. Both now name only working
    alternatives.
    """
    import inspect

    from surfel.data import parser as parser_mod

    src = inspect.getsource(parser_mod.parse_feedforward)
    assert "init_points=none" not in src, (
        "parse_feedforward still recommends init_points='none', which cannot "
        "train"
    )


# --------------------------------------------------------------------------
# Bundle path must honour the same resolution/config knobs as COLMAP
# --------------------------------------------------------------------------
def test_bundle_path_honours_data_factor(tmp_path: Path):
    """``data.factor`` used to be silently ignored on bundle input.

    The bundle path only read ``target_long_edge``, so a run asking for a 2x
    downscale silently trained at full resolution -- a config that does not
    describe the run, which is the class of error this project treats as most
    expensive (a resolution/K mismatch once cost 12 dB, silently). Both paths now
    go through ``ResolutionPolicy``.
    """
    from surfel.data import parse_dataset

    bundle = _minimal_bundle(tmp_path)
    base = _bundle_cfg(bundle, tmp_path / "a")
    split_a, report_a = parse_dataset(base)
    cam_a = split_a.train.cameras[0]

    halved = _bundle_cfg(bundle, tmp_path / "b", **{"data.factor": "2"})
    split_b, report_b = parse_dataset(halved)
    cam_b = split_b.train.cameras[0]

    assert (cam_a.width, cam_a.height) == (16, 12)
    assert (cam_b.width, cam_b.height) == (8, 6), (
        f"data.factor=2 produced {cam_b.width}x{cam_b.height}, not half of 16x12"
    )
    # K must describe the grid actually rendered, not the source grid.
    assert abs(cam_b.K[0, 0] - cam_a.K[0, 0] / 2.0) < 1e-6


def test_bundle_path_honours_target_long_edge_over_factor(tmp_path: Path):
    """``target_long_edge`` overrides ``factor``, as it does on the COLMAP path."""
    from surfel.data import parse_dataset

    bundle = _minimal_bundle(tmp_path)
    cfg = _bundle_cfg(
        bundle,
        tmp_path / "out",
        **{"data.factor": "2", "data.target_long_edge": "8"},
    )
    split, _ = parse_dataset(cfg)
    cam = split.train.cameras[0]
    assert max(cam.width, cam.height) == 8


def test_bundle_path_refuses_image_dir_instead_of_ignoring_it(tmp_path: Path):
    """A bundle pins image paths, so ``data.image_dir`` cannot be honoured.

    Silently ignoring it made a run whose config said one thing and whose data
    came from another. The project's rule is to refuse rather than guess.
    """
    from surfel.data import parse_dataset

    bundle = _minimal_bundle(tmp_path)
    cfg = _bundle_cfg(bundle, tmp_path / "out", **{"data.image_dir": "/nope"})
    with pytest.raises(Exception, match="not used by a feed-forward bundle"):
        parse_dataset(cfg)


def test_scal3r_patterns_cover_uppercase_extensions(tmp_path: Path):
    """Mip-NeRF 360 ships ``*.JPG``; Scal3R's default patterns are lowercase.

    ``collect_image_paths`` globs the configured patterns from the config file,
    and ``scal3r.run`` exposes no flag for them, so a folder of uppercase
    extensions failed with "No images found" while every file was present. The
    wrapper generates a derived config that inherits the upstream one and adds
    the extensions actually present -- Scal3R's source stays untouched.
    """
    from surfel.data.feedforward_run import _derived_config_for_extensions

    imgs = tmp_path / "images"
    imgs.mkdir()
    for i in range(3):
        _real_image(imgs / f"{i:03d}.JPG", size=(16, 12))

    root = tmp_path / "fake_scal3r"
    (root / "configs" / "models").mkdir(parents=True)
    (root / "configs" / "models" / "scal3r.yaml").write_text("model: {}\n")

    derived = _derived_config_for_extensions(root, imgs, tmp_path / "out")
    assert derived is not None, "uppercase extensions were not detected"
    text = derived.config_path.read_text(encoding="utf-8")
    assert "*.JPG" in text
    # Quoted, because a leading '*' is YAML alias syntax and would not parse.
    assert "image_patterns: '" in text


def test_scal3r_patterns_are_left_alone_for_lowercase_images(tmp_path: Path):
    """The common case must not generate a config at all."""
    from surfel.data.feedforward_run import _derived_config_for_extensions

    imgs = tmp_path / "images"
    imgs.mkdir()
    for i in range(2):
        _real_image(imgs / f"{i:03d}.jpg", size=(16, 12))

    root = tmp_path / "fake_scal3r"
    (root / "configs" / "models").mkdir(parents=True)
    (root / "configs" / "models" / "scal3r.yaml").write_text("model: {}\n")

    assert _derived_config_for_extensions(root, imgs, tmp_path / "out") is None
