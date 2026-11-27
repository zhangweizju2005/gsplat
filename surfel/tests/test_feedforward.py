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
    export_vggt,
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
# VGGT adapter
# --------------------------------------------------------------------------
def test_export_vggt_produces_valid_bundle(tmp_path: Path):
    n, h, w = 3, 12, 16
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    c2ws = _ring_poses(n)
    ext = np.stack([np.linalg.inv(c)[:3, :4] for c in c2ws])
    ixt = np.stack([np.array([[50.0, 0, 8], [0, 50.0, 6], [0, 0, 1]])] * n)
    depth = np.full((n, h, w), 1.5, np.float32)
    conf = np.full((n, h, w), 0.9, np.float32)
    npz = tmp_path / "pred.npz"
    np.savez(npz, extrinsic=ext, intrinsic=ixt, depth=depth, depth_conf=conf)
    for i in range(n):
        _real_image(imgs / f"{i:03d}.jpg", size=(16, 12))

    out = export_vggt(npz, tmp_path / "bundle", imgs, depth_scale=2.0)
    b = read_bundle(out.parent)
    assert len(b.names) == n
    assert b.depth_grid == (w, h)
    np.testing.assert_allclose(b.load_depth(0), 3.0)  # 1.5 * scale
    R = b.cam_to_world[0, :3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-6


def test_export_vggt_rejects_short_image_list(tmp_path: Path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _real_image(imgs / "0.jpg", size=(16, 12))
    npz = tmp_path / "p.npz"
    np.savez(npz, extrinsic=np.stack([np.eye(4)[:3, :4]] * 2),
             intrinsic=np.stack([np.eye(3)] * 2), depth=np.ones((2, 4, 4), np.float32))
    with pytest.raises(ValueError, match="images for"):
        export_vggt(npz, tmp_path / "b", imgs)


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
