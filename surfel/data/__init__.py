"""Data layer: COLMAP parsing, resolution policy, depth handling, gauge."""

from .colmap_io import (
    ColmapCamera,
    ColmapImage,
    ColmapModel,
    qvec_to_rotmat,
    read_colmap_model,
    rotmat_to_qvec,
    write_colmap_text,
)
from .dataset import CameraBatch, SceneDataset, collate
from .depth import (
    DepthCalibration,
    depth_to_normal,
    find_depth_file,
    load_depth_scales,
    needs_ray_remap,
    read_depth,
    remap_depth,
    resize_depth_nearest,
    scale_K_2d,
)
from .dgi import DgiConfig, build_dgi_cloud, filter_cloud_by_extent, unproject_depth
from .feedforward import (
    BundleWriter,
    LoadedBundle,
    export_scal3r,
    export_vggt,
    load_depth_any,
    read_bundle,
    read_opencv_yaml,
    save_depth_npy,
)
from .gauge import (
    Gauge,
    GaugeApplication,
    GaugeMismatchError,
    GaugeSource,
    Sim3,
    apply_gauge,
    camera_extent,
    check_compatible,
    check_pose_point_consistency,
    depth_quantiles,
    derive_scale,
    derive_scene_scale,
    estimate_sim3,
    needs_alignment,
    projected_footprint_px,
    sim3_residual,
    trajectory_span,
)
from .parser import (
    DatasetError,
    ParseReport,
    align_scal3r_to_colmap,
    align_to_reference,
    parse_colmap,
    parse_dataset,
    parse_feedforward,
)
from .resolution import (
    ResolutionPolicy,
    ResolvedResolution,
    probe_image_size,
    resolve_image_dir,
)
from .scene import Camera, Scene, Split, scene_subset

__all__ = [
    # colmap
    "ColmapCamera",
    "ColmapImage",
    "ColmapModel",
    "read_colmap_model",
    "write_colmap_text",
    "qvec_to_rotmat",
    "rotmat_to_qvec",
    # dataset
    "SceneDataset",
    "CameraBatch",
    "collate",
    # depth
    "DepthCalibration",
    "read_depth",
    "find_depth_file",
    "load_depth_scales",
    "remap_depth",
    "resize_depth_nearest",
    "needs_ray_remap",
    "scale_K_2d",
    "depth_to_normal",
    # dgi
    "DgiConfig",
    "unproject_depth",
    "build_dgi_cloud",
    "filter_cloud_by_extent",
    # gauge
    "Gauge",
    "GaugeSource",
    "GaugeApplication",
    "GaugeMismatchError",
    "Sim3",
    "apply_gauge",
    "estimate_sim3",
    "sim3_residual",
    "trajectory_span",
    "derive_scale",
    "derive_scene_scale",
    "camera_extent",
    "depth_quantiles",
    "needs_alignment",
    "check_compatible",
    "check_pose_point_consistency",
    "projected_footprint_px",
    # parser
    "parse_colmap",
    "parse_feedforward",
    "parse_dataset",
    "ParseReport",
    "DatasetError",
    "align_to_reference",
    "align_scal3r_to_colmap",
    # feedforward bundle
    "BundleWriter",
    "LoadedBundle",
    "read_bundle",
    "export_scal3r",
    "export_vggt",
    "save_depth_npy",
    "load_depth_any",
    "read_opencv_yaml",
    # resolution
    "ResolutionPolicy",
    "ResolvedResolution",
    "probe_image_size",
    "resolve_image_dir",
    # scene
    "Camera",
    "Scene",
    "Split",
    "scene_subset",
]
