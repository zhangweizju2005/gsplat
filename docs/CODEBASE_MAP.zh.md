# gsplat 功能 — 代码说明文档

> 本文档基于 `codebase-memory-mcp` 对 `/home/aic/Github/gsplat` 的完整索引（26,853 节点 / 109,713 边 / 2,169 文件）
> 以及源码逐文件精读整理，用于回答两个问题：**「这个项目能做什么」** 和 **「对应实现在哪」**。
>
> 版本：`gsplat 1.6.0`（`gsplat/version.py:16`） · 分支 `main`

---

## 目录

1. [项目定位与能力全景](#1-项目定位与能力全景)
2. [代码规模与目录结构](#2-代码规模与目录结构)
3. [分层架构与调用链](#3-分层架构与调用链)
4. [构建系统：五个原生扩展](#4-构建系统五个原生扩展) — 含 [源码编译安装要点](#45-源码编译安装要点)
5. [核心渲染入口 `gsplat/rendering.py`](#5-核心渲染入口-gsplatrenderingpy)
6. [CUDA 后端：Python 绑定层](#6-cuda-后端python-绑定层)
7. [CUDA 后端：C++ 编排层](#7-cuda-后端c-编排层)
8. [CUDA 后端：设备端 kernel](#8-cuda-后端设备端-kernel)
9. [训练基础设施：strategy / 优化器 / 损失](#9-训练基础设施strategy--优化器--损失)
10. [场景与阶段抽象：scene / stage](#10-场景与阶段抽象scene--stage)
11. [传感器库 `gsplat/sensors`](#11-传感器库-gsplatsensors)
12. [几何库 `gsplat/geometry`](#12-几何库-gsplatgeometry)
13. [推理渲染（HiGS）：`gsplat/experimental`](#13-推理渲染higsgsplatexperimental)
14. [工具模块：压缩 / 导出 / 分布式 / 性能剖析](#14-工具模块压缩--导出--分布式--性能剖析)
15. [动态高斯（G-SHARP 移植）：`gsplat/contrib/dynamic`](#15-动态高斯g-sharp-移植gsplatcontribdynamic)
16. [示例与训练脚本](#16-示例与训练脚本)
17. [测试体系](#17-测试体系)
18. [关键数据约定](#18-关键数据约定)
19. [扩展指南](#19-扩展指南)
20. [索引数据与热点分析](#20-索引数据与热点分析)
21. [已知限制与注意事项](#21-已知限制与注意事项)

---

## 1. 项目定位与能力全景

gsplat 是一个 **CUDA 加速的可微 3D 高斯光栅化库**，以 Python 包形式发布，灵感来自 SIGGRAPH 2023 的
3D Gaussian Splatting，但在显存、速度与功能上做了大量扩展。

### 1.1 核心能力矩阵

| 能力 | 功能说明 | 主要入口 |
| --- | --- | --- |
| **3DGS 光栅化** | 经典 EWA（椭圆加权平均）投影 + tile 排序 + alpha 合成 | `gsplat.rasterization` (`rendering.py:234`) |
| **2DGS 光栅化** | 面元（surfel）光线-平面求交，输出法线与中值深度 | `gsplat.rasterization_2dgs` (`rendering.py:1358`) |
| **3DGUT** | Unscented Transform 投影 + 3D 空间高斯响应，支持畸变相机与卷帘快门 | `rasterization(with_ut=True, with_eval3d=True)` |
| **任意批维度** | `[..., N, ...]` 高斯 × `[..., C, ...]` 相机一次性渲染 | `rendering.py:234` 的 `...` 语义 |
| **多 GPU 分布式** | 各 rank 持有部分高斯，协作渲染同一批相机 | `rasterization(distributed=True)` |
| **稀疏光栅化** | 只渲染指定像素（active tile），前反向均支持 | `isect_tiles_sparse` / `rasterize_to_pixels_sparse` |
| **高斯 ID 查询** | 逐像素贡献高斯数量 / id 列表 / top-K 权重 | `rasterize_num_contributing_gaussians` 等 |
| **N 维特征渲染** | 任意通道数颜色或特征，支持 SH 系数 | `colors` 为 `[N,K,D]` + `sh_degree` |
| **深度 / 命中距离** | 9 种 `render_mode`（`RGB` / `D` / `ED` / `d` / `Ed` / 组合） | `RenderMode` (`rendering.py:64`) |
| **相机模型** | pinhole / ortho / fisheye / FTheta / **LiDAR** | `CameraModel` (`cuda/_wrapper.py:38`) |
| **相机畸变** | OpenCV radial+ tangential+thin-prism、FTheta 多项式、双变量挡风玻璃外畸变 | `radial_coeffs` / `ftheta_coeffs` / `external_distortion_coeffs` |
| **卷帘快门** | 4 种扫描方向 + Newton 迭代求解 | `rolling_shutter` / `viewmats_rs` |
| **反锯齿** | Mip-Splatting 式不透明度补偿 | `rasterize_mode="antialiased"` |
| **AbsGrad** | AbsGS 式绝对梯度用于致密化判定 | `absgrad=True` + `means2d.absgrad` |
| **稀疏梯度** | means/quats/scales 梯度以 COO 稀疏张量返回 | `sparse_grad=True`（需 `packed=True`） |
| **致密化策略** | 原版 3DGS 复制/分裂/剪枝、3DGS-MCMC 重定位/增殖/扰动 | `DefaultStrategy` / `MCMCStrategy` |
| **压缩** | PNG 量化 + 并行线性分配排序 + KMeans(SH) | `PngCompression` (`compression/`) |
| **导出** | PLY / 压缩 PLY / `.splat`，含 Morton 排序 | `export_splats` (`exporter.py:588`) |
| **传感器库** | 相机射线↔图像点、世界点↔图像、滚动快门、LiDAR 角度/射线 | `gsplat.sensors.functional` |
| **几何库** | 四元数、SE(3)、轨迹插值（CUDA 可微） | `gsplat.geometry.functional` |
| **场景抽象** | `GaussianScene`（训练）/ `GaussianInferenceScene`（推理打包） | `gsplat.scene` |
| **推理渲染（HiGS）** | macro-tile 融合光栅化 + fp16 场景打包，仅前向 | `gsplat.experimental.render_scene` |
| **动态高斯** | HexPlane 4D 特征场 + MLP 形变 + 动态策略 | `gsplat.contrib.dynamic` |
| **性能工具** | NVTX 标注、输入捕获、独立 replay harness | `gsplat.trace` / `gsplat.profile` |

### 1.2 支持论文的落地位置

| 论文/技术 | 代码位置 |
| --- | --- |
| 3DGS (Kerbl et al. 2023) | `strategy/default.py`、`cuda/csrc/ProjectionEWA3DGS*.cu` |
| AbsGS | `rendering.py` `absgrad`、`cuda/csrc/RasterizeToPixels*.cu` |
| Mip-Splatting | `compensations` 字段、`_constants.py:MIN_COMPENSATION` |
| 3DGS-MCMC | `strategy/mcmc.py`、`relocation.py`、`cuda/csrc/MCMCPerturbCUDA.cu` |
| 3DGUT (NVIDIA) | `cuda/_torch_impl_ut.py`、`cuda/csrc/ProjectionUT3DGSFused.cu`、`RasterizeToPixelsFromWorld3DGS*.cu` |
| 2DGS | `cuda/csrc/Projection2DGS*.cu`、`RasterizeToPixels2DGS*.cu` |
| On Scaling Up 3DGS Training（分布式） | `distributed.py`、`cuda/csrc/DistributedCollectives.cpp`、`Rendering.cpp` Seam A/B |
| Taming 3DGS（可见性 Adam） | `optimizers/selective_adam.py`、`cuda/csrc/AdamCUDA.cu` |
| AccuTile / SNUGBOX | `isect_tiles(conics=..., opacities=...)`、`IntersectTile.cu` |
| Self-Organizing Gaussian Grids（压缩） | `compression/png_compression.py`、`compression/sort.py` |
| HiGS | `experimental/render/kernels/cuda/csrc/gaussian_inference/` |
| G-SHARP v0.2（动态手术场景） | `contrib/dynamic/`、`init_utils.py`、`regularizers.py`、`training/schedulers.py` |
| PPISP / 双边网格外观补偿 | `examples/lib_bilagrid.py`、`examples/simple_trainer.py` |

---

## 2. 代码规模与目录结构

### 2.1 规模统计

| 区域 | 文件数 | 行数 |
| --- | --- | --- |
| `gsplat/**/*.py` | 125 | ~40,700 |
| `gsplat/**` 原生代码（`.cu/.cuh/.cpp/.h`，排除 vendored glm） | 150 | ~86,600 |
| `examples/**/*.py` | 28 | 12,823 |
| `tests/**/*.py` | 102 | 42,839 |

各包分布（Python / 原生）：

```
gsplat/cuda          11,865  /  44,237     主光栅化扩展
gsplat/sensors       11,647  /  26,584     相机 + LiDAR 投影
gsplat/geometry       2,694  /   7,943     四元数 / SE3
gsplat/experimental   1,624  /   7,534     HiGS 推理渲染
gsplat/scene          1,688  /     290     场景容器 + 打包算子
gsplat/stage            117  /       0     场景-渲染函数编排
gsplat/strategy       1,210  /       0     致密化策略
gsplat/contrib          965  /       0     动态高斯（实验）
gsplat/compression      518  /       0     压缩
gsplat/training         151  /       0     两阶段调度
gsplat/optimizers       117  /       0     SelectiveAdam
gsplat/*.py（顶层）    8,100  /       0     rendering/losses/exporter/profile/...
```

### 2.2 目录树（按职责标注）

```
gsplat/
├── __init__.py             # 公共 API 门面：__all__ 共 67 个导出符号
├── rendering.py            # ★ 光栅化主入口（3DGS / 2DGS / Inria 兼容层）
├── losses.py               # 42 个 def（38 公开 + 4 内部）、1 个类，逐元素不归约
├── losses_fused.py         # 4 个高斯正则损失的融合 CUDA 版 + 纯 PyTorch 回退
├── regularizers.py         # G-SHARP 风格 TV 损失 / 掩码膨胀 / 不可见区域
├── init_utils.py           # SfM-free 初始化：多帧深度反投影 + kNN 尺度
├── exporter.py             # PLY / splat / 压缩 PLY 导出与 PLY 读取
├── color_correct.py        # 评测用仿射 / 二次颜色对齐
├── relocation.py           # MCMC 重定位的解析公式（新不透明度 & 尺度）
├── distributed.py          # 分布式训练工具（all-gather / all-to-all / CLI）
├── trace.py                # NVTX 标注（trace_range / trace_function）
├── profile.py              # ★ 独立的性能 replay harness（1,657 行）
├── utils.py                # save_ply / depth_to_points / depth_to_normal / 投影矩阵
├── _helper.py              # 测试断言工具（860 行，含 soft-check）
├── _lazy_backend.py        # 子包统一的懒加载原生扩展机制
├── version.py
│
├── cuda/                   # ★ 主原生后端
│   ├── ext.cpp             # TORCH_LIBRARY(gsplat) + PYBIND11_MODULE
│   ├── build.py            # JIT 构建参数与 build_and_load_gsplat
│   ├── _backend.py         # 先 import 预编译 .so，否则 JIT
│   ├── _wrapper.py         # ★ 3,218 行：算子 Python 绑定 + autograd 注册
│   ├── _torch_impl*.py     # 5 个纯 PyTorch 参考实现（测试基准）
│   ├── _math.py            # 多项式求逆 / safe normalize / 四元数参考实现
│   ├── _torch_cameras.py   # 2,204 行：相机模型 Python oracle
│   ├── _torch_lidars.py    # LiDAR 模型 Python oracle
│   ├── _lidar.py           # LiDAR 参数 dataclass + tiling 工具
│   ├── _losses_wrapper.py  # 融合损失 autograd 桥
│   ├── _constants.py       # ALPHA_THRESHOLD / MAX_ALPHA / ...
│   ├── include/            # Cameras.cuh / Lidars.cuh / Dispatch.h / TorchUtils.h ...
│   └── csrc/               # 50+ .cu/.cpp 设备端实现（见 §8）
│
├── geometry/               # 四元数与 SE(3)：functional/ + kernels/
├── sensors/                # 相机与 LiDAR：functional/ + kernels/ + models/
├── scene/                  # GaussianScene / GaussianInferenceScene + 打包算子
├── stage/                  # Stage：scene × render_fn 的路由
├── strategy/               # Strategy 基类 + Default + MCMC + 原地算子
├── compression/            # PngCompression + PLAS 排序
├── optimizers/             # SelectiveAdam
├── training/               # TwoStageScheduler
├── contrib/dynamic/        # HexPlane / DeformNetwork / 动态策略 / 平面正则
└── experimental/render/    # HiGS 推理渲染
```

### 2.3 随包发布的模块设计文档

仓库自带一组 `design.md`，是理解子包契约的**权威补充**（本文档不重复其细节，只做索引）：

| 文档 | 覆盖内容 |
| --- | --- |
| `docs/modules-design.md` | `gsplat/<name>` 共享模块规范：`functional/ → kernels/ → kernels/cuda/` 分层、`.cuh`+`.cu` 配对、测试要求 |
| `gsplat/geometry/design.md` | 几何模块：Python 层校验策略、CUDA 命名（`_cuda` / `_bwd_cuda`） |
| `gsplat/sensors/design.md` | 传感器总览（三层：functional / kernels / models） |
| `gsplat/sensors/design-functional.md` | 8 个相机 op + 5 个 LiDAR op 的签名与返回 dataclass |
| `gsplat/sensors/design-kernels.md` | 三层 C++ 表示、`(projection, distortion)` 分发表、scratch 反向契约 |
| `gsplat/sensors/design-models.md` | `CameraModel` / `LidarModel` / `Frame` 族 |
| `gsplat/scene/design.md` | `Scene` / `GaussianScene` 存储模型、拓扑钩子、序列化权衡 |
| `gsplat/stage/design.md` | `Stage` 只做 scene↔render_fn 路由 |
| `docs/3dgut.md` | 3DGUT 使用指南（`--with_ut --with_eval3d`） |
| `docs/batch.md` | 任意批维度渲染用法与基准 |

---

## 3. 分层架构与调用链

### 3.1 四层结构

```
┌──────────────────────────────────────────────────────────────────┐
│ L4  用户 API                                                       │
│     gsplat.rasterization() / rasterization_2dgs()                  │
│     gsplat.sensors.functional.* / gsplat.geometry.functional.*     │
│     gsplat.scene / stage / experimental.render_scene               │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Python 语义、形状/设备校验、dataclass 包装
┌───────────────────────────▼──────────────────────────────────────┐
│ L3  算子绑定层                                                     │
│     cuda/_wrapper.py：_make_lazy_cuda_func("op") → torch.ops.gsplat│
│     torch.library.register_autograd 注册反向                        │
└───────────────────────────┬──────────────────────────────────────┘
                            │ 裸 Tensor / 自定义 class 句柄
┌───────────────────────────▼──────────────────────────────────────┐
│ L2  C++ 编排（host 侧）                                            │
│     Rendering.cpp: rasterization_3dgs()  串联全部阶段              │
│     Rasterization.cpp / Projection.cpp：单算子 host 包装            │
└───────────────────────────┬──────────────────────────────────────┘
                            │ POD 结构 + cudaStream_t
┌───────────────────────────▼──────────────────────────────────────┐
│ L1  CUDA kernel（device 侧）                                       │
│     ProjectionEWA* / RasterizeToPixels* / IntersectTile* / ...     │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 一次 3DGS 前向的完整调用链

```
rasterization(...)                                   rendering.py:234
  └─ _make_lazy_cuda_func("rasterization_3dgs")(...)  → torch.ops.gsplat.rasterization_3dgs
       └─ gsplat::rasterization_3dgs(...)             cuda/csrc/Rendering.cpp:745
            ├─ check_rasterization_3dgs_inputs(...)   Rendering.cpp:97（定义）/ :803（调用）
            ├─ Seam A：gather_cameras_for_distributed  Rendering.cpp:852   （仅 distributed）
            ├─ 投影
            │    ├─ with_ut → projection_ut_3dgs_fused   ProjectionUT3DGSFused.cu
            │    └─ 否则   → projection_ewa_3dgs_fused   ProjectionEWA3DGSFused.cu
            │                （packed 模式走 ProjectionEWA3DGSPacked.cu）
            ├─ 特征装配（SH 求值 + extra signals + depth 列）Rendering.cpp:1051-1200
            │    └─ 融合快路径：SHPostOp assembler kernel（一次 coalesced 读写）
            ├─ Seam B：all-to-all scatter 投影结果        Rendering.cpp:1217   （仅 distributed）
            ├─ intersect_tile                            IntersectTile.cu
            │    └─ AccuTile：提供 conics+opacities 时用保守椭圆求交
            ├─ intersect_offset                          （前缀和）
            ├─ rasterize_to_pixels_3dgs（按 channel_chunk 分块）  RasterizeToPixels3DGSSerialBatchFwd.cu
            └─ 重组输出通道                              Rendering.cpp:1449
```

反向：

```
rasterize_to_pixels_3dgs_bwd   RasterizeToPixels3DGSSerialBatchBwd.cu
  └─ rasterize_to_indices_3dgs（重放 tile 遍历以重建合成顺序）
       └─ projection_ewa_3dgs_fused_bwd → 各输入梯度
```

### 3.3 层间依赖（来自索引的 `boundaries` 视图）

```
cuda       → _helper        46 次调用   ← 唯一"核心被依赖"模块
rendering  → cuda           16 次调用
sensors    → cuda           15 次调用
geometry   → cuda            8 次调用
experimental → cuda          7 次调用
losses_fused → losses        4 次调用
profile    → cuda            3 次调用
rendering  → utils           3 次调用
compression → utils          2 次调用
strategy   → relocation      2 次调用
```

**判读**：`_helper` 与 `cuda` 是图层意义下的 core（高扇入），`relocation` 是叶子
（只被调用），`rendering` / `sensors` / `geometry` / `experimental` 是入口层。
顶层模块**不反向依赖**子包，依赖方向始终单向向下。

---

## 4. 构建系统：五个原生扩展

### 4.1 扩展清单

| 扩展模块名 | 注册机制 | Torch op 命名空间 | `build.py` 入口 |
| --- | --- | --- | --- |
| `gsplat.csrc` | `PYBIND11_MODULE` + `TORCH_LIBRARY(gsplat)` | `gsplat::*` | `cuda/build.py:351` `build_and_load_gsplat` |
| `gsplat_geometry_cuda` | 纯 pybind11（**无** `TORCH_LIBRARY`） | 直接属性访问 `_GEOMETRY_CUDA.quat_multiply` | `geometry/kernels/cuda/build.py` |
| `gsplat_sensors_cuda` | `PYBIND11_MODULE` + `TORCH_LIBRARY(gsplat_sensors)` | `gsplat_sensors::*` | `sensors/kernels/cuda/build.py:60` |
| `gsplat_scene_cuda` | 纯 pybind11 | `pack_gaussian_inference_scene` | `scene/kernels/cuda/build.py:38` |
| `experimental_gaussian_render_inference_scene_cuda` | `PYBIND11_MODULE` + `TORCH_LIBRARY(experimental)` | `experimental::*` | `experimental/render/kernels/cuda/build.py:68` |

> 反向注册细节：`gsplat` 命名空间还额外注册了 `TORCH_LIBRARY_IMPL(gsplat, CUDA/PrivateUse1/AutogradCUDA/AutogradPrivateUse1, ...)`
> （`cuda/ext.cpp:1284-1341`），其中 `PrivateUse1` 分支是给非 CUDA 后端（如昇腾等）预留的派发槽位。

### 4.2 两种加载模式

```python
# AOT（pip install 时编译）——setup.py:218 get_extensions()
gsplat.csrc                     = CUDAExtension(...)
gsplat.experimental.render.kernels.csrc = CUDAExtension(...)

# JIT（首次 import 时编译，用于开发）
gsplat/cuda/_backend.py:
    try:  from gsplat import csrc as _C        # 优先预编译
    except ImportError:
        if cuda_toolkit_available(): _C = build_and_load_gsplat()   # 否则 JIT
        else: 打印 "gsplat will be disabled"
```

子包统一走 `gsplat/_lazy_backend.py:84` 的 `make_lazy_backend()`：
**在首次访问哨兵属性时才加载**（PEP 562 `__getattr__`），因此 CPU-only 机器上
`import gsplat.sensors.functional` 不会触发编译。加载失败会被缓存为 `ImportError`，
后续访问快速失败，不会反复重跑数分钟的 JIT。

### 4.3 构建参数与开关

`gsplat/cuda/build.py`：

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `DEBUG` | `0` | `1` → `-g -O0 -Werror`（并打开 `WITH_SYMBOLS`、`BUILD_CAMERA_WRAPPERS`） |
| `FAST_MATH` | `1` | 是否启用 `--use_fast_math` |
| `WITH_SYMBOLS` | `DEBUG?1:0` | 是否保留 `-lineinfo` |
| `NVCC_FLAGS` | 空 | 追加给 nvcc 的参数 |
| `MAX_JOBS` / `NINJA_STATUS` / `VERBOSE` | — | ninja 并行度 / 状态行 / 详细日志 |
| `NUM_CHANNELS` | `1,2,3,4,...,513` | **模板实例化的通道数**；`rasterize_to_pixels` 的 `colors.shape[-1]` 必须命中其中某个值 |
| `BUILD_3DGS` / `BUILD_3DGUT` / `BUILD_2DGS` / `BUILD_ADAM` / `BUILD_RELOC` / `BUILD_LOSSES` | 全开 | 显式指定任一个则**只**构建指定的那些（`csrc/Config.h:28-38`） |
| `BUILD_CAMERA_WRAPPERS` | `0` | 相机/畸变的 Python 测试封装，默认关闭 |
| `BUILD_NO_CUDA` | `0` | `1` → 安装时不编译原生扩展 |
| `BUILD_EXPERIMENTAL` | `1` | 是否构建 HiGS 推理扩展 |
| `GSPLAT_SENSORS_FORCE_JIT` | — | `1` → 跳过 sensors 预编译包，强制 JIT |

`pytest.ini` 在测试时固定 `NUM_CHANNELS=1,3,4,6,8,21,23,24,32,128`、
`BUILD_CAMERA_WRAPPERS=1`、`VERBOSE=1`。

`get_build_parameters()`（`build.py:98`）返回 `SimpleNamespace(name, sources, flags...)`，
被 JIT 与 `setup.py` **共用**，保证两条路径产出相同的二进制；flag 变化会通过
`build_params.json` 快照比对触发目录清空重建（防止 stale `.so` 存活）。

`build.py:321` `_exclude_dispatcher_objects()` 把 `ext.cpp` 的 `.o` 从核心对象列表中剔除，
因为原生 C++ 测试二进制不能重复注册 `gsplat` 命名空间；`build.py:653`
`get_gsplat_link_inputs()` 提供 ABI 匹配的链接输入给 `tests/test_cpp.py`。

### 4.4 发行打包

`MANIFEST.in` + `setup.py:294 package_data` 双管齐下：所有子包的 `kernels/cuda/**`
源码都会进入 sdist 与 wheel，因此**装 wheel 也能在首次运行时 JIT 编译**。
`config.yaml` 里的 `CLANG_FORMAT_VERSION: 22.1.5` 被 CI 与 `lint/format-code.sh` 共同读取。

### 4.5 源码编译安装要点

本节记录在本仓库上实测得到的安装路径与坑，环境为：
**Python 3.12.3 / torch 2.15.0+cu134 / CUDA 13.4 toolkit / GCC 13.3 / sm_120（Blackwell）**。

#### 4.5.1 AOT 只覆盖两个扩展

`setup.py:284` 的 `get_extensions()` 只返回两个 `CUDAExtension`：

| AOT 编译 | 源 | 产物位置 |
| --- | --- | --- |
| `gsplat.csrc` | `gsplat/cuda/build.py` 的 50 个源文件 | 源码树 `gsplat/csrc*.so` |
| `gsplat.experimental.render.kernels.csrc` | `gsplat/experimental/render/kernels/cuda/build.py` | 源码树对应包目录 |

**其余三个扩展只能 JIT**：`gsplat_geometry_cuda`、`gsplat_sensors_cuda`、`gsplat_scene_cuda`
（`setup.py` 未把它们列为 ext_modules）。它们首次访问时才在
`~/.cache/torch_extensions/py<ver>_cu<cuda>/` 下编译，此后复用。

实践含义：**装 AOT 之后，仍会有一次性的子扩展 JIT 编译**，这是预期行为，不是安装失败。
好在三个子扩展都不读 `NUM_CHANNELS` / `BUILD_CAMERA_WRAPPERS`，因此不会被下节的参数抖动牵连。

#### 4.5.2 构建参数抖动会导致全量重建（最容易踩的坑）

`build.py:362-402` 的逻辑是：把当前 `get_build_parameters()` 序列化成
`build_params.json` 落盘；**下次加载时若与快照不一致，就 `shutil.rmtree(build_dir)`
删掉整个构建目录重新来过**（防止旧 `.so` 冒充新配置）。

`pytest.ini` 的 `env` 段（`pytest-env` 是 `setup.py:176` 声明的依赖，默认生效）注入了：

```ini
env =
    VERBOSE=1
    BUILD_CAMERA_WRAPPERS=1
    NUM_CHANNELS=1,3,4,6,8,21,23,24,32,128
```

实测对比「普通运行」与「pytest 运行」的 `get_build_parameters()`，有 **3 个字段不同**：

| 字段 | 普通运行 | pytest 运行 |
| --- | --- | --- |
| `extra_cflags` | 无 `-DGSPLAT_*` | `-DGSPLAT_BUILD_CAMERA_WRAPPERS=1`、`-DGSPLAT_NUM_CHANNELS=...` |
| `extra_cuda_cflags` | 同上 | 同上（逗号转义为 `\,`） |
| `sources` | **50** 个 | **51** 个（多 `csrc/CameraWrappers.cu`） |

于是 **在「普通运行」和「pytest」之间来回切换，每次都会触发 50 文件全量重建**。
叠加两个 eval3d kernel 各需约 25 分钟，一次切换就是一小时级的浪费。

加剧因素：`import gsplat` 会先执行 `gsplat/__init__.py`，其中导入 `cuda._wrapper`，
**任何** gsplat 子模块的 import 都会触发主扩展加载。所以仅「换一组环境变量再跑一条命令」
就足以重启重建。

**规避办法（三选一）**：

1. **AOT 定死（推荐）**：用 `pip install -e .` 真正编译，让 `from gsplat import csrc` 成功。
   `cuda/_backend.py` 一旦 import 成功就**永不进入 JIT 分支**，也就不再比较参数、不再删目录。
   编译时把 pytest 需要的两项直接编进去即可（见下节命令）。
2. **让两份参数一致**：shell 里恒设
   `NUM_CHANNELS=<Config.h 默认全集>` 与 `BUILD_CAMERA_WRAPPERS=1`，
   使 JIT 与 pytest 得到同一个快照（`Config.h:72` 的默认通道集包含 pytest 那 10 个，
   是安全的超集）。
3. **临时关闭 pytest-env**：跑测试时加 `-p no:env`，让 `pytest.ini` 的 `env` 段失效。

#### 4.5.3 推荐命令

先补齐构建期依赖（不触碰 torch）：

```bash
~/.venv/3DGen/bin/python -m ensurepip --upgrade          # 若 venv 缺 pip
~/.venv/3DGen/bin/python -m pip install --no-deps ninja wheel "rich>=12" jaxtyping nvtx
```

再 AOT 源码安装（**前台运行以便监控**）：

```bash
cd /home/aic/Github/gsplat && \
NUM_CHANNELS=1,2,3,4,5,6,8,9,16,17,21,23,24,32,33,64,65,128,129,256,257,512,513 \
BUILD_CAMERA_WRAPPERS=1 \
MAX_JOBS=6 \
~/.venv/3DGen/bin/python -m pip install -e . \
  --no-build-isolation --no-deps --no-cache-dir --verbose \
  2>&1 | tee /tmp/gsplat_build.log
```

各参数的必要性：

| 参数 | 原因 |
| --- | --- |
| `NUM_CHANNELS=<默认全集>` | `Config.h:72` 的默认值。pytest 的 10 个通道是其子集，编全集可同时满足两者 |
| `BUILD_CAMERA_WRAPPERS=1` | 把 `CameraWrappers.cu` 编入。这是 pytest 相对普通运行唯一多出的源文件 |
| `MAX_JOBS=6` | **必设**。`RasterizeToPixelsFromWorld3DGS{Serial,Parallel}BatchFwd.cu` 单个编译峰值约 7GB RSS；32 核并发放开到 62GB 内存会 OOM |
| `--no-build-isolation` | 使用当前 venv 的 torch/setuptools 构建。若不加，pip 会新建隔离环境并**装一份无关版本的 torch** |
| `--no-deps` | 完全不解析 `install_requires`，保证**不改动已装 torch** |
| `--no-cache-dir` | 避免复用陈旧 wheel，确保干净构建 |

**耗时**：50 个 CUDA 文件，其中两个 eval3d 前向 kernel 各约 25 分钟并占绝大部分，
总计约 **40–60 分钟**。若不需要 HiGS，可加 `BUILD_EXPERIMENTAL=0` 跳过第二个扩展。

**torch 保护的三重保证**：`--no-deps`（不解析 torch 依赖）+ `--no-build-isolation`
（不建隔离环境）+ `find_packages(exclude=["tests","tests.*"])`（只打包 `gsplat`，
共 36 个包，**不含 `examples/`、不含 `tests/`**）。

#### 4.5.4 验证安装

```bash
cd /tmp && ~/.venv/3DGen/bin/python -c "
import gsplat, torch
print('version  ', gsplat.__version__)
print('backend  ', gsplat.cuda._backend._C is not None)
for f in ['has_3dgs','has_3dgut','has_2dgs','has_adam','has_reloc','has_losses','has_camera_wrappers']:
    print(f'  {f:22s}', getattr(gsplat,f)())
print('torch    ', torch.__version__)
"
```

判读要点：

- `has_camera_wrappers` 为 `True` → `BUILD_CAMERA_WRAPPERS=1` 已编入，pytest 不会再重建。
- **import 应在秒级完成**。若耗时数分钟，说明 `gsplat/csrc*.so` 缺失、
  `_backend.py` 静默回退到了 JIT——此时参数冲突会重现，需检查 AOT 是否真的产出了 `.so`。
- 在源码树外（`cd /tmp`）执行，可确认是装到了 site-packages 而不是靠当前目录遮蔽。

功能验证（前向+反向、3DGUT、2DGS、SH、深度模式）：

```python
import torch, gsplat
N, C, H, W = 1000, 2, 64, 64
means = torch.randn(N,3,device="cuda",requires_grad=True)
quats = torch.randn(N,4,device="cuda")
scales = torch.rand(N,3,device="cuda")*0.1
opa = torch.rand(N,device="cuda").contiguous()
col = torch.rand(N,3,device="cuda")
vm = torch.eye(4,device="cuda")[None].repeat(C,1,1); vm[:,2,3]=4.0
K = torch.tensor([[80.,0,W/2],[0,80.,H/2],[0,0,1]],device="cuda")[None].repeat(C,1,1)

o,a,_ = gsplat.rasterization(means,quats,scales,opa,col,vm,K,W,H,packed=False)
(o.mean()+a.mean()).backward()
print("3DGS", o.shape, "grad ok:", torch.isfinite(means.grad).all().item())

# 3DGUT：注意 eval3d 不支持 packed=True
o2,_,_ = gsplat.rasterization(means.detach(),quats,scales,opa,col,vm,K,W,H,
                              packed=False, with_ut=True, with_eval3d=True)
print("3DGUT", o2.shape)

# 2DGS：packed 与 render_mode 均可自由组合（两种组合模式的反向已修复，见 §5.6）
sh = torch.rand(N,16,3,device="cuda")
r = gsplat.rasterization_2dgs(means.detach(),quats,scales,opa,sh,vm,K,W,H,
                              sh_degree=3, packed=True, render_mode="RGB+ED")
r[0].sum().backward()
print("2DGS", r[0].shape, "surf_normals:", None if r[3] is None else r[3].shape)
```

#### 4.5.5 已验证的上游兼容性问题

安装过程中遇到的三类失败，**均非 gsplat 主扩展的问题**：

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| `gsplat_scene_cuda` 编译报 `c10::List<c10::SymInt>::iterator` 无匹配比较 | `gsplat/scene/kernels/cuda/build.py:54` 用 `-std=c++17`，而 torch 2.15 头文件已要求 C++20；另外四个扩展都用 C++20 | 已在本仓库改为 `-std=c++20`（Win32 分支同步改为 `/std:c++20`）。改后 5 秒编译通过 |
| `nerfacc` 0.5.3 编译报 `float lerp(...) conflicts with std::lerp` | 其 `nerfacc/cuda/csrc/include/utils_math.cuh:1145` 在全局命名空间定义了 `lerp`，与 GCC 13 `<cmath>` 的 `std::lerp` 冲突。该 `lerp` 在 nerfacc 全库**零调用点** | 已把 4 个重载改名为 `nerfacc_lerp`（零调用点，安全）。改后可编译 |
| `NerfAcc: No CUDA toolkit found` 且 `_C=None` | nerfacc 的 `cuda_toolkit_available()` 用 `call(["nvcc"])` 探测，要求 `nvcc` 在 `PATH`；本机 CUDA 仅在 `/usr/local/cuda/bin` 且未加入 `PATH`（gsplat 自己的探测器走 `_find_cuda_home()`，不受影响） | 运行时设 `PATH=/usr/local/cuda/bin:$PATH`，或建软链 |

> 注意后两项修改落在 **site-packages**（`~/.venv/3DGen/lib/python3.12/site-packages/nerfacc/`）
> 与仓库工作区，重装/升级这两个包会丢失，需要重新施加。

#### 4.5.6 排错清单

| 症状 | 检查点 |
| --- | --- |
| 反复出现 `gsplat: rebuilding due to build parameter change` | 参数快照抖动，见 §4.5.2。比对 `build_params.json` 的 `extra_cflags` / `sources` 长度 |
| 编译被 `Killed`（OOM） | `MAX_JOBS` 过大。按 `峰值 7GB × 并发数 < 可用内存 × 0.8` 设 |
| `import gsplat` 极慢 | 正在 JIT。查 `~/.cache/torch_extensions/py312_cu134/gsplat_cuda/*.so` 是否存在 |
| `RuntimeError: Packed mode is not supported with Eval3D` | 3DGUT（`with_eval3d=True`）不支持 `packed=True`，属用法错误 |
| 2DGS 报 `opacities must be contiguous` | dense 路径的 `expand()` 缺陷（已在 §5.6 修复）。若在旧版本/未编译本修复时遇到，临时用 `packed=True` |
| 2DGS 组合模式报 `grad.depths must be contiguous` | 2DGS 投影反向缺 `.contiguous()`（已在 §5.6.2 修复）。旧版本可先对 `render_colors` 整体求导规避通道切片 |
| `colors must have shape [..., N, channels]` | 张量布局契约。底层算子 `rasterize_to_pixels*` 要求 `[..., C, N, D]`；高层 `rasterization()` 接受 `[..., N, D]` |
| 中断编译后 `.so` 消失 | 中断会让 torch 的 ninja 构建处于半成品状态；重新触发即可续编（已完成的 `.o` 会复用） |
| 构建目录残留 `lock` | `build.py:359` 会自动删除；若进程被 `kill -9` 可能残留，手动删 `lock` 即可 |


---

## 5. 核心渲染入口 `gsplat/rendering.py`

文件 1,717 行，是整个库的中枢。按行号分区：

| 行号 | 内容 |
| --- | --- |
| 25–56 | 从 `cuda._wrapper` 导入全部底层算子 + `trace_function` / `capture_inputs` 装饰器 |
| 58–59 | `_POST_CODE`：`none/shift/shift_relu` 三种后处理码的 Python 侧对照表（当前仅定义未引用；C++ 侧在 `Rendering.cpp:1124` 硬编码 `color_post=2`） |
| 64–65 | 类型别名 `RenderMode`（9 种）、`RasterizeMode`（`classic` / `antialiased`） |
| 68–104 | `RendererConfig` 基类 + `RendererConfig_MixedBatch` + `RendererConfig_ParallelBatch` |
| 106–133 | `_validate_renderer_config` / `_renderer_config_type`（Python 对象 → CUDA 枚举） |
| 136–161 | 7 个 `render_mode_has_*()` 谓词（颜色 / 深度 / 期望深度 / 命中距离 / 仅深度） |
| 164–200 | `_validate_3dgut_rasterize_mode`、`_get_default_nccl_process_group_name` |
| 201–232 | `_resolve_tile_size`：`None` → 按路径选默认 tile 尺寸 |
| **234–690** | **`rasterization()`：3DGS 主入口（43 个参数，其中 31 个有默认值；完整 docstring）** |
| 693–719 | `_maybe_evaluate_sh`：把 `[N,K,D]` SH 系数按视角求值成 `[..., C, N, D]` |
| 722–1106 | `_rasterization()`：**纯 PyTorch autograd** 的研究用版本（需 nerfacc） |
| 1199–1354 | `rasterization_inria_wrapper()`：与官方 Inria 实现对齐的兼容层 |
| **1358–1568** | **`rasterization_2dgs()`：2DGS 入口，返回 7 元组** |
| 1571–1717 | `rasterization_2dgs_inria_wrapper()` |

### 5.1 `rasterization()` 关键参数分组

```python
rasterization(
    means, quats, scales, opacities, colors,   # 高斯本体
    viewmats, Ks, width, height,               # 相机
    near_plane=0.01, far_plane=1e10, radius_clip=0.0, eps2d=0.3,
    sh_degree=None,
    packed=True, tile_size=None, backgrounds=None,
    render_mode="RGB",
    sparse_grad=False, absgrad=False, rasterize_mode="classic",
    channel_chunk=32,           # 通道数 > 此值时循环分块渲染
    distributed=False,          # 多 GPU 协作
    camera_model="pinhole",     # pinhole|ortho|fisheye|ftheta|lidar
    segmented=False,            # 分段 radix sort
    covars=None,                # 直接给协方差，忽略 quats/scales
    with_ut=False, with_eval3d=False,   # ← 3DGUT 的两个开关
    return_normals=False, global_z_order=True,
    rays=None,                  # [..., C, H, W, 6] 逐像素射线（eval3d 专用）
    # 畸变
    radial_coeffs=None, tangential_coeffs=None, thin_prism_coeffs=None,
    ftheta_coeffs=None, lidar_coeffs=None, external_distortion_coeffs=None,
    # 卷帘
    rolling_shutter=RollingShutterType.GLOBAL, viewmats_rs=None,
    ut_params=None,
    extra_signals=None, extra_signals_sh_degree=None,
    renderer_config=None,
)
```

**返回**：`(render_colors, render_alphas, meta)`，其中 `meta` 键集合（`rendering.py:663-681`）：

```
batch_ids, camera_ids, gaussian_ids,   # packed 模式下的 COO 索引（非 packed 时为 None）
radii, means2d, depths, conics, opacities,
tile_width, tile_height, tiles_per_gauss,
isect_ids, flatten_ids, isect_offsets,
width, height, tile_size, n_batches, n_cameras
[+ render_extra_signals] [+ normals]
```

> **致密化策略正是依赖 `meta["means2d"]` 与 `meta["radii"]`**，所以策略必须能拿到完整
> `info` 字典。`absgrad=True` 时 `meta["means2d"].absgrad` 会被挂上绝对梯度属性。

### 5.2 tile 尺寸选择逻辑（`rendering.py:201`）

```python
if tile_size is not None: return tile_size           # 用户显式指定优先
if with_eval3d:  return 16 if min(width, height) >= 1080 else 8
return 16                                            # 3DGS 只编译了 TILE_SIZE=16
```

3DGUT 的 kernel 对 `tile_size ∈ {8, 16}` 做**编译期**派发：`tile=8` 是紧凑 CTA 路径
（CTA=32、每线程 2 像素），在 1080p 以下胜出；`tile=16` 是一线程一像素，在 1080p+
时因求交+排序开销更小而胜出。LiDAR 网格宽而浅（如 128×3600），`min(W,H)=128` 所以
总是落在 `tile=8`。

### 5.3 渲染模式语义（重要，易混淆）

| 模式 | 深度含义 | 公式 |
| --- | --- | --- |
| `D` / `RGB+D` | **高斯投影深度**累积 | Σ wᵢ zᵢ |
| `ED` / `RGB+ED` | **期望**高斯投影深度 | Σwᵢzᵢ / Σwᵢ |
| `d` / `RGB-d` | **沿射线命中距离**累积 | Σ wᵢ dᵢ |
| `Ed` / `RGB-Ed` | **期望**命中距离 | Σwᵢdᵢ / Σwᵢ |

`global_z_order` 控制排序依据：`True` 用相机空间 z，`False` 用欧氏距离。
后者是渲染 **FTheta 相机 >180° 视场带**的必要条件，也让 `D/ED` 通道改用欧氏距离。

### 5.4 2DGS 返回签名与条件字段

```python
colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(...)
```

7 元组**位置固定**，但 `surf_normals`（第 4 项）**条件性返回 `None`**（`Rendering.cpp:1735`）：

```cpp
const bool compute_normals_from_depth = render_mode_has_depth(render_mode) && has_color;
```

实测各 `render_mode` 下第 4 项：

| `render_mode` | `surf_normals` |
| --- | --- |
| `RGB` / `D` / `ED` / `d` / `Ed` | **`None`** |
| `RGB+D` / `RGB+ED` / `RGB-d` / `RGB-Ed` | `(C, H, W, 3)` |

即**必须同时具备颜色与深度通道**才产出。解包后直接 `.shape` 会崩，需判空：

```python
colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(...)
if surf_normals is not None:
    ...
```

两路法线的来源不同，不要混淆：

| 字段 | 来源 | 坐标系 |
| --- | --- | --- |
| `normals`（第 3 项） | 光栅化 kernel 内逐像素累积（`RasterizeToPixels2DGSSerialBatchFwd.cu:404,453`，输入是投影阶段产出的相机空间法线），随后在 `Rendering.cpp:1930` 用 `inv(viewmats)` 的旋转块转世界 | kernel 输出相机空间 → 返回前转世界 |
| `surf_normals`（第 4 项） | CUDA 侧 `depth_to_normal_2dgs()`（`Rendering.cpp:1686`）：先 `depth_to_points_2dgs` 反投影成点云，再对相邻像素差分叉积 | 世界 |

> ⚠️ **`depth_to_normal()`（Python，`gsplat/utils.py:223`）不参与此路径**。
> 它只出现在 `rasterization_2dgs_inria_wrapper`（`rendering.py:1699`，函数起于 1571），
> 那是官方 Inria 后端的兼容层，与 `rasterization_2dgs` 是两条独立实现。

### 5.5 `renderer_config`（仅 eval3d 路径）

| 配置 | 前向 | 反向 | 内存 |
| --- | --- | --- | --- |
| `RendererConfig_MixedBatch`（默认） | 串行批：每 tile 一个 CTA，前到后合成 | 批并行（每 batch 一 CTA） | 低 |
| `RendererConfig_ParallelBatch` | 批并行 + partials/scan/replay 流水线 | 同一个批并行反向 kernel | 高（需持久化每批前向状态） |

非 eval3d 路径只接受 `MixedBatch`，否则 `ValueError`（`rendering.py:551-556`）。

### 5.6 2DGS 两处已修复的缺陷（历史记录 + 防回归说明）

本节记录两个曾使 2DGS 在特定配置下不可用的缺陷。**二者均已在本仓库修复并回归验证**，
保留记录是为了说明代码为什么要那样写，避免后续重构把它们改回去。

#### 5.6.1 dense（`packed=False`）路径不可用 —— 已修复

**曾经的症状**：`rasterization_2dgs(..., packed=False)` 必定抛
`RuntimeError: opacities must be contiguous`，且无调用侧绕过手段。

**根因**（`Rendering.cpp` dense 分支）：

```cpp
std::vector<int64_t> opacity_shape = batch_shape_with_2dgs(means, {C, N});
projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape);
//                                                     ^^^^^^^^ stride=0 视图
```

`expand()` 只广播 stride、不复制数据，产出**非连续**张量；它作为 `opacities` 传给
`rasterize_to_pixels_2dgs_fwd`，后者的 `CHECK_INPUT(opacities)` →
`CHECK_CONTIGUOUS` 必然失败。

**修复**：在 `expand` 后立刻 `.contiguous()` 把广播实体化。

```cpp
// expand() broadcasts with stride 0, but the rasterizer's CHECK_INPUT
// demands contiguous storage. Materialize the broadcast instead.
projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape).contiguous();
```

> ⚠️ 不要改成 `repeat(opacity_shape)`：`repeat` 的参数是**每维复制倍数**而非目标形状，
> 传形状向量会得到 `[C, N*N]` 这类错误结果（实测踩过）。

**曾经的绕过方式**（现已不需要）：`torch.no_grad()`、显式 `.contiguous()` 全部输入、
改用后置颜色 `[C,N,D]` 都无效——因为非连续性产生在库内部。
当时只能强制 `packed=True`，而 `examples/simple_trainer_2dgs.py:146` 的默认值恰是
`packed=False`，所以按默认参数训练会直接失败。

#### 5.6.2 组合模式（`RGB+D` / `RGB+ED`）反向崩溃 —— 已修复

**曾经的症状**：前向正常，`backward()` 抛 `RuntimeError: grad.depths must be contiguous`。
纯 `RGB` / `D` / `ED` 模式不受影响。

**根因**：2DGS 前向把深度拼到特征末尾（`Rendering.cpp`）：

```cpp
feature = at::cat({feature, depths.unsqueeze(-1)}, -1);
```

反向时 autograd 从拼接梯度里切出深度部分，得到 **stride 非连续**的视图
（实测末维 stride 为 4 而非 1）。而 `Projection.cpp` 中 2DGS 的两个反向入口
**只做检查、不做归一化**：

```cpp
CHECK_DENSE(grad.depths);
CHECK_INPUT(grad.depths);      // ← 非连续即失败
```

同文件里 3DGS 的 `projection_ewa_3dgs_fused_bwd` / `..._packed_bwd` 则是先转换再检查
（`at::Tensor grad_depths = grad.depths.contiguous();`），2DGS 漏了这一步。

**修复**：`projection_2dgs_fused_bwd` 与 `projection_2dgs_packed_bwd` 两处，
把 `grad.{means2d,depths,normals,ray_transforms}` 统一先 `.contiguous()` 再
`CHECK_INPUT`，并把归一化后的张量传给 `launch_*_bwd_kernel`。写法与 3DGS 路径对齐。

**回归测试**：`tests/test_2dgs.py`

| 测试 | 覆盖 |
| --- | --- |
| `test_rasterization_2dgs_depth_channel_backward[packed × {RGB+D, RGB+ED}]` | 端到端：4 个组合，校验全张量反向且所有输入的梯度有限 |
| `test_projection_2dgs_bwd_accepts_non_contiguous_grad` | 底层：直接喂 stride 非连续的 `grad.depths`，确认算子自身能归一化 |

#### 5.6.3 修复后的实测状态

2 种 `packed` × 5 种 `render_mode` 全部前向+反向通过：

| `packed` | `RGB` | `D` | `ED` | `RGB+D` | `RGB+ED` |
| --- | --- | --- | --- | --- | --- |
| `True` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `False` | ✅ | ✅ | ✅ | ✅ | ✅ |

全量测试：`2626 passed, 141 skipped, 0 failed`。

> `D` / `ED` 模式下 `colors` 的梯度为 `None`（该模式不使用颜色通道），属预期行为。

---

## 6. CUDA 后端：Python 绑定层

### 6.1 `cuda/_wrapper.py`（3,218 行）

这是 Python 与 `torch.ops.gsplat` 之间的唯一桥梁，分为三类设施：

**(a) 懒解析器**

| 函数 | 行 | 作用 |
| --- | --- | --- |
| `_make_lazy_cuda_func(name)` | 48 | 返回一个闭包，调用时才 `from ._backend import _C`（触发编译）再 `getattr(torch.ops.gsplat, name)(...)` |
| `_make_lazy_cuda_cls(name)` | 114 | 同上，用于 `torch.classes.gsplat.*` 自定义类 |
| `_make_lazy_cuda_obj(name)` | 150 | 用于枚举值，如 `CameraModelType.PINHOLE` |
| `_has_schema(op)` | 65 | 检查 `gsplat::<op>` 是否被编译进来（构建开关可能裁掉） |
| `has_2dgs/has_3dgs/has_3dgut/has_adam/has_reloc/has_losses/has_camera_wrappers` | 270–296 | 特性探测，用于在算子缺失时给出友好降级 |

**(b) 公共算子**（可直接 `from gsplat import ...`）

| 算子 | 行 | 说明 |
| --- | --- | --- |
| `spherical_harmonics` | 436 | `[...,N,3]` 均值 + `[N,K,D]` 系数 → `[...,C,N,D]`；`D` 任意正整数 |
| `spherical_harmonics_l0` | 493 | 只算 l=0（`[N,1,D] → [N,D]`） |
| `spherical_harmonics_l1_plus` | 508 | 只算 l≥1（`shN` 从 1 阶开始） |
| `quat_scale_to_covar_preci` | 657 | 四元数+尺度 ↔ 协方差/精度矩阵 |
| `proj` | 750 | 相机空间高斯 → 2D 均值+协方差（单算子） |
| `fully_fused_projection` | 819 | **融合** `quat_scale_to_covar` + `world_to_cam` + `proj`，支持 packed/dense |
| `world_to_cam` | 381 | ⚠️ **已弃用**，现在转调纯 PyTorch `_torch_impl._world_to_cam` |
| `isect_tiles` | 1196 | 高斯 → tile 交集；提供 `conics`+`opacities` 时启用 AccuTile 椭圆求交 |
| `isect_tiles_lidar` | 1271 | LiDAR 专用求交 |
| `isect_offset_encode` | 1328 | 交集 id → 每 tile 偏移表 |
| `isect_tiles_sparse` | 1352 | 只对 active tile 求交 |
| `build_sparse_tile_layout` | 1433 | 稀疏渲染的 tile/像素簿记（位掩码 + 前缀和 + 像素置换） |
| `rasterize_to_pixels` | 1497 | 主合成算子（**通道数必须命中 `GSPLAT_NUM_CHANNELS`**） |
| `rasterize_to_pixels_sparse` | 1566 | 稀疏合成 |
| `rasterize_num_contributing_gaussians(_sparse)` | 1668 / 1710 | 逐像素贡献计数 + 累积 alpha |
| `rasterize_contributing_gaussian_ids(_sparse)` | 1776 / 1825 | 逐像素全部贡献 id 与权重 |
| `rasterize_top_contributing_gaussian_ids(_sparse)` | 1895 / 1943 | 逐像素 top-K（按辐射权重） |
| `rasterize_to_pixels_eval3d(_extra)` | 2263 / 2356 | 3D 空间高斯响应合成（3DGUT） |
| `rasterize_to_indices_in_range` | 2488 | 反向重放用的索引区间 |
| `fully_fused_projection_with_ut` | 2547 | UT 投影 |
| `fully_fused_projection_2dgs` / `rasterize_to_pixels_2dgs` / `rasterize_to_indices_in_range_2dgs` | 2635 / 2920 / 3156 | 2DGS |
| `adam` | 419 | 融合 Adam 单步（被 `SelectiveAdam` 调用） |

**(c) Python 侧 autograd**

原先在 C++ 里写的 autograd `Function` 已迁移到 Python：`_register_autograd(register)`（第 74 行）
用 `torch.library.register_autograd` 把 `<base>_bwd` 挂到 `<base>` 前向上，共注册 **12 组**
（`_ensure_autograd_registrations`，第 91–112 行）：

```
SphericalHarmonics, SphericalHarmonicsL0, SphericalHarmonicsL1Plus,
QuatScaleToCovarPreci, ProjectionEWASimple, ProjectionEWA3DGSFused,
ProjectionEWA3DGSPacked, Projection2DGSFused, Projection2DGSPacked,
RasterizeToPixels3DGS, RasterizeToPixels2DGS, RasterizeToPixelsSparse
```

这么做的收益：**直接调 `torch.ops.gsplat.*` 的用户也自动获得反向**，而不只是用
`gsplat.*` 封装的那条路径。`_has_schema` 保证被编译开关裁掉的算子不会注册失败。

### 6.2 参考实现（PyTorch oracle）

这 6 个文件是"数值真值"，CUDA kernel 的测试都拿它们做对比：

| 文件 | 行数 | 覆盖的 CUDA 算子 |
| --- | --- | --- |
| `_torch_impl.py` | 1,067 | `_persp_proj` / `_fisheye_proj` / `_ortho_proj` / `_world_to_cam` / `_fully_fused_projection` / `_isect_tiles` / `_isect_offset_encode` / `_isect_tiles_sparse` / `_build_sparse_tile_layout` / `accumulate` / `_rasterize_to_pixels(_sparse)` / `_spherical_harmonics` |
| `_torch_impl_ut.py` | 644 | UT 权重、sigma 点生成、`_fully_fused_projection_with_ut` |
| `_torch_impl_eval3d.py` | 871 | 射线生成、高斯变换、射线-高斯距离、`accumulate_eval3d` |
| `_torch_impl_2dgs.py` | 334 | 2DGS 投影 / 合成 / 射线变换 |
| `_torch_impl_lidar.py` | 394 | LiDAR tile 采样、`_isect_tiles_lidar` |
| `_math.py` | 791 | Horner 多项式求值/求逆（Newton）、`SafeNormalize`（自带与 CUDA 一致的反向）、四元数全套、`_quat_scale_to_covar_preci`、`compute_inverse_polynomial` |

`_math.py` 中的 `PolynomialProxy` / `FullPolynomialProxy` / `OddPolynomialProxy` /
`EvenPolynomialProxy` 是相机畸变多项式的类型化求值器，`_eval_poly_inverse_horner_newton`
用 Newton 迭代求逆——这套逻辑与 `sensors` 的 C++ 侧保持镜像。

### 6.3 相机与 LiDAR 的 Python 模型

- `_torch_cameras.py`（2,204 行）：`_BaseCameraModel.create(...)` 工厂 + 五个子类
  （`_PerfectPinhole` / `_Orthographic` / `_OpenCVPinhole` / `_OpenCVFisheye` / `_FTheta`），
  以及 `world_point_to_image_point_shutter_pose` 的卷帘迭代求解、
  `_compute_undistortion_newton` / `_compute_undistortion_iterative` 两条反畸变路径。
- `_torch_lidars.py`（374 行）：`_RowOffsetStructuredSpinningLidarModel` 的 Python 版。
- `_lidar.py`（882 行）：**参数 dataclass + tiling 算法**，与 CUDA 侧共享语义：
  `RowOffsetStructuredSpinningLidarModelParametersExt`（含 `to_cpp()`）、`LidarTiling`、
  `compute_angles_to_columns_map`、`angles_to_dense_ray_mask_cdf`、`angles_to_tile_indices`、
  `compute_tiles_to_elements_map`、`compute_histogram_equalization`、`compute_tiling`。
  这些是 LiDAR tile 划分（按角度-CDF 而非像素网格）的核心。
- `_torch_external_distortion.py`（159 行）：挡风玻璃双变量多项式的 Python 镜像 +
  `make_identity_horizontal_poly` / `make_identity_vertical_poly` / `make_params` 构造工具。

### 6.4 常量（`cuda/_constants.py`）

```python
ALPHA_THRESHOLD        = 1/255      # alpha 提前终止阈值
MAX_ALPHA              = 0.99       # 最大不透明度上限
TRANSMITTANCE_THRESHOLD = 1e-4      # = (1 - MAX_ALPHA)^2，保证反向数值稳定
MIN_COMPENSATION       = 0.005      # 反锯齿补偿因子下限
```

注释说明设计意图：让"最大不透明度高斯"需要被合成两次才达到阈值，同时不把透射率
压到反向不稳定。

---

## 7. CUDA 后端：C++ 编排层

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `Rendering.cpp` | 1,995 | **总编排**：`rasterization_3dgs`（:745）、`rasterization_2dgs`（:1705）、渲染模式/形状辅助、`torch` 注册 |
| `Rasterization.cpp` | 3,630 | 单算子 host 包装：`rasterize_to_pixels_3dgs_fwd/bwd`、稀疏版、`rasterize_to_indices_3dgs`、贡献查询、2DGS 全套、`rasterize_to_pixels_from_world_3dgs` |
| `Projection.cpp` | 2,220 | `projection_ewa_simple`、`projection_ewa_3dgs_fused/packed`、`projection_2dgs_fused/packed`、`projection_ut_3dgs_fused` |
| `SphericalHarmonics.cpp` | 1,114 | SH 前反向 host 包装 |
| `Intersect.cpp` | 822 | tile 求交 / offset 编码 / 稀疏求交 host 包装 |
| `Adam.cpp` / `Relocation.cpp` / `MCMCPerturb.cpp` / `GaussianLosses.cpp` / `Null.cpp` | 小 | 训练相关算子 host 包装 |
| `Utils.cpp` | — | 共享 host 工具 |
| `DistributedCollectives.cpp` | — | NCCL all-gather / all-to-all（分布式渲染 Seam A/B） |

### 7.1 `rasterization_3dgs()` 的阶段注释（源码里标了 `// ---`）

```
L745   函数签名（含 renderer_config / process_group_name / world_size）
L798   distributed = process_group_name.has_value()
L802   check_rasterization_3dgs_inputs(...)   ← 所有形状/组合合法性集中在此
L828   batch_ndim / N / C / B / I 推导；normalize_covars_for_3dgs；expand_rays_for_3dgs
L852   【Seam A】gather_cameras_for_distributed → 把别 rank 的相机收集过来
        （顺便把 C 改成全局相机数，I 仍是本地图像数）
L886   // --- Project Gaussians to 2D ---    UT 或 EWA，packed 走稀疏分支
L1022  // --- Prepare projected opacities --- 乘以 compensation（反锯齿）
L1051  // --- Prepare feature channels ---
L1059  // --- Fused fast-path ---             一次 kernel 完成 [SH颜色|直通extra|depth] 装配
L1202  记录 meta（**scatter 之前**的 rank-local 投影结果，策略按 gaussian_id 索引）
L1217  【Seam B】all-to-all scatter 到相机所属 rank；C 回到本地
L1264  // --- Append requested depth channel ---
L1298  // --- Identify intersecting tiles ---   AccuTile 打开
L1343  // --- Rasterize feature chunks ---       按 channel_chunk 分块
L1449  // --- Reassemble output channels ---
L1467  填充返回值
```

**关键设计点**：
1. **单一 C++ 编排器**：Python 只调一个 `rasterization_3dgs`，避免多次跨界与
   中间张量的 Python 侧生命周期管理开销。
2. **distributed 的边界很窄**：只支持 unbatched + pinhole + 无 UT/eval3d + 无畸变 +
   无卷帘 + 无 sparse_grad/absgrad + 逐高斯颜色（详见 `Rendering.cpp:187-234` 的
   一串 `TORCH_CHECK`）。校验在"分发限制"层先报错，让错误信息有指向性。
3. **meta 必须在 scatter 前快照**：否则策略会看到被 scatter 覆盖后的张量。

---

## 8. CUDA 后端：设备端 kernel

`gsplat/cuda/csrc/` 下按功能分组（排除 vendored `third_party/glm`）：

### 8.1 投影（Projection）

| 文件 | 说明 |
| --- | --- |
| `ProjectionEWASimple.cu` | 单算子投影（`proj`） |
| `ProjectionEWA3DGSFused.cu` | fused 投影（dense 输出），最常用 |
| `ProjectionEWA3DGSPacked.cu` | packed 投影（只输出可见高斯，COO） |
| `ProjectionUT3DGSFused.cu` | Unscented Transform 投影（3DGUT 前半） |
| `Projection2DGSFused.cu` / `Projection2DGSPacked.cu` | 2DGS 光线变换 |
| `QuatScaleToCovarCUDA.cu` | 四元数+尺度 → 协方差/精度 |
| `Projection.h` / `Projection2DGS.cuh` | host 声明与 device 数学 |

### 8.2 求交与排序（Intersect）

| 文件 | 说明 |
| --- | --- |
| `IntersectTile.cu` | 高斯×tile 求交；**AccuTile** 椭圆保守测试；radix/merge sort（`cub::DeviceRadixSort`） |
| `IntersectTileSparse.cu` | active tile only |
| `IntersectTileLidar.cu` | LiDAR 角度网格求交 |
| `SparseTileLayout.cu` | 稀疏 tile 布局：位掩码 + 前缀和 + 像素置换 |
| `PrimingChainEncoding.cu` + `PrimingChainEncoding.*` | 交集 id 的位打包（image | tile | depth） |

> 交集 id 是 **64 位**整数（`28e794ca [NV] Expand intersection offsets to 64-bit`），
> 位宽分配为 `image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits)`。

### 8.3 光栅化（Rasterize）

| 文件 | 说明 |
| --- | --- |
| `RasterizeToPixels3DGSSerialBatchFwd.cu` | 3DGS 前向：每 tile 一 CTA，前到后 alpha 合成 |
| `RasterizeToPixels3DGSSerialBatchBwd.cu` | 3DGS 反向 |
| `RasterizeToIndices3DGSSerialBatch.cu` | 反向的索引重放 |
| `RasterizeToPixels2DGS{SerialBatchFwd,SerialBatchBwd}.cu` | 2DGS 前反向 |
| `RasterizeToPixelsSparse{Fwd,Bwd}.cu` | 稀疏合成 |
| `RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu` | **eval3d 前向**：串行批（MixedBatch 用） |
| `RasterizeToPixelsFromWorld3DGSParallelBatch{Fwd,Bwd}.cu` | eval3d 批并行前向 + 共享反向 |
| `RasterizeNumContributingGaussians.cu` | 逐像素贡献计数 |
| `RasterizeContributingGaussianIds.cu` | 逐像素贡献 id 列表 |
| `RasterizeTopContributingGaussianIds.cu` | 逐像素 top-K |
| `RasterizeToPixels3DGSDevice.cuh` / `RasterizeToPixelsFromWorld3DGS.cuh` / `RasterizeCSR.cuh` | 共享 device 逻辑 |

`eval3d` 路径里 `ParallelBatch` 的前向是**三阶段流水**：partials kernel → scan → replay，
代价是必须持久化每批前向状态供反向使用。

### 8.4 其他

| 文件 | 说明 |
| --- | --- |
| `SphericalHarmonicsCUDA.cu` / `SphericalHarmonicsL1PlusCUDA.cu` / `SphericalHarmonicsViewDirectionCUDA.cu` | SH 前反向；**影像/视图方向融合**与 **l0 / l1+ 拆分**是 2026 年的优化项 |
| `CameraWrappers.cu` / `ExternalDistortionWrappers.cu` | Python 测试用相机封装（`BUILD_CAMERA_WRAPPERS`） |
| `ExternalDistortion.cuh` | 挡风玻璃双变量多项式 device 数学 |
| `Lidar.cpp` + `Lidars.cuh` / `Lidars.h` | LiDAR 模型 host 侧与 device 侧 |
| `AdamCUDA.cu` | 融合 Adam（支持 `valid` 掩码 → 选择性更新） |
| `RelocationCUDA.cu` | MCMC 重定位解析式 |
| `MCMCPerturbCUDA.cu` | MCMC 噪声注入（`inject_noise`） |
| `GaussianLossesCUDA.cu` | 4 个高斯正则损失的融合 kernel |
| `DistributedCollectives.cpp` | NCCL 集合通信 |
| `TensorView.h` | 多设备张量访问抽象（CUDA / PrivateUse1 共用） |
| `Dispatch.h` / `TypeList.h` | 编译期类型派发 |
| `include/Cameras.cuh`（1,991 行） | 投影/畸变/卷帘的 device 数学总库 |
| `include/Utils.cuh` / `MathUtils.h` / `Common.h` | 数值工具 |

---

## 9. 训练基础设施：strategy / 优化器 / 损失

### 9.1 策略抽象（`gsplat/strategy/`）

```python
# base.py:23
@dataclass
class Strategy:
    def check_sanity(self, params, optimizers) -> None: ...      # 校验 keys 一一对应、每组 1 个 param_group
    def step_pre_backward(self, *args, **kwargs) -> None: ...    # 反向前的钩子
    def step_post_backward(self, *args, **kwargs) -> None: ...   # 反向后的钩子
```

**参数契约**（`docs/source/apis/strategy.rst`）：`params` 是 `dict` 或 `nn.ParameterDict`，
至少含 `{"means", "scales", "quats", "opacities"}`；`optimizers` 是同键字典，**每个参数
一个 optimizer 且只有一个 param_group**。策略会**原地**替换 `params[name]` 里的
`Parameter` 并同步迁移 optimizer state（见 `ops._update_param_with_optimizer`，`strategy/ops.py:96`）。

#### `DefaultStrategy`（`strategy/default.py:32`，原版 3DGS + AbsGS）

| 字段 | 默认 | 语义 |
| --- | --- | --- |
| `prune_opa` | 0.005 | 不透明度低于此值剪枝 |
| `grow_grad2d` | 2e-4 | 图像平面梯度阈值（AbsGS 建议 8e-4） |
| `grow_scale3d` | 0.01 | 3D 尺度（按 scene_scale 归一）小于此值→复制，大于→分裂 |
| `grow_scale2d` / `prune_scale2d` | 0.05 / 0.15 | 2D 尺度阈值（默认关闭，见 `refine_scale2d_stop_iter=0`） |
| `prune_scale3d` | 0.1 | 3D 尺度上限 |
| `refine_start_iter` / `refine_stop_iter` / `refine_every` | 500 / 15000 / 100 | 致密化窗口与频率 |
| `reset_every` | 3000 | 每 N 步把不透明度重置成 `prune_opa*2` |
| `absgrad` | False | 用 `means2d.absgrad` 代替平均梯度 |
| `revised_opacity` | False | arXiv:2404.06109 的修正不透明度启发式 |
| `key_for_gradient` | `"means2d"` | 2DGS 用 `"gradient_2dgs"` |

执行流（`strategy/default.py:172`）：
```
step_post_backward:
  step >= refine_stop_iter → return
  _update_state(...)                      # 累加 grad2d / count / radii
  若 命中 refine 窗口:
      _grow_gs(...)  → duplicate(高梯度&小尺度) 然后 split(高梯度&大尺度)
      _prune_gs(...) → remove(低不透明度 | 过大)
      清零 running stats + empty_cache
  若 step % reset_every == 0: reset_opa(prune_opa * 2)
```

`step_pre_backward` 只做一件事：`info[key_for_gradient].retain_grad()`——因为要读梯度，
必须保证该中间张量保留 `.grad`。

#### `MCMCStrategy`（`strategy/mcmc.py:40`, 3DGS-MCMC）

| 字段 | 默认 | 语义 |
| --- | --- | --- |
| `cap_max` | 1e6 | 高斯数量上限 |
| `noise_lr` | 5e5 | 位置扰动噪声强度 = `lr * noise_lr` |
| `refine_start_iter` / `refine_stop_iter` / `refine_every` | 500 / 25000 / 100 | 重定位+增殖窗口 |
| `min_opacity` | 0.005 | 视为"死"的阈值 |
| `noise_injection_stop_iter` | -1（不停止） | 单独控制噪声停止（控制器蒸馏用） |
| `noise_opacity_t` / `noise_opacity_k` | 0.005 / 100 | 按不透明度抑制噪声的过渡点与陡度 |

执行流（`strategy/mcmc.py:139`）：
```
初始化：binoms (51×51 二项式系数表，供重定位解析式用)
step_post_backward:
  命中窗口 → _relocate_gs(死高斯迁到高不透明度处) → _add_new_gs(增殖到 1.05x，上限 cap_max)
  step < noise_stop → inject_noise_to_position(...)  # 优先走 fused CUDA kernel
```

`inject_noise_to_position`（`strategy/ops.py:463`）优先调用 `_cuda_fused_mcmc_perturb`
（`strategy/ops.py:405` → `torch.ops.gsplat.mcmc_perturb_positions`），不可用时回退纯 PyTorch。

#### 原地算子（`ops.py`）

| 函数 | 行 | 语义 |
| --- | --- | --- |
| `_update_param_with_optimizer` | 96 | **所有**拓扑变更的统一实现：替换 Parameter、迁移 Adam `exp_avg/exp_avg_sq`、保留 `step` |
| `duplicate` | 141 | 按 mask 复制行（optimizer state 追加零） |
| `split` | 175 | 按 mask 分裂：采样 2 个位置（`revised_opacity` 可切换不透明度公式） |
| `remove` | 238 | 按 mask 删除 |
| `reset_opa` | 271 | 把不透明度设为给定 post-sigmoid 值 |
| `relocate` | 302 | 死高斯 ← 按不透明度多项分布采样的活高斯；用 `compute_relocation` 算新不透明度/尺度 |
| `sample_add` | 360 | 追加采样的高斯行 |
| `inject_noise_to_position` | 463 | 位置噪声（含不透明度门控） |

**`scene` 钩子**：每个算子末尾都有 `if scene is not None: scene.on_*()`，让
`GaussianScene` 的 `signal` / `component_index` 与 `splats` 保持行对齐。

`relocation.py:25 compute_relocation()` 是 3DGS-MCMC 论文的解析公式（给定不透明度、
尺度、采样次数 `ratio` 与二项式表，返回新的 `(opacity, scale)`），独立成模块以便单测。

### 9.2 优化器（`gsplat/optimizers/selective_adam.py:21`）

`SelectiveAdam(torch.optim.Adam)`：`step(visibility=mask)` 只更新可见参数。
内部调用融合 CUDA kernel `gsplat::adam`（`cuda/csrc/AdamCUDA.cu`），一次 kernel 完成
一阶/二阶矩更新 + 参数写入。这是 Taming3DGS 提到的两种优化器之一。

### 9.3 损失（`gsplat/losses.py`，1,183 行）

全部是**函数式 API**：输入裸 Tensor，输出 Tensor；约定"逐元素/未归约"，由调用方或
`reduce_*` 辅助函数决定归约方式。

| 分组 | 函数 |
| --- | --- |
| 基础 | `l1_loss`、`mse_loss`、`huber_loss`、`smooth_l1_loss`、`log_l1` |
| 图像结构 | `create_ssim_window`、`torch_ssim_loss`、`ssim_loss`、`masked_l1`、`masked_ssim` |
| 深度 | `depth_l1_loss`（视差空间）、`binocular_disparity_l1`、`pearson_depth_loss`、`depth_inverse_mse` |
| LiDAR | `lidar_distance_loss`、`lidar_intensity_loss`、`lidar_raydrop_loss`、`lidar_background_loss` |
| 正则 | `total_variation_loss`、`total_variation_temporal`、`opacity_reg_loss`、`scale_reg_loss`、`weights_reg`、`relu_sum`、`identity_distance` |
| 高斯（G-SHARP） | `gaussian_scale_reg`、`gaussian_density_reg`、`gaussian_z_scale_reg`、`out_of_bound_loss`、`bilateral_grid_drift_loss` |
| 分类 | `bce_loss`、`bce_with_logits_loss`、`cross_entropy_loss`、`bce_clipped` |
| 几何 | `normal_cosine_loss` |
| 归约 | `reduce_mean` / `reduce_quantile` / `reduce_sum` |
| 调度 | `LinearLambdaScheduler`（损失权重线性插值） |
| 内部 | `_resolve_loss_fn` / `_apply_loss_fn` / `_validate_lidar_shapes` |

### 9.4 融合损失（`gsplat/losses_fused.py:32`）

`FusedGaussianLosses(nn.Module).forward(scales, densities, z_scales, positions, cuboid_dims, visibility)`
一次 kernel 算出 `(loss_scale, loss_density, loss_z_scale, loss_oob)` 四项。
分发条件：**编译期**有 `BUILD_LOSSES` + **运行期**所有输入都在同一 CUDA 设备上；
否则回退到纯 PyTorch 四项组合。注意 kernel 只支持 fp32/fp64（AMP 的 fp16/bf16 会报错）。

autograd 桥在 `cuda/_losses_wrapper.py:31 _FusedGaussianLosses`（`forward` 输出 4 个标量 +
`backward` 调 `gaussian_losses_bwd`）。

### 9.5 其它训练工具

| 模块 | 内容 |
| --- | --- |
| `regularizers.py` | `compute_tv_loss_targeted`（各向异性 TV + 掩码）、`dilate_mask`（max-pool 膨胀）、`create_invisible_mask`（手术器械/动态物体掩码并集） |
| `init_utils.py` | `multi_frame_depth_unprojection`（多帧掩码深度 → 世界点云 + 每点 RGB）、`knn_scale_init`（按 k 近邻平均距离给初始 log-scale） |
| `training/schedulers.py` | `TwoStageScheduler` + `TwoStageScheduleStep`：粗→细两阶段训练（G-SHARP 移植） |
| `color_correct.py` | `color_correct_affine` / `color_correct_quadratic`：评测时把渲染图颜色对齐到 GT（用于 `cc_psnr` 等指标） |

---

## 10. 场景与阶段抽象：scene / stage

### 10.1 `Scene` 基类（`scene/components/base.py:23`）

```python
class Scene(ABC):
    id: str                                        # Stage 上的唯一键
    def put(self, name, component) -> None: ...     # 抽象
    def get(self, component) -> object: ...         # 抽象
    # 拓扑钩子（默认全为 no-op）
    def on_duplicate(sel) / on_split(sel, rest) / on_remove(remove_mask)
    def on_relocate(dead_indices, sampled_indices) / on_sample_add(sampled_indices) / on_permute(order)
```

刻意**不含**共享存储、`names()`、`items()`、`Scene[T]` 泛型——比 mapping 接口窄得多。

### 10.2 `GaussianScene`（`scene/components/gaussian_scene.py:23`）

字段：
```python
scene.id                 # str，必填
scene.splats             # nn.ParameterDict：means/scales/quats/opacities/[colors|sh0+shN|features]
scene.signal             # dict[str, Tensor]：逐高斯旁路数据（不参与渲染，但要与行对齐）
scene.component_names    # list[str]
scene.component_index    # LongTensor[N]，每行恰好属于一个 component
```

API：

| 方法 | 行 | 说明 |
| --- | --- | --- |
| `put(name, splats)` | 33 | **仅初始化期可用**。首次直接存引用；后续按现有 keys 做 `torch.cat` 追加，并给已有 `signal` 补零对齐 |
| `from_splats(splats, id, signal=None)` | 92 | 从既有 `ParameterDict` 构造 |
| `validate()` | 109 | 必填 keys、行数一致、`component_index` 形状与取值范围。⚠️ 内部 `.item()` 会同步 CUDA，**不可放进 per-step 热路径** |
| `num_gaussians()` | 139 | |
| `get(component: str|int)` | 154 | 返回 `{name, index, mask, splats{...}, signal{...}}`（选取视图，非可写视图） |
| `state_dict()` / `from_state_dict()` | 165 / 180 | 完整序列化（含 `requires_grad` 标志、signal、component 簿记）；兼容缺失 `component_names`/`component_index` 的旧格式 |
| `on_duplicate` / `on_split` / `on_remove` / `on_relocate` / `on_sample_add` / `on_permute` | 223–266 | 与策略算子一一对应，保持 `signal` 与 `component_index` 行对齐 |

**重要权衡**（`scene/design.md`）：训练脚本的 checkpoint 目前只存 `splats + scene_id`，
**不存** `scene.state_dict()`，因此恢复后会丢失 `signal` 与多 component 簿记。若使用这两者，
需把保存处改成 `"scene": self.scene.state_dict()` 并用 `GaussianScene.from_state_dict` 读取。

### 10.3 `GaussianInferenceScene`（`scene/components/gaussian_inference_scene.py:32`）

推理专用：存**已激活并打包**的张量。

```python
scene.means_planar      # [3, N]  float32 CUDA（转置布局，便于 macro-tile 预取）
scene.qso_packed        # [N, 8]  float16（quat 4 + scale 3 + opacity 1）
scene.colors_packed     # 依 sh_degree / sh_compression_mode 变化
scene.sh_degree, scene.sh_compression_mode, scene.num_gaussians
scene.component_names / component_index
```

构造入口：
- `from_gaussian_scene(scene, ...)`（第 261 行）：从训练用 `GaussianScene` 自动完成激活
  （`exp(scales)` / `sigmoid(opacities)` / 四元数归一）与打包；
- `from_gaussian_tensors(...)`（第 382 行）：从已激活张量构造，会校验激活契约
  （`_check_activation_contract`，第 574 行）与 fp16 取值范围（`_check_fp16_range`，第 639 行）。

底层打包由 `gsplat_scene_cuda` 的 `pack_gaussian_inference_scene` 完成，
Python 包装在 `scene/kernels/gaussian_inference_ops.py:85`（前置完整校验）；
公开 functional 入口 `scene/functional/gaussian_inference.py:28`（懒加载）。

`SHCompressionMode`（`scene/sh_compression.py:21`）：`NONE=0` / `PACKED_32B=1` / `PACKED_16B=2`，
控制 `colors_packed` 的布局（**打包**，不是量化——真正的 YCoCg 量化在渲染 kernel 的 COMPRESS 模式里）。

### 10.4 `Stage`（`stage/components/stage.py:23`）

极简路由层：`dict[scene_id] → (scene, render_fn)`。

```python
stage.add_scene(scene, render_fn)     # scene.id 重复 → ValueError
stage.scene_ids() -> list[str]
stage.get_scene(scene_id) -> GaussianScene
stage.render(scene_id, **kwargs)      # 实际调用 render_fn(splats=scene.splats, **kwargs)
```

**只做三件事**：持有 scene 引用、绑定 render_fn、按 id 分发。不算 loss、不建 optimizer、
不采样相机、不组合多场景输出、不编排策略钩子。`splats` 一律以**关键字**传入，
避免与 `rasterize_splats(self, camtoworlds, ...)` 这类首参不是 splats 的函数冲突。
返回值原样透传（3DGS 3 元组 / 2DGS 7 元组都行）。

---

## 11. 传感器库 `gsplat/sensors`

严格三层，设计契约见 `gsplat/sensors/design*.md`。

```
gsplat/sensors/
├── functional/          # L1 公开无状态 API（唯一产生命名返回类型的地方）
│   ├── cameras.py       # 8 个相机 op
│   ├── lidars.py        # 5 个 LiDAR op
│   └── return_types.py  # 8 个 return dataclass（单一真源）
├── kernels/             # L0 后端：CUDA 源码 + autograd glue + 分发表
│   ├── projective_sensor_ops.py   # 相机 (projection, distortion) 分发表
│   ├── cameras/{ops,types,windshield}.py
│   ├── lidars/{dispatch,ops,types}.py
│   ├── common/{pose,utils}.py     # Pose/DynamicPose/Trajectory + wxyz↔xyzw
│   └── cuda/            # gsplat_sensors_cuda（26,584 行原生代码）
└── models/              # L2 nn.Module 封装
    ├── cameras/{camera_model,image_frame}.py
    ├── lidars/{lidar_model,lidar_frame}.py
    └── common/{frame,utils}.py
```

### 11.1 服务范围

**相机**：3 种投影 × 2 种外畸变 = 6 个派发对

| 投影 | 参数要点 |
| --- | --- |
| `OpenCVPinholeProjection` | focal_length(2), principal_point(2), radial(6), tangential(2), thin_prism(4) |
| `FThetaProjection` | reference_poly, pixeldist_to_angle / angle_to_pixeldist 多项式, max_angle |
| `OpenCVFisheyeProjection` | focal_length(2), principal_point(2), radial(4) |

| 外畸变 | 参数 |
| --- | --- |
| `NoExternalDistortion` | 空（**必须显式传入实例，`None` 会 `TypeError`**） |
| `BivariateWindshieldDistortion` | `(42,)` 系数：`[h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)]`，h 度 ≤2、v 度 ≤4 |

**快门类型**：`GLOBAL` / `ROLLING_TOP_TO_BOTTOM` / `ROLLING_LEFT_TO_RIGHT` /
`ROLLING_BOTTOM_TO_TOP` / `ROLLING_RIGHT_TO_LEFT`（C++ `shutter_type.h` 为唯一真源，
Python `IntEnum` 在 import 时逐值校验，漂移则拒绝加载）。

**LiDAR**：`RowOffsetStructuredSpinningLidarProjection`（每行俯仰角 + 每列方位角 +
可选每行方位偏移 + 垂直/水平 FOV + 旋转方向）。

### 11.2 13 个公开算子

| 相机 op（`functional/cameras.py`） | 行 | 返回 |
| --- | --- | --- |
| `camera_rays_to_image_points` | 44 | `ImagePointsReturn` |
| `image_points_to_camera_rays` | 77 | `Tensor (N,3)` |
| `project_world_points_mean_pose` | 109 | `WorldPointsToImagePointsReturn` |
| `project_world_points_shutter_pose` | 191 | `WorldPointsToImagePointsReturn`（含 Newton 卷帘求解，上限 12 次迭代） |
| `image_points_to_world_rays_static_pose` | 294 | `WorldRaysReturn` |
| `image_points_to_world_rays_shutter_pose` | 355 | `WorldRaysReturn` |
| `pixel_grid_to_world_rays_shutter_pose` | 431 | `WorldRaysReturn`（内部先建像素栅格） |
| `generate_image_points` | 506 | `Tensor (H,W,2)`（唯一允许非 CUDA device 的 op） |

| LiDAR op（`functional/lidars.py`） | 行 | 返回 |
| --- | --- | --- |
| `sensor_rays_to_sensor_angles` | 41 | `SensorAnglesReturn` |
| `sensor_angles_to_sensor_rays` | 64 | `SensorRayReturn` |
| `elements_to_sensor_angles` | 87 | `SensorAnglesReturn` |
| `generate_spinning_lidar_rays` | 120 | `WorldRaysReturn` |
| `inverse_project_spinning_lidar` | 179 | `WorldPointsToSensorAnglesReturn` |

### 11.3 返回类型（`functional/return_types.py`）

| Dataclass | 必需字段 | 可选字段（由 `return_*` 开关控制） |
| --- | --- | --- |
| `ImagePointsReturn` | `image_points (N,2)`, `valid_flag (N,)` | `jacobians (N,2,3)` |
| `WorldPointsToImagePointsReturn` | `image_points (N,2)` | `T_sensor_world (N,4,4)`, `valid_flag`, `valid_indices (M,)`, `timestamps_us (N,)` |
| `WorldRaysReturn` | `world_rays (N,6)`（打包 origin+direction） | `T_sensor_world`, `timestamps_us` |
| `PixelsReturn` | `pixels (N,2) int`, `valid_flag` | — |
| `WorldPointsToPixelsReturn` | `pixels (N,2) int` | 同 `WorldPointsToImagePointsReturn` |
| `SensorAnglesReturn` | `sensor_angles (N,2)` | `valid_flag` |
| `SensorRayReturn` | `sensor_rays (N,3)` | `valid_flag` |
| `WorldPointsToSensorAnglesReturn` | `sensor_angles (N,2)` | 同上可选集 |

`valid_flag` / `valid_indices` / `timestamps_us` 是**不可微**的状态输出。

### 11.4 分发表与一致性保障

```python
# kernels/projective_sensor_ops.py:221
_DISPATCH_TABLES = {
    "camera_rays_to_image_points": {...6 项...},
    "image_points_to_camera_rays": {...},
    "project_world_points_mean_pose": {...},
    "project_world_points_shutter_pose": {...},
    "image_points_to_world_rays_static_pose": {...},
    "image_points_to_world_rays_shutter_pose": {...},
    "pixel_grid_to_world_rays_shutter_pose": {...},
}
```

每张表必须覆盖 `REGISTERED_CAMERA_PROJECTIONS × REGISTERED_DISTORTIONS`（3×2=6）。
`kernels/test_projective_sensor_ops.py::test_dispatch_table_keys_match_registered_pairs`
会遍历并断言——**新增一个注册类型却漏填任一张表，测试直接失败**。
LiDAR 同理（`kernels/lidars/dispatch.py:38-50` 的 5 张单键表，由 `test_lidar_dispatch.py` 守护）。

因为 `torch.classes.*` 实例的 Python 类名统一是 `CustomClassHolder`，`_lookup()`（第 245 行）
改为**匹配注册字符串名**（`script_class_name`）而不是 `isinstance`。

### 11.5 C++ 三层表示（严格隔离）

| 层 | 文件 | 编译器 | 允许的 include |
| --- | --- | --- | --- |
| Torch host | `*_torch.h/.cpp` | host C++ | `*_params.h` + ATen / `torch::CustomClassHolder` |
| 桥接 POD | `*_params.h` | 两者 | `<cstdint>` + 兄弟 `*_params.h` + 前置声明的 `cudaStream_t` |
| CUDA device | `*_kernel.cuh/.cu` | nvcc | `*_params.h` + 其它 `.cuh` + `<cuda_runtime.h>` + geometry 头 |

**为什么要隔离**：让 nvcc 编译 kernel 时完全不必拉入 ATen 的模板图。
唯一的缝是两个声明：opaque 的 `CUstream_st` 前置声明（别名 `cudaStream_t`）和 POD 参数结构。

每个注册类型的三种形态（命名不完全统一，这是有意的）：

| 类型 | Torch host | 桥接 POD | 每线程 |
| --- | --- | --- | --- |
| OpenCV pinhole | `OpenCVPinholeProjection` | `OpenCVPinholeProjection_KernelParameters` | `OpenCVPinholeParams` |
| FTheta | `FThetaProjection` | `FThetaProjection_KernelParameters` | `FThetaParams` |
| OpenCV fisheye | `OpenCVFisheyeProjection` | `OpenCVFisheyeProjection_KernelParameters` | `OpenCVFisheyeParams` |
| 无外畸变 | `NoExternalDistortion` | `NoExternalDistortion_KernelParameters`（空） | `NoExternalDistortion_Parameters`（空 tag） |
| 挡风玻璃 | `BivariateWindshieldDistortion` | `BivariateWindshieldDistortion_KernelParameters` | `BivariateWindshieldParams` |
| 旋转 LiDAR | `RowOffsetStructuredSpinningLidarProjection` | `..._KernelParameters` | 直接表/POD 访问 |

### 11.6 注册规模

- 相机：**6 个需要派发的 op × 6 个 (projection, distortion) 对 = 36 个前向 + 36 个反向 = 72 个** torch op
  （命名如 `camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward`）；
  外加不派发的 `generate_image_points`，相机侧共 **73 个**。
- `pixel_grid_to_world_rays_shutter_pose` **没有**自己的 torch op：它在
  `sensors/kernels/cameras/ops.py:4947` 里先调 `generate_image_points(...).reshape(-1, 2)`，
  再复用 `_image_points_to_world_rays_shutter_pose_impl`（见 `sensors/kernels/cameras/ops.py:4995-5000`）。
- LiDAR：5 个 op + 5 个反向 = **10 个**（无畸变轴，所以名字不加后缀）。
- 合计 `gsplat_sensors` 命名空间下 **83 个** torch op（与 `ext.cpp` 中 83 处 `m.def` 一致）。
- `camera_params.h` 中有 **72 个** `_launch` 原型（36 前向 + 36 反向 + `generate_image_points_launch`）。

### 11.7 scratch 反向契约

相机前向的返回元组**最后一项是 `scratch` 张量**（形状 `(N,K)`，K 逐算子不同，例如
`camera_rays_to_image_points` 无外畸变是 `(N,6)`，快门位姿是 `(N,14)`）。
`torch.autograd.Function` 保存 `(主输入, scratch)`，反向直接复用 scratch，
**不重跑 OpenCV 畸变数学或卷帘 Newton 迭代**。

梯度裁剪通过 `ctx.needs_input_grad` → C++ 侧每个输入的 `need_*_grad` 布尔；
未请求的梯度传 `nullptr`（host 侧补空 `(0,)` 张量），避免无谓写显存。

### 11.8 模型层

`CameraModel`（`models/cameras/camera_model.py:126`）是**单类覆盖所有投影**（靠分发表，
不靠继承）。构造：`projection`（非 None）、`external_distortion`（非 None）、
`resolution=(w,h)`、`shutter_type`。因为 TorchScript 类实例不是 `nn.Module` 的子模块，
它**重写 `_apply`** 让 `.to()/.cuda()` 也能搬运投影内部的张量。

方法按"域 × 位姿风格"分组（static / mean / shutter）：

| 域 | 方法 |
| --- | --- |
| 世界点 → 图像点 | `world_points_to_image_points_{static,mean,shutter}_pose` |
| 世界点 → 像素 | `world_points_to_pixels_{static,mean,shutter}_pose` |
| 图像点 → 世界射线 | `image_points_to_world_rays_{static,mean,shutter}_pose` |
| 像素 → 世界射线 | `pixels_to_world_rays_{static,mean,shutter}_pose` |
| 相机射线 ↔ 图像点 | `camera_rays_to_image_points`, `image_points_to_camera_rays` |
| 相机射线 ↔ 像素 | `camera_rays_to_pixels`, `pixels_to_camera_rays` |
| 像素 ↔ 图像点 | `pixels_to_image_points`（`+0.5`）, `image_points_to_pixels`（`floor→int32`） |
| 时序 | `image_points_relative_frame_times` |
| 图像域变换 | `transform(scale, offset, new_resolution)` → **新实例**（不原地改） |

`return_jacobians=True` 时会在 detached 叶子上跑投影并用 `torch.autograd.grad` 求
`(N,2,3)` 雅可比，返回的 `image_points` 是 **detached** 的（避免雅可比图泄漏给调用方）。

`LidarModel`（`models/lidars/lidar_model.py:60`）同理，另有 `elements_to_sensor_points`、
`valid_sensor_angles`、`sensor_angles_relative_frame_times`（模型层的线性最近列回退）。
两者都**不实现 `forward()`**，调用方必须用显式方法。

帧容器：`Frame`（`models/common/frame.py:34`）→ `ImageFrame`（buffer 化 `image (H,W,C)`）
/ `LidarFrame`（`distance_m`/`intensity` buffer；稠密 `(H,W,R)` 或稀疏 `(N,R)`+`model_element`；
无效返回用距离 `NaN` / 强度 `0.0` 编码）。`Frame._apply` 会把 device/dtype 变更
传播到 `Pose`/`DynamicPose` 的普通属性张量（它们不是 `Parameter`，`nn.Module._apply` 本来会漏掉）。

`allow_device_transfer` 契约：**默认严格**，任何非目标 CUDA 设备/精度输入会抛 `RuntimeError`；
设为 `True` 才允许一次隐式 `.to()`（代价是一次主机同步）。

---

## 12. 几何库 `gsplat/geometry`

三层结构（`functional/` → `kernels/` → `kernels/cuda/`），公开面只是 `functional/`。

### 12.1 四元数算子（12 个）

| 函数 | 说明 |
| --- | --- |
| `quat_normalize_safe` | 安全归一化（零模长返回安全值） |
| `quat_conjugate` / `quat_inverse` | 共轭 / 逆（逆在 Python 侧复用共轭） |
| `quat_multiply` | Hamilton 积 |
| `quat_rotate_vector` | 用四元数旋转向量 |
| `quat_to_matrix` | → 3×3 旋转矩阵 |
| `quat_slerp` / `quat_lerp` / `quat_manifold_interp` | 球面 / 线性 / 流形插值 |
| `quat_from_axis_angle` | 轴角 → 四元数 |
| `quat_angular_distance` | 角距离 |
| `quat_identity` | 生成单位四元数（纯 Python，`geometry/kernels/quaternion_ops.py:712`） |

### 12.2 位姿与轨迹算子（13 个：8 个 `se3pose_*` + `se3_interpolate_tracks` + 3 个 `trajectory_*` + `frame_transform_poses_tquat`）

`se3pose_transform_point/_direction`、`se3pose_inverse_transform_point/_direction`、
`se3pose_to_matrix`、`se3pose_to_inverse_matrix`、`se3pose_from_matrix`、`se3pose_compose`、
`se3_interpolate_tracks`、`trajectory_transform_point_1pose`、
`trajectory_transform_point_2poses`、`trajectory_get_rotation_2poses`、
`frame_transform_poses_tquat`。

- 前向/反向入口命名：`<op>` / `<op>_bwd`（**pybind11 直接导出**，不是 torch op）。
- 公开 API 校验在**kernel wrapper 层**（`quaternion_ops.py` / `pose_ops.py`）完成，
  即使 C++ 也有 `TORCH_CHECK`——这样用户拿到的是明确的 `TypeError`/`ValueError`，
  而不是 kernel 启动后的不透明错误。
- 内部实现：`quaternion.cuh` 提供全部 device 标量助手（`quat_multiply_impl`、
  `quat_rotate_vector_*_impl`、`quat_slerp_pair_*`、`quat_to_matrix_fwd_write`、
  `quat_normalize_safe_*_write`）；`pose.cuh` **include** `quaternion.cuh` 复用，
  不重复实现，并补充 Shepperd 矩阵→四元数、SE3 变换、轨迹编排。
- `coordinate_conversions.cuh` 提供 `sincos_t` 与球坐标↔笛卡尔转换，被 `sensors` 的
  LiDAR kernel **跨包直接 include**（构建时把 geometry 的 `csrc` 加进 include path）。
- `se3_interpolate_tracks` 有两套 slerp 反向变体：`..._bwd`（带时间梯度，供几何轨迹用）
  与 `quat_slerp_pair_bwd_no_time_grad`（供传感器用，因为时间戳不可微）。

`design.md` 中明确的原生布局规范：一个 `.cuh` 对应一个同名 `.cu`；`.cuh` 放 device
助手与 `*_device` 主体，`.cu` 放 `__global__`、launch、`AT_DISPATCH_FLOATING_TYPES`
与导出入口。`quaternion.cuh` 里的 `QuatNormEps` 等常量也定义在此。

---

## 13. 推理渲染（HiGS）：`gsplat/experimental`

只前向、不参与训练的低延迟渲染路径。基于 HiGS（Hierarchically Tiled Gaussian Splatting）：
**macro-tile 融合光栅化 + fp16 场景打包 + 分层分块**。

### 13.1 公开面

```python
from gsplat.experimental import (
    GaussianInferenceScene,          # ← 实际来自 gsplat.scene
    GaussianInferenceRenderer,       # 有状态渲染器（缓存 buffer）
    RenderReturn,                    # dataclass: frame + metadata
    rasterize_gaussian_inference_scene,   # 无状态低层入口
    render_scene,                    # 带类型检查的通用入口
)
```

顶层 `__init__.py` 用 `__getattr__` 做**按名懒加载**，避免 import 就触发编译。

### 13.2 两个入口

| 入口 | 位置 | 特点 |
| --- | --- | --- |
| `render_scene(scene, *, out=None, **request)` | `experimental/render/functional/render_scene.py:33` | 校验是 `GaussianInferenceScene`，转发后在 metadata 里打 `render_path="inference"` |
| `rasterize_gaussian_inference_scene(scene, *, out, **request)` | `experimental/render/functional/gaussian_inference.py:292` | 直接低层调用，7 步校验流程 |

`rasterize_gaussian_inference_scene` 的 7 步（源码注释顺序）：
1. 类型检查 → 2. 空场景守卫（`is_empty()`）→ 3. 梯度模式门禁（`torch.inference_mode()` 或
禁用 autograd，见 `_common.check_inference_grad_mode`）→ 3b. 设备一致性 →
4. 请求子集校验（白名单 `_INFERENCE_ACCEPTED_KWARGS` / 黑名单
`_INFERENCE_UNSUPPORTED_FEATURES` 各 27 项）→ 5. 相机归一化（`viewmat`/`viewmats`、`K`/`Ks`）→
6. `out=` buffer 校验（形状/dtype/device/连续性）→ 7. 构建输出视图。

**明确不支持**（传入即 `TypeError`）：`with_ut`、`with_eval3d`、`absgrad`、`sparse_grad`、
`distributed`、`packed`、`segmented`、`return_normals`、`covars`、`rays`、所有畸变/卷帘参数、
`extra_signals`、`rasterize_mode`、`channel_chunk`、`global_z_order`、`ut_params`、`colors`。

### 13.3 有状态渲染器

`experimental/render/components/gaussian_inference_renderer.py:81` `GaussianInferenceRenderer`：
缓存每场景/每帧的 GPU buffer，支持上下文管理器、`release()`、`num_gaussians()`、
`__del__` 自动释放，以及**原生 half4 输出 buffer 复用**（`_validate_half4_out_buffer`）。

### 13.4 原生实现

扩展 `experimental_gaussian_render_inference_scene_cuda`，op
`experimental::gaussian_render_inference_only`。关键源文件：

| 文件 | 作用 |
| --- | --- |
| `MacroTileIntersect.cu` | macro-tile 级求交 |
| `MacroTileRasterize.cu` | **macro-tile 融合光栅化 kernel**（974 行，索引热点 top-8） |
| `IntersectMTFused.cu` | 融合求交 |
| `SegmentedSort.cu` | 分段排序（1,120 行） |
| `SphericalHarmonics.cu` + `SHCompression.cu` | SH 求值与 YCoCg 量化压缩 |
| `Projection.cu` | 投影 |
| `GaussianRenderInferenceScene.cu` | 场景状态与调度 |
| `Constants.h` / `InferenceTypes.h` / `IntersectMTConfig.h` | 配置与类型 |

反向被**显式禁用**：`ext.cpp:77` 注册 `TORCH_LIBRARY_IMPL(experimental, Autograd, m)` 并
`makeFallthrough()`，防止 PyTorch 2.x 自动包一层 `grad_fn` 让调用方的 `grad_fn is None`
检查失效。

---

## 14. 工具模块：压缩 / 导出 / 分布式 / 性能剖析

### 14.1 压缩（`gsplat/compression/`）

`PngCompression`（`compression/png_compression.py:31`）—— 论文：Self-Organizing Gaussian Grids。

```
compress(dir, splats):
  1. means → log_transform；quats → 归一化
  2. 裁到完全平方数（n_sidelen = sqrt(n_gs)），丢弃最低不透明度的余数
  3. use_sort → sort_splats()（PLAS 并行线性分配排序，让相邻高斯在空间上相邻）
  4. 逐参数选压缩器：
       means     → _compress_png_16bit   (16 位量化 + PNG)
       scales    → _compress_png         ( 8 位量化 + PNG)
       quats     → _compress_png
       opacities → _compress_png
       sh0       → _compress_png
       shN       → _compress_kmeans      (KMeans 聚类 + 质心/标签存 npz)
       其它       → _compress_npz
  5. meta.json 记录每参数的压缩方式
```

解码是镜像的 `decompress()`。文档给出的效果：100 万高斯 236 MB → 16.5 MB，
PSNR 仅从 29.18 降到 28.65 dB。

依赖（可选）：`imageio`、`PLAS`、`torchpq`。

### 14.2 导出（`gsplat/exporter.py`, 666 行）

| 函数 | 行 | 说明 |
| --- | --- | --- |
| `export_splats(...)` | 588 | **主入口**：校验形状 → 过滤 NaN/Inf 行 → 按 `format` 分派 |
| `splat2ply_bytes` | 378 | 标准 PLY（几乎所有 viewer 支持） |
| `splat2ply_bytes_compressed` | 209 | 压缩 PLY（Supersplat viewer） |
| `splat2splat_bytes` | 533 | `.splat` 格式（antimatter15 viewer） |
| `load_ply_to_splats` | 435 | **反向**：读 PLY 回参数张量 |
| `sh2rgb` | 25 | SH DC → RGB |
| `encode_morton3_vec` / `sort_centers` | 56 / 71 | Morton 码 + 空间排序（压缩前预处理） |
| `pack_unorm` / `pack_111011` / `pack_8888` / `pack_rotation` | 103–162 | 定点打包（11/10/11、8×4、四元数 32 位） |

`format ∈ {"ply", "splat", "ply_compressed"}`；`save_to` 可选，返回 bytes 供 HTTP 直接下发。

### 14.3 分布式（`gsplat/distributed.py`, 375 行）

| 函数 | 行 | 说明 |
| --- | --- | --- |
| `all_gather_int32` | 25 | 收集一个 int32 |
| `all_to_all_int32` | 70 | 多对多交换 int32 |
| `all_gather_tensor_list` | 117 | 收集不等长张量列表 |
| `all_to_all_tensor_list` | 185 | 交换不等长张量列表 |
| `cli(fn)` | 319 | 把函数包成分布式入口（自动找空闲端口、spawn worker） |
| `_find_free_port` / `_distributed_worker` | 275 / 287 | 辅助 |

这些是 `rasterization(distributed=True)` 之外的**训练侧**集合通信工具；
渲染内部的 gather/scatter 走 `cuda/csrc/DistributedCollectives.cpp`。

### 14.4 性能剖析（`gsplat/profile.py` + `trace.py`）

**`trace.py`（179 行）**：轻量 NVTX 标注。

```python
trace_range("name", **kwargs)        # 上下文管理器
trace_function("render2D-fwd")       # 装饰器（_wrapper.py 里大量使用）
```

优雅降级：没有 NVTX 时是 no-op；`nvtx.push_range` 不支持的 kwargs 会被过滤掉而不是抛错；
按 domain 探测 API 完整性（`_get_valid_gsplat_nvtx_domain`）。

**`profile.py`（1,657 行）**：独立的 **replay harness**。

```bash
# 1) 训练时捕获输入（环境变量开启）
TIMEIT=1 ...           # timeit 计时
# capture_inputs 装饰器：envvar 设置时把某次调用的输入 dump 成 .pt
# 2) 离线 replay + nsys
nsys profile --capture-range=cudaProfilerApi \
  python -m gsplat.profile --input gsplat_inputs.pt --warmup 5 --iterations 20
```

CLI 选项（`profile.py:1193-1340`）：`--input`（必需）、`--warmup`、`--iterations`、
`--grad/--nograd`、`--sync`、`--describe`、`--ensure-rays`/`--no-rays`、
`--losses`、`--channels`、`--renderer-config` 等。

内部机制：
- `_select_replay_inputs`：把通用输入字典投影到单个算子的关键字契约；
- `_apply_3dgs_replay_preset` / `_apply_3dgut_replay_preset`：强制走经典 3DGS / 3DGUT 路径；
- `_expected_kernel_families`：列出该工作负载**应当**出现的 kernel 族（用于判定 profile 是否走了预期分支）；
- `_build_profile_loss_contribution`：可选地把 CUDA 损失并入被测区间；
- `_clear_replay_grads` / `_detach_for_capture`：隔离 autograd 图；
- `main()` 是唯一被索引标记为 entry point 的函数。

⚠️ `profile.py:160` 有一条重要注释：**模块顶层不得 import gsplat**，否则会触发 JIT 编译，
而 `capture_inputs` 的使用者（`rendering.py`）只想拿装饰器，不想付出编译代价。
所有 gsplat import 都在函数内部惰性进行。

---

## 15. 动态高斯（G-SHARP 移植）：`gsplat/contrib/dynamic`

`gsplat.contrib` 整体标注为**实验性**，API 可能随版本变化。

| 模块 | 类/函数 | 行 | 作用 |
| --- | --- | --- | --- |
| `hexplane.py` | `HexPlaneField(nn.Module)` | 159 | 多分辨率 6 平面 4D `(x,y,z,t)` 特征场（K-Planes / 4DGaussians 形式） |
| | `_normalize_aabb` / `_grid_sample_wrapper` / `_init_grid_param` / `_interpolate_ms_features` | 48–114 | AABB 归一化、网格采样、参数初始化、多尺度插值 |
| | `.forward(pts)` → 特征 | 237 | |
| | `.spatial_planes()` / `.temporal_planes()` | 277 / 285 | 供正则化使用的平面列表 |
| `deformation.py` | `DeformNetwork(nn.Module)` | 49 | 消费 HexPlane 特征、输出 means/quats/opacities 增量的 MLP |
| | `DeformationTable` | 165 | 每高斯的动态/静态布尔表；`set_indices` / `prune` / `duplicate` / `split` 与策略算子同步 |
| `regulation.py` | `plane_smoothness` / `time_smoothness` / `time_l1` / `hexplane_regularization` | 78–138 | 平面二阶差分平滑、时间平滑、时间 L1 偏离 1.0、聚合包装 |
| `strategy.py` | `DynamicStrategy(DefaultStrategy)` | 50 | 扩展 `initialize_state`（带每高斯动态标记）；`step_post_backward` 中把动态表同步给策略算子 |

端到端示例是 `examples/dynamic_surgical_trainer.py`（775 行，包含
`build_splats_from_parser` / `build_deform_modules` / `_deform_with_mask` / `train_step` /
`render_frame` / `render_gif` / `train`）。

`_deform_with_mask` 是关键设计：**只有被标记为 dynamic 的高斯**才走 HexPlane + DeformNet，
静态高斯保持原值，避免无意义的 MLP 计算。

---

## 16. 示例与训练脚本

`examples/`（28 个文件，12,823 行）既是使用示例，也是事实上的集成测试目标
（`tests/test_dataset_colmap.py`、`tests/test_av_example.py`、`tests/test_dynamic_surgical_trainer.py` 都直接 import 它们）。

### 16.1 主要训练脚本

| 脚本 | 行数 | 定位 |
| --- | --- | --- |
| `simple_trainer.py` | 1,627 | **主力 3DGS 训练器**：COLMAP / NCore 数据、Default & MCMC 策略、3DGUT、外观优化、PPISP、压缩、导出、轨迹渲染 |
| `simple_trainer_2dgs.py` | 1,078 | 2DGS 训练器（7 元组渲染输出） |
| `av_trainer.py` | 1,272 | 自动驾驶多相机：NCore v4（FTheta + 3DGUT）或 PandaSet NPZ（pinhole）；含 LiDAR 渲染损失 |
| `dynamic_surgical_trainer.py` | 775 | 动态手术场景（G-SHARP 全套模块的装配示范） |
| `image_fitting.py` | 204 | 最小可运行示例：随机高斯拟合一张图 |
| `sample_inference.py` | 319 | 用 `Stage` + `GaussianScene` 从 checkpoint 渲染多视角画廊 |

### 16.2 `simple_trainer.py` 结构

```
Config (L79)                       # tyro 数据类，~60 个字段，见 §16.3
adjust_steps(factor) (L267)        # 按 steps_scaler 缩放 eval/save/ply 步数
create_splats_with_optimizers (L292)
Runner (L385)
  ├─ __init__ (L388)               # 数据加载（colmap / ncore）、GaussianScene、Stage、策略、损失
  ├─ freeze_gaussians (L640)       # 控制器蒸馏时冻结高斯参数
  ├─ rasterize_splats (L655)       # ← Stage 绑定的 render_fn（接受 splats kwarg）
  ├─ train (L801)                  # 主循环
  ├─ eval (L1211)                  # 指标：PSNR / SSIM / LPIPS (+ cc_* 颜色校正指标)
  ├─ render_traj (L1313)           # interp / ellipse / spiral / raw 四种轨迹
  ├─ export_ppisp_reports (L1391)  # 导出 PPISP 可视化 PDF + 参数 JSON
  ├─ run_compression (L1422)       # PngCompression 封装
  └─ _viewer_render_fn (L1439)
main (L1513)
```

`rasterize_splats`（L655）是理解数据流的样板：

```python
means     = splats["means"]                  # 原始参数
quats     = splats["quats"]                  # 不归一化，rasterization 内部会归一
scales    = torch.exp(splats["scales"])      # ← 激活
opacities = torch.sigmoid(splats["opacities"])  # ← 激活
colors    = torch.cat([splats["sh0"], splats["shN"]], 1)   # [N,K,3]
# 或 app_opt 路径：app_module(features, embed_ids, dirs) + colors → sigmoid
render_colors, render_alphas, info = rasterization(
    ..., viewmats=torch.linalg.inv_ex(camtoworlds).inverse,
    packed=self.cfg.packed,
    absgrad=(strategy.absgrad if isinstance(strategy, DefaultStrategy) else False),
    distributed=self.world_size > 1,
    **kwargs,
)
# 可选后处理：bilateral_grid / ppisp
```

注意 **`camtoworlds` → `viewmats` 的求逆方向**，以及 **激活函数留在 render_fn 里**
（Stage 与 GaussianScene 都不做激活——这是 `stage/design.md` 的明确决定）。

### 16.3 `Config` 关键字段

| 分组 | 字段 |
| --- | --- |
| 数据 | `data_type`(colmap/ncore)、`data_dir`、`data_factor`、`test_every`、`patch_size`、`cache_images`、`normalize_world_space`、`global_scale` |
| NCore | `ncore_camera_ids`、`ncore_lidar_ids`、`ncore_seek_offset_sec`、`ncore_duration_sec`、`ncore_max_lidar_points`、`ncore_*_component_group` |
| 训练 | `max_steps`、`batch_size`、`steps_scaler`、`eval_steps`、`save_steps`、`ply_steps`、`backend`(cuda/dgx) |
| 初始化 | `init_type`(sfm/random)、`init_num_pts`、`init_extent`、`init_opa`、`init_scale` |
| 外观 | `sh_degree`、`sh_fp16`、`sh_degree_interval`、`ssim_lambda`、`app_opt*`、`post_processing`(bilateral_grid/ppisp)、`ppisp_*` |
| 策略 | `strategy`（`DefaultStrategy` 或 `MCMCStrategy` 实例，直接从命令行 `--strategy.cap-max` 覆盖） |
| 渲染 | `packed`、`sparse_grad`、`antialiased`、`near_plane`、`far_plane`、`camera_model`、`with_ut`、`with_eval3d`、`visible_adam`、`random_bkgd` |
| 学习率 | `means_lr`、`scales_lr`、`opacities_lr`、`quats_lr`、`sh0_lr`、`shN_lr` |
| 正则 | `opacity_reg`、`scale_reg`、`depth_loss`/`depth_lambda` |
| 位姿/外观优化 | `pose_opt*`、`pose_noise`、`app_opt*` |
| 输出 | `result_dir`、`disable_viewer`、`disable_video`、`port`、`compression`、`render_traj_path`、`lpips_net` |

### 16.4 数据集与工具

| 文件 | 说明 |
| --- | --- |
| `datasets/colmap.py`（585 行） | `Parser`（pycolmap 解析、相机模型映射、畸变提取、图像缩放）+ `Dataset`（`__getitem__` 惰性解码、`preload_images()` 预热） |
| `datasets/ncore.py`（1,140 行） | NCore v4 解析：`NCoreParser`（相机内参、位姿批加载、世界归一化、点云、刚性动态轨迹）、`NCoreDataset`、`CameraRenderData`、`RigidDynamicTrack` |
| `datasets/endonerf.py` | 内窥镜手术数据集（动态场景用） |
| `datasets/normalize.py` | 世界空间归一化 |
| `datasets/traj.py` | 四种相机轨迹生成（interp / ellipse / spiral / raw） |
| `datasets/download_dataset.py` / `download_ncore.py` | 数据下载 |
| `utils.py` | `CameraOptModule`（位姿可学习增量）、`AppearanceOptModule`（外观嵌入）、`rotation_6d_to_matrix`、`knn`、`rgb_to_sh`、`set_random_seed`、`colormap`、`apply_depth_colormap` |
| `lib_bilagrid.py`（503 行） | 独立可复制的 3D 双边网格 + CP 分解 4D 双边网格实现（Bilateral Guided Radiance Field Processing） |
| `exif.py` / `prepare_pandaset.py` | EXIF 曝光读取 / PandaSet 预处理 |

### 16.5 Viewer 与基准脚本

| 文件 | 说明 |
| --- | --- |
| `gsplat_viewer.py` | 基于 viser + nerfview 的完整 viewer（可直接调 SH 阶数、near/far、eps2d、渲染模式） |
| `gsplat_viewer_2dgs.py` | 2DGS 专用 viewer |
| `simple_viewer.py` / `simple_viewer_2dgs.py` / `simple_viewer_3dgut.py` | 轻量 viewer；`simple_viewer.py --use_gaussian_render_inference_scene` 可切到 HiGS 推理路径 |
| `benchmarks/basic.sh` | MipNeRF360 七场景批量训练+评估（7k/30k），产出 PSNR/SSIM/LPIPS/内存/时间表 |
| `benchmarks/mcmc.sh` / `mcmc_4gpus.sh` | MCMC 策略（单卡 / 4 卡） |
| `benchmarks/basic_2dgs.sh` / `basic_4gpus.sh` | 2DGS / 4 卡 3DGS |
| `benchmarks/3dgut/mcmc.sh`、`mcmc_zipnerf.sh` | 3DGUT 基准 |
| `benchmarks/fisheye/` | ZipNeRF 鱼眼（含 undistorted 对照） |
| `benchmarks/compression/mcmc.sh`、`mcmc_tt.sh` | 压缩基准（含 TT 数据集） |
| `benchmarks/bilarf/mcmc_bilarf.sh` | 双边网格外观 |
| `benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py` | HiGS vs 默认 `rasterization()` 对比 |
| `benchmarks/geometry/se3_interpolate_tracks_bench.py` | SE(3) 轨迹插值基准 |

> 3DGUT 只支持 **MCMC** 策略（`docs/3dgut.md`）：
> `python examples/simple_trainer.py mcmc --with_ut --with_eval3d ...`

---

## 17. 测试体系

### 17.1 规模与组织

- 102 个测试文件，42,839 行，**793 个 `def test_*`**。
- `pytest.ini`：`testpaths = tests`、`pythonpath = .`、`addopts = -p pytest_check`、
  marker `gradcheck`（CUDA 双精度数值梯度检查，慢）。
- 统一设置 `NUM_CHANNELS=1,3,4,6,8,21,23,24,32,128` 与 `BUILD_CAMERA_WRAPPERS=1`。

| 目录/文件 | 覆盖 |
| --- | --- |
| `tests/test_basic.py` | **核心算子与参考实现的逐项对照**（最大文件）：`quat_scale_to_covar_preci`、`proj`、`projection`(fused/packed)、`isect`、`rasterize_to_pixels`、贡献查询、`eval3d`、UT、LiDAR isect |
| `tests/test_rasterization.py` | `rasterization()` 端到端：renderer_config 校验、3DGUT-only 构建、分布式单 rank 一致性、外畸变需 UT |
| `tests/test_2dgs.py` / `test_2dgs_scene_integration.py` | 2DGS 算子与场景集成 |
| `tests/test_sparse_*.py`（6 个） | 稀疏光栅化全家桶：`intersect` / `tile_layout` / `rasterize` / `num_contributing` / `contributing_ids` / `top_contributing` |
| `tests/test_losses*.py`（3 个） | 损失函数、深度损失、融合损失（含与 PyTorch 逐项对照与反向对照） |
| `tests/test_strategy.py` / `test_mcmc_perturb.py` / `test_relocation.py` | 策略、MCMC 扰动、重定位 |
| `tests/test_cameras.py` / `test_ftheta.py` / `test_external_distortion.py` / `test_pickle.py` | 相机模型、FTheta、外畸变、自定义类序列化 |
| `tests/geometry/` | `functional/test_{quaternion,pose}.py` |
| `tests/sensors/` | `functional/`（公开 API 契约）、`kernels/cameras/test_ops.py`（4,489 行，逐对前后向 + gradcheck）、`kernels/lidars/test_ops.py`（970 行）、`models/`（nn.Module 语义） |
| `tests/scene/` / `tests/stage/` | `GaussianScene` 钩子、`GaussianInferenceScene`、`Stage` |
| `tests/experimental/render/` | 推理渲染组件/functional/kernels/集成 |
| `tests/test_cpp.py` | 原生 C++ 测试二进制（用 `build.get_gsplat_link_inputs()` 取 ABI 匹配对象） |
| `tests/test_dataset_*.py` / `test_av_*.py` / `test_dynamic_surgical_trainer.py` | 示例脚本级集成 |
| `tests/test_packaging.py` | wheel/sdist 是否带上 CUDA 源码 |
| `tests/test_profile.py` / `test_trace.py` | 剖析与 NVTX |
| `tests/test_pickle.py` | `UnscentedTransformParameters` 等自定义类的 pickle 版本兼容 |

### 17.2 测试基础设施

**`gsplat/_helper.py`（860 行）**——不是普通工具模块，而是"断言 DSL"：

| 能力 | 函数 |
| --- | --- |
| 参数化展开 | `expand_named_params` |
| 测试数据加载 | `load_test_data` |
| 形状 | `assert_shape`（精确）、`ensure_shape`（可广播则扩展，否则报错） |
| 数值比较 | `assert_close`、`assert_mismatch_ratio`、`assert_close_with_boundary_band`（针对算法不连续量的两级比较） |
| 梯度专项 | `assert_grad_sparsity`（逐行幅值比稀疏性）、`assert_grad_reference_close`（值与向量双视角对比参考梯度） |
| 软断言 | `expect_group`（作用域内收集失败、退出时统一硬断言）、`expect_call` / `expect_true` / `expect_close` / `expect_mismatch_ratio` / `expect_grad_sparsity` / `expect_grad_reference_close` / `expect_close_with_boundary_band` |

软断言的意义：一次运行报告**所有**失败项，而不是第一个失败就中断——这对 GPU kernel
的数值调试非常关键。

**`tests/_backend_collect.py`**：`cuda_collect_ignore_glob()` —— 在没有 CUDA 的主机上
把 CUDA 专属测试**反收集**（而不是 import 时报 ERROR）。`tests/{geometry,sensors,experimental}/conftest.py`
都用它。

**根目录 `conftest.py`**：`GPU_CI_XFAIL` 环境变量开启时，把已知的"GPU CI runner 边缘
FP 不匹配"用例标 `xfail`（当前 20 条）。

**`tests/sparse_test_helpers.py`**：稀疏算子的统一验证范式——构造合成 2D 高斯场景、
选一组像素、把稀疏结果与稠密结果在那些像素上做 gather 后对比。

### 17.3 测试方法论（`docs/modules-design.md` 规定）

1. **前向**：形状、dtype、数值容差、边界情形（空 batch、退化输入、越界状态位、已知脆弱点）。
2. **反向**：不能只查"有限"——必须与可信参考对比。证据优先级：
   **PyTorch 参考梯度 > 有限差分 > 仅有限性**。
3. **分支覆盖**：前向的每条有意义分支都要测；若前向分支导致不同的反向行为，测试输入要显式覆盖。
4. **数值回归**：随机对照 + 固定回归种子 + 容差检查。
5. **导入与后端**：`functional/` 公开导入可用、后端加载行为正确、失败信息清晰。

### 17.4 CI

| 工作流 | 内容 |
| --- | --- |
| `.github/workflows/core_tests.yml` | CPU 上跑：格式检查（`black==22.3.0` + `clang-format==22.1.5`，`lint/format-code.sh --check --full`）+ `pytest tests/`（`BUILD_NO_CUDA=1 pip install .[examples]`） |
| `.github/workflows/gpu_tests.yml` | 自托管 GPU runner + `nvidia/cuda:12.9.0-devel` 容器；ccache 加速；`GPU_CI_XFAIL=1`；含 JIT debug 构建等多作业 |
| `.github/workflows/doc.yml` | Sphinx 文档构建 |
| `.github/workflows/building.yml` / `publish.yml` | 打包与发布 |
| `.github/workflows/generate_simple_index_pages.py/.yml` | 文档索引页生成 |

`lint/format-code.sh` 支持 `--check`（只检查）与 `--full`（全仓库）；
`config.yaml` 是 clang-format 版本的唯一来源，CI 与脚本都从这里读。

---

## 18. 关键数据约定

来自 `docs/source/conventions/data_conventions.rst` 与源码：

### 18.1 旋转

四元数采用 **`(w, x, y, z)`** 顺序，无需归一化（kernel 内部会归一）。
对应旋转矩阵：

```
R = [[1-2(y²+z²), 2(xy - wz), 2(xz + wy)],
     [2(xy + wz), 1-2(x²+z²), 2(yz - wx)],
     [2(xz - wy), 2(yz + wx), 1-2(x²+y²)]]
```

`gsplat.sensors` / `gsplat.geometry` 的公开 API 统一 **wxyz**；到 `gsplat/geometry`
的 SE(3) 原语时在 `kernels/common/utils.py` 里用 `wxyz_to_xyzw` / `xyzw_to_wxyz` 转换。
`sensors` 的 kernel 边界只接受拆分后的 `(translation (N,3), rotation (N,4) wxyz)`，
**从不接受 `(N,4,4)` 视图矩阵**。

### 18.2 相机

- `viewmats` = **world → camera**（w2c），`[..., C, 4, 4]`；样例里用
  `torch.linalg.inv_ex(camtoworlds).inverse` 得到。
- `Ks` = `[..., C, 3, 3]` 内参。
- 相机空间 +z 是**前向**，`depths` 即相机空间 z。

### 18.3 高斯参数

- `means [..., N, 3]`、`quats [..., N, 4]`、`scales [..., N, 3]`、`opacities [..., N]` 全部是
  **激活前**的原始参数（`rasterization` 期望调用方已做 `exp` / `sigmoid`，见
  `examples/simple_trainer.py:672-676`）。
- 或用 `covars [..., N, 6]`（上三角）替代 `quats`+`scales`（`rendering.py:539-543`
  会把 `(...,3,3)` 转成 6 维）。
- `colors` 二义性：
  - `sh_degree=None` → `[..., N, D]` 或 `[..., C, N, D]`，**已激活**的颜色/特征；
  - `sh_degree=k` → `[N, K, D]` **SH 系数**（跨 batch/camera 共享），要求 `(k+1)² ≤ K`。
- 通道数 `D` 必须命中编译期 `GSPLAT_NUM_CHANNELS`；`channel_chunk`（默认 32）控制分块渲染。

### 18.4 tile 与交集 id

- tile 默认 16×16（eval3d 下按分辨率 8 或 16，见 §5.2）。
- 交集 id 是 **64 位**整数：`image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits)`，
  `Xc`/`Xt` 是能表示对应 id 上界所需的最少位数。
- `flatten_ids` 是 `[I*N]`（dense）或 `[nnz]`（packed）下的全局索引，
  按 `(image_id, tile_id, depth)` 由近到远排序。

### 18.5 稀疏渲染的行序

`build_sparse_tile_layout` 约定 `pixels[p] = (row, col)`，`row ∈ [0, height)`、
`col ∈ [0, width)`；稠密 tile id =
`image_id * (tile_height*tile_width) + (row//tile_size)*tile_width + (col//tile_size)`。
调用方必须**去重**（不能有重复的 `(image,row,col)`）。空输入返回长度为 1 的
`tile_pixel_cumsum`（值为 0）。

### 18.6 dtype 与设备

- 主力 dtype 是 **float32**；`_math.py` 的参考实现与 gradcheck 支持 float64；
  融合损失 kernel 只支持 fp32/fp64。
- `-1` 是 `sh_degree` / `extra_signals_sh_degree` 的"未设置"哨兵（Python 侧转成
  `sh_degree_value = sh_degree if not None else -1` 传给 C++）。

---

## 19. 扩展指南

### 19.1 给主光栅化加一个新特性

1. **`gsplat/rendering.py`**：在 `rasterization()` 加参数 → 传给 `_make_lazy_cuda_func("rasterization_3dgs")`
   （注意它是**位置参数**调用，新增参数必须加在末尾以免错位）。
2. **`gsplat/cuda/ext.cpp`**：更新 `m.def("rasterization_3dgs(...)")` 的 schema 字符串
   （`csrc` 侧 `ext.cpp:1145` 起）。schema 与 C++ 签名不一致会在加载时报注册错误。
3. **`gsplat/cuda/csrc/Rendering.cpp`**：`rasterization_3dgs` 签名 + 校验
   （把新组合的合法性集中加到 `check_rasterization_3dgs_inputs`）+ 阶段逻辑。
4. **`gsplat/cuda/csrc/`**：新增 kernel（`.cu` + `.cuh` 配对），在对应 `register_*_impl` 里加 `m.def`/`m.impl`。
5. **`gsplat/cuda/_wrapper.py`**：如果新 kernel 是独立算子，加 Python wrapper +
   一个 `RegisterXxx` 类并在 `_ensure_autograd_registrations` 注册。
6. **测试**：`tests/test_basic.py` 加前向对照（vs `_torch_impl.py` 的参考实现）+ 梯度对照。

### 19.2 给 `gsplat/sensors` 加一个相机模型

1. **C++ 类型注册**（`sensors/kernels/cuda/ext.cpp:51` 的 `TORCH_LIBRARY(gsplat_sensors)`）：
   `m.class_<XxxProjection>(...)`，带命名 kwargs 的 `torch::init`、`def_readwrite` 属性、
   `def_pickle`、以及 `transform(scale, offset, new_resolution)`。
2. **桥接 POD**（`csrc/camera_params.h`）：`XxxProjection_KernelParameters` + 每个 op 的
   `_launch` 原型（前向 + 反向）。
3. **CUDA 实现**：`csrc/xxx_kernel.cuh`（device 数学 + 每线程 struct）、`.cu`（`__global__` + launch）、
   `_backward.cu`。注意**严格隔离**：`.cu/.cuh` 不得 include ATen。
4. **host 侧**：`csrc/camera_torch.h/.cpp` 加 host struct、`to_kernel_params()`、校验器。
5. **Python 句柄**：`kernels/cameras/types.py` 加类句柄、加进
   `REGISTERED_CAMERA_PROJECTIONS`、更新 `_REGISTERED_CLASS_NAMES`。
6. **分发表**：`kernels/projective_sensor_ops.py` 的 **7 张表全部补上 × 2 个畸变**
   （漏填会被 `test_projection_*` 一致性测试挡住）。
7. **autograd**：`kernels/cameras/ops.py` 为每个 `(op, 新投影)` 加 `torch.autograd.Function`。
8. **模型层**：`models/cameras/camera_model.py` 的 `_move_projection` 分发加分支。
9. **测试**：`kernels/cameras/test_ops.py` 逐对前后向 + gradcheck；
   `models/cameras/` 加模型级语义测试。

### 19.3 加一个新致密化策略

```python
# gsplat/strategy/my_strategy.py
from .base import Strategy
from .ops import duplicate, split, remove, reset_opa, relocate, sample_add

@dataclass
class MyStrategy(Strategy):
    def initialize_state(self, scene_scale: float = 1.0) -> dict: ...
    def check_sanity(self, params, optimizers): super().check_sanity(params, optimizers) ...
    def step_pre_backward(self, params, optimizers, state, step, info): ...
    def step_post_backward(self, params, optimizers, state, step, info, scene=None):
        ...  # 调 ops.* 并传 scene=scene 让旁路数据同步
```

要点：
- 拓扑变更**必须**走 `ops.*`（内部 `_update_param_with_optimizer` 会同步 Adam state）；
- 每次变更都要 `scene=scene`，否则 `signal` / `component_index` 会与 `splats` 错位；
- 若需要梯度，在 `step_pre_backward` 里 `retain_grad()`；
- 最后在 `gsplat/strategy/__init__.py` 与 `gsplat/__init__.py` 导出。

### 19.4 加一个新损失

`gsplat/losses.py` 的函数式约定：**逐元素/未归约**，布尔掩码可选，返回值不做 `.mean()`。
这样调用方可以自由组合 `reduce_mean` / `reduce_quantile` / `reduce_sum`。
若要做融合 kernel，参照 `losses_fused.py` + `_losses_wrapper.py` + `csrc/GaussianLossesCUDA.cu`，
并且**必须提供纯 PyTorch 回退路径**（`FusedGaussianLosses.forward` 里的 `all_cuda` 判断）。

---

## 20. 索引数据与热点分析

### 20.1 本次索引的构建方式

```
工具：codebase-memory-mcp / index_repository
参数：repo_path=/home/aic/Github/gsplat, mode=full, persistence=true
结果：state=succeeded, 26,853 nodes, 109,713 edges, 2,169 files, 用时 ~2s
```

**索引内容**：

| 维度 | 数据 |
| --- | --- |
| 节点标签（19 种） | Function 8,044 / Method 3,568 / Type 2,641 / File 2,169 / Variable 2,038 / Module 2,007 / Field 1,849 / Class 1,767 / Macro 1,262 / Section 1,206 / Folder 144 / Enum 59 / EnvVar 42 / Package 26 / Decorator 18 / Route 7 / Process 4 / Branch 1 / Project 1 |
| 边类型（25 种） | IMPORTS 30,123 / DEFINES 27,833 / USAGE 17,909 / CALLS 17,400 / SIMILAR_TO 3,328 / DEFINES_METHOD 3,165 / TESTS 2,564 / WRITES 2,264 / CONTAINS_FILE 2,169 / DECORATES 816 / MENTIONS 582 / INHERITS 382 / FILE_CHANGES_WITH 271 / SEMANTICALLY_RELATED 227 / REFERENCES_FILE 202 / ... |
| 语言分布 | HTML 826 / C++ 232 / Python 125 / JavaScript 98 / C 57 / CSS 4 / YAML 2（HTML/JS 主要来自 vendored `glm/doc`） |
| 包节点数 | cuda 5,320 / sensors 779 / geometry 393 / experimental 149 / scene 58 / losses 45 / strategy 29 / profile 29 / contrib 29 |
| 文档链接 | 582 处 doc-comment 引用（310 ambiguous / 245 missing / 89 below-bar） |
| entry point | 仅 1 个：`gsplat.profile.main` |

### 20.2 索引覆盖的已知缺口（诚实声明）

| 类别 | 内容 |
| --- | --- |
| `parse_partial`（5 个文件，214 处区间） | `cuda/csrc/CameraWrappers.cu`、`Rasterization.cpp`、`SphericalHarmonicsCUDA.cu`、`TensorView.h`、`third_party/glm/glm/common.hpp` —— 这些文件**已索引**，但列出的行区间内构造可能有缺失（tree-sitter 错误恢复仍救回一部分）。这些位置应以**文本搜索**为准 |
| `parse_unusable`（1 个） | `third_party/glm/glm/gtx/type_aligned.hpp` 整体不可用 |
| `not_indexed` | `.git` + 79 个二进制/资源文件（gif/png 等），**按设计排除** |
| 未索引的第三方 | `cuda/csrc/third_party/glm` 是 vendored 的 GLM 头文件库（含完整的 doc/api HTML/JS 站点），会**显著污染**统计：语言分布里的 826 个 HTML、98 个 JS 都来自它；hotspot 排行榜前三（`search.init_search`、`SearchBox`、`initMenu` 各 733 扇入）也是它的文档 JS |
| 交叉仓库 | `cross_repo.status = never_run`（未做跨仓库服务链接） |
| watch | 未开启自动同步（不是 MCP 会话的活动项目）；代码变更后需重跑 `index_repository` |

> 因此本文档中的结构判断以**源码阅读**为主、索引为辅：索引给出模块边界（`boundaries`）、
> 层级（`layers`）、聚类（`clusters`）与扇入/复杂度信号，具体语义细节全部回源码核对。

### 20.3 集群（clusters）解读

索引把代码聚成 12 个高内聚簇，其中有意义的几个：

| 簇 | 成员数 | 内聚度 | 代表节点 | 含义 |
| --- | --- | --- | --- | --- |
| — | 347 | 0.81 | `vec4/vec3/vec2/epsilonEqual` | vendored GLM 数学库 |
| — | 283 | 0.91 | `to_torch_args/dispatch/as_optional_tensor/chunk` | **`TensorView.h` 派发层**：C++ host ↔ device 的模板桥 |
| — | 201 | 0.92 | `read_quat_xyzw_from_wxyz/write_vec3/load_bivariate_windshield_params` | **外部畸变 + 四元数 device 读写** |
| — | 156 | 0.91 | `check_cuda_float_contiguous/empty_scratch/current_stream` | **sensors 的 C++ 校验/scratch 样板**（也是 hotspot 榜上 76/74/72 扇入的那几个） |
| — | 136 | 0.71 | `ivec4/ivec2/ivec3` | GLM 整型向量 |
| — | 127 | 1.00 | `_to_dev/project_world_points_mean_pose/script_class_name/_lookup` | **sensors 的 Python 分发表与设备搬运** |
| — | 119 | 0.73 | `length/test_make_pointer_mat/test_compile` | GLM 单元测试 |
| — | 116 | 0.97 | `quat_rotate_vector_fwd_impl/check_se3_pose_vec3_inputs/...` | **geometry：四元数 + SE(3) CUDA impl** |
| — | 109 | 0.95 | `quat_slerp_pair_fwd/normalize_shape/se3_interpolate_tracks_bwd_kernel` | **geometry：轨迹插值** |
| — | 108 | 0.97 | `assert_shape/_safe_normalize/_fully_fused_projection_with_ut` | **UT 投影的 Python 参考实现 + 测试断言** |
| — | 106 | 0.88 | `rasterize_to_pixels_from_world_3dgs_fwd_batch_scan_kernel/...` | **eval3d 的 ParallelBatch 三阶段流水**（partials→scan→replay） |
| — | 104 | **0.986** | `_make_lazy_cuda_func/_rasterization/rasterization/_make_lazy_cuda_obj/to_cpp` | **主渲染 Python 入口层**（最高内聚，说明 `rendering.py` + `_wrapper.py` 耦合非常紧） |

**结论**：`_wrapper.py` ↔ `rendering.py` 是最紧的一对（0.986），这也解释了为什么给
`rasterization()` 加参数必须同步改 `ext.cpp` schema——三者必须逐字对齐。

### 20.4 复杂度热点（排除 vendored 与测试）

| 函数 | 文件 | 复杂度 | 行数 |
| --- | --- | --- | --- |
| `profile.main` | `gsplat/profile.py` | 66 | 474 |
| `rasterize_to_pixels_from_world_3dgs_fwd_batch_scan_kernel` | `RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu` | 45 | 424 |
| `rasterize_gradient_bwd_kernel` | `RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu` | 40 | 682 |
| `rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel` | `RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu` | 38 | 477 |
| `rasterize_to_pixels_2dgs_bwd_kernel` | `RasterizeToPixels2DGSSerialBatchBwd.cu` | 37 | 664 |
| `macro_tile_rasterize_kernel`（HiGS） | `MacroTileRasterize.cu` | 36 | 531 |
| `get_build_parameters` | `gsplat/cuda/build.py` | 32 | 178 |
| `check_rasterization_3dgs_inputs` | `Rendering.cpp` | 30 | 416 |
| `rasterization_3dgs` | `Rendering.cpp` | 29 | 737 |
| `profile.capture_inputs` | `gsplat/profile.py` | 27 | 192 |
| `_rasterization` | `gsplat/rendering.py` | 26 | 385 |

**读代码的优先级建议**：想理解渲染 → 从 `rasterization()`(Python, 234) 读
`rasterization_3dgs`(C++, 745)；想理解 3DGUT → `ProjectionUT3DGSFused.cu` +
`RasterizeToPixelsFromWorld3DGS*.cu`；想调性能 → `profile.py`（它本身复杂度最高，
说明它把大量分支逻辑压在了一个入口里）。

---

## 21. 已知限制与注意事项

### 21.1 功能限制（源码/文档明确声明的）

| 限制 | 依据 |
| --- | --- |
| **不对相机内参 `Ks` 可微** | `rendering.py` 的 `.. warning::`（3DGS 与 2DGS 均是） |
| 分布式渲染只支持 unbatched + pinhole + 无 UT/eval3d/畸变/卷帘/sparse_grad/absgrad | `Rendering.cpp:187-234` 的 11 条 `TORCH_CHECK` |
| 3DGUT 只支持 `rasterize_mode="classic"` | `Rendering.cpp:243` `TORCH_CHECK(classic_rasterize_mode \|\| (!with_eval3d && !with_ut))` |
| 3DGUT 只支持 **MCMC** 策略（示例层面） | `docs/3dgut.md` |
| `ParallelBatch` 仅限 `with_eval3d=True` | `rendering.py:551-556` |
| `sparse_grad` 仅在 `packed=True` 时生效 | `_wrapper.py:fully_fused_projection` |
| 通道数必须命中 `GSPLAT_NUM_CHANNELS` | `_wrapper.py:rasterize_to_pixels` docstring |
| 融合损失 kernel 只支持 fp32/fp64 | `losses_fused.py` docstring |
| `GaussianScene.put()` 仅限初始化期 | `scene/components/gaussian_scene.py:33` docstring（之后调用会让旧 Parameter 成为孤儿） |
| `GaussianScene.validate()` 会同步 CUDA | `gaussian_scene.py` 内联注释 |
| 训练 checkpoint 不存 `scene.state_dict()` | `scene/design.md`（会丢 signal 与多 component 簿记） |
| `PngCompression` 会丢弃非平方数的低不透明度高斯 | `png_compression.py` docstring |
| `world_to_cam()` 已弃用 | `_wrapper.py:381` |
| `generate_image_points` 非 CUDA device 需 `allow_device_transfer=True` | `sensors/kernels/cameras/ops.py:4260` |
| 推理路径不支持 27 项特性（见 §13.2） | `gaussian_inference.py` 的 `_INFERENCE_UNSUPPORTED_FEATURES` |
| `LidarModel` / `LidarFrame` 不实现 `forward()` | `sensors/models/lidars/lidar_model.py:464` |
| ~~2DGS `packed=False` 路径不可用~~ | **已修复**（§5.6.1）。曾因 `expand()` 产出非连续张量被 `CHECK_CONTIGUOUS` 拒绝 |
| 2DGS `surf_normals` 仅在「颜色 + 深度」同时存在时产出，否则为 `None` | `Rendering.cpp:1735` `compute_normals_from_depth`，见 §5.4 |
| 2DGS 缺 23 项 3DGS 参数（3DGUT / 相机模型 / 畸变 / 卷帘 / LiDAR / 分布式 / 反锯齿 / 分块通道等） | 见 §5.4 之后的 API 对照；2DGS 独有 `depth_mode` 与 `distloss` |
| 2DGS `distort` 是 L1 版本，与 2DGS 论文的 L2 不一致 | `rendering.py` `rasterization_2dgs` docstring |

### 21.2 数值/工程注意事项

1. **`meta["means2d"]` 的生命周期**：策略读它拿梯度，所以 `step_pre_backward` 必须
   `retain_grad()`；且 `absgrad=True` 时读的是 `.absgrad` 而不是 `.grad`。
2. **meta 是 scatter 前快照**：分布式模式下不要假设 `meta` 里的张量与最终渲染张量同形。
3. **`eps2d` 语义**：加到 2D 协方差特征值上的 epsilon，`0.3` 约等于"最小 3 像素"。
4. **`radius_clip` 是双刃剑**：能跳过远处微小高斯，但**近处的小高斯也会被跳过**。
5. **`MIN_COMPENSATION = 0.005`**：反锯齿补偿因子的下界，防止极小高斯补偿到 0。
6. **反锯齿会略微降低量化指标**（`EXPLORATION.md` 有具体数字）。
7. **`glm` 在 vendored 目录下**：`gsplat/cuda/csrc/third_party/glm` 是头文件库，
   不要改；但也不要在代码搜索/索引统计里把它当项目代码。
8. **JIT 缓存位置**：`~/.cache/torch_extensions/py*-cu*/`；改 flag 会触发**整个构建目录被删除重建**。
   最隐蔽的触发源是 `pytest.ini` 的 `env` 段与日常 shell 环境不一致（详见 §4.5.2）。
9. **AOT 与 JIT 的优先级**：`cuda/_backend.py` 先试 `from gsplat import csrc`，成功即**永不 JIT**。
   因此只要 AOT 产物在，参数快照比较逻辑根本不会执行——这是规避重建抖动最彻底的办法。
10. **`VERBOSE=1 DEBUG=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"`
    是触发一次纯编译的最快方式**（`_backend.py` docstring 里写的）。注意 `DEBUG=1` 会连带打开
    `BUILD_CAMERA_WRAPPERS`，又是一次参数变化。
11. **三个子扩展（geometry / sensors / scene）不在 AOT 范围内**，首次访问仍会 JIT 一次；
    它们不读 `NUM_CHANNELS`/`BUILD_CAMERA_WRAPPERS`，因此不会被测试参数抖动牵连。
12. **`sensors` 的 `None` 畸变会被拒**：必须显式 `NoExternalDistortion()`。
13. **`sensors` 的分发表一致性由测试守门**，不是运行时检查——新增类型后**先跑那个测试**。
14. **2DGS 的 `packed` 可自由选择**：dense 路径的 `expand()` 缺陷与组合模式的反向缺陷
   均已修复（§5.6），`packed=True/False` × 5 种 `render_mode` 全部前向+反向通过。
   `examples/simple_trainer_2dgs.py:146` 的默认 `packed=False` 现在可以正常训练。
15. **3DGUT（`with_eval3d=True`）不支持 `packed=True`**，会报
    `Packed mode is not supported with Eval3D`。
16. **`packed` 的真实代价是「稳态省、峰值费」**（实测 N=150k、C=8、可见率 31%）：

    | | meta 张量合计 | fwd 峰值 | fwd+bwd 峰值 | 纯前向后驻留 |
    | --- | --- | --- | --- | --- |
    | `packed=False` | 54.2 MB | 90.8 MB | 145 MB | 25.7 MB |
    | `packed=True` | **30.1 MB** | **124.6 MB** | **158 MB** | **82.9 MB** |

    meta 与驻留占用确实下降（`[C,N,...]` 网格 → `[nnz,...]` + 3 个索引向量，
    `nnz ≈ C·N·可见率`），但**峰值更高**——packed 投影是两遍扫描
    （`Projection.cpp:918-929`）：第一遍只计数，`cumsum` 后 `.item()` 同步取回 `nnz`，
    再分配输出跑第二遍，额外引入 `block_cnts`/`block_accum` 临时缓冲与一次设备同步。
    速度上也稳定慢 10–20%。**按峰值显存做容量规划时不要假设 packed 更省。**

---

## 附录 A：功能 → 入口 → 实现 速查表

| 我想… | Python 入口 | 核心实现 |
| --- | --- | --- |
| 渲染一张图 | `gsplat.rasterization(...)` | `rendering.py:234` → `Rendering.cpp:745` → `RasterizeToPixels3DGSSerialBatchFwd.cu` |
| 带畸变/卷帘渲染 | `rasterization(with_ut=True, with_eval3d=True)` | `ProjectionUT3DGSFused.cu` + `RasterizeToPixelsFromWorld3DGS*.cu` |
| 渲染 2D 高斯（面元） | `gsplat.rasterization_2dgs(...)` | `rendering.py:1358` → `Rendering.cpp:1705` → `RasterizeToPixels2DGS*.cu` |
| 只渲染部分像素 | `isect_tiles_sparse` + `build_sparse_tile_layout` + `rasterize_to_pixels_sparse` | `SparseTileLayout.cu` + `RasterizeToPixelsSparse*.cu` |
| 查每个像素被哪些高斯覆盖 | `rasterize_contributing_gaussian_ids` | `RasterizeContributingGaussianIds.cu` |
| 训练（原版 3DGS） | `DefaultStrategy` + `rasterization` | `strategy/default.py` + `strategy/ops.py` |
| 训练（MCMC） | `MCMCStrategy` | `strategy/mcmc.py` + `relocation.py` + `MCMCPerturbCUDA.cu` |
| 稀疏更新优化器 | `SelectiveAdam(..., step(visibility=mask))` | `optimizers/selective_adam.py` + `AdamCUDA.cu` |
| 算损失 | `gsplat.losses.*` / `FusedGaussianLosses` | `losses.py` / `losses_fused.py` + `GaussianLossesCUDA.cu` |
| 把高斯打包进推理场景 | `GaussianInferenceScene.from_gaussian_scene(...)` | `scene/components/gaussian_inference_scene.py:261` → `gsplat_scene_cuda.pack_gaussian_inference_scene` |
| 低延迟推理渲染 | `gsplat.experimental.render_scene(scene, viewmat=..., K=..., width=..., height=...)` | `experimental/render/functional/` → `experimental::gaussian_render_inference_only` |
| 世界点投影到相机 | `gsplat.sensors.functional.project_world_points_shutter_pose(...)` | `sensors/functional/cameras.py:191` → `sensors/kernels/cameras/ops.py` → `camera_kernel.cu` |
| LiDAR 生成射线 | `gsplat.sensors.functional.generate_spinning_lidar_rays(...)` | `sensors/functional/lidars.py:120` → `lidar_kernel.cu` |
| 四元数运算（可微） | `gsplat.geometry.functional.quat_*` | `geometry/kernels/quaternion_ops.py` → `quaternion.cu` |
| 位姿/轨迹插值 | `gsplat.geometry.functional.se3_*` | `geometry/kernels/pose_ops.py` → `pose.cu` |
| 压缩模型 | `PngCompression().compress(dir, splats)` | `compression/png_compression.py` + `compression/sort.py` |
| 导出给 viewer | `export_splats(..., format="ply"/"splat"/"ply_compressed")` | `exporter.py:588` |
| 多卡训练 | `torchrun ... ` 或 `distributed.cli(fn)` | `distributed.py` + `rasterization(distributed=True)` |
| 抓性能数据 | `capture_inputs` 装饰 + `nsys` + `python -m gsplat.profile --input ...` | `profile.py` + `trace.py` |
| 训练动态场景 | `DynamicStrategy` + `HexPlaneField` + `DeformNetwork` | `contrib/dynamic/` + `examples/dynamic_surgical_trainer.py` |

---

## 附录 B：术语表

| 术语 | 含义 |
| --- | --- |
| **EWA** | Elliptical Weighted Average，3DGS 的投影近似（Jacobian 线性化） |
| **UT** | Unscented Transform，3DGUT 用来处理非线性投影/畸变 |
| **eval3d** | 在 3D 世界空间沿射线评估高斯响应（而非 2D 图像空间），3DGUT 的后半部分 |
| **packed** | 中间结果压成 COO 稀疏布局（`[nnz, ...]` + `batch_ids/camera_ids/gaussian_ids`）。**省 meta 张量与常驻占用，但峰值反而更高**，见 §21.2 第 16 条 |
| **tile** | 屏幕空间分块（默认 16×16），光栅化的调度单位 |
| **isect** | intersection，高斯与 tile 的交集记录 |
| **flatten_id** | 全局展平索引（`[I*N]` dense / `[nnz]` packed） |
| **AccuTile / SNUGBOX** | 用不透明度阈值化的保守椭圆做 tile 求交，跳过一定不贡献的 tile |
| **compensation** | 反锯齿补偿因子 `sqrt(det Σ / det(Σ+εI))` |
| **absgrad** | `means2d` 梯度的**绝对值**，AbsGS 用它做致密化判据 |
| **scratch** | 前向返回、供反向复用以免重算的中间张量 |
| **Seam A / Seam B** | 分布式渲染中「收集全体相机」与「散射投影结果」两个接缝 |
| **Seam / 缝** | 在 sensors 文档里指 CUDA 与 Torch 两个编译单元之间唯一的桥接声明 |
| **NCore** | NVIDIA 的自动驾驶多传感器数据集格式（v4），支持 FTheta 相机与点云 |
| **HiGS** | Hierarchically Tiled Gaussian Splatting，本项目的推理专用光栅化方案 |
| **G-SHARP** | 动态手术场景重建方案，`contrib/dynamic` 与 `init_utils`/`regularizers`/`schedulers` 均移植自它 |
| **PPISP** | 一种补偿训练视角的控制器方案（与双边网格并列的外观后处理选项） |

---

*文档基于 gsplat 1.6.0（commit `512d366b`）编写。索引数据为一次性快照；
代码变动后建议重跑 `index_repository(repo_path="/home/aic/Github/gsplat", mode="full")` 并更新 §20。*
