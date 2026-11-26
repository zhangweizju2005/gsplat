# gsplat 现有代码复用清单（surfel 架构）

> 读完 `docs/CODEBASE_MAP.zh.md`、`examples/` 全部训练脚本、`examples/datasets/`、
> `gsplat/init_utils.py`、`gsplat/losses.py`、`gsplat/exporter.py` 后的结论。
>
> 核心判断：**大量我手写的东西，gsplat 上游已有成熟实现。应尽量替换。**

---

## 0. 一句话结论

| 类别 | 结论 |
|---|---|
| **可以直接用、应当删掉自己的实现** | kNN 尺度初始化、深度反投影、普通损失原语、LR 衰减、导出、指标 |
| **可以直接用、但必须改一处接线** | `DefaultStrategy.initialize_state(scene_scale=...)`、`key_for_gradient` |
| **不能直接用，必须自己实现** | 分辨率 policy、gauge/Sim3、深度网格 remap、深度源接入、COLMAP 无 pycolmap 读取 |
| **可以考虑用、但代价明显** | `examples/datasets/colmap.py` 的 Parser、`Stage`/`GaussianScene`、`CameraOptModule` |

---

## 1. 明确可复用（应删掉我的重复实现）

### 1.1 `gsplat.init_utils` —— 我有两个模块是它的重复实现

#### `knn_scale_init(xyz, k=3, eps=1e-7, chunk_size=1024)` → `(N,)` **log** 距离

对比我写的 `surfel/model/splats.knn_mean_distance_rms`：

| 项 | 上游 `knn_scale_init` | 我的实现 |
|---|---|---|
| 数学 | `sqrt(mean(d²))` over k neighbours，`k=3` **不含自身** | 同（`k-1` 最近邻 RMS） |
| 返回 | **已取 log** | 原始距离（我自己再 log） |
| 实现 | `torch.cdist` 分块 | scipy `cKDTree` |
| 依赖 | 纯 torch | scipy |

**实测内存**：

```
N=  1,000:  518 ms, peak 0.039 GB
N= 50,000:  482 ms, peak 0.446 GB
N=1,800,000: 预测单块 7.37 GB（chunk=1024 → (1024, 1.8M) float32）
```

**结论**：
- **小场景（< 5 万点）直接用上游**，省一个 scipy 依赖，代码更少。
- **大场景（> 50 万点）必须保留 scipy 路径** —— 上游 7.4 GB 峰值在统一内存机器上会进 swap（本项目已因此重启过一次机器；当年是 90 GB，这个是 7.4 GB，虽不致命但会拖慢）。
- 建议：`surfel` 保留一个薄包装，按点数分派：`N <= 200_000` → `knn_scale_init`，否则 scipy。

#### `multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics, max_points)` → `(xyz, rgb)`

这是 DGI 的上游实现，我写的 `surfel/data/dgi.py` 功能重叠。

| 能力 | 上游 | 我的 |
|---|---|---|
| 逐帧用**自己的 K** 反投影 | ✅ | ✅ |
| mask + `depth > 0` 过滤 | ✅ | ✅ |
| 颜色取同一像素 | ✅ | ✅ 但按**归一化坐标**采样（可处理 depth/image 网格不同） |
| `max_points` 随机下采样 | ✅ | ✅ |
| 支持 **depth 网格 ≠ image 网格** | ❌ **假设两者相同**（`h, w = images.shape[1:3]`，直接用同一 `u,v` 索引 depth 与 rgb） | ✅ |
| 支持 border_trim / conf_quantile / min/max depth | ❌ | ✅ |
| 返回 `flat_index` | ❌ | ✅ |

**关键限制**：上游 `multi_frame_depth_unprojection` 在 `images.shape[1:3]` 取 `h,w`，
然后 `u_coords[valid]` 同时索引 `depths_f[i]` 和 `images_f[i]`。
**若深度图与图像尺寸不同，会静默错位或直接越界**。

而本项目恰恰有这个场景：
- Scal3R 深度 **518×280**，图像 **1920×1080**
- Scal3R 内部非等比拉伸（`sx=3.707, sy=3.857`）

**结论**：**不能用上游版本**做 DGI。保留我的 `dgi.py`，因为"深度网格 ≠ 图像网格"是这里的常态，
不是边缘情况。但可以复用它的 API 形状（`(xyz, rgb)` 返回、`max_points` 语义）以减少差异。

---

### 1.2 `gsplat.losses` —— 我的 `losses.py` 大部分是包装

已有且应当直接调：

| 我的函数 | 上游对应 | 说明 |
|---|---|---|
| `image_terms` 里的 L1/SSIM | `l1_loss` / `ssim_loss` | 已直接调用 ✅ |
| `depth_term` 的非归一化分支 | `depth_l1_loss(pred, gt, scene_scale)` | 已直接调用 ✅ |
| `normal_term` | **`normal_cosine_loss(pred, gt)`** | 我手写了 `(1 - cos)`，应改用上游（带 unit-norm 契约断言） |
| `opacity_reg` | `opacity_reg_loss(logits)` | 已直接调用 ✅ |
| `scale_reg` | `scale_reg_loss(log_scales)` | 已直接调用 ✅ |
| 深度权重 ramp | **`LinearLambdaScheduler(start, end, lambda_init, lambda_end)`** | 我手写了线性插值，可替换 |

**`LinearLambdaScheduler` 实测**（`start=0, end=7000, init=0.05, end=1.0`）：

```
step    0 -> 0.0500
step 1000 -> 0.1857
step 3500 -> 0.5250
step 6000 -> 0.8643
step 7000 -> 1.0000
step 8000 -> 1.0000   ← 自动 clamp
```

与 GSL 的 `depth_loss_weight: {init, final_factor, max_steps}` 语义完全一致
（GSL 是 `final_factor` 乘子，上游是绝对值 → 传 `lambda_end = init * final_factor`）。
**建议替换**，少一个手写调度器。

**`normal_cosine_loss` 的额外好处**：`ENFORCE_CONTRACTS` 开启时会断言两侧都是单位向量。
我的 `normal_term` 没有这个检查，而 `surf_normals` 在 `C=1` 且某些模式下形状不对时可能不是单位长。

**必须保留的**：`depth_term` 的另外两个分支
- `per_frame_median_norm`（逐帧中位归一化）—— 上游**没有**
- alpha 掩码 + `depth_eps` 保护 —— 上游 `depth_l1_loss` 用 `where(pred>0, 1/pred, 0)`，
  没有 `clamp_min(eps)`，遇到极小的正深度仍会爆炸

---

### 1.3 `gsplat.exporter.export_splats` —— 我完全没有，但必然需要

```python
export_splats(means, scales, quats, opacities, sh0, shN,
              format="ply" | "splat" | "ply_compressed", save_to=...)
```

**实测**：2DGS 风格的 `scales=[N,3]`（第三轴 `1e-6`）可以正常导出：

```
export ply:   OK 25074 bytes
export splat: OK 3200 bytes
```

**注意**：上游要求
- `opacities` 是 **`(N,)` 一维**（不是 `(N,1)`）
- `sh0` 是 `(N,1,3)`、`shN` 是 `(N,K,3)`
- 传入的应是**激活后**的 `scales`（`exp`）与 `opacities`（`sigmoid`）

导出 PLY 里第三尺度轴会被写成 `1e-6`。这对 2DGS 的消费端是正确的（hbb1/gsplat 都只读前两轴）。

---

### 1.4 评估指标 —— 上游示例用 torchmetrics，我手写了 PSNR

**实测等价性**：

```
PSNR 手写 -10log10(MSE): 26.312656  vs  torchmetrics: 26.312654  diff 1.9e-06   ✅
SSIM gsplat 1-ssim_loss :  0.985227  vs  torchmetrics:  0.985389   diff 1.6e-04
SSIM (256x384)          :  0.985370  vs              0.985362      diff 7.7e-06
```

**结论**：
- PSNR 手写与 torchmetrics 等价（差 1e-6）
- SSIM 在**小图**（64×64）差 1.6e-4，**大图**差 7.7e-6 —— 差异来自窗口与 padding 处理。
  归档基线用的是 `torchmetrics.StructuralSimilarityIndexMeasure`，
  **为了可比性应当用 torchmetrics**，不要用 `1 - ssim_loss`。
- 环境里 `torchmetrics 1.9.0` 和 LPIPS 都可用。

**建议**：评估走 torchmetrics（`PeakSignalNoiseRatio` / `StructuralSimilarityIndexMeasure` /
`LearnedPerceptualImagePatchSimilarity`），训练内 loss 走 `gsplat.losses`。这正是官方示例的分工。

---

### 1.5 LR 衰减 —— 官方**已有**，与我手写的等价（更正我先前的说法）

先前文档写"官方无衰减"是**错的**。实际在 `examples/simple_trainer_2dgs.py:530`：

```python
ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / max_steps))
```

逐步对拍（`lr_init=1.6e-4`）：

| step | 官方 | GSL 对数插值 | 比值 |
|---|---|---|---|
| 0 | 1.6000e-04 | 1.6000e-04 | 1.000000 |
| 1750/7000 | 5.0596e-05 | 5.0596e-05 | 1.000000 |
| 3500/7000 | 1.6000e-05 | 1.6000e-05 | 1.000000 |
| 7000/7000 | 1.6000e-06 | 1.6000e-06 | 1.000000 |

**7k / 30k 预算下都逐位一致**。所以：
- 用 `torch.optim.lr_scheduler.ExponentialLR` 即可，**不需要**我手写的 `means_lr_scheduler`
- 好处：它会自动存进 `optimizer.state_dict()`，checkpoint 往返不需要额外处理

---

## 2. 可直接用但要改一处接线

### 2.1 `DefaultStrategy` —— 唯一必须改的是 `scene_scale`

上游代码本身完全可用，包括 `key_for_gradient="gradient_2dgs"` 这个参数（文档里写明了）。

**问题只在示例的调用处**：

```python
# examples/simple_trainer_2dgs.py:364-370  ← 3DGS 示例正确传了，2DGS 示例没传
self.strategy.check_sanity(self.splats, self.optimizers)
self.strategy_state = self.strategy.initialize_state()          # scene_scale = 1.0 ❌
```

对照 `examples/simple_trainer.py:517`：

```python
if isinstance(self.cfg.strategy, DefaultStrategy):
    self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)  # ✅
```

**这不是 gsplat 库的 bug，是 2DGS 示例的遗漏**。我们照 3DGS 示例写即可。

### 2.2 官方 Parser 的 `scene_scale` 语义与 strategy 需要的不完全一致

`examples/datasets/colmap.py:436-441`：

```python
camera_locations = camtoworlds[:, :3, 3]
scene_center = np.mean(camera_locations, axis=0)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)
self.scene_scale = np.max(dists)
```

而 `similarity_from_cameras` 归一化的目标是 **中位数**相机距离 = 1（`normalize.py:86`：
`scale = 1.0 / np.median(...)`）。

所以归一化后：

```
紧凑块（等间距）    median=1.000 max=1.020 -> parser.scene_scale ≈ 1.02
长条航带            median=1.000 max=2.400 -> parser.scene_scale ≈ 2.40
```

**含义**：`parser.scene_scale` = "最远离中心的相机距离"，是个**长宽比敏感的**量。
对 `DefaultStrategy` 的尺寸阈值它可用（2.4 倍偏差在可接受范围），
但**不能同时**当 LR 缩放与 loss 乘子 —— 这三个用途需要三个不同的数。

**建议**：
- `strategy.scene_scale` ← 上游 Parser 的值可以接受（`* 1.1 * global_scale` 与官方一致）
- `loss.depth_scale` ← 必须用**中位深度**，不能用这个
- `means_lr *= spatial_lr_scale` ← 归一化后固定 `1.0`，不需要再乘 `scene_scale`

### 2.3 `examples/utils.py` 的 `knn` —— 别用

```python
def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
```

- 依赖 `sklearn`（环境有 1.9.1，但为这一个函数引入不值得）
- 返回的是**欧氏距离**不是平方，`create_splats_with_optimizers` 里再 `**2` 后 `.mean()`
- 无分块，182 万点会吃满内存

**建议**：用 `gsplat.init_utils.knn_scale_init`（已含 RMS + 分块 + log）或我的 scipy 版。

---

## 3. 不能直接用（必须自己实现的部分）

以下这些**上游没有**，是 `surfel` 存在的理由：

| 能力 | 为什么上游不行 |
|---|---|
| **分辨率 policy** | 上游 `Parser` 假设 `images/` 存在、`factor` 从那里下采样、且用 `images_<n>` 后缀。真实数据里 `images_4/` 的内容可能是全尺寸、`images/` 可能不存在。这个假设造成过 8-12 dB 的静默误差。 |
| **无 pycolmap 读 COLMAP** | 上游 `Parser` 硬依赖 `pycolmap`（`import pycolmap` + `Reconstruction(colmap_dir)`）。官方包无 aarch64 wheel，PyPI 同名包是 2018 年无关项目。 |
| **gauge / Sim(3)** | 上游只有 `similarity_from_cameras`（一个固定的"摆正 + 居中 + 中位归一化"流程），没有：显式 gauge 类型、跨源 Sim(3) 拟合、残差门控、反射保护、"位姿+点+深度一起变换"的原子性。 |
| **深度网格 remap** | 上游 `Dataset` 只做 COLMAP 稀疏点投影深度（`load_depths=True`），没有"把外部深度图对齐到渲染网格"的概念，更没有 ray-remap。 |
| **深度源接入** | 上游没有 EXR 读取、没有逐帧 `{scale,offset}` 标定、没有 `depth_K` 概念。 |
| **大场景 near/far 派生** | 上游 `near_plane=0.2 / far_plane=200` 是常量。 |
| **致密化失效检测** | 上游策略静默运行，没有任何"是否真的增点了"的统计。 |
| **`absgrad` 的 2DGS 拒绝** | 上游会直接 `AttributeError`，没有配置层的前置校验。 |
| **checkpoint 带完整 config + 协议** | 上游 `torch.save({"step", "scene_id", "splats"})`，恢复后丢失配置。CODEBASE_MAP §10.2 明确记录了这个限制。 |
| **多源帧名对齐** | 上游按 COLMAP 名字直接拼路径。 |

---

## 4. 可以考虑用、但有明显代价

### 4.1 `examples/datasets/colmap.py` 的 `Parser`（585 行）

**能给的**：
- `pycolmap` 解析 + 相机模型映射 + 畸变参数提取
- `mapx/mapy` 预计算 + 运行时 `cv2.remap` 去畸变（这部分做得比我好，我只取 pinhole K 不处理畸变）
- `alignment/归一化`（`similarity_from_cameras` + `align_principal_axes` + 翻转修正）
- `point_indices`（image_name → point3D 索引），用于稀疏深度投影

**代价**：
1. **硬依赖 pycolmap** —— 环境里已有（4.3.0，源码编译），但换机器是负担
2. **`images/` + `images_<factor>/` 的假定** —— 正是分辨率 bug 的来源
3. `imsize_dict` 用 `cam.width // factor` 整数除法，与文件实际尺寸无关
4. `Dataset` 的 split 是 `indices % test_every == 0`，与 GSL 的 `split_mode`/`eval_ratio` 语义不同

**建议**：
- **去畸变那段值得移植**（`cv2.getOptimalNewCameraMatrix` + `initUndistortRectifyMap` +
  `roi_undist` 裁剪 + `K` 修正）。我目前的 `colmap_io` 只取 pinhole `K`，
  若数据有畸变会错。**这是一个真实缺口。**
- 分辨率判定、split 语义、depth 接入必须用自己的。
- 折中方案：写一个 `ColmapParserUpstream` 适配器，**只在显式配置 `data.use_undistort=True` 时**调它做去畸变 + K 修正，其余走我的路径。

### 4.2 `Stage` + `GaussianScene`

**能给的**：
- `scene.on_duplicate/on_split/on_remove` 钩子 —— 策略算子会调用它们保持行对齐
- `GaussianScene.state_dict()` 完整序列化（含 `signal`、`component_index`）
- `Stage.render(scene_id, **kwargs)` 统一入口

**代价**：
- 引入一层间接（`stage.render` → `render_fn(splats=..., **kwargs)`），
  而我们只有一个场景、一个渲染函数
- `scene.validate()` 会 `.item()` 同步 CUDA，不能放进热路径
- 若只用 `splats` 而不用 `signal`/多 component，这层是纯开销

**但有一个实际收益**：`strategy.step_post_backward(..., scene=self.scene)` 的 `scene` 参数是
**可选**的（`default.py:195` 等所有调用点都是 `scene=scene` 默认 `None`）。
若不用 `GaussianScene`，就必须自己保证 `params` 里每个张量在拓扑变更后同步 ——
而 `strategy.ops._update_param_with_optimizer` 已经负责这件事（替换 Parameter + 迁移 Adam state）。

**结论**：**不需要 `Stage`/`GaussianScene`**。
- `Stage` 是纯路由，单场景无收益
- `GaussianScene` 的主要价值是 `signal`/多 component，我们不用
- 策略的拓扑同步由 `ops.py` 内部完成，不依赖 scene

（唯一注意：不用 scene 时**不能**传 `scene=` 参数，用默认 `None`。）

### 4.3 `examples/utils.py` 的 `CameraOptModule`

```python
class CameraOptModule(nn.Module):
    def __init__(self, n): self.embeds = nn.Embedding(n, 9)   # 3 trans + 6 rot
    def forward(self, camtoworlds, embed_ids):
        rot = rotation_6d_to_matrix(drot + identity)
        return camtoworlds @ transform    # right-multiply ✅
```

- 用的正是 **right-multiplication**（与我 `surfel/train/pose.py` 一致）✅
- 6D 旋转表示 + Gram-Schmidt（我用的 Rodrigues 指数映射）
- 零初始化是精确 no-op ✅

**建议**：**直接用它**。100 行成熟代码，6D 表示保证了连续性与可控的更新幅度，
而且它是官方示例的默认，做对照实验时更可控。我的 `pose.py` 可以删掉。
（注意：`embeds` 里面是 9 维，前 3 平移、后 6 旋转。）

### 4.4 `examples/utils.py` 的 `AppearanceOptModule`

比我的 `ExposureAffine` 强得多：embedding + MLP 输出逐高斯颜色，能表达逐视角颜色变化。
但它与 `app_opt=True` 的 `features` 参数强耦合（`create_splats_with_optimizers` 里
`feature_dim=32 if app_opt else None`，会让参数集多一个 `features`）。

**对航拍场景的判断**：
- 需求是"补偿逐图曝光/光照变化"，`ExposureAffine` 的 2 自由度就够，且**不会**让网络吸收几何误差
- `AppearanceOptModule` 表达力过强，在小重叠的航拍块上容易过拟合
- **建议保留 `ExposureAffine`**，但把 `AppearanceOptModule` 作为可选项，如果将来需要

### 4.5 `examples/datasets/normalize.py` 的 `normalize()`

```python
def normalize(camtoworlds, points=None):
    T1 = similarity_from_cameras(camtoworlds)      # 摆正 z-up + 居中 + 中位归一
    if points is not None:
        T2 = align_principal_axes(points)          # PCA 主轴对齐
        ...
        return camtoworlds, points, T2 @ T1
```

这是 **Mip-NeRF 360 的既定流程**，对物体级场景是对的。

**为什么航拍不能直接用**：
1. `similarity_from_cameras` 用"相机 up 轴平均"定世界 up —— 航拍俯视时所有 `up` 都指向地面，
   平均向量可能接近零或噪声主导
2. 归一化目标是**中位相机距离 = 1** —— 长条航带会被压成细长条（前面实测 `max/median = 2.4`）
3. `align_principal_axes` 用 PCA 主轴 —— 城市block的 PCA 主轴通常沿街道，但**符号不定**，
   同一场景两次运行可能差 90° 或 180°
4. 它**不缩放深度** —— 正是"规则 2"违反的地方

**建议**：**不用**。用 `surfel/data/gauge.py` 的显式 Sim(3)：
只有 `s`（+ 可选 `R`），不做 PCA，且 `apply_gauge` 保证深度一起缩放。
若需要"摆正"，用 GNSS/航向显式给 `R`，而不是从相机 up 轴推断。

---

## 5. 具体替换清单

| 我的文件/函数 | 处置 | 替换为 |
|---|---|---|
| `model/splats.knn_mean_distance_rms` | **保留**（大场景需要 scipy），小场景转发上游 | `gsplat.init_utils.knn_scale_init` |
| `model/splats.rgb_to_sh0` | **保留**（上游 `examples/utils.rgb_to_sh` 在 examples 里不可 import） | — |
| `model/splats.means_lr_scheduler` | **删除** | `torch.optim.lr_scheduler.ExponentialLR(gamma=0.01**(1/max_steps))` |
| `model/splats.create_splats_2dgs` | **保留**（加了第三轴压制，上游没有） | — |
| `train/losses.normal_term` | **改调上游** | `gsplat.losses.normal_cosine_loss` |
| `train/losses` 权重 ramp | **删除手写** | `gsplat.losses.LinearLambdaScheduler` |
| `train/losses.depth_term` | **保留**（median-norm + eps + alpha mask 上游没有） | — |
| `train/pose.CameraOptModule` | **删除** | `examples/utils.CameraOptModule` |
| `train/appearance.ExposureAffine` | **保留**（航拍更合适） | — |
| `train/trainer` 里的 PSNR | **改用 torchmetrics** | `torchmetrics.image.PeakSignalNoiseRatio` |
| 导出 | **新增** | `gsplat.exporter.export_splats` |
| `data/dgi.py` | **保留**（上游不支持 depth 网格 ≠ image 网格） | — |
| `data/gauge.py` | **保留**（上游无跨源 Sim3） | — |
| `data/resolution.py` | **保留**（上游的这套假设就是 bug 源） | — |

---

## 6. 一个新发现的真实缺口：去畸变

上游 `examples/datasets/colmap.py:376-427` 有完整的去畸变流程：

```python
K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (width, height), cv2.CV_32FC1)
# 运行时
image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi_undist
image = image[y:y+h, x:x+w]
# K 修正
K_undist[0, 2] -= x_min   # fisheye 分支
K_undist[1, 2] -= y_min
```

**我当前的 `surfel/data/colmap_io.py` 只取 pinhole `K`，忽略失真系数**。
若数据本身有畸变（`OPENCV` / `RADIAL` 等），会静默产生几何错误。

**这必须补上**，两个选项：
1. **移植上游这段**（约 50 行，`cv2` 已可用）
2. 要求输入数据已经过 `colmap image_undistorter`，且在配置里断言
   `cameras` 全是 `PINHOLE`/`SIMPLE_PINHOLE`

**建议**：两者都做 —— 默认断言输入已去畸变（fail fast），
提供 `data.undistort=True` 走移植的 `cv2` 路径。

MatrixCity / Scal3R 导出的数据通常是 pinhole，所以这对当前数据集不阻塞，
但换数据集时会成为静默错误源。

---

## 7. 复用后的代码量估计

| 模块 | 当前 | 复用后 | 说明 |
|---|---|---|---|
| `model/splats.py` | 260 | ~180 | 删 LR 调度器，kNN 转发 |
| `train/losses.py` | 400 | ~280 | normal/ramp 改调上游 |
| `train/pose.py` | 90 | **0** | 用 `examples/utils.CameraOptModule` |
| 其它 | 不变 | | 上游无对应能力 |
| **总计** | ~2000 | ~1700 | 净减约 300 行 |

减得不多，因为**真正有价值的部分（gauge、resolution、depth remap、DGI）上游都没有**。
复用的主要收益不是行数，而是：
1. `normal_cosine_loss` 的 unit-norm 契约断言
2. `ExponentialLR` 自动进 `optimizer.state_dict()`
3. `export_splats` 三种格式
4. torchmetrics 指标与归档基线可比

---

## 8. 一条重要提醒

`examples/` **不是可 import 的包**（没有 `examples/__init__.py`）。
官方测试的做法是：

```python
# tests/test_dataset_colmap.py:18-22
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES))
from datasets.colmap import Dataset
```

所以若复用 `examples/utils.py`（`CameraOptModule`），有两个选择：
1. `sys.path` 注入后 `import utils`（脆弱，`utils` 是极通用的名字）
2. **复制这 ~50 行到 `surfel/train/pose.py`**，注明来源与许可（Apache-2.0）

**建议选 2**：`utils` 这个名字在 `sys.path` 里冲突风险太高，
而且 `examples/utils.py` 顶层还 import 了 `matplotlib`/`sklearn`，为了一个 Module 不值得。
