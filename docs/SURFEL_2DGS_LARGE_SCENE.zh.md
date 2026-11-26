# 大场景 2DGS 训练架构：关键要点与配置契约

> 适用范围：以 **gsplat 1.6.0** 为骨干、面向航拍/城市级大场景的 2DGS 训练管线的设计与配置依据。
> 所有数值均为本机实测或从既有实验归档中核出的结果，不是文献值或估计值。
>
> 环境：`gsplat 1.6.0`（commit `512d366b`，editable 安装于 `/home/cngc209/Github/gsplat`）、
> `torch 2.12.1+cu132`、Python 3.12、aarch64（GB10 统一内存）、`pycolmap 4.3.0`、`scipy 1.17.1`。
> 解释器：`~/.venv/3DGen/bin/python`。
>
> **交付状态**：本文档描述的设计已在 `surfel/` 包中实现，但**该包尚未随本提交入库**
> （计划作为独立的 feat 分支提交）。因此文中 `surfel/...` 形式的路径引用是设计记录，
> 当前仓库中不存在。§1-§3、§5-§8、§12 的结论只依赖 gsplat 源码与实测，可独立成立。

---

## 目录

1. [环境事实与本地补丁](#1-环境事实与本地补丁)
2. [后端内置预/后处理 → 前端必须做的调整](#2-后端内置预后处理--前端必须做的调整)
3. [渲染器配置契约](#3-渲染器配置契约)
4. [训练参数配置](#4-训练参数配置)
5. [密度控制](#5-密度控制)
6. [世界系与 gauge](#6-世界系与-gauge)
7. [分辨率与深度网格对齐](#7-分辨率与深度网格对齐)
8. [损失函数语义](#8-损失函数语义)
9. [数据格式与解析](#9-数据格式与解析)
10. [已量化的失败案例库](#10-已量化的失败案例库)
11. [验证协议与判据](#11-验证协议与判据)
12. [配置速查表](#12-配置速查表)

---

## 1. 环境事实与本地补丁

### 1.1 版本与能力开关

| 项 | 值 |
|---|---|
| gsplat 版本 | **1.6.0**（不是 `requirements/gsplat.txt` 里写的 1.5.3，该文件已过时） |
| commit | `512d366b`（`[perf] 3DGS training ... 13.9 ms to 9.8 ms per step`） |
| `csrc.so` | 64 MB，重编于 2026-11-25 16:22 |
| 编译参数 | `NUM_CHANNELS=1,2,3,4,5,6,8,9,16,17,21,23,24,32,33,64,65,128,129,256,257,512,513`<br>`BUILD_CAMERA_WRAPPERS=1 MAX_JOBS=6 TORCH_CUDA_ARCH_LIST=12.1` |
| 能力开关 | `_C` / `has_2dgs` / `has_3dgs` / `has_3dgut` / `has_adam` / `has_camera_wrappers` / `has_losses` / `has_reloc` 全 True |

**通道白名单是硬约束**：1.6.0 只接受编译期 `GSPLAT_NUM_CHANNELS` 白名单里的通道数。
自定义特征维度（如 7/10/13）会直接 `ValueError: Unsupported number of color channels: 7`，
而在 1.5.3 下是静默补零可用的。若需要新通道数必须重编。

### 1.2 两处上游 C++ 缺陷（已修复并提交，`45e0760b`）

这两处是上游 1.6.0 的真实缺陷，已在本仓库修复、加测试并提交。
**换机器/重装时若基于未打补丁的 1.6.0，必须重新应用。**

#### 补丁 A：`Rendering.cpp` — 2DGS dense 路径 `expand()` 产生 stride-0

```cpp
// gsplat/cuda/csrc/Rendering.cpp:1813
std::vector<int64_t> opacity_shape = batch_shape_with_2dgs(means, {C, N});
- projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape);
+ // expand() broadcasts with stride 0, but the rasterizer's CHECK_INPUT
+ // demands contiguous storage. Materialize the broadcast instead.
+ projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape).contiguous();
```

- **症状**：`packed=False` 且 `C ≥ 2` 时报 `opacities must be contiguous`
- **陷阱**：不能用 `repeat(shape)` —— 它的参数是每维**复制倍数**而非目标形状，会得到 `[C, N*N]`

#### 补丁 B：`Projection.cpp` — 组合模式反向的深度梯度是非连续视图

2DGS 前向把深度 `cat` 到特征末尾（`[C,H,W,3] + [C,H,W,1] → [C,H,W,4]`）。
反向时 autograd 对 `cat` 做切片 `grad[..., 3:4]`，`squeeze(-1)` 后得到 `[C,H,W]` 且末维 stride=4，
是真正的非连续视图，被 `CHECK_INPUT` 拒绝。

修复模式与 3DGS 既有路径同构（先 `.contiguous()` 再 `CHECK_INPUT`），涉及
`projection_2dgs_fused_bwd` 和 `projection_2dgs_packed_bwd` 两个入口，共 4 个梯度张量
（`means2d` / `depths` / `normals` / `ray_transforms`）。

- **注意**：两处的 kernel 实参顺序**不同**（fused 是 `means2d, depths, normals, ray_transforms`，
  packed 是 `means2d, depths, ray_transforms, normals`），不要"顺手对齐"，否则是静默的梯度错位。
- `contiguous()` 对已连续张量是 no-op（返回 self，不拷贝），正常情况下零额外开销。

**修复后验证**：`packed` × 5 种 `render_mode` 前向+反向 **10/10 通过**。

### 1.3 可用的辅助能力

| 能力 | 说明 |
|---|---|
| `rasterize_num_contributing_gaussians` | 1.6.0 原生，替代旧 fork 专有的 `hit_pixel_count`，用于 LOD / 可见性剪枝，**不再需要 `gspl_extras` 自编译扩展** |
| `scipy.spatial.cKDTree` | 1.17.1 可用。kNN 尺度初始化用它是 O(N) 内存、约 1 秒；torch 暴力法在 182 万点上需 **90 GB** |
| `diff_surfel_rasterization`（hbb1 后端） | 已安装在 3DGen venv，可通过 `rasterization_2dgs_inria_wrapper` 对拍 |

---

## 2. 后端内置预/后处理 → 前端必须做的调整

**这是最容易静默算错的部分**。gsplat 的 C++ 编排层已经做了一批"用户会以为要自己做"的处理。
若前端按直觉再补一遍，结果**不会报错，只会错**。以下四类全部经过数值对拍确认。

### 2.1 深度通道已按 alpha 归一（E 模式）

`Rendering.cpp:1914-1919`：

```cpp
if (expected_depth) {
    at::Tensor expected = render_colors.narrow(-1, channels-1, 1) / render_alphas.clamp_min(1e-10);
    render_colors = ... at::cat({narrow(...,0,channels-1), expected}, -1) : expected;
}
```

**前端错误做法**：渲染器里再除一次 `alpha`。
**后果**：`surf_depth` 偏大 `1/alpha` 倍，直接毒化深度监督。
**实测**：错误版偏差 **1.72**；正确版 `|HL_depth − acc/alpha| = 0`。

**注意区分**：
- `expected_depth`（`Ed`/`ED`/`RGB-Ed`/`RGB+ED`）→ **已归一**
- `render_median`（中值深度，独立返回）→ **未归一**

判据：`render_mode_has_expected_depth(mode)`（`Rendering.cpp:1570`）。

### 2.2 法线已转到世界系

`Rendering.cpp:1931-1933`：

```cpp
at::Tensor camtoworlds_rotation = at::linalg_inv(viewmats).narrow(-2,0,3).narrow(-1,0,3);
render_normals = at::einsum("...ij,...hwj->...hwi", {camtoworlds_rotation, render_normals});
```

**前端错误做法**：再乘一次 `w2c` 旋转矩阵（"把相机系法线转到世界系"）。
**后果**：双重变换，`normal_loss` 完全错误。
**实测**：相机=世界系时两者 diff = 0（此时正确实现必须无可见差异）。

### 2.3 SH 的 `+0.5` 与 `clamp_min` 在 kernel 内

`Rendering.cpp:1647` 附近对 SH 求值后做 `clamp_min(sh + 0.5, 0)`。

**前端错误做法**：Python 侧再 `+0.5`。
**后果**：颜色整体偏亮。
**实测**：RGB maxdiff = **0.0**（一致）。

### 2.4 `render_normals_from_depth` 的产出条件比文档更窄

`Rendering.cpp:1735`：

```cpp
const bool compute_normals_from_depth = render_mode_has_depth(render_mode) && has_color;
```

| render_mode | `surf_normals` |
|---|---|
| `RGB` | `None` |
| `d` / `Ed` | **`None`**（有深度通道但没颜色） |
| `D` / `ED` | **`None`** |
| `RGB-d` / `RGB-Ed` | **`None`**（有深度通道但用的是 hit-distance 语义，`render_mode_has_depth` 为 False） |
| **`RGB+D` / `RGB+ED`** | **有** |

**形状陷阱**：`C == 1` 时返回 `(H, W, 3)`，缺 batch 维；`C > 1` 时返回 `(C, H, W, 3)`。
若直接与 `[1,H,W,3]` 项广播，会静默错位。

### 2.5 其它需注意的语义

| 项 | 事实 |
|---|---|
| `packed` 影响 `gradient_2dgs` 形状 | `packed=True` → `[N,2]`；`packed=False` → `[C,N,2]` |
| `distloss=True` 必须有深度通道 | 否则 `RuntimeError: distloss requires a depth render mode` |
| `scales` 必须是 `[N,3]` | `[N,2]` 报 `scales must have shape [..., N, 3]` |
| `render_colors` 已 clamp | 无需 Python 侧再 clamp（评估时仍建议 clamp 以防数值抖动） |
| `backgrounds` 需 `.contiguous()` | wrapper 里已处理，但自行构造时要注意 |

---

## 3. 渲染器配置契约

### 3.1 调用签名

```python
from gsplat.rendering import rasterization_2dgs

(colors, alphas, normals, surf_normals, distort, median, meta) = rasterization_2dgs(
    means,          # [..., N, 3]
    quats,          # [..., N, 4]  wxyz，内部会归一化
    scales,         # [..., N, 3]  第 3 轴在 kernel 内被丢弃
    opacities,      # [..., N]
    colors,         # [N, K, D]（SH）或 [..., N, D]（后激活颜色）
    viewmats,       # [..., C, 4, 4]  world→camera
    Ks,             # [..., C, 3, 3]
    width, height,
    near_plane=0.01, far_plane=1e10,
    radius_clip=0.0,
    eps2d=0.3,
    sh_degree=None,
    packed=False,
    tile_size=16,
    backgrounds=None,
    render_mode="RGB",
    sparse_grad=False,
    absgrad=False,
    distloss=False,
    depth_mode="expected",   # "expected" | "median"
)
```

**注意**：
- `rasterization_2dgs` **没有** `rasterize_mode` 参数（那是 3DGS 的）
- `tile_size` 是普通 `int`
- 内部已对所有主参数 `.contiguous()`，但 `Ks`/`viewmats` 的 batch 维度推断依赖 `means.dim()`

### 3.2 `meta` 字典的键

```
camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, opacities,
normals, tile_width, tile_height, tiles_per_gauss, isect_ids, flatten_ids,
isect_offsets, width, height, tile_size, n_cameras, render_distort,
gradient_2dgs   ← 密度控制的梯度来源
```

**关键**：2DGS 必须用 `key_for_gradient="gradient_2dgs"`，**不是** `"means2d"`。

### 3.3 已验证的渲染矩阵（当前 `csrc.so`）

| packed | RGB | D | ED | RGB+D | RGB+ED |
|---|---|---|---|---|---|
| True | ✅ | ✅ | ✅ | ✅ | ✅ |
| False | ✅ | ✅ | ✅ | ✅ | ✅ |

多相机维 `C ∈ {1,2,4}` × 上述全部 × 前向+反向：**全部通过**。

`gradient_2dgs.grad` 非零（`nz > 0`），`means.grad` 非零。

### 3.4 `absgrad=True` 在 2DGS 上是**不可用**的

实测：

```
absgrad=False: gradient_2dgs 有 .absgrad? False | means2d 有 .absgrad? False
absgrad=True:  gradient_2dgs 有 .absgrad? False | means2d 有 .absgrad? True
DefaultStrategy(absgrad=True, key_for_gradient="gradient_2dgs")
  → AttributeError: 'Tensor' object has no attribute 'absgrad'
```

**根因**：2DGS 的 CUDA 反向只写 `v_densify`（有符号累加），没有绝对梯度的实现分支；
`means2d_absgrad` 虽然被计算并挂到 `meta["means2d"].absgrad`，但它不是密度控制的消费对象。

**处理**：配置层显式拒绝 `density.absgrad=True`，报错说明原因，而不是让它在中途 `AttributeError`。

### 3.5 多 batch 轴约束（B > 1）

若确实需要 batch 维（本项目训练不用，只用 C 轴）：

```python
means    = torch.randn(B, N, 3)     # [B,N,3]
colors   = torch.rand(B, N, K, 3)   # ❌ 报 "SH coefficients must have shape [N,K,D]"
colors   = torch.rand(N, K, 3)      # ✅ SH 系数共享，不带 batch
viewmats = torch.randn(B, C, 4, 4)  # 必须匹配 means 的 batch 维
```

两个易踩坑：
1. **SH 不能带 batch**。这是有意契约（`Rendering.cpp:1621`，注释写明 "shared across batch/camera dims"），一套高斯只有一份颜色表。错误信息容易被误读成"不支持多 batch"。
2. **`viewmats` 的维数由 `means.dim()` 推导**（`batch_ndim = means.dim() - 2`，要求 `viewmats.dim() == batch_ndim + 3`）。`means=[N,3]` 必须配 `viewmats=[C,4,4]`，给 `[1,C,4,4]` 会被拒。

### 3.6 hbb1 后端对拍入口

```python
from gsplat.rendering import rasterization_2dgs_inria_wrapper
out, meta = rasterization_2dgs_inria_wrapper(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    near_plane=0.01, far_plane=100.0, eps2d=0.3, sh_degree=None,
    backgrounds=None, depth_ratio=0
)
# out[0] = [C,H,W,4]（RGB + depth 已拼接）
# meta keys: normals_rend, normals_surf, render_distloss, means2d,
#            width, height, radii, n_cameras, gaussian_ids
```

**差异**：wrapper 内部自己做 `quats = F.normalize(...)` 和 `scales = scales[:, :2]`，
且 `eps2d` 断言为 0.3（CUDA 内硬编码）。它**不返回** `gradient_2dgs`，
密度梯度只能从 `meta["means2d"].grad` 取。

---

### 3.7 接口数据规范审计：后端预/后处理全清单

**这是本项目反复踩坑的根源**：gsplat 的 C++ 编排层做了大量"用户会以为要自己做"的处理，
而且**同一个语义在不同层实现**。前端若按直觉补一遍，结果不报错、只是错。

#### 3.7.1 `rasterization_2dgs` 的 Python 侧几乎什么都不做

`rendering.py:1509-1535` 的 wrapper 只做两件事：

```python
_make_lazy_cuda_func("rasterization_2dgs")(
    means.contiguous(), quats.contiguous(), scales.contiguous(),
    opacities.contiguous(), colors.contiguous(), viewmats.contiguous(),
    Ks.contiguous(), width, height, tile_size, eps2d, near_plane, far_plane,
    radius_clip, backgrounds.contiguous() if ... else None,
    packed, sparse_grad, absgrad, distloss, sh_degree, render_mode, depth_mode,
)
```

1. 全部参数 `.contiguous()`
2. `if absgrad: means2d.absgrad = means2d_absgrad`（仅挂属性）

**除此之外 Python 侧零处理**。所有归一化、SH 偏移、法线旋转全在 C++。

> **注意**：`F.normalize(quats)` 出现在 `rendering.py:1283` 和 `:1608`，但那是
> `rasterization_2dgs_inria_wrapper`（hbb1 后端）里的 —— hbb1 不做内部归一化。
> **主路径 `rasterization_2dgs` 不需要**（文档明确写 "It's not required to be
> normalized"）。照抄 hbb1 wrapper 的代码到主路径会造成双重归一化（数值上无害，
> 但会多一次不必要的 kernel）。

#### 3.7.2 expected depth 的归一化在**三个不同位置**

这是最容易出错的一处：同一个语义（`Ed`/`ED`/`RGB-Ed`/`RGB+ED` 的 alpha 归一化）
有三份实现，走哪一条取决于 render_mode 与是否有 extra signals：

| # | 位置 | 代码 | 何时生效 |
|---|---|---|---|
| **A** | `Rendering.cpp:1914-1919`（2DGS C++） | `render_colors.narrow(-1,ch-1,1) / render_alphas.clamp_min(1e-10)` | **2DGS 所有 expected 模式** |
| **B** | `Rendering.cpp:718-740`（3DGS C++，`postprocess_3dgs_outputs`） | 同式，但只在 `has_extra_signals` 分支 | 3DGS + extra signals |
| **C** | `rendering.py:1074-1080`（3DGS Python） | `render_depth / render_alphas.clamp(min=1e-10)` | **3DGS 无 extra signals** |

**对 `surfel` 的含义**：

- 2DGS 走路径 **A**，所以前端**绝对不能再除 alpha**。实测错误版偏差 **1.72**。
- 若将来支持 3DGS：路径 C 意味着**Python 侧已经归一化了**，同样不能再除。
  但路径 B 说明带 extra signals 时 C++ **也**会归一化一次 —— 两条路径的边界
  由 `has_extra_signals` 决定，**不能靠"3DGS 在 Python 归一化"这条记忆**。

**统一规则：拿到 `render_colors[..., 3]` 时，假定它已经是 expected depth。**
若要 raw accumulation，用 `render_median`（它**永不**归一化）。

#### 3.7.3 全部后端内置处理清单

| 处理 | 位置 | 前端若重复做的后果 |
|---|---|---|
| **depth 按 alpha 归一**（expected 模式） | 2DGS `Rendering.cpp:1914`；3DGS `:725`/`:737` 或 `rendering.py:1074` | `surf_depth` 偏大 `1/alpha`，实测偏差 **1.72** |
| **法线转世界系** | `Rendering.cpp:1931-1933` `einsum(camtoworlds_rotation, render_normals)` | 双重变换，`normal_loss` 全错 |
| **SH `+0.5` 与 `clamp_min(0)`** | 2DGS `Rendering.cpp:1653`；3DGS `:642-643`（由 `clamp_after_bias` 控制） | 颜色偏亮，实测 RGB maxdiff **0.0** 时才是对的 |
| **四元数归一化** | kernel 内（`quat_to_rotmat`） | 主路径多做一次无害但多余；**hbb1 wrapper 需要**（`:1283`/`:1608`） |
| **scales 只用前两轴** | `Projection2DGSFused.cu:166`，构造 `mat3(sx,0,0, 0,sy,0, 0,0,1)` | 第三轴永远不渲染 |
| **背景合成** | kernel 内（`backgrounds` 参数） | 前端再加背景会重复 |
| **`render_median` 不归一化** | 独立返回 | 误当 expected depth 用会偏小 `alpha` 倍 |
| **`render_colors` 已 clamp 到 [0,1]**（RGB） | SH 求值后 | 评估时仍建议 clamp（防数值抖动），但训练 loss 不需要 |

#### 3.7.4 `viewmats` 的方向：只接受 w2c

三个入口都在内部做 `at::linalg_inv(viewmats)` 或 `torch.linalg.inv_ex(camtoworlds)`
来取 c2w（`Rendering.cpp:1927`、`:1932`）。所以：

- 传 `viewmats` = **world→camera**
- 传 `camtoworlds` 给示例的 `rasterize_splats` = camera→world，由它自己求逆

**混用是静默错误**：一个转置的 4×4 矩阵仍然是合法输入，会渲染出"看起来有结构但几何错位"的图像。

#### 3.7.5 三个 `clamp_min` 的 epsilon 不同

| 位置 | epsilon |
|---|---|
| 2DGS C++ expected depth | `1e-10` |
| 3DGS C++ expected depth | `1e-10` |
| 3DGS Python expected depth | `1e-10` |
| 2DGS normal-from-depth | `1e-12`（`Rendering.cpp:1698`） |

前三个一致；**第四个是 1e-12 且只 clamp 前向不 clamp 导数** —— 这正是
§10.1 #26 那个 NaN 的来源（零覆盖像素处 `d(sqrt(x))/dx → ∞`）。

#### 3.7.6 `depth_mode` 影响的不只是深度

`rasterization_2dgs(..., depth_mode="expected"|"median")` 同时决定：

1. `expected_depth` 是否归一化
2. **`render_normals_from_depth` 用哪个深度**（`Rendering.cpp:1924-1926`）：
   ```cpp
   at::Tensor depth_for_normal = depth_mode_is_median ? render_median
                                                      : render_colors.narrow(-1, channels-1, 1);
   ```

所以切到 `median` 会同时改变 normal-consistency 项的目标 —— 两个量耦合，
不能只改一个。

---

## 4. 训练参数配置

### 4.1 学习率

| 参数 | gsplat 官方 | GSL 默认 | 说明 |
|---|---|---|---|
| `means` | `1.6e-4 × scene_scale` | `means_lr_init 1.6e-4` | 官方**乘 scene_scale**，GSL 乘 `spatial_lr_scale` |
| `means` 衰减 | **有**：`ExponentialLR(gamma=0.01**(1/max_steps))` | 1.6e-4 → 1.6e-6 / 30k 步 | **两者数值等价**（已实测，见下） |
| `scales` | `5e-3` | `5e-3` | 一致 |
| `quats` | `1e-3` | `1e-3` | 一致 |
| `opacities` | `5e-2` | `5e-2` | 一致 |
| `sh0` | `2.5e-3` | `shs_dc_lr 2.5e-3` | 一致 |
| `shN` | `2.5e-3 / 20` | `shs_rest_lr 2.5e-3/20` | 一致 |
| Adam betas | `(0.9, 0.999)`，`eps=1e-15/√B` | 同 | 官方按 batch 缩放 |

**`means` LR 衰减：两套实现数值等价**（此前误记为"官方无衰减"，实测更正）：

```python
# gsplat 官方 examples/simple_trainer_2dgs.py:530-535
ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / max_steps))
```

逐步对拍（`lr_init=1.6e-4`，终点 `0.01×`）：

| step | 官方 `lr_init·gamma^step` | GSL `exp(lerp(log lr_init, log lr_final, t))` | 比值 |
|---|---|---|---|
| 0 | 1.6000e-04 | 1.6000e-04 | 1.000000 |
| 1750 / 7000 | 5.0596e-05 | 5.0596e-05 | 1.000000 |
| 3500 / 7000 | 1.6000e-05 | 1.6000e-05 | 1.000000 |
| 7000 / 7000 | 1.6000e-06 | 1.6000e-06 | 1.000000 |

两者数学上是同一个函数（指数衰减 = 对数空间线性插值），**7k 与 30k 预算下都逐位一致**。
所以 `train.lr.decay_means` 不是"官方 vs 归档"的差异项，而是两者共有的行为。

**batch 缩放公式**（gsplat 官方）：

```python
lr_effective = lr * sqrt(batch_size)
eps          = 1e-15 / sqrt(batch_size)
betas        = (1 - batch_size*(1-0.9), 1 - batch_size*(1-0.999))
```

`B=1` 时退化为普通 Adam。

### 4.2 `spatial_lr_scale` 三档实测（GT raw gauge，extent=531）

| 值 | 实际 means LR | 位移（% span） | PSNR |
|---|---|---|---|
| `0.2` | 3.2e-5 | 0.00037% ≡ **位置冻结** | 21.42（只损失 0.24 dB） |
| `50` | 8.0e-3 | — | **22.03（最优）**，但 63% 新生点 |
| `-1`（auto=extent 531） | 8.5e-2 | — | 21.42（**过冲**，几何恶化 11.7%） |

**结论**：raw gauge 下这个参数没有好答案（冻结和过冲都能跑出"看得过去"的数字）。
归一化 gauge 下应固定为 `1.0`（means LR = 设计值 1.6e-4）。

GSL 语义：`spatial_lr_scale <= 0` 时取 `camera_extent`，
`camera_extent = max‖cᵢ − mean(c)‖ × 1.1`（`internal/dataparsers/dataparser.py:75-84`）。

### 4.3 步数预算

| profile | `max_steps` | densify 窗口 | 依据 |
|---|---|---|---|
| 归档 MatrixCity | **7000** | 500 → 7000（全程） | `outputs/f0_gtpose`、`strip0` 等全部为 7k |
| gsplat 官方 Mip-NeRF360 | 30000 | 500 → 15000 | `simple_trainer*.py` |
| 续训（不增点） | — | 必须设 `densify_until_iter = 当前步数` | 否则续训期仍在增点 |

**7k 与 30k 不可混比**：续训 7k→14k 只值 **+0.48 dB**，且来自致密化（+38% 高斯），
几何反而恶化（NN-to-GT p50 3.626→3.734，mean +12.2%）。

### 4.4 位姿优化（默认关）

| 实验 | 结果 |
|---|---|
| strip0 `s3rpose_opt` | 17.697 vs 17.973 → **−0.27 dB** |
| flight0 归一化 N1 | 20.61 vs 21.81 → **−1.20 dB** |
| mixedD trainable | 19.30 vs 21.60 → **−2.30 dB** |
| frozen 纯光度 BA | rot 1.15° → **12.87°** @30k，**0/264 改善**，但 loss −4% 且 val PSNR 上升 |

**统一规律**：位姿误差与"跨帧一致性地板"的比值。
Scal3R 在 MatrixCity 上的残差（0.60°/0.48% 量级）**已在地板内**，纯光度优化没有信息可加，
只会把位姿推向"光度最优"而非"几何正确"。

**Adam 预算定律**（定量解释为何大场景下位姿"看起来冻结"）：

```
delta ≈ 渲染次数 × 有效步幅系数 × pose_lr
raw gauge 系数实测 3.48，归一化 gauge ≈ 1.0
flight0: 26 渲染 × 3.48 × 3e-4 ≈ 0.0277/相机
raw gauge 平移需 7.9 → 预算差 300 倍（平移实际冻结）
归一化 gauge 需 0.0292 ≈ 预算 0.0277（95% 可达）
```

**推论**：gauge 决定 pose_lr 是否够用；归一化使 `pose_lr=3e-4` 可跨场景迁移。

### 4.5 图像缓存

| 项 | 数值 |
|---|---|
| 1080p 单帧 uint8 | 1920×1080×3 ≈ 6.2 MB |
| 1000 帧 | ≈ 6 GB |
| 7k 步 × batch 1 | 每帧约 7 次解码（30k 步则 30 次） |

gsplat 上游 `d5e65e14` 引入 `--cache_images`。大场景下缓存常是"时间花在渲染还是解码"的分水岭，
但它占用的是用户可能没有的内存 → **默认关，由配置显式开启**。

---

## 5. 密度控制

### 5.1 阈值口径：已澄清的历史悬案

`DefaultStrategy._update_state`（`gsplat/strategy/default.py:243-249`）：

```python
grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

早期分析怀疑这可能与 GSL/hbb1 的 `0.5 * [W,H]` 口径不一致。**实测结论：一致，不需要重标定。**

同一输入（3000 个平面高斯、960×540、斜视航拍相机、随机纹理目标）：

| 来源 | 梯度定义 | p50 | p90 | p99 | >2e-4 | >6e-4 |
|---|---|---|---|---|---|---|
| gsplat | `gradient_2dgs × (W/2, H/2)` | **6.464e-4** | 1.958e-3 | 3.550e-3 | 80.2% | 53.1% |
| hbb1 | `means2d` 原始（像素单位） | **6.542e-4** | 1.959e-3 | 3.551e-3 | 80.6% | 53.3% |

差 1%。**因此 GSL 的 `densify_grad_threshold = 6e-4` 在 gsplat 后端下直接有效，
不能乘除任何系数。**

> 反例警示：在纯色目标 + 全覆盖的极端配置下，梯度会掉到 `1e-8` 量级（无残差 → 无梯度），
> 此时任何阈值都不触发。这不是口径问题，是"没有可学的信息"。

### 5.2 `scene_scale` 三个不同含义（**必须分离**）

这是大场景最容易崩的一处。同一个名字在三个地方指三件不同的事：

| 用途 | 含义 | 数量级（flight0） |
|---|---|---|
| `world.normalize_scale` | 把世界缩到 ~1 的相似变换因子 | `1/d_p99 ≈ 3.5e-3` |
| `strategy.scene_scale` | `DefaultStrategy` 除高斯尺寸的参考 | ~1（归一化后） |
| `loss.depth_scale` | 视差 L1 的乘子 | ~中位深度，归一化后 ~1 |

**上游 2DGS 示例的实际 bug**：`examples/simple_trainer_2dgs.py:370` 调用
`self.strategy.initialize_state()` **不传参**，`scene_scale` 默认 1.0；
而 3DGS 示例（`simple_trainer.py:517`）正确传了 `scene_scale=self.scene_scale`。

实测后果（`grow_scale3d=0.01`、`prune_scale3d=0.1`、真实场景 extent 530 m）：

| `scene_scale` | 探针尺寸 0.05 / 0.2 / 1.0 / 5.0 m |
|---|---|
| `1.0`（示例默认） | duplicate-eligible = `[F, F, F, F]`<br>prune-as-too-big = `[F, T, T, T]` |
| `530.0`（真实） | duplicate-eligible = `[T, T, T, T]`<br>prune-as-too-big = `[F, F, F, F]` |

即：**默认值会把所有结构性高斯当"过大"剪掉，且 duplicate 分支永不可达**。

### 5.3 `grow_scale3d` 的语义在 DGI 下会反转

`is_small = exp(scales).max(-1) <= grow_scale3d * scene_scale`

SfM 稀疏点 → kNN 间距大 → 多数高斯被判 "large" → 走 **split**。
DGI 稠密点 → kNN 间距 = 像素足迹 → 多数被判 "small" → 走 **duplicate**。

**matrixcity strip 实测**（300k DGI 点，`scene_scale=3.60`）：

```
点间距（kNN RMS）中位数     0.0055
默认阈值 0.01 × 3.60        0.0360   ← 大 6.6 倍
被判 "small" 的比例         99.9%
```

**注意**：这个阈值**不决定高斯总数**，只决定候选走哪个分支（两个分支都增点）。
增长速率由 `grow_grad2d` 决定（同一场景实测：step 500 时 11% 的高斯超 2e-4，
145 次 refine 事件后仍涨到 4.2M）。

**garden 过度致密化是 gsplat 1.6.0 固有行为**：SfM 点云 RMS 0.039 < `grow_scale3d × scene_scale` = 0.049
→ 多数走 duplicate；7k 步涨到 176 万（归档基线 31.3 万），用官方代码同样涨到 75.4 万。

### 5.4 2DGS 的第三尺度轴：kein 渲染，但参与决策

`Projection2DGSFused.cu:166`：

```cpp
R * quat_to_rotmat(quats) * mat3(scales[0], 0.0, 0.0,
                                0.0, scales[1], 0.0,
                                0.0, 0.0, 1.0);
```

第 3 轴**完全不参与渲染**，梯度恒为 0，永远是初值。但：

1. `DefaultStrategy` 用 `exp(scales).max(dim=-1)` 判 duplicate/split → 第 3 轴会污染 `max()`
2. `split` op 用 `rotmat @ scales @ randn` 采样子高斯位置 → 过大的第 3 轴会把子高斯甩出去

**处理**：初始化时把第 3 轴压到 `1e-6`（`dead_axis_scale`），使其不参与 `max()`。
**不要**用 kNN 间距初始化它。

### 5.5 失效检测（必须有的护栏）

本项目代价最大的一次静默失败：旧代码从 `proj.means2d` 读致密化梯度，
而 1.6.0 的 2DGS kernel **从不写这个张量**。结果：

```
bench_a_gsplat:  step 500 → 3000 只掉 0.55%（6,534 个）
gsl_bench_a:    同期涨 37.9%
```

致密化全程哑火，唯一症状是模型高斯数偏小（525K vs 2.1M），PSNR 低 5.4 dB。

**双变量污染的教训**：`strip0_gsplat` 报 22.22 dB（比 vanilla 高 1.89），但它的
`spatial_lr_scale=0.2` 而对照组是 `-1`，LR 差 **18.7 倍**（3.2e-5 vs 5.98e-4），
且配置文件是在跑完之后 13 分钟才改的。那个 "+1.89 dB" **不可引用**。

**护栏设计**：

```python
# 每次 refine 事件记录：
#   grad_p50 / p90 / max、超过阈值的比例、高斯数变化
# 连续 N 次 refine 事件零增长 → raise DeadDensifierError
# 并在错误信息里给出诊断分支：
#   over_frac == 0  → 梯度没有到达策略（key 错了 / pre_backward 顺序错）
#   over_frac 很高  → grow/prune 尺寸阈值把所有候选都拒了（scene_scale 不匹配）
```

### 5.6 MCMC 策略

`MCMCStrategy` 的参数与 `DefaultStrategy` 不重叠：

| 参数 | 默认 | 说明 |
|---|---|---|
| `cap_max` | 1,000,000 | **硬上限**，大场景的主要显存杠杆 |
| `noise_lr` | 5e5 | 采样噪声的学习率 |
| `refine_start_iter` | 500 | |
| `refine_stop_iter` | 25,000 | |
| `min_opacity` | 0.005 | |
| `noise_injection_stop_iter` | -1 | |

**`MCMCStrategy.initialize_state()` 不接受 `scene_scale`**（与 DefaultStrategy 不同），
且它自己维护高斯数上限，不需要外部的尺寸阈值。

### 5.7 `refine_scale2d_stop_iter`

默认 0（关闭）。上游注释说明屏幕尺寸分支在原实现里**因 bug 从未生效**
（graphdeco-inria/gaussian-splatting#123），gsplat 为完整性实现但默认关闭以保持等价。
**若要与 2DGS 官方结果对齐，保持 0。**

---

## 6. 世界系与 gauge

### 6.1 三类源的 gauge 差异

| 源 | 尺度由什么固定 | 参考系 |
|---|---|---|
| COLMAP / PixelSfM | 任意 | 自己的原点 |
| Scal3R | 任意 | 自己的原点 |
| VGGT / VGGT-X | 任意，且深度是**归一化逆深度** | 第一台相机 |
| GT / GNSS / RTK | 公制（米） | 世界 / ECEF |

### 6.2 两条规则（违反后训练不报错、渲染错误）

**规则 1：不同源的位姿与深度属于不同 gauge。**
Scal3R 的深度与它**自己估计的焦距**及自己恢复的尺度自洽。
把它与 SfM 位姿配对，得到两套不一致的几何：深度反投影出的点不落在相机看向的位置。
（实测某航拍场景两者焦距差 **10.4%**。）

**规则 2：对位姿施加相似变换时，深度也必须一起缩放。**
若 `x → s·R·x + t` 施加于相机中心，相机空间深度变为 `s` 倍。
深度图与逐帧 `{scale, offset}` 必须同乘 `s`，否则损失与几何在不同单位。

**规则 2 的实证代价**（strip0 两支）：

| 臂 | cloud_z / depth_map | 所需因子 | 视差地板 |
|---|---|---|---|
| s3rpose | 0.9644 | 0.964411 | 0.133 |
| gtpose | **0.3611** | **0.470753** | **1.941** |

→ 渲染/监督深度比必须为 1.0。gtpose 臂在 step 1 之前就损失了 **2/3** 的深度监督：

| 臂 | 实际 depth loss | 修正后 | 降低 |
|---|---|---|---|
| s3rpose | 0.5277 | 0.4413 | 1.2× |
| gtpose | 1.7449 | 0.5102 | **3.4×** |

**另一处同类 bug**：Mode B 实现里位姿**已经在 pose source 的 frame**，却又被 `s·(R@c+t)` 变换了一次
（双重变换）→ PSNR 15.63；去掉多余的位姿变换 → **19.42 dB（+3.8）**。
深度 `scale` 仍须乘 `s`（因为深度图每步重读）。
修正后 `scene_scale=1.0897` 与 Mode A 的 `1.0919` 几乎一致 —— 这是判断对齐是否正确的最好信号。

### 6.3 归一化只改数值条件，不改位姿误差（数学恒等）

R 不变（角度是 R 的不变量）、相对平移 `‖t‖/span` 是比值、投影像素只依赖 K 与**相对**位姿
→ **归一化后渲染逐像素等价，位姿误差完全不变**。

归一化唯一改变的：NDC 深度精度、`dist_loss` 量级、`means_lr` 量级、位姿学习率是否够用。

"能在统一世界系算误差"是 **Sim(3) 锚定**的功能，与归一化无关。
Sim(3) 群封闭 ⇒ "前馈/GT 直接到归一化 gauge"与"先锚 GT 再缩放"数学等价；
两步走的实际价值 = 留住 GT 参照系（可算误差、可跨航段比）。

### 6.4 归一化常数的选取：必须用深度分位数

`dist_loss` 的 `z_ndc` 是 `1/z` 的仿射函数，`L_dist ∝ (Δ(1/z))²`。
探针实测响应比（满值 64）：

| 深度 | 响应比 |
|---|---|
| 150 | 1.04（**噪声底**，fp32 耗尽） |
| 50 | 2.50 |
| 30 | 7.68 |
| 15 | 24.63 |
| 10 | 40.41 |
| 5 | 52.12 |
| 1.5–0.3 | 56–57（**满值**） |
| 0.20 | 27–31 |
| ≤0.15 | 0（`alpha=0`，无高斯被渲染） |

两端机制不同：高端是 NDC `1/z²` 精度（`alpha` 恒 0.992 时仍单调衰减）；
低端是渲染覆盖（`z ≤ 0.15` 时 `alpha=0`，缩小屏幕覆盖到 0.5 也救不回）。

三种尺度因子的实测（flight0）：

| 因子 | 失效像素占比 | 满值窗 |
|---|---|---|
| `1/extent`（531.18） | 6.8% | 72.8% |
| `1/bbox_diag` | 14.5% | **0%** |
| **`1/d_p99`** | **6.7%**（天空/近零无效像素，正常） | **92.7%** |

**建议公式**：`c = 1/d_p99`（更保守可 `0.6/d_p50`），并检查 `d_p1·c ≥ 0.15`。

**`zfar/znear` 不是根因**：实测改 `zfar` 100→1000→4000 或 `znear` 0.01→10，
`dist` 逐位不变（4.097e-08）。`cameras.py:159` 注释 "near and far ignored" 指不做裁剪，易误导。

### 6.5 归一化 gauge 契约（可直接复用的常数）

flight0 系数据集：`scene_scale = c = 1/d_p99`：

| 量 | 值 |
|---|---|
| `c` | `3.536e-3`（全 264 帧复测 3.710e-3，差异 5%，取文档值） |
| parser 作用范围 | 只缩放 c2w 平移与 points3D；**旋转/内参不动** |
| `spatial_lr_scale` | `1.0`（means LR = 设计值 1.6e-4） |
| `pose_lr` | `3e-4` |
| dist_loss 深度窗 | `[0.099, 1.0]`，满值 92.7% |

**契约常数不再随 gauge 重推**。新增数据集只需算自己的 `c`。
两种取值按用途选：
- dist 正则在场 → `c = 1/d_p99`（满值窗约束）
- 纯光度 pose-opt 预算场景 → `c = 1/extent` 即够（预算算术最直观）

### 6.6 Sim(3) 拟合

**用相机中心拟合**，不用深度：两边都可观测，且位姿误差直接体现在中心上。

**必须有反射保护**（Umeyama 1991）：`det(R) < 0` 时翻转 `R[2,2]`。
否则会静默返回镜像世界 —— 残差很小、位姿看起来正常、每张渲染图都是镜像的。

**残差是门控，不是装饰**。相似变换总能拟合出答案，残差才说明"这是不是同一个场景"：

| 配对 | Sim(3) 残差 |
|---|---|
| `scal3r_depth/*.npy` + 同 run 的 `sparse/0` | **8.6e-08** |
| 原始 EXR + 已对齐 `sparse/0`（跨 gauge） | **11.1%** |

相差 6 个数量级 → 5% 阈值既不误放也不误杀。

**旋转块保持刚性**：`R' = R·Δ`，不要给旋转乘 `s`（否则 `R^T R ≠ I`，
下游所有 `inv` 都静默失效）。

---

## 7. 分辨率与深度网格对齐

### 7.1 分辨率错配是本项目最贵的 bug 类

同一 ckpt，只改渲染分辨率：

| 渲染分辨率 | PSNR | SSIM |
|---|---|---|
| 1297×840 | 17.47 | 0.4257 |
| 648×420 | 19.44 | 0.6234 |
| **324×210** | **25.34** | **0.8665** |

**7.9 dB 完全由分辨率造成**，而评估器全程算得正确。

### 7.2 根因：`images_<n>` 目录的内容不一定是 `<n>` 倍下采样

真实的 garden 数据集：

- COLMAP 模型声明 **5187×3361**
- `images_4/` 里的文件实际是 **1297×840**（等效 /4）
- `images/` **不存在**

gsplat 官方 `Parser` 要求 `images/` 存在并从那里按 `factor` 下采样；
若 `images/` 缺失而代码取实际文件大小且"永不放大"，`factor=4` 就停在 1297，
而按语义应该是 5187/4/4 = 324。

### 7.3 正确的 policy（三条硬规则）

1. **`factor` 永远相对 COLMAP 模型声明的尺寸解释**，与文件实际大小无关
2. **`K` 按最终实际读取的网格逐轴缩放**（`sx = render_w / model_w`），并**保持 `K[2,2] = 1`**
3. **永不上采样**：若请求的目标大于文件实际尺寸，以文件尺寸获胜

`K` 缩放不能把整个矩阵除以 `s`（看似省事的做法）——那是投影对象的齐次归一化被破坏，
所有反投影都会失准。

### 7.4 深度 remap vs resize：`f` 是否一致决定用哪个

**情形 A：内参一致，仅分辨率不同** → `resize` 是**精确**的（纯缩放）。
COLMAP 投影深度、已 remap 的前馈深度属于这类。

**情形 B：内参不同**（前馈自报 `f` 与 SfM 不一致，实测差 **10.4%**）→ 必须**逐像素射线 remap**：

```python
# 目标像素 → 用渲染 K 反投影 → 用深度 K 投影 → 采样
cam_dirs = rays @ inv(K_render).T
src_uv   = (cam_dirs @ K_depth.T) / z
```

用 `resize` 代替等于假设两张图**视野相同**；当 `f` 不一致时，采样点系统性偏移，
**越靠边缘越严重**（已有测试证明误差随边缘增大）。

**采样必须用 nearest**：深度在物体边界处有阶跃，双线性会插出不属于任何表面的值。

**判据**（`depth_grid_differs`）：

```
if depth_hw == render_hw: False
sx = render_w / depth_w; sy = render_h
fx_implied = depth_K[0,0] * sx; fy_implied = depth_K[1,1] * sy
differs = |fx_implied - K[0,0]|/K[0,0] > 1% or |fy_implied - K[1,1]|/K[1,1] > 1%
```

### 7.5 Scal3R 的非等比缩放（必须逐轴处理）

1920×1080 → 518×280：

```
sx = 1920/518 = 3.707
sy = 1080/280 = 3.857
```

**sx ≠ sy**（Scal3R 把输入拉伸到固定短边）。用单一标量缩放 `K` 会让每一根射线都倾斜。

### 7.6 前馈分辨率与训练分辨率的关系

| 事实 | 结论 |
|---|---|
| 初始化点数与训练分辨率**完全解耦**（实测 factor=4/8/16 均为 138,766 点，位置/颜色相同） | 可以在任意分辨率初始化 |
| 训练可达精度受**渲染分辨率**限制（`1 px ↔ z/fx`） | 低分辨率是精度的硬上界 |
| 同一 SfM 初值，3000 步：324×210 → 341K 高斯（2.46×）、scale p50 0.0220；1297×840 → 1.10M（7.94×）、0.0143 | 提高分辨率确实促进细化 |
| 但 garden 上高分辨率 PSNR **低 3.3 dB**（用了 4.6 倍高斯） | 图像信息量不足时，多出的高斯去拟合噪声 |

**推荐**：训练分辨率 ≈ 数据源分辨率（前馈 518×392 → 训练 518×392 或整数倍 1036×784）。

### 7.7 主流实现的实际训练分辨率（都不是原分辨率）

| 实现 | 分辨率 |
|---|---|
| gsplat `simple_trainer.py` / `simple_trainer_2dgs.py` | `data_factor = 4` |
| CityGaussian `matrixcity_aerial` | `down_sample_factor: 1.2` |
| CityGaussian `lfls_coarse` | `3.4175` |
| GSL `garden_colmap_ref.yaml` | `down_sample_factor: 4` |
| INRIA 3DGS | 常用 `-r 2` / `-r 4` |
| 2DGS 原文 | 通常 `images_2`，DTU 800×600 |

**报指标必须连同分辨率**：跨分辨率比较毫无意义。

---

## 8. 损失函数语义

### 8.1 总目标

```
L = lerp(L1, 1−SSIM, λ_ssim)          λ_ssim = 0.2
  + w_depth(t) · L_disparity
  + w_dist(t)  · L_distortion
  + w_normal(t)· L_normal
  (+ opacity_reg / scale_reg，若配置)
```

### 8.2 深度：视差空间，不是米制

`gsplat.losses.depth_l1_loss(pred, gt, scene_scale)`：

```python
disp    = where(pred > 0, 1/pred, 0)
disp_gt = where(gt   > 0, 1/gt,   0)
return F.l1_loss(disp, disp_gt) * scene_scale
```

**为什么用 `1/d`**：深度误差是乘性的，米制 L1 会被最远处的像素主导，
而航拍场景的"远处"就是天空。

**两个已踩过的坑**：

1. **`1/pred_depth` 会爆**。`depth_eps` 若很小（1e-6），一个无覆盖的像素就产生 `1e6` 量级的值。
   MatrixCity f0_gtpose 在 step 3300 附近 `train/depth_loss` 从 **0.001 跳到 81,124**（7 个数量级），
   `train/loss` 同步从 0.14 爆到 174,693，训练彻底发散。
   → 必须有 `depth_eps` 配置 + 可选的 alpha 掩码（排除未被覆盖的像素）。
2. **`scene_scale` 乘子的取值**。上游把它当"场景尺度"，但在归一化世界里它应该是
   **中位监督深度**。用 camera extent 会让量级错（见 §6.4）。

### 8.3 逐帧中位数归一化（仅用于 VGGT 类源）

VGGT 预测的是 min-max 归一化**逆深度**，即使拟合了逐帧 scale，
可信的也只是**相对**结构，逐帧仍有几个百分点误差。不归一化时该 term 曾达到 **29.5**
（总 loss 才 ~0.2），压过光度项，训练反而变差。

**做法**：两侧各除以自己帧内有效像素的中位视差，返回无 `scene_scale` 乘子的纯均值。
（归一化后已无量纲，再乘场景 extent 就是错。）

**互斥约束**：`per_frame_median_norm` 与 `rescale` 不能同时开 —— 中位数归一化已经消除逐帧尺度，
先拟合再归一化是死工作，还会掩盖 gauge 问题。
同理，`per_frame_median_norm=True` 时 `gauge.apply_to_depth` 必须显式设 False，
不能"隐约不管"。

### 8.4 Distortion（2DGS 专用）

把每个 surfel 的深度分布往单值拉。

- **作用**：航拍城市里，不做这一项时每面建筑立面会变成"一叠平行半透明薄片"。
- **权重**：`0.1`，`dist_start_iter = 3000`（归档 strip/f0 配方）。
- **注意**：`distloss=True` 要求渲染模式带深度通道，否则直接报错。
- **与 PSNR 的关系**：flight0 实测 `dist` 正则把该项从 `1.3e-9`（噪声底）救到 `1.7e-4` 有效值，
  但 **5 个数量级的改善只值 −0.05 dB**（航拍城市几何本身近平面，没有多层叠加可抑制）。
  它的价值在几何质量，不在 PSNR。

### 8.5 Normal consistency（2DGS 专用，**不需要 GT 深度**）

```python
gt_normal = depth_to_normal(rendered_depth, c2w, K).detach()   # 模型自己的渲染深度
loss = mean(1 − cos(surf_normals, gt_normal) * alpha)
```

**两侧都来自模型本身**，是**自一致性正则**，作用是让扁平 surfel 贴平在"它自己的深度场隐含的表面"上。

**这是本项目第二个静默失效点**：旧代码用「有没有 GT 深度」决定「要不要渲染深度」：

```python
want_depth = cfg.data.depth.enabled and "depth" in sample   # ❌ 只看 GT
```

在 rubble（1573 帧，**0 帧有 GT 深度**）上 → `render_depth=False` → 纯 RGB →
`surf_normals` 返回 `None` → 该项**全程权重 0**，只每步刷一条警告。

**正确逻辑**是两个独立理由的并集：

```python
want_depth = want_gt_depth or want_normal or want_dist
```

**修复后实证**：rubble 无 GT 深度，60 步内 normal 项在 50 步激活，权 0.05，
张量 `(1, 432, 576, 3)` 正常传入；PSNR **19.28 → 19.46（+0.18 dB）**。

**权重与起点**：归档配方 `lambda_normal = 0.05`，`normal_start_iter = 2000`
（2DGS 官方 `simple_trainer_2dgs.py` 默认 0.05 / 7000）。

### 8.6 深度权重：**实际是常数 0.05，不是 ramp**（重要更正）

归档配方（`matrixcity_f0_gtpose.yaml`、`strip0_vanilla.yaml` 等 20 个配置）写的是：

```yaml
depth_loss_weight:
  init: 0.05
  final_factor: 1.0
  max_steps: 7000
```

**此前本文档写作"0.05 → 1.0 线性 ramp"，这是错的。** 真实公式在
`internal/metrics/gs2d_depth_metrics.py:123-126`：

```python
w = self.config.depth_loss_weight.init * (
    self.config.depth_loss_weight.final_factor
    ** min(step / self.config.depth_loss_weight.max_steps, 1))
```

是**几何（指数）插值**，不是线性：

```
w = init * final_factor ** t        t = min(step / max_steps, 1)
```

由于全部 20 个配置都用 `final_factor: 1.0`，而 `1.0 ** t ≡ 1.0`：

| step | t | w |
|---|---|---|
| 0 | 0.0 | 0.050000 |
| 3500 | 0.5 | 0.050000 |
| 7000 | 1.0 | 0.050000 |

**即深度权重自始至终恒为 0.05。** 若要实现 0.05→1.0 的几何 ramp，
`final_factor` 必须是 **20.0**（此时 step 3500 得 0.2236，step 7000 得 1.0）。
现有配置里没有任何一个这样做。

**这条更正的两个后果：**

1. **不要把 `final_factor` 当"终值倍数"直接搬到线性 ramp 实现上。** `surfel` 早期把
   `weight_final = weight * final_factor` 理解为线性终值，与 GSL 语义不同。
   正确映射是 `weight_final = weight * final_factor`（终值 = init × final_factor），
   但**插值方式必须是几何**，否则中间过程差很多（线性在 t=0.5 给 0.525，几何给 0.2236）。
2. 既然实际权重恒定，那么"深度权重 ramp"**不是**归档配方与 1080p 实验之间的差异项。
   strip0 那 ~2.4 dB 的协议差异只能归给：`split_mode`（reconstruction vs experiment）
   与正则项的有无。原先把 ramp 也算进去是过度归因。

### 8.6.1 GSL 的深度损失内置了逐帧中位归一化

`internal/metrics/gs2d_depth_metrics.py:82-87`：

```python
gt_n   = gt_inv   / gt_inv[valid].median().detach()
pred_n = pred_inv / gt_inv[valid].median().detach()   # 注意：用的是 GT 的中位数
```

**两侧都除以 GT 逆深度的中位数**，所以 GSL 的深度项**恒为无量纲**，
与 `scene_scale` / `depth_rescaling` 无关。这解释了两件此前看起来矛盾的事：

- 为什么 `f0_scal3r_norm.yaml` 注释说"深度图不需要随 scene_scale 缩放"——
  因为损失层已经把它归一化了；
- 为什么 `depth_rescaling: false` 还能正常工作。

代码注释还记录了一个**反面模式**：预测侧不能用 `pred` 自身的中位数归一化，
否则退化像素占比一高就会抬高预测中位数，导致正常像素被系统性压低，
损失对退化反而更宽容（"越坏越宽容"）。

**对 `surfel` 的影响**：`per_frame_median_norm` 不是"VGGT 专用"选项。
GSL 实际上是**对所有深度源都开**的。因此在复现归档基线时，
`surfel` 必须设 `data.depth.per_frame_median_norm=True`（且
`gauge.apply_to_depth=False`），否则损失量级与 GSL 不同、权重 0.05 不可直接迁移。
另外 `surfel` 用的是"各自中位数"，与 GSL 的"都用 GT 中位数"有差异——
复现时应改用 GSL 的语义（见 §8.6.2）。

### 8.6.2 `depth_outlier_clip` 是必需的第三道保护

同一文件 `:33` 有 `depth_outlier_clip: float = 10.0`，在算 `mean` 之前
`clamp(d, ±clip)`。注释直接指向 f0_gtpose step 3300 那次爆炸（0.001 → 81,124）。

`surfel` 目前有 `depth_eps` 与 `depth_alpha_mask` 两道保护，但**没有**离群栅栏。
理论上 alpha 掩码已排除 `pred_depth→0` 的像素，但若掩码阈值以下的像素仍有极小
正值（例如 alpha=0.005），`1/pred` 依然会给出尖峰。**建议补上**：

```python
if clip and clip > 0:
    diff = torch.clamp(diff, -clip, clip)
```

---

## 9. 数据格式与解析

### 9.1 COLMAP 模型

**读取不依赖 `pycolmap`**。原因：PyPI 上的 `pycolmap` 是一个 2018 年的无关旧包；
官方包无 aarch64 wheel，只能源码编译。直接读格式约 200 行，去掉这个可移植性税。

| 文件 | 内容 |
|---|---|
| `cameras.bin/.txt` | `camera_id → (model, width, height, params)`，**多图共享** |
| `images.bin/.txt` | `image_id → (qvec, tvec, camera_id, name)` |
| `points3D.bin/.txt` | `xyz, rgb, error, track` |

**约定**：
- 四元数是 **`(w, x, y, z)`**
- `images.bin` 的 `qvec`/`tvec` 是 **world→camera**
- OpenCV 相机约定：`+x` 右、`+y` 下、`+z` 前
- 模型目录可以是 `sparse/0/`、`sparse/<n>/`，或直接含 `cameras.bin`

**排序必须按名字，不能按 `image_id`**：COLMAP 的 id 按摄取顺序分配，
重新处理的模型不一定跟随飞行路径。轨迹来回跳会破坏所有"相邻帧对"统计（漂移、RRA）。

**内参恢复**：所有失真模型的前 4 个参数都是 `fx, fy, cx, cy`；
`SIMPLE_PINHOLE` 是 `f, cx, cy`。取 pinhole `K` 时忽略失真项，
但若数据本身有失真，必须使用 undistort 后的图像。

### 9.2 Scal3R 输出格式

| 文件 | 内容 | 陷阱 |
|---|---|---|
| `intri.yml` | `names[]`、逐帧 `K_*`、`H_*`、`W_*`、`D_*` | `K` 是**深度网格**的内参 |
| `extri.yml` | `Rot_*`（3×3，**world→camera**）、`T_*`（w2c 平移）、`R_*`（3×1 旋转向量） | 用 `Rot`，别用 `mat.txt` |
| `depths/NNNNNN.exr` | float32 z-depth，与 `K` 同网格 | |
| `mat.txt` | c2w，每行 16 个 float | **R 块含 `cbrt(det)` 因子，不是刚性矩阵**。`mat.txt` 的 R = `Rot^T / s`，逐帧 s 在 rubble 达 0.518–1.951 |
| `scal3r_depth/*.npy` | 已对齐版本 | 与 EXR 差 **108.805×**（已复算确认） |
| `estimated_depth_scales.json` | 逐帧 scale | rubble 上**全为 1.0**（属于已缩放的 `scal3r_depth`）；非 1.0 时应告警 |

**`K` 属于深度网格**：实测某场景 Scal3R 自报 `fx=379.92`（518×322 网格），
而 1920×1080 渲染网格对应 `fx=2316.39`。DGI 必须用深度网格的 `K` 反投影。

**`width/height` 必须是图像尺寸，不是深度网格尺寸**。曾把 `Camera.width/height`
设成 518×280 而实际图像是 1920×1080，导致分辨率归属彻底错乱。

**帧名映射**：Scal3R 用零填充索引（`000000`），原图是 `DSC07956.JPG` 或 `train_000001.jpg`。
需要 `name_map.json` 或 `runtime/image_manifest.txt` 自动推导。
**并且映射必须同步到 `Camera.name` 字段**，否则跨源匹配 0 帧命中（且无提示）。

### 9.3 MatrixCity `transforms.json`

**GT 位姿的 `rot_mat` 不是刚性的**：

| 项 | 实际 | 正确处理 |
|---|---|---|
| R 块 | 含 ×0.01 因子，`det = 1e-6` | `R *= 100`（得 `det=1.0`） |
| t | 单位是 **cm** | `t *= 0.01`（得米） |
| 列 1:3 | 符号约定与 OpenCV 不同 | `cols 1:3 *= -1` |

修好后：`det=1.0`、中心 `[-10, -0.76, 1.5]` 米制、`scene_scale=6.0`。
**修正前 `det=0` 的位姿根本不能用于训练或 Sim(3)**，而且不会报错。

**另一个坑**：cm→m 转换与通用 `scene_scale` 参数曾**重复应用**（227 行和 332 行各一次），
把轨迹缩了 100 倍。这两个语义必须分离。

### 9.4 深度文件匹配

导出器对文件名的处理不一致，需要依次尝试：

```
{stem}{suffix}   →  train_000001.exr
{name}{suffix}   →  train_000001.jpg.exr
{stem}.exr/.npy/.png
{name}.exr/.npy/.png
```

### 9.5 深度读取的格式细节

| 格式 | 处理 |
|---|---|
| `.exr` | OpenCV 需 `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"`，且**必须在 imgcodecs 初始化前设置** → 每次调用都 defensive set |
| `.npy` | 直接用 |
| 16-bit `.png` | `/65535` 归一到 [0,1]；**无声明单位**，调用方必须提供 scale |

---

## 10. 已量化的失败案例库

按"症状 → 根因 → 量化 → 检测手段"排列。这些是架构设计必须防住的。

| # | 症状 | 根因 | 量化 | 检测 |
|---|---|---|---|---|
| 1 | garden 13.4 dB（应 ~26），**无报错** | 324×210 图像配 1297×840 内参 | 12 dB | 渲染尺寸与 K 必须同源（§7） |
| 2 | 同上，同一 ckpt 不同分辨率 | 纯分辨率差 | 17.47 → 25.34（7.9 dB） | 报告必须带分辨率 |
| 3 | PSNR 低 5.41 dB、高斯数只有 1/2 | `viewspace_points` 指向 `proj.means2d`，1.6.0 从不写它 | step 500→3000 只掉 0.55%（对照 +37.9%） | 致密化计数器（§5.5） |
| 4 | 2DGS "退化成普通 3DGS" | dist/normal 权重传 0 | — | 权重打印 + 非零断言 |
| 5 | `normal_weight>0` 但该项全程无效 | 用 GT 深度开关控制"是否渲染深度" | PSNR 19.28 → 19.46 | `want_depth = gt ∪ normal ∪ dist`（§8.5） |
| 6 | 剪枝永不生效 | 可见性统计用错 `isect_tiles` 签名（返回 3 值，后两参是 tile 数而非像素，`means2d` 必须保留相机维） | — | 剪枝计数器 |
| 7 | 初始高斯偏小 | kNN 用算术平均而非 **RMS** | 破坏与参考的等价性 | RMS 对应 `distCUDA2` |
| 8 | **机器重启** | kNN 暴力 `[4096,N,3]` 在 182 万点上需 **90 GB**；统一内存下进 swap | 90 GB → 1.3 GB / 1 s（scipy） | 内存上限断言 |
| 9 | 崩在 `_update_state`（`None` 索引） | runner 硬编码 `packed=True` 传给 `step_post_backward`，渲染器按配置走 `packed=False` | — | 两处读同一个配置值 |
| 10 | `surf_normals` 广播错位 | `C=1` 时返回 `(H,W,3)`，缺 batch 维 | — | 形状归一化 |
| 11 | `gsl-eval` 必崩 | checkpoint 里的 config 返回裸 dict | — | 用同一个 validated setter 重建 |
| 12 | `--dump-config` 后 override 静默失效 | 同上 | — | 同上 |
| 13 | 整个数据层从未入库 | `.gitignore` 里未锚定的 `data/` 命中 `gsplat_lightning/data/` | 18 个文件 | 干净 clone 验证 |
| 14 | Mode B 0 帧匹配，无提示 | Scal3R `name` 用零填充索引，未同步 name_map | — | 跨源帧名对齐断言 |
| 15 | `Camera.width` 586×280 而图像 1920×1080 | reader 把宽高设成深度网格尺寸 | — | 尺寸与文件实测对比 |
| 16 | 深度 remap 写了但没被调用 | `remap_depth` 存在但 dataset 走纯缩放 | — | `depth_grid_differs` 判据 |
| 17 | `per_frame_median_norm` 静默失效 | 未转发进 `total_loss` | — | 参数穿透测试 |
| 18 | Mode B 只有 15.63 dB | 位姿**已对齐**却又被 Sim(3) 变换一次（双重变换） | → 19.42（+3.8） | `scene_scale` 与 Mode A 对比 |
| 19 | gtpose 臂深度监督丢 2/3 | 深度未随 gauge 缩放 | depth loss 1.7449 → 0.5102 | 渲染/监督深度比必须 = 1 |
| 20 | `absgrad=True` 崩 | `meta["gradient_2dgs"]` 无 `.absgrad` | — | 配置层显式拒绝（§3.4） |
| 21 | 曾误改随机四元数为单位四元数 | 误判"致密化过度"的原因 | 等价性 2437 vs 5915 → 回退 | 等价性门禁 |
| 22 | 任意通道数崩溃 | 1.6.0 移除补零，只接受编译期白名单 | C=7/10/13 直接 fail | 重编或避免 |
| 23 | `distloss` 报错 | 用了 `RGB` 模式 | — | 模式自动升级 |

### 10.1 端到端验证暴露的 4 个新 bug（已修复，含回归判据）

用解析式合成数据集（平面 + 环视相机，100% 可解的几何）跑通 400 步训练时逐一暴露：

| # | 症状 | 根因 | 量化 | 修复 |
|---|---|---|---|---|
| **24** | `depth_loss = 9984`，总 loss 从 0.27 飙到 1e4 | `depth_alpha_mask` 默认 `0.0`（不启用掩码）。无覆盖像素渲染深度=0 → `clamp_min(1e-6)` → 视差 `1e6` | **稠密 GT 下未掩码 = 2.57e5，掩码后 = 0.139**（差 6 个数量级） | 默认改为 `0.01` |
| **25** | `DeadDensifierError` 在第 4 步误报，但 `over_frac=78.7%` 梯度正常 | 仪表把**每个 step** 都计为 refine 事件，而策略只在 `step % refine_every == 0` 工作 | 5 步 vs 5 个 refine 事件 | 加 `_is_refine_event()` 按策略的真实窗口判定 |
| **26** | 训练 11 步后 `means/scales/opacities` 全部 NaN | **`surf_normals` 未 detach**，autograd 经 `depth_to_normal_2dgs` 回传；该函数 `sqrt(cross²).clamp_min(1e-12)` 在零覆盖像素处反向梯度为 **inf**（前向被 clamp 保护，所以不报错） | 前向无 NaN，仅反向产生 | `normal_term` 内 `sn.detach()` |
| **27** | `normal term` 恒为 0.958，被背景主导 | 未覆盖像素的目标法线为 0 → `1-cos = 1`；对全图求平均等于在度量"画面多大比例是空的" | 23% 未覆盖 → 该项 0.958 | 改为**按 alpha 加权归一化** |

**#26 的细节值得记录**：这是 gsplat 1.6.0 的一个真实数值缺陷 —— `Rendering.cpp:1698`：

```cpp
at::Tensor norm = at::sqrt((normals * normals).sum(-1, true)).clamp_min(1e-12);
normals = normals / norm;
```

`clamp_min` 只 clamp 了**前向的值**，没有 clamp 导数。当 `normals == 0`（零覆盖像素）时，
`d(sqrt(x))/dx = 1/(2·sqrt(x)) → ∞`。所以只要经这个张量反传就会得到 inf，进而污染全参数。
`depth_to_normal_2dgs` 的设计意图是产出**目标**（用于自一致性约束），不该被优化穿透 ——
detach 在语义上也是正确的。

**#27 的判断依据**：这里不能照抄 2DGS 官方示例的写法（`normals_from_depth *= alphas` 后
再 `.mean()`）。乘 alpha 只把未覆盖像素的贡献置 0，但分母仍是全图像素数，
所以结果依然被"空画面比例"缩放。正确做法是分子权重和分母同用 alpha 掩码：

```python
weight = (alphas.detach() > 1e-3).to(normals.dtype)
per_pixel = 1.0 - (normals * surf_normals.detach()).sum(-1, keepdim=True)
return (per_pixel * weight).sum() / weight.sum()
```

---

## 11. 验证协议与判据

### 11.1 PSNR 绝不可跨以下任一维度比较

1. **分辨率**（同一 ckpt 差 7.9 dB）
2. **`split_mode`**
   - `reconstruction`：全部图像都训练（含 val），报告的 val PSNR 衡量的是**拟合**，不是泛化
   - `experiment`：留出一部分（如 27/262 帧）
   - 归档 MatrixCity 基线用的是 `reconstruction`
3. **损失协议**（`split_mode` + 正则项有无；深度权重实际恒为 0.05，不是 ramp —— 见 §8.6）
4. **步数**（7k vs 30k）

→ 配置与 checkpoint 必须记录：`split_mode` / `n_train` / `n_val` / `resolution` / `eval_on_train`。

### 11.2 可用的基线数字（同协议）

| 输出 | 配置 | 分辨率 | PSNR | 高斯数 |
|---|---|---|---|---|
| `outputs/f0_gtpose` | GT 位姿 + Scal3R 深度 | 518×280 | **26.31** | 2.19M |
| `outputs/bench_a` | GT 位姿 + GT 深度 | 1080p | 25.11 | 2.16M |
| `outputs/strip0` | Scal3R 位姿+深度 | 518×280 | 20.28 | 1.04M |
| `gsl_bench_a_repro`（vanilla 复现） | 同 bench_a | 1080p | 25.10 | 1.03M |
| `strip0_highlevel`（gsplat 高层 API） | 同 strip0（唯一变量=renderer） | 518×280 | 20.20 | 1.01M |

**`strip0_highlevel` vs vanilla 基线：PSNR 差 0.08 dB、SSIM 差 0.004、高斯数差 2.9%**
→ 这是"gsplat 后端与 hbb1 等价"的实证（差异来自光栅器实现，属预期）。

### 11.3 等价性门禁（比 PSNR 基线更可信的判据）

仓库里归档的 garden 28.45 dB 基线**无法复现**（用它自己的 ckpt 经独立评估器只得 16.5 dB，
PCA 各轴比值 1.33/1.31/1.64 不一致 → 属于另一个世界系）。

因此验证判据应与**官方算法等价性**挂钩：

| 测试 | 实测 |
|---|---|
| 合成场景（与 gsplat 参考对比） | 3.9%（2437 vs 2533 高斯） |
| garden 2000 步 | 1.6%（742k vs 754k 高斯） |
| garden factor=16（324×210） | **29.001 dB / SSIM 0.9141 / 389k**（归档 28.450 / 0.9048 / 313k） |

### 11.4 启动即应打印的项

一个可信的运行，日志开头必须能看到：

```
render grid  : 1297x840 (161 train / 24 val)
model        : 138766 2DGS surfels
depth range  : near=0.056 far=3.89 (derived from measured camera-space depth)
scene_scale  : 0.512
depth_scale  : 1.83 (disparity L1 multiplier)
depth render : on (gt=True normal=True dist=True)
densify size thresholds: duplicate below 0.00512, split above 0.00512, prune above 0.0512 (scene_scale 0.512)
gauge: normalize, s=0.00282, R=..., t=..., depth scaled=True
```

缺少任何一行都意味着某个耦合点没接上。

---

## 12. 配置速查表

### 12.1 大场景 profile vs 官方 2DGS profile

| 项 | `object`（对齐上游） | `aerial`（7k，对齐归档） | `aerial_long`（30k） |
|---|---|---|---|
| `max_steps` | 30000 | **7000** | 30000 |
| `density.grow_grad2d` | 2e-4 | **6e-4** | 6e-4 |
| `density.refine_start_iter` | 500 | 500 | 500 |
| `density.refine_stop_iter` | 15000 | **7000** | 20000 |
| `density.refine_every` | 100 | 100 | 100 |
| `density.reset_every` | 3000 | 3000 | 3000 |
| `density.prune_opa` | 0.005 | 0.005 | 0.005 |
| `loss.dist_weight` / start | 0.01 / 3000 | **0.1 / 3000** | 0.1 / 7000 |
| `loss.normal_weight` / start | 0.05 / 7000 | **0.05 / 2000** | 0.05 / 5000 |
| `train.lr.decay_means` | True（官方**也有**） | True | True |
| `train.spatial_lr_scale` | 1.0 | **1.0** | 1.0 |
| `gauge.scale_source` | `extent` | **`depth_p99`** | `depth_p99` |
| `model.sh_degree` | 3 | 3 | 3 |
| `model.near/far` | `None`（派生） | `None`（派生） | `None`（派生） |

> 注意：官方 2DGS 示例的 `prune_opa` 默认是 **0.05**，而 2DGS 论文/归档配方是 **0.005**。
> 这是一个真实的默认值分歧，不是笔误 —— 两者不可混用同一份配置对比。

### 12.2 GSL 归档配方的原始数值（对照用）

`configs/matrixcity_f0_gtpose.yaml` / `strip0_*.yaml`：

```yaml
model.metric:
  lambda_normal: 0.05           # from 2000
  lambda_dist: 0.1              # from 3000
  depth_loss_type: l1
  depth_output_key: surf_depth
  depth_loss_weight: {init: 0.05, final_factor: 1.0, max_steps: 7000}
model.density:
  cull_opacity_threshold: 0.005
  densify_grad_threshold: 0.0006
model.gaussian.optimization:
  spatial_lr_scale: -1          # = auto extent
data.parser:
  split_mode: reconstruction
  down_sample_factor: 1
  depth_rescaling: false
```

`configs/matrixcity/gsplat-aerial.yaml`（多 block 大场景）：

```yaml
max_steps: 60000
data.parser.depth_read_step: 4
model.gaussian.optimization.spatial_lr_scale: 0.2   # avoid large xyz lr
model.gaussian.sh_degree: 0                          # avoid CUDA OOM
model.density.densify_grad_threshold: 0.0006         # avoid CUDA OOM
model.density.densify_until_iter: 30000
```

注意 `sh_degree: 0` 与 `depth_read_step: 4` 都是**显存约束的妥协**，不是质量最优。

### 12.3 建议的判据阈值（用于自检）

| 检查 | 阈值 | 依据 |
|---|---|---|
| Sim(3) 残差 / 轨迹 span | ≤ 5% | 实测好配 8.6e-8、坏配 11.1% |
| 渲染/监督深度比 | 1.0 ± 1% | 偏离即为 gauge 错 |
| `depth_p1 · c` | ≥ 0.15 | 低于此 `alpha=0` |
| 初始点间距 / (`grow_scale3d × scene_scale`) | ≲ 0.3 | DGI 下 0.0055/0.0360 = 0.15 |
| 连续零增长 refine 事件 | ≤ 5 | 超过即判定致密化失效 |
| `val` 集与 `train` 集帧名交集 | 空 | 否则等于没有留出 |

### 12.4 GSL 配置方言审计结论（子代理调研 + 本机复核）

#### 12.4.1 方言：jsonargparse subclass spec，有四种写法

GSL 用 `LightningCLI` + jsonargparse，`class_path`/`init_args` 是它的
**subclass spec** 语法。同一件事有四种可接受写法，这解释了为什么同一个
`configs/` 目录里风格看起来不一致：

| 写法 | 例子 | 何时可用 |
|---|---|---|
| 裸字符串 | `model.renderer: internal.renderers.vanilla_2dgs_renderer.Vanilla2DGSRenderer` | 总是 |
| 完整 spec | `{class_path: ..., init_args: {...}}` | 总是 |
| 仅 `init_args` | `{init_args: {...}}` | 该 section 的默认类已确定时 |
| 扁平 kwargs | `density: {densify_grad_threshold: 0.0006}`（默认类是 dataclass 时可直接写字段） | 默认类是 dataclass / 可解析时 |

**这对 `surfel` 的含义**：识读 GSL 配置时**不能假设某个 section 是 dict**。
实测 `yaml.safe_load` 后 `model.renderer` / `data.parser` / `data` 都可能是字符串。
任何遍历代码都必须先判类型。

#### 12.4.2 顶层键清点（151 个 YAML：85 根 + 66 子目录）

出现频次：`model` 141、`data` 109、`max_steps` 27、`cache_all_images` 20、
`trainer` 12、`save_iterations` 11，另有 `strategy`/`devices`/`logger`/`pbar_rate`。

**没有 `_base_` / 继承 / include 机制**。组合靠三种手段：
1. 多个 `--config` 深合并
2. `key+:` 追加（如 `model.plugins+:`）
3. `configs/plugins/*.yaml` 覆盖层

**实测确认**：`max_steps` 是**顶层键**，通过 `internal/cli.py:78-79`
（`config.trainer.max_steps = config.max_steps`）转发。而
`depth_loss_weight.max_steps` 是**每个文件手写的独立值**，两者不联动 ——
`--max_steps` 不会传播到深度权重或 means LR 调度器。这是一个真实的陷阱。

#### 12.4.3 `internal/configs/*.py` 是死代码

`internal/configs/{model,optimization,dataset}.py` 定义了 dataclass，
但**没有任何模块 import 它们**，且 `optimization.py` 用的是已废弃的
`position_lr_init` 等名字。唯一在用的是 `instantiate_config.py` 的
`InstantiatableConfig` 基类。

**不要照抄这些 dataclass**，它们的字段名与真实 CLI 表面不一致。

#### 12.4.4 `spatial_lr_scale` 全量普查（19 处）

| 值 | 次数 | 场景 |
|---|---|---|
| `0.2` | 7 | MatrixCity bench_a/b（注释 "avoid large xyz learning rate"） |
| `-1` | 4 | 自动 = `camera_extent` |
| `1.0` | 3 | 归一化后的 gauge |
| `5` | 5 | 其他 |

注释里的算术全部复核正确：`1.6e-4 × 531.18 = 0.0849888`、
`1.6e-4 × 3.738 = 5.98e-4`、`1.6e-4 × 0.2 = 3.2e-5`、
`1/531.18 = 1.8826e-3`。

#### 12.4.5 两个跑不起来的配置（勿参考）

| 文件 | 问题 |
|---|---|
| `larger_dataset.yaml` | `class_path: Colmap` 放在 `data:` 层而非 `data.parser:`；且用了已移除的 `position_lr_init`/`scaling_lr`。实测 CLI 报 `Group 'data' does not accept option 'init_args.split_mode'` |
| `rubble_partition_src.yaml` | `data.parser.init_args` 无 `class_path`，只给 `block_dim`/`content_threshold`。这两个键不属于任何 parser——它是被 `utils/partition_citygs.py` 用裸 `yaml.safe_load` 读的旁路文件，不是训练配置 |

#### 12.4.6 `configs/.gitignore` 是白名单模式

`:1` 是 `*`，随后 133 行 `!` 例外。**即绝大多数配置默认不入库**，
只白名单放行 118 个 yaml。这解释了为什么目录里能看到未跟踪的文件。

#### 12.4.7 对 `surfel` 的设计结论

**推荐：扁平 schema 为主 + `gsl_compat` 转换器，不实现通用 `class_path` 解析。**

理由：
1. `surfel` 的价值在**校验**（未知键报错、类型强约束）。通用 spec 解析会把任意
   键放进来，等于放弃这个价值。
2. 2DGS 架构是固定的（只有 `model.type="2dgs"`），不需要在运行时选类。
3. 转换器可以**对无法映射的键显式报错**，而不是静默忽略 —— 这正是本项目历史上
   反复吃亏的地方（"override 静默失效"）。

转换器需要处理的映射（GSL → surfel）：

| GSL 键 | surfel 键 | 转换规则 |
|---|---|---|
| `max_steps`（顶层） | `train.max_steps` | 直传 |
| `model.gaussian.optimization.spatial_lr_scale` | `train.spatial_lr_scale` | **`-1` → 先算 extent 再填具体值**；不可直传 |
| `model.density.densify_grad_threshold` | `density.grow_grad2d` | 直传（§5.1 已证同口径） |
| `model.density.cull_opacity_threshold` | `density.prune_opa` | 直传 |
| `model.density.densify_from_iter` | `density.refine_start_iter` | 直传 |
| `model.density.densify_until_iter` | `density.refine_stop_iter` | 直传 |
| `model.density.densification_interval` | `density.refine_every` | 直传 |
| `model.density.opacity_reset_interval` | `density.reset_every` | 直传 |
| `model.density.percent_dense` | `density.grow_scale3d` | 直传（同为 `0.01` 语义） |
| `metric.init_args.lambda_dist` | `loss.dist_weight` | 直传 |
| `metric.init_args.lambda_normal` | `loss.normal_weight` | 直传 |
| `metric.init_args.dist_regularization_from_iter` | `loss.dist_start_iter` | 直传 |
| `metric.init_args.normal_regularization_from_iter` | `loss.normal_start_iter` | 直传 |
| `metric.init_args.depth_loss_weight.init` | `data.depth.weight` | 直传 |
| `metric.init_args.depth_loss_weight.final_factor` | `data.depth.weight_final` | **须乘 init**：`weight_final = init × final_factor` |
| `metric.init_args.depth_eps` | `loss.depth_eps` | 直传 |
| `metric.init_args.depth_outlier_clip` | `loss.depth_outlier_clip` | 直传 |
| `data.parser.init_args.split_mode` | `data.split_mode` | 直传 |
| `data.parser.init_args.down_sample_factor` | `data.factor` | 直传（语义已对齐，见 §7） |
| `data.parser.init_args.depth_dir` | `data.depth.dir` | 直传 |
| `data.parser.init_args.depth_rescaling` | `data.depth.rescale` | 直传；但若 GSL 语义下恒开中位归一化，应同时设 `per_frame_median_norm=True` |
| `data.parser.init_args.scene_scale` | `gauge.scale`（`scale_source=manual`） | 直传 |
| `model.renderer`（class_path） | `model.type="2dgs"` | 只接受 2DGS 三个渲染器 |
| `model.metric`（class_path） | `data.depth.enabled` | `GS2DDepthMetrics` → True，`VanillaMetrics` → False |
| `model.gaussian.sh_degree` | `model.sh_degree` | 直传 |
| `data.train_max_num_images_to_cache` | `train.cache_images` | `-1` → True |
| `model.save_iterations` | `train.save_steps` | 直传 |
| — | `loss.depth_alpha_mask` | **GSL 无对应**（它靠 `depth_outlier_clip`），保持默认 0.01 |
| — | `gauge.mode` / `scale_source` | **GSL 无对应**，需按数据推断 |
| — | `model.near_plane` / `far_plane` | **GSL 无对应**（用 gsplat 默认 0.01/1e10），建议派生 |

**必须显式拒绝的键**（不能静默忽略）：`model.renderer` 指向非 2DGS 渲染器、
`model.type` 非 `2dgs`、任何 `avg_*`/`distributed`/`swag`/`deformable` 相关项。

---

### 12.5 密度控制器审计结论（子代理调研 + 本机复核）

#### 12.5.1 GSL 与 gsplat 的 split 在 N=2 时数值等价

| 维度 | GSL `GS2DDensityController` | gsplat `split()` | 判定 |
|---|---|---|---|
| 位置采样 | `normal(0, exp(scales))` 再 `R @` | `einsum(R, exp(scales), randn(2,N,3))` | **分布等价**（非同 RNG 序列） |
| 尺度系数 | `/(0.8*N)`，N=2 → `/1.6` | `/1.6` 固定 | **完全相同** |
| 不透明度 | 原样 repeat | 原样 repeat（除非 `revised_opacity`） | 等价 |
| 第三轴 | std 补 0 → 子在盘面内 | 三轴都采样 | **GSL 更安全**（见 12.5.2） |

**结论**：GSL 不是"更好的公式"，而是"2DGS 安全的公式"。

#### 12.5.2 第三轴：两种解法等价，但必须加断言

GSL 从**数据结构上消除**第三轴（`Gaussian2D` 的 scales 就是 `[N,2]`，
`gaussian_2d.py:25,36`）。`surfel` 用 `[N,3]` + 死轴 `log(1e-6)`。

两者数值等价：`exp(-13.8) ≈ 1.02e-6`，比 `grow_scale3d × scene_scale`（1e-2 量级）
小 4 个数量级，不影响 `max()`；split 沿该轴位移 ~1e-6·randn 可忽略；
`RelocationCUDA.cu:75-77` 逐轴独立，死轴保持不变。

**但需要防回归**：任何把第三轴写成 0（→ `log 0 = -inf`）或从 3D 点云初始化后
忘了压小的路径都会破坏它。建议断言：

```python
assert (torch.exp(params["scales"])[:, 2] < 1e-3 * scene_scale).all()
```

#### 12.5.3 唯一 gsplat 完全没有的能力：预算式致密化

`Taming3DGSDensityController` / `GNS` 有 hard cap + 梯度/重要性优先采样：
- 预算按 `counts_array` 分配到 clone/split（`taming:194-232`）
- 二次曲线预算表（`taming:377-396`）或 √ 预算（`gns:109-119`）
- 多通道打分 + `multinomial` 采样（`gns:256-295`）

gsplat 只有 MCMC 的 `cap_max`，`DefaultStrategy` 无上限。

**但有两个障碍**：
1. edge-aware 打分依赖 GSL fork 的 `rasterize_to_weights` CUDA 内核，gsplat 无此算子
2. **本仓库无量化实验记录**（`docs/LOG.md` 无 budget/Taming/GNS 条目），
   README 的 "fewer primitives" 说法未经验证

**建议**：若要做，只移植 `edge_aware=False` 分支（退化为纯梯度采样），
并自行做消融验证收益。

#### 12.5.4 两条低成本、已采纳的能力

| 能力 | 实现 | 验证 |
|---|---|---|
| **`opacity_reset_value` 独立可调** | 包装 `ops.reset_opa`，拦截 `DefaultStrategy` 硬编码的 `prune_opa*2` | 实测均值 opacity **0.0245（默认 0.01）vs 0.1012（0.05）**，4 倍差异 |
| **策略状态进 checkpoint** | `strategy_state` 序列化进 ckpt，恢复时形状匹配则复制、不匹配则重建 | 实测 `grad2d (3839,)` 保存/恢复，`count sum=64014` 保留，续训正常 |

**为什么第二条是实质缺陷**：gsplat 的 `grad2d`/`count`/`radii` 是纯 Python dict，
其官方示例都不保存。resume 后：
- 第一批 refine 事件的梯度统计只有 resume 后的样本
- `max_radii2D` 归零 → 屏幕空间剪枝短暂失效

**实现细节（踩过的坑）**：形状不匹配时**不能跳过**，必须重建零张量。
跳过会让 key 变成 `None`，而 gsplat 的 `_update_state` 之后做
`state["grad2d"][gs_ids] = ...`，直接 `AttributeError: 'NoneType' has no attribute shape`。

#### 12.5.5 MCMC 路径的三个真实 bug（已修复）

都是"只在真正跑 MCMC 时才暴露"：

| # | bug | 症状 |
|---|---|---|
| 1 | 误传 `packed` 给 `MCMCStrategy.step_post_backward` | `TypeError: unexpected keyword argument 'packed'` |
| 2 | 未传必需的 `lr` | `TypeError: missing required positional argument 'lr'` |
| 3 | `initialize_state(**{"scene_scale": None})` | `TypeError: unexpected keyword argument 'scene_scale'` |

**根因**：两个策略的签名不兼容。`DefaultStrategy` 收 `packed`，`MCMCStrategy` 收 `lr`；
`initialize_state` 一个收 `scene_scale` 一个不收。修复方式是**按签名分派**而不是假设。

**另外补齐**：MCMC 必需 `init_opa=0.5` / `init_scale=0.1` + `opacity_reg=0.01` / `scale_reg=0.01`
（gsplat 自己的 preset，`simple_trainer.py:1593-1599`）。用 split 的默认值
（0.1 / 1.0、无正则）跑 MCMC 不是"略有不同的配置"，而是配置失真。

**`cap_max <= n_initial` 必须启动即拒绝**：此时 `n_target = min(cap_max, 1.05*n) ≡ n`，
`add_new_gs` 永远返回 0（`mcmc.py:227`），静默空转。DGI 点云 125 万 > 默认 `cap_max=1e6`
会直接踩中。

#### 12.5.6 明确不移植

| 项 | 理由 |
|---|---|
| `Utils` 全套张量手术 | 绑定 GSL 的 `List[Optimizer]` + `param_group["name"]` 惯例；gsplat 的 `_update_param_with_optimizer` 已覆盖同等功能且按名字索引 |
| `MCMCDensityController` 整体 | gsplat `MCMCStrategy` 功能更全（`noise_injection_stop_iter`、融合 CUDA 噪声、>2²⁴ 的 numpy 采样回退）；GSL 在大场景会因 `multinomial` 超限抛错 |
| 屏幕空间剪枝 | GSL 用 20 px（`vanilla:87,154`），gsplat 用 `radii/max(W,H)`。**两个项目独立地都指向"别开"**：GSL commit 6651b34 以"avoid metric drops"删除；gsplat 注释说明上游因 bug 实际未启用 |
| `cull_by_max_opacity` | 为 view-dependent opacity（GlossyGS）设计，2DGS 无此属性 |
| `opacity_reset_at` | 死变量，全仓库无人读取 |
| 前景优先 / 距离衰减 / 最低海拔剪枝 | 依赖 GSL 分块 store，航拍 2DGS 无需求 |

---

## 13. 双重变换防护：一个必须用 API 而非检查来防的错误类

### 13.1 现象：gauge 错误伪装成"初始化质量差"

同一个坑在本项目出现**三次**，每次都是静默的几何错误：

| # | 场景 | 错误 | 代价 |
|---|---|---|---|
| 1 | strip0 Mode B | 位姿**已对齐**却又被 Sim(3) 变换一次 | PSNR 15.63 → 19.42（+3.8） |
| 2 | 1297 DGI 实验 | 深度乘 `sim.s`，而 `c2w` 已被 `sim` 变换 | 点云深度是应有值的 5.13 倍 |
| 3 | 1297 DGI 验证脚本 | 对 **COLMAP** 位姿施加 `Scal3R→COLMAP` 的 Sim(3) | 相机被推远 5.13 倍 → 投影半径 < 1px → 41% 画面无覆盖 → **PSNR 12.3 而非 ~24** |

共同特征：**变换被施加到了已经变换过的量上**。

### 13.2 为什么归一化救不了（数学证明）

两个关键量都是相似变换不变量：

```
radius_px = focal · scale_3d / z
  施加 x → s·R·x + t：f·(s·scale)/(s·z) = f·scale/z     不变

is_small  = scale_3d ≤ grow_scale3d × scene_scale
  两侧同乘 s → 不等式不变
```

实测（`s` 从 0.25 到 4.0）：`radius_px` 恒为 **2.5627**，`is_small` 恒为 `False`。

**所以归一化只能改优化器条件数、NDC 精度、loss 量级，不能把"0.4 px"变成"1.5 px"。**
这与记忆里那条结论一致：*归一化只改数值条件，不改位姿误差*。

### 13.3 实测：没有任何纯内部统计量能检测它

正确 DGI 云 vs 双重变换后的云（garden）：

| 统计量 | 正确 | 双重变换 | 相对变化 |
|---|---|---|---|
| 点云 extent / 相机 extent | 0.99 | 0.79 | **−20%** |
| `z_p50 / 相机最近邻间距` | 9.49 | 7.33 | **−23%** |
| 点云半径 / 相机半径 | 0.99 | 0.79 | **−20%** |
| 相机半径（绝对值） | 4.549 | **22.908** | **5.0×** |
| 相机系深度（绝对值） | 4.827 | **19.169** | **4.0×** |
| `radius_p50` @324×210 | 2.71 px | 0.78 px | 3.5× |
| `radius_p50` @1297×840 | 8.44 px | 3.25 px | 2.6× |

**所有比值变化都 < 25%**（落在数据集间正常波动范围内），而**绝对量变化 4–5 倍**。而归一化恰好把这些绝对量除回去 —— 这就是它无法修复的原因。

### 13.4 三道防线（按有效性排序）

#### (a) API 方向守卫 —— 唯一真正有效的防线

`Sim3` 现在记录方向，误用会**报错而不是静默缩放**：

```python
sim = estimate_sim3(src, dst, source="scal3r", target="colmap")

sim.apply_to_scal3r_poses(c2w)     # ✅ 源侧位姿
sim.apply_to_reference_poses(c2w)  # ✅ 目标侧位姿：验证方向后原样返回
                                   #    误用则 ValueError
sim.inverse()                      # 方向自动交换
```

`apply_to_reference_poses` 是**原样返回**的设计：让正确的代码路径也成为最自然的写法。
无方向的 `Sim3`（`source=None`）只能走无守卫的 `apply_points`/`apply_cam_to_world`，
用守卫方法会明确报错提示。

#### (b) 启动几何报告 —— 只能抓极端情形（诚实标注）

`check_pose_point_consistency()` 在 trainer 启动时打印：

```
geometry     : median footprint 5.48 px, visible 100.0%, camera radius 0.8565
```

- **能抓**：亚像素（< 0.2 px，会被 `eps2d` 丢弃）、点几乎全在相机后方
- **不能抓**：上表那种"所有比值只变 20%"的情形
- **默认不 raise**（`raise_on_fail=False`）：一个不完整的检查，带数字的警告比硬中断更有用

同时**记录绝对 `camera_radius`** —— 与参考 run 对比是唯一有判别力的信号。

#### (c) 足迹驱动的 DGI stride

DGI 的 stride 原本按点数预算反解。改为按**投影足迹**：

```
相邻深度像素的 3D 间距 = stride · z / f_depth
投影到渲染网格的半径   = f_render · spacing / z = stride · f_render / f_depth
```

`z` 抵消 → **纯网格属性**，与场景尺度无关。

garden 实测（`f_render=961.4`、`f_depth=379.9`，比值 2.53）：

| stride | 预测半径 | 实测 |
|---|---|---|
| 1 | 2.53 px | **8.44 px** |
| 5 | 12.65 px | **10.95 px**（修正前） |

stride=5 的 10.95 px 使初始 splat 覆盖十个像素，同时也抬高了 kNN 派生的 3D 尺度，
进而与 `grow_scale3d` 阈值交互。改为 `stride = round(target_px · f_depth / f_render)`
（默认 `target_px=1.5`），点数预算作为上界兜底。

### 13.5 工程教训

**当一类错误在数值上不可观测时，唯一可靠的防法是让它在类型/API 层面无法表达。**
把 `Sim3` 从"一个 4×4 矩阵"提升为"一个有方向的映射"代价只有约 40 行，
但把三次踩坑的共同根因从"靠人记住"变成了"编译器/运行时检查"。

**守卫本身也会写反。** 第一版实现里 `apply_to_reference_poses()` 调用的是
`_require_direction(self.source, "colmap")`，即检查**源端**而不是**目标端**：
于是"正确用法"（把观点传进去、拿回原样）抛异常，而"错误用法"（对已在
target gauge 的观点再变换一次）被放行——守卫的作用被完全反转，且不会有任何
数值异常暴露它。现在的实现分成 `_require_source()` 与 `_require_target()` 两个
函数，`apply_to_scal3r_poses` 只查 source、`apply_to_reference_poses` 只查
target；`test_sim3_direction_guard_rejects_wrong_side` 同时构造正向与反向拟合
两组变换，双向都断言，任何一侧写反都会失败。

---

## 14. 分辨率提高到底改变了什么（实测，2026-11-27）

用户提出的假设是：**用低分辨率初始化的点云，无法在高分辨率下渲染出有意义
（没有空洞）的图像，因此 rgb_loss 的驱动能力不足**。这个假设在机制上合理，
但在 garden 上被三项实测证伪，真实机制是**反过来的**。以下数据均来自
1297×840 / 324×210 两档、深度网格 518×322、Scal3R 深度、COLMAP 位姿、
Sim(3) 定标后（`s=5.130892`，残差 0.300 % 轨迹跨度）。

### 14.1 没有空洞：覆盖由 scale 决定，不由点数决定

2DGS 是 alpha 累积的 splat 渲染，空洞与否取决于**屏幕空间 splat 半径与点间距
之比**，而不是点云点数。`create_splats_2dgs` 把初始 scale 设为 **kNN 间距**，
所以点云变稀 → splat 自动变大，覆盖率基本不变。第 0 步实测（无任何训练）：

| 分辨率 | stride | 点数 | alpha 中位数 | 覆盖 α>0.05 | 覆盖 α>0.5 |
|---|---|---|---|---|---|
| 324×210 | 1 | 1.50 M | 0.995 | 100.0 % | 99.8 % |
| 324×210 | 5 | 1.25 M | 0.992 | 100.0 % | 100 % |
| 1297×840 | 1 | 1.50 M | 0.977 | 100.0 % | 100 % |
| 1297×840 | 5 | 1.25 M | 0.976 | 100.0 % | 100 % |

**没有任何"没有空洞也无法渲染"的窗口。** 1297×840 第 0 步 PSNR 为 14.69 dB
（324×210 为 15.87 dB），不是无意义图像，rgb_loss 有充足信号。

### 14.2 稠密化不是"太稀"，而是"太密且大量不可见"

DGI 从 185 帧反投影，同一块表面被重复写入很多次。第 0 步在 324×210 上，
1.5 M 高斯只有 **62.4 %** 可见（`radii > 0`），却要覆盖 68 k 个像素——
即 **22× 过绘**。可见性是**与分辨率无关**的 62.4 %，说明多视重复是点云属性，
不是渲染网格属性。训练日志里 `grad p50 = 0.000e+00`、超过阈值的比例仅 0.6 %，
就是因为一半以上的高斯永远拿不到梯度，随后被 prune（1.50 M → 1.33 M）。

> **结论：feed-forward/DGI 初始化的问题方向是"去重"，不是"加密"。**

### 14.3 加密点云会让第 0 步**变差**

在 1297×840 上打破 1.5 M 预算上限，观察第 0 步 PSNR：

| 点云 | 点数 | 第 0 步 PSNR | alpha |
|---|---|---|---|
| stride 1 | 30.86 M | **13.33 dB** | 1.000 |
| stride 2 | 7.71 M | 14.01 dB | 0.998 |
| stride 4 | 1.95 M | **14.64 dB** | 0.976 |

点数增加 16×，PSNR **下降** 1.3 dB。原因是多视重复采样：每个表面元有几十份
各自的深度噪声与各自的颜色（曝光、镜面、重采样差异不同），alpha 合成等于把
这些互不一致的样本平均起来——点越多，叠加的层数越多。单帧→185 帧的对照
（同为 stride 1）印证了这条曲线：1 帧 11.46 dB（alpha 0.457，覆盖不足）→
5 帧 12.64（0.676）→ 50 帧 13.25（0.999）→ 185 帧 13.37。**覆盖在 50 帧
之后就饱和了**，此后增加帧数只增加几何抖动。

### 14.4 深度网格给出的细节上限

深度网格 518×322 = 166,796 样本/帧，渲染网格 1297×840 = 1,089,480 像素/帧，
**少 6.5 倍**。一个深度像素在渲染图上占 `f_render / f_depth` 像素：

| 渲染分辨率 | f_render | 单个深度像素的屏幕足迹 |
|---|---|---|
| 324×210 | 240.43 | 0.634 px |
| 1297×840 | 962.11 | **2.538 px** |

把 §14.2 的可见性和像素数放在一起，得到本场景**最锋利的那个数**——
每像素可见高斯数：

| 渲染分辨率 | 像素数 | 点数 | 总点/像素 | 可见点/像素 |
|---|---|---|---|---|
| 324×210 | 68,040 | 1.50 M | 22.0× | **13.78×**（严重过绘） |
| 1297×840 | 1,089,480 | 1.50 M | 1.4× | **0.86×**（欠采样） |

**同一个 1.5 M 点云，在 324 下过绘 13.8 倍，在 1297 下欠采样。** 交叉点大约在
1200×780，而 garden 的 1297×840 正好落在这条线上（0.86×）。这统一解释了此前
所有看似矛盾的现象：

* 低分辨率下过绘 → 绝大多数高斯拿不到梯度、被 prune、`grad p50 = 0`，
  稠密化"看起来不工作"；
* 高分辨率下欠采样 → 每个像素由跨度 2.5 px 的 splat 插值而来，**高频细节在
  原理上无法表示**，这才是第 0 步只有 14.7 dB、最终收敛到 22.5 dB 而非 27.3 dB
  的原因；
* 但两者都**不影响覆盖率**（alpha 仍为 0.977），所以不是"渲染不出有意义的图像"。

这就是"分辨率提高"真正改变的量：**采样率从过绘翻转到欠采样**。它解释收敛速度与
最终精度上限，不解释崩溃。而要真正提高高分辨率细节，必须提高**深度网格**的
采样密度或改用稠密化的多视融合，单纯把 stride 调小（重复反投影同样的 518×322
样本）只会加剧 §14.3 的多视不一致。

### 14.5 稠密化速率随分辨率反向变化（这就是"高斯增长异常快"）

同一个 1.5 M 点云、同一份超参，只改渲染分辨率，稠密化行为完全相反：

| 分辨率 | kNN 足迹 | `grow_grad2d` 超阈比例 | 1000 步 | 3000 步 | 5000 步 | 7000 步 |
|---|---|---|---|---|---|---|
| 324×210 | 0.634 px | **0.6–0.7 %** | 1.53 M | 1.58 M | 1.36 M | 1.22 M |
| 1297×840 | 2.538 px | **2.5–3.3 %** | 1.69 M | 3.39 M | **4.31 M** | — |

机制是纯几何的，与超参无关：

* **超阈比例 ∝ 每高斯覆盖的像素数。** `grad2d` 是每个高斯在屏幕上的梯度范数，
  按 `width/2`、`height/2` 归一化。足迹 2.54 px 的高斯覆盖约 4 倍于 0.634 px 的
  高斯，且高分辨率下相邻高斯不再互相掩盖，因此更多高斯拿到高梯度。
* **`prune_scale3d` 在高分辨率下更难触发。** 它按 `exp(scales).max() > 0.1 ×
  scene_scale` 剪枝，而 `scene_scale` 几乎不随分辨率变化（1.0654 vs 1.0663）。
  `create_splats_2dgs` 的初始 scale 来自 kNN 间距，**不随渲染分辨率缩放**——
  所以 1297 下的高斯相对 `scene_scale` 一点都不"大"，prune 没有抓手。
* 结果：1297 在 3000→5000 步间 PSNR **停在 22.52 不动**（22.520 → 22.521），
  高斯却从 3.39 M 涨到 4.31 M（+27%），最终可能到 4.5 M+。**"高斯狂长但
  PSNR 不涨"正是这个组合的特征**，而不是初始化坏了。

这不是 bug，是 `grow_grad2d` / `grow_scale3d` 的**口径本身带一个隐式分辨率
依赖**。GSL 与 gsplat 的阈值等价性（§5.1）成立的前提是"梯度按图像尺寸归一化"，
而超阈比例同时受足迹影响——足迹一变，同一个 `2e-4` 就落在分布的不同分位上。

**高分辨率下的对策：按足迹比放大 `grow_grad2d`。**

实测超阈比例随足迹的幂次：

```
F(0.634 px) = 0.65 %      F(2.538 px) = 2.9 %
k = log(2.9/0.65) / log(2.538/0.634) = 1.08
```

要把 1297 的超阈比例拉回 324 的水平，阈值必须**增大**：

```
grow_grad2d(1297) = 2e-4 × (2.538/0.634)^1.08 = 8.9e-4
```

这个数有一个独立校验：**gsplat 自己在 `DefaultStrategy` 文档里为 AbsGS 推荐的
`grow_grad2d` 就是 0.0008**（原文："typically leads to better results but
requires you to set the `grow_grad2d` to a higher value, e.g., 0.0008"），理由
同样是高分辨率/绝对梯度下图像平面梯度更大。两条完全独立的推导落到同一个量级，
说明方向与幅度都对。

> **一个必须记住的反例**：把阈值**调小**不会降低稠密化速率，而是把它放大。
> 实测 5e-5 时超阈比例从 2.5 % 升到 **8.6 %**，第 500→1000 步高斯数从 1.50 M
> 冲到 **2.72 M**（2e-4 时为 1.69 M），第 1000 步 duplicate 从 12,613 变成
> **134,227**（10.7 倍）。**阈值越低，通过筛选的高斯越多**——这是"阈值是下限"
> 的直接推论，但调参时极易写反方向。本轮分析中我先写反过一次，正是这个反例
> 把它纠正过来。

**`grow_grad2d` 单独不够——它会撞上 prune 的固定基线。**

在 1297×840 上扫描 `grow_grad2d`，三次实测给出一个很窄的可用带：

| `grow_grad2d` | 超阈比例 | 第 1 个 refine 事件 | 结果 |
|---|---|---|---|
| 5e-5 | 8.6 % | +134,227 dup | 失控（比 2e-4 更糟） |
| 2e-4（原值） | 2.5–3.3 % | +12,613 dup | 失控，5500 步到 4.73 M |
| **4e-4** | **1.0–1.1 %** | **+3,091 dup / +4,645 split** | **可控** |
| 9e-4（足迹补偿值） | 0.3–0.4 % | +236 dup / +879 split | `DeadDensifierError` |

9e-4 按 §14.5 的足迹补偿公式是正确的方向，却**饿死**了稠密化：`prune` 每轮
固定删除 19–33 k，而 0.4 % × 1.47 M ≈ 5.9 k 候选，新增打不过删除，连续 5 个
refine 事件零净增长，防护直接抛错。

> **因此足迹补偿必须与 `prune` 一起调，不能只调 `grow_grad2d`。**
> 单独提高阈值会把"失控"换成"饿死"，两个极端都不工作。可用带宽度约
> 2e-4 → 9e-4（4.5×），而 4e-4 落在中间且实测稳定（1.0–1.1 %，
> 每轮 26k+26k 新增 vs 33k 删除，接近平衡）。

**其他对策（按有效性）：**

1. 对 `prune_scale3d` 做同样的足迹放大，给高分辨率一个真正的剪枝抓手（当前
   `0.1` 在 `scene_scale ≈ 1.065` 下意味着 3D 半径 > 0.107 才剪，而初始 kNN
   间距远小于此，永远不触发）。这是把上表的"接近平衡"变成"稳定"的关键补丁。
2. **在高分辨率下提前 `refine_stop_iter`**：如果细节上限由深度网格决定
   （§14.4），那么持续致密化只会在无法表示的频率上堆积高斯。
3. 提高初始点云密度前必须先解决 §14.3 的多视不一致（去重后加密才有意义）。
4. 用点数上限（`MCMCStrategy` 的 `cap_max` 思路）给 `DefaultStrategy` 也加一道
   保险。

> 已实测：voxel 去重（0.5–4 px 体素）在第 0 步只让 PSNR 在 14.35–14.91 dB 之间
> 移动（基线 14.81），即冗余本身是良性的，不是高分辨率上限的来源。上限来自
> 深度网格采样率（§14.4）。

### 14.6 正确初始化下的收敛轨迹（7000 步，val 24 帧）

`DGI_ok_324`（1.50 M 点，stride 2，`scene_scale=1.06631`）：

| 步数 | 500 | 1000 | 2000 | 3000 | 5000 | 7000 |
|---|---|---|---|---|---|---|
| PSNR | 24.505 | 25.209 | 26.438 | 26.936 | **27.274** | 27.255 |
| SSIM | 0.804 | 0.823 | 0.846 | 0.858 | **0.861** | 0.858 |

单调上升并在 5000 步后进入平台，最终 1.22 M 高斯。对比此前破损构造的
1297 臂（`scene_scale=0.5901`，第 600 步就 39,237 个 duplicate，最终 12.29 dB），
**同一份数据、同一份超参，唯一差别是 gauge 构造**——`scene_scale` 从 0.5901
纠正为 1.0663 后，第 600 步 duplicate 从 39,237 降到 723。

### 14.7 已排除的其他候选原因

| 候选 | 实测 | 结论 |
|---|---|---|
| 初始朝向（随机四元数） | 随机 14.84 dB vs 深度法线定向 14.89 dB | 无影响，不是原因 |
| 深度尺度错误 | 反向 warp 光度扫描（lag 20）在 k=1.0 取最优 15.54 dB，落点深度不一致 0.72 %（其他 k 为 5–8 %） | 深度已在正确 gauge |
| sim3 拟合方向错误 | 残差 0.300 % 轨迹跨度 | 拟合正确 |
| gauge 未归一化 | `radius_px` 与 `is_small` 都是 gauge 不变量（所有 s 下测得 2.5627） | 归一化救不了（§13.2） |

### 14.8 诊断量自身的 bug（已修）

`projected_footprint_px()` 曾把 1297 下的足迹报成 **8.51 px**，而解析答案是
2.54 px。两处独立错误叠加：

1. **在抽样后的点云上测间距。** 函数先把点云随机降到 `max_points=50_000`（供
   相机循环用），然后在这个子样本上建 kNN 树。1.5 M → 50 k 使最近邻间距膨胀
   **3.1×**（理想 ∝ √(N/50k)）。
2. **统计量口径不一致。** 用最近邻中位数，而 `create_splats_2dgs` 用 k−1 个
   邻居的 **RMS**（对齐 `distCUDA2`），后者在不规则点云上约大 1.3×。

后果不是美观问题：8.51 px 会被读成"初始 splat 有 8 像素宽，稠密化阈值全错"，
把排查方向引向 seed 配置，而真实点云是正确的。修复后在合成规则网格上
**精确等于解析值**（0.6327 vs 0.6327；2.5319 vs 2.5319），两条回归测试
`test_projected_footprint_does_not_inflate_on_large_clouds` 与
`test_projected_footprint_accepts_single_shared_intrinsic` 在旧代码上均失败、
在新代码上通过。第二个测试同时修掉了 `Ks` 为共享单相机（`[1,3,3]`）时的越界：
单相机多视图拍摄是常态，不是边缘情况。

---

## 附录 A：待验证项（诚实清单）

| 项 | 状态 |
|---|---|
| gsplat 高层 API 在**归档 MatrixCity 配置**上的完整 7k 复现 | ✅ 已做（`strip0_highlevel` 20.20 vs 20.28） |
| `densify_grad_threshold` 口径标定 | ✅ 已解决（§5.1，无需重标定） |
| `scene_scale` 默认值 bug 的大场景影响 | ✅ 已量化（§5.2） |
| 2DGS `absgrad` | ✅ 已确认不可用（§3.4） |
| 全分辨率 1080p 训练的稳定配置 | ⚠️ 归档的 1080p 臂（17.97 / 16.87）与 518×280 臂（20.28）**不可比**（协议不同） |
| 前馈 Scal3R/VGGT 直连（非 GSL 中间结果） | ⚠️ 未做，但 DGI / gauge / 对齐原语已就绪 |
| 分区 / 合并 / LOD | ⚠️ 未做（不在当前阶段） |
| MCMC 策略在大场景的实测 | ⚠️ 仅有 smoke |
| `surfel/` 包的端到端验证 | ⚠️ 进行中，见附录 B |

## 附录 B：`surfel/` 包结构（配套实现）

写在 `/home/cngc209/Github/gsplat/surfel/`（与 `gsplat/` 平级，不侵入上游 tree）：

```
surfel/
├── config/
│   ├── schema.py       # 三向 gauge 分离的 dataclass 配置
│   ├── loader.py       # YAML + 点分 override，typo 即报错
│   └── profiles.py     # aerial / aerial_long / object / smoke
├── data/
│   ├── colmap_io.py    # 二进制+文本读写，不依赖 pycolmap
│   ├── resolution.py   # 12 dB 那类 bug 的防线
│   ├── depth.py        # EXR/npy/png、逐帧标定、ray-remap
│   ├── scene.py        # Camera 带独立 depth_K
│   ├── gauge.py        # Sim3（含反射保护）、atomic apply
│   ├── dgi.py          # 深度反投影初始化
│   ├── parser.py       # COLMAP → Split（含 split / 归一化 / 深度接线）
│   ├── scal3r.py       # Scal3R 读取（intri/extri + EXR + name_map）→ Scene + DGI
│   └── dataset.py      # torch Dataset + CameraBatch
├── model/
│   ├── splats.py       # 2DGS 初始化（第三轴压制、RMS kNN）
│   └── renderer.py     # rasterization_2dgs 前端（§2 的全部调整）
└── train/
    ├── losses.py       # 视差深度 / distortion / normal 自一致 / 几何权重调度
    ├── density.py      # 策略工厂 + 仪表 + 死致密化护栏
    ├── pose.py         # 位姿优化（默认关）
    ├── appearance.py   # 逐图曝光仿射
    └── trainer.py      # 官方 gsplat 顺序的训练环
```

**尚未完成**：Scal3R ingest CLI、完整测试套件、`f0_gtpose` 端到端验证。

---

## 附录 C：关键文件定位

| 内容 | 位置 |
|---|---|
| 2DGS 渲染入口 | `gsplat/rendering.py:1358` |
| `gradient_2dgs` 定义 | `gsplat/rendering.py:1557`、`gsplat/cuda/csrc/Rendering.cpp:1818` |
| 深度 alpha 归一 | `gsplat/cuda/csrc/Rendering.cpp:1914-1919` |
| 法线转世界系 | `gsplat/cuda/csrc/Rendering.cpp:1931-1933` |
| render_mode 判定 | `gsplat/cuda/csrc/Rendering.cpp:1550-1572` |
| 2DGS 第三轴丢弃 | `gsplat/cuda/csrc/Projection2DGSFused.cu:166` |
| 梯度归一 `W/2, H/2` | `gsplat/strategy/default.py:243-249` |
| `is_small` / `is_too_big` | `gsplat/strategy/default.py:301` / `:355` |
| `initialize_state` 签名 | `gsplat/strategy/default.py:116-130` |
| 2DGS 示例的 scene_scale 缺口 | `examples/simple_trainer_2dgs.py:370` |
| 3DGS 示例的正确写法 | `examples/simple_trainer.py:517` |
| 视差深度损失 | `gsplat/losses.py:209` |
| 官方初始化 + 优化器 | `examples/simple_trainer_2dgs.py:222-290` |
| 官方训练环 | `examples/simple_trainer_2dgs.py:560-720` |
| GSL 优化默认值 | `internal/models/vanilla_gaussian.py:24-52` |
| GSL 密度默认值 | `internal/density_controllers/vanilla_density_controller.py:14-26` |
| GSL `camera_extent` | `internal/dataparsers/dataparser.py:75-84` |
| GSL 深度损失 | `internal/metrics/gs2d_depth_metrics.py:52-73` |
