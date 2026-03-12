# TraceForge 轨迹计算流水线完整分析

## 概述

本文档详细分析 TraceForge 系统从输入视频到输出 3D 轨迹的完整计算流水线，包括深度估计、位姿估计、3D 点跟踪和轨迹重采样等关键步骤。

## 流水线概览

```
原始视频 (MP4/图像序列)
    ↓
[阶段1: 输入加载与预处理] → video_tensor (T, 3, H, W)
    ↓
[阶段2: 深度与位姿估计] → depth, intrinsics, extrinsics
    ↓
[阶段3: 查询点采样] → 2D grid points (N, 3): [t, x, y]
    ↓
[阶段4: 2D→3D提升] → 3D query points (N, 4): [t, X, Y, Z]
    ↓
[阶段5: 分段跟踪] → 多个独立跟踪段
    ↓
[阶段6: 3D轨迹推理] → 3D轨迹 (T, N, 3)
    ↓
[阶段7: 轨迹时间重采样] → 弧长均匀的轨迹 (N, max_length, 3)
    ↓
[阶段8: 结果保存] → NPZ文件
```

---

## 阶段1：输入加载与预处理

### 输入数据
- **RGB视频**：图像序列或视频文件
- **深度图**（可选）：已知深度数据
- **掩码**（可选）：前景/背景分割
- **外部几何**（可选）：预计算的相机参数

### 处理流程

```python
# 加载视频和掩码
video_tensor, video_mask, original_filenames = load_video_and_mask(
    video_path, mask_dir, fps, max_num_frames
)
# 输出：
# - video_tensor: (T, 3, H, W), 范围 [0, 1]
# - video_mask: (T, H, W)
# - original_filenames: 帧文件名列表
```

### 自适应采样

为控制总帧数，系统会根据视频长度自动计算采样步长：

```python
# 计算采样步长
target_frames = 150  # 目标帧数
stride = max(1, ceil(n_frames / target_frames))

# 按步长采样
video_tensor = video_tensor[::stride]
depth_tensor = depth_tensor[::stride]  # 如果有深度图
```

**设计目的**：
- 控制内存使用
- 保持合理的计算时间
- 适应不同长度的视频

---

## 阶段2：深度与位姿估计

系统目前通过 `--depth_pose_method` 参数支持两种主方案：

### 方案A：VGGT4Wrapper（深度+位姿联合估计）

使用 VGGT4Track 模型同时估计深度和相机位姿。

```python
# 预处理视频
video_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H, W)

# VGGT4Track 推理
with torch.no_grad():
    predictions = model(video_processed)

# 输出（略去 batch 维）：
# - depth_map : (T, H, W)   # 预测深度
# - depth_conf: (T, H, W)   # 深度置信度
# - extrinsics: (T, 4, 4)   # 相机外参（VGGT 自身约定的位姿，实际在 TraceForge 中视作 w2c 使用）
# - intrinsics: (T, 3, 3)   # 相机内参
```

#### 深度尺度对齐

如果提供了已知深度，需要对齐预测深度的尺度：

```python
# 使用中值比率对齐尺度
valid_mask = (known_depth > 0) & (pred_depth > 0)
scale = median(known_depth[valid_mask]) / median(pred_depth[valid_mask])

# 缩放深度和平移向量
depth_map = pred_depth * scale
extrinsics[:, :3, 3] *= scale  # 平移向量也要缩放
```

#### 外部外参替换（VGGT 模式下，可选）

在 `VGGT4Wrapper` 内部，如果提供了 `--external_geom_npz`，当前实现允许**仅用外部外参替换 VGGT 估计的外参**，深度仍由 VGGT 预测并（可选）做尺度对齐：

```python
if self.external_extrs is not None:
    # 将 VGGT 估计的 extrs_npy 替换为外部外参（帧数按最短对齐）
    extrs_npy = self.external_extrs[:T_use].copy()
    # depth_npy / depth_conf / intrs_npy / video_ten 也会按 T_use 截断
```

### 方案B：ExternalGeomWrapper（纯外部几何）

当 `--depth_pose_method external` 时，完全跳过 VGGT，只使用外部提供的几何与深度：

```python
# 入口参数
--depth_pose_method external \
--depth_path <外部深度 PNG 目录> \
--external_geom_npz <含 intrinsics/extrinsics 的 NPZ/H5> \
--external_extr_mode {w2c,c2w}
```

- 外部深度：通过 `--depth_path` 提供，脚本用 `load_video_and_mask(..., is_depth=True)` 读取 16-bit PNG（mm），内部统一转换为米 (m)。
- 外部几何：`_load_external_geom(geom_path, camera_name)` 读取：
  - `intrinsics`: (T, 3, 3)
  - `extrinsics`: (T, 4, 4)
- `--external_extr_mode` 用于说明外参的语义：
  - `w2c`（默认）：外部矩阵被视为 world→camera（w2c），**直接作为 w2c 使用**；
  - `c2w`：外部矩阵被视为 camera→world（c2w），TraceForge 在 wrapper 内先做一次 `np.linalg.inv`，得到 w2c。

在 `ExternalGeomWrapper.__call__` 中，视频 / 深度 / 外部几何会在时间维度上对齐到相同的帧数 `T_use`，并生成：

```python
video_ten  # (T_use, 3, H, W), 归一化到 [0,1]
depth_npy  # (T_use, H, W), 外部深度（米）
intrs_npy  # (T_use, 3, 3), 外部内参
extrs_npy  # (T_use, 4, 4), 外部外参（统一规范为 w2c）
depth_conf # (T_use, H, W), 简单的有效深度掩码 (depth>0)
```

### 统一输出格式

无论使用 VGGT 方案还是 external 方案，传给后续流水的接口格式统一为：

```python
video_ten  # (T, 3, H, W), torch.Tensor, [0,1]
depth_npy  # (T, H, W), numpy.ndarray, 深度（米）
depth_conf # (T, H, W), numpy.ndarray, 深度置信度 / 有效掩码
extrs_npy  # (T, 4, 4), numpy.ndarray, 外参，统一视作 world→camera (w2c)
intrs_npy  # (T, 3, 3), numpy.ndarray, 内参
```

**注意**：此处约定 `extrs_npy` 为 **w2c（世界→相机）**，在 2D→3D 提升时会显式对它做一次 `np.linalg.inv`，得到 c2w（相机→世界）用于反投影。

---

## 阶段3：查询点采样

在选定的关键帧上生成均匀网格点作为查询点。

### 采样策略

```python
# 确定查询帧（每隔 frame_drop_rate 帧）
query_frames = list(range(0, video_length, frame_drop_rate))
# 例如：frame_drop_rate=5 → [0, 5, 10, 15, ...]

# 为每个查询帧生成 uniform grid
for frame_idx in query_frames:
    grid_points = create_uniform_grid_points(
        height=H,
        width=W,
        grid_size=grid_size  # 例如 20x20 = 400个点
    )
    # 设置时间索引
    grid_points[:, 0] = frame_idx
    query_point.append(grid_points)
```

### 网格点生成

```python
def create_uniform_grid_points(height, width, grid_size=20):
    """
    在图像上生成均匀分布的网格点

    返回: (grid_size*grid_size, 3) - [t, x, y]
    """
    # 在 [0, width-1] 和 [0, height-1] 范围内均匀采样
    x = linspace(0, width-1, grid_size)
    y = linspace(0, height-1, grid_size)

    # 生成网格
    xx, yy = meshgrid(x, y)
    points = stack([zeros_like(xx), xx, yy], axis=-1)

    return points.reshape(-1, 3)
```

### 输出

```python
query_point: List[ndarray]
# 每个元素: (N, 3) - [t, x, y]
# N = grid_size * grid_size
```

**设计考虑**：
- 均匀网格覆盖整个图像
- 密度可调（grid_size参数）
- 查询帧间隔可调（frame_drop_rate参数）

---

## 阶段4：2D查询点→3D世界坐标

将2D像素坐标的查询点提升到3D世界坐标系。

### 数学原理

从2D像素坐标 `(u, v)` 到3D世界坐标 `(X, Y, Z)` 的转换分两步：

**步骤1：反投影到相机坐标系**

```
P_cam = K^(-1) * [u, v, 1]^T * d
```

其中：
- `K^(-1)` 是相机内参的逆矩阵
- `d` 是该像素的深度值
- `P_cam = (X_cam, Y_cam, Z_cam)` 是相机坐标系下的3D点

**步骤2：转换到世界坐标系**

```
P_world = R * P_cam + t
```

其中：
- `R, t` 来自相机外参的逆矩阵（c2w变换）
- `P_world = (X, Y, Z)` 是世界坐标系下的3D点

### 实现代码

```python
def prepare_query_points(query_xyt, depths, intrinsics, extrinsics):
    """
    输入：
        query_xyt: List[ndarray] - 每个元素 (N, 3): [t, x, y]
        depths: (T, H, W) - 深度图
        intrinsics: (T, 3, 3) - 相机内参
        extrinsics: (T, 4, 4) - 相机外参（w2c格式，world→camera）

    输出：
        query_point_3d: (N_total, 4) - [t, X, Y, Z]
    """
    final_queries = []

    for query_i in query_xyt:
        # 获取时间索引
        t = int(query_i[0, 0])

        # 获取该帧的几何参数
        depth_t = depths[t]
        K_inv = np.linalg.inv(intrinsics[t])
        c2w = np.linalg.inv(extrinsics[t])  # 将 w2c 反转为 c2w（camera→world）

        # 提取2D坐标
        xy = query_i[:, 1:]  # (N, 2)

        # 查询深度值
        ji = np.round(xy).astype(int)
        d = depth_t[ji[:, 1], ji[:, 0]]  # (N,)

        # 反投影到相机坐标系
        xy_homo = np.concatenate([xy, np.ones_like(xy[:, :1])], axis=-1)  # (N, 3)
        local_coords = K_inv @ xy_homo.T  # (3, N)
        local_coords = local_coords * d[None, :]  # 乘以深度

        # 转换到世界坐标系
        world_coords = c2w[:3, :3] @ local_coords + c2w[:3, 3:]  # (3, N)

        # 组合时间索引和3D坐标
        final_queries.append(
            np.concatenate([query_i[:, :1], world_coords.T], axis=-1)
        )

    return np.concatenate(final_queries, axis=0)  # (N_total, 4)
```

### 关键点

1. **外参格式转换**：输入的外参是 w2c，需要求逆得到 c2w
2. **深度查询**：使用最近邻插值（round）获取深度值
3. **齐次坐标**：添加1作为第三维进行矩阵运算
4. **批量处理**：所有查询点一起处理，提高效率

---

## 阶段5：分段跟踪

将长视频分成多个重叠的跟踪段，每个段独立处理。

### 分段策略

```python
# 创建跟踪段
tracking_segments = []
for frame_idx in query_frames:
    end_frame = min(frame_idx + future_len, video_length)
    tracking_segments.append((frame_idx, end_frame))

# 例如：future_len=16, query_frames=[0, 5, 10, ...]
# → segments = [(0, 16), (5, 21), (10, 26), ...]
```

### 段内处理

对每个跟踪段 `(start_frame, end_frame)`：

```python
# 1. 提取段数据
video_seg = video_ten[start_frame:end_frame]      # (T_seg, 3, H, W)
depth_seg = depth_npy[start_frame:end_frame]      # (T_seg, H, W)
intrs_seg = intrs_npy[start_frame:end_frame]      # (T_seg, 3, 3)
extrs_seg = extrs_npy[start_frame:end_frame]      # (T_seg, 4, 4)

# 2. 调整查询点到段内坐标系
segment_query_point = query_point[seg_idx].copy()
segment_query_point[:, 0] = 0  # 时间索引重置为0（段内相对坐标）

# 3. 计算支持网格大小
support_grid_size = int(grid_size * 0.8)  # 支持网格密度为查询网格的80%

# 4. 准备输入
video, depths, intrinsics, extrinsics, query_tensor, support_grid_size = prepare_inputs(
    video_seg, depth_seg, intrs_seg, extrs_seg,
    segment_query_point, inference_res, support_grid_size
)
```

### 设计优势

1. **内存效率**：每次只处理16帧，避免长视频的内存问题
2. **并行化**：不同段可以并行处理
3. **灵活性**：支持任意长度的视频
4. **重叠覆盖**：查询帧间隔小于跟踪长度，保证轨迹连续性

---

## 阶段6：3D轨迹推理（PointTracker3D）

使用 PointTracker3D 模型在3D空间中跟踪查询点。

### 模型架构

PointTracker3D 包含三个核心组件：

1. **Encoder（编码器）**：提取RGB图像的视觉特征
2. **Corr Processor（相关性处理器）**：计算点与图像特征之间的3D相关性
3. **Point Updater（点更新器）**：迭代优化点的3D位置

### 6.1 坐标系转换（世界→相机）

为了提高跟踪稳定性，模型在局部相机坐标系中工作：

```python
if eval_mode == "local":
    # 获取查询帧的外参
    extrinsics_at_query = extrinsics[query_frames]  # (N, 4, 4)

    # 将世界坐标转换到查询帧的相机坐标系
    query_coords_cam = apply_homo_transform(
        query_coords_world,
        transform=extrinsics_at_query
    )

    # 后续在相机坐标系中跟踪
```

**为什么转换到相机坐标系？**
- 减少数值误差（局部坐标范围更小）
- 更符合视觉特征的局部性
- 提高跟踪稳定性

### 6.2 特征提取与点云生成

```python
# 1. 提取RGB特征
image_feats = encoder(video_seg)  # (T_seg, C, H, W)

# 2. 深度图反投影为3D点云
pcds = batch_unproject(depth_seg, intrinsics, extrinsics)  # (T_seg, H, W, 3)
```

点云用于后续的3D空间相关性计算。

### 6.3 深度范围估计

使用 IQR（四分位距）估计有效深度范围，用于约束预测：

```python
# 提取有效深度值
valid_depths = depth_seg[depth_seg > 0]

# 计算四分位数
q25 = percentile(valid_depths, 25)
q75 = percentile(valid_depths, 75)
iqr = q75 - q25

# 定义深度感兴趣区域（过滤异常值）
depth_roi = [1e-7, q75 + 1.5 * iqr]
```

**注意**：这只是计算一个范围值，不会改变深度图的尺寸或内容。

### 6.4 添加支持点

在第一帧生成额外的网格点作为支持点：

```python
# 生成支持网格点（密度为查询网格的80%）
support_queries = get_grid_queries(
    grid_size=support_grid_size,  # 如 16x16
    depths=depth_seg[0],
    intrinsics=intrinsics[0],
    extrinsics=extrinsics[0]
)

# 合并查询点和支持点
all_queries = torch.cat([query_point_3d, support_queries], dim=1)
```

**支持点的作用**：
- 提供密集的场景结构信息
- 帮助模型理解相机运动和场景几何
- 最终输出时会被移除，只保留原始查询点的轨迹

### 6.5 滑动窗口迭代跟踪

模型使用滑动窗口策略处理视频段：

```python
# 初始化预测
pred_coords = repeat(query_coords, "n c -> t n c", t=T_seg)
pred_visibs = zeros(T_seg, N)

# 滑动窗口参数
seq_len = 12  # 窗口大小
stride = 6    # 滑动步长

for window_end in range(seq_len, T_seg + 1, stride):
    window_start = window_end - seq_len

    # 提取窗口数据
    window_video = video_seg[window_start:window_end]
    window_depth = depth_seg[window_start:window_end]
    window_intrs = intrinsics[window_start:window_end]
    window_extrs = extrinsics[window_start:window_end]

    # 初始化窗口预测（使用上一窗口的结果）
    coords_init = pred_coords[window_start:window_end]
    visibs_init = pred_visibs[window_start:window_end]

    # 窗口内迭代优化
    for iter_idx in range(num_iters):  # 通常6次
        # 1. 计算3D相关性
        corr_feats = corr_processor(
            coords=coords_pred,
            pcds=pcds[window_start:window_end],
            feats=image_feats[window_start:window_end],
            shared_ctx=shared_corr_ctx
        )

        # 2. 更新坐标和可见性
        delta_coords, delta_visibs = point_updater(
            corr_features=corr_feats,
            coords=coords_pred,
            visibs=visibs_pred
        )

        coords_pred = coords_pred + delta_coords
        visibs_pred = visibs_pred + delta_visibs

    # 更新全局预测
    pred_coords[window_start:window_end] = coords_pred
    pred_visibs[window_start:window_end] = visibs_pred
```

**关键机制**：
- **相关性计算**：在3D空间中查找点周围的特征
- **迭代优化**：类似RAFT，逐步精细化预测
- **深度约束**：depth_roi 限制预测范围
- **窗口重叠**：保证平滑过渡

### 6.6 移除支持点

```python
# 只保留原始查询点的跟踪结果
N_original = query_point_3d.shape[1]
N_support = support_queries.shape[1]

coords = coords[:, :N_original]
visibs = visibs[:, :N_original]
```

### 6.7 坐标系转换（相机→世界）

将结果转换回世界坐标系：

```python
if eval_mode == "local":
    inv_extrinsics = torch.linalg.inv(extrinsics)
    coords_world = apply_homo_transform(
        coords_cam,
        transform=inv_extrinsics[:, :, None, :, :]
    )
```

### 输出

```python
{
    "coords": (T_seg, N, 3),  # 3D轨迹坐标（世界坐标系）
    "visibs": (T_seg, N)      # 可见性概率
}
```

---

## 阶段7：轨迹时间重采样（Retarget）

将基于视频帧率的轨迹重采样为基于弧长的均匀轨迹，适配机器人执行需求。

### 为什么需要Retarget？

**问题**：
- 3D跟踪输出的轨迹是按视频帧率采样的（如30fps）
- 不同轨迹段的运动速度不同（快速运动 vs 静止）
- 机器人控制需要等弧长间隔的轨迹点

**解决方案**：
- 基于弧长（arc-length）重采样轨迹
- 保证轨迹点之间的空间距离均匀
- 适配机器人执行的时间要求

### Retarget算法流程

#### 步骤1：全局归一化

```python
# 使用最后一条轨迹的第一帧作为参考
scale_x = trajectory[-1, 0, 0]
scale_y = trajectory[-1, 0, 1]

# 归一化 x, y 坐标
traj_norm[:, :, 0] = trajectory[:, :, 0] / scale_x
traj_norm[:, :, 1] = trajectory[:, :, 1] / scale_y

# 裁剪到 [0, 1] 范围
traj_norm[:, :, 0:2] = clip(traj_norm[:, :, 0:2], 0, 1)

# 注意：z 坐标不归一化
```

#### 步骤2：计算鲁棒段长度

```python
# 计算每个时间段的所有轨迹长度
for t in range(H-1):
    lengths = ||traj_norm[:, t+1] - traj_norm[:, t]||  # (N,)

    # 取 top k% 的轨迹长度求平均（鲁棒估计）
    k = ceil(top_percent * N)  # 例如 top_percent=0.02 → 取最快的2%
    robust_seglen[t] = mean(top_k(lengths, k))
```

**为什么用 top k%？**
- 避免被静止或遮挡的轨迹影响
- 使用运动最快的轨迹代表真实运动速度
- 提高对异常值的鲁棒性

#### 步骤3：构建累积弧长

```python
# 计算累积弧长
cumulative_length[0] = 0
for t in range(1, H):
    cumulative_length[t] = cumulative_length[t-1] + robust_seglen[t-1]

total_length = cumulative_length[-1]

# 每隔 interval 放置一个目标点
num_targets = floor(total_length / interval) + 1
targets = [0, interval, 2*interval, ..., min(num_targets*interval, total_length)]
```

#### 步骤4：同步插值

```python
for i, target_length in enumerate(targets):
    # 找到该弧长对应的时间段
    t = searchsorted(cumulative_length, target_length) - 1

    # 计算段内插值比例
    alpha = (target_length - cumulative_length[t]) / robust_seglen[t]

    # 对所有 N 条轨迹使用相同的 (t, alpha) 进行插值
    for n in range(N):
        retargeted[n, i] = (1-alpha) * traj_norm[n, t] + alpha * traj_norm[n, t+1]
```

**关键**：所有轨迹使用相同的时间插值参数（同步插值），保持相对位置关系。

#### 步骤5：反归一化

```python
# 只对 x, y 坐标反归一化
retargeted[:, :, 0] *= scale_x
retargeted[:, :, 1] *= scale_y

# z 坐标保持线性插值结果
```

### 函数签名

```python
def retarget_trajectories(
    trajectory: np.ndarray,  # (N, H, D) with D in {2, 3}
    interval: float = 0.05,  # 目标弧长间隔
    max_length: int = 64,    # 输出最大长度
    top_percent: float = 0.02 # 鲁棒速度计算的百分比
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
        retargeted: (N, max_length, D) - 重采样后的轨迹
        valid_mask: (max_length,) - 有效点的掩码
    """
```

### Retarget的关键特性

1. **弧长均匀**：输出轨迹点之间的空间距离近似相等
2. **鲁棒速度**：使用 top k% 轨迹避免异常值
3. **同步插值**：所有轨迹在同一时刻采样
4. **自适应采样**：
   - 快速运动段：细分为多个点
   - 缓慢运动段：合并为少量点

---

## 阶段8：结果保存

将处理后的轨迹数据保存为 NPZ 格式。

### 保存内容

```python
save_data = {
    # 原始轨迹
    "coords": coords,                    # (T, N, 3) - 3D轨迹坐标
    "visibs": visibs,                    # (T, N) - 可见性

    # 重采样轨迹
    "coords_retargeted": retargeted,     # (N, max_length, 3)
    "retarget_mask": valid_mask,         # (max_length,)

    # 几何信息
    "depths": depth_seg,                 # (T, H, W)
    "intrinsics": intrs_seg,             # (T, 3, 3)
    "extrinsics": extrs_seg,             # (T, 4, 4)

    # 视频数据
    "video": video_seg,                  # (T, 3, H, W)

    # 元数据
    "query_frame": start_frame,
    "grid_size": grid_size,
    "future_len": future_len
}

# 保存为 NPZ 文件
np.savez_compressed(output_path, **save_data)
```

### 文件命名

```python
# 每个查询帧一个文件
filename = f"{video_name}_query_{start_frame:04d}.npz"
# 例如：video001_query_0000.npz, video001_query_0005.npz, ...
```

### 输出目录结构

```
output_dir/
├── video001/
│   ├── video001_query_0000.npz
│   ├── video001_query_0005.npz
│   ├── video001_query_0010.npz
│   └── ...
├── video002/
│   └── ...
└── ...
```

---

## 关键设计特点

### 1. 3D空间跟踪

- 在世界坐标系中跟踪点，而非2D像素空间
- 利用深度和相机参数提供几何约束
- 更鲁棒，能处理相机运动和场景变化

### 2. 几何约束

- 深度图提供3D位置信息
- 相机内外参约束投影关系
- 深度范围（depth_roi）过滤异常值

### 3. 分段独立处理

- 每个查询帧独立跟踪 future_len 帧
- 避免长视频的内存问题
- 支持并行化处理

### 4. 迭代优化

- 类似 RAFT 的迭代精细化策略
- 通常6次迭代达到收敛
- 每次迭代计算相关性并更新坐标

### 5. 支持点机制

- 额外的密集网格点提供场景上下文
- 帮助模型理解相机运动
- 推理后移除，不影响输出

### 6. 坐标系灵活切换

- 世界坐标系：输入输出
- 相机坐标系：模型内部跟踪
- 提高数值稳定性

### 7. 自适应采样

- 视频采样：根据长度自动计算 stride
- 查询帧采样：frame_drop_rate 控制密度
- 轨迹重采样：基于弧长的自适应插值

### 8. 鲁棒性设计

- 深度尺度对齐：处理深度估计误差
- Top-k 速度估计：避免异常轨迹影响
- IQR 深度范围：过滤深度异常值

---

## 完整数据流总结

```
输入：原始视频 + 可选（深度/掩码/外部几何）
  ↓
[加载] → video_tensor (T, 3, H, W)
  ↓ stride采样
[深度位姿] → depth (T, H, W) + intrinsics (T, 3, 3) + extrinsics (T, 4, 4)
  ↓
[查询点采样] → 2D points (N, 3): [t, x, y]
  ↓ 反投影
[3D提升] → 3D points (N, 4): [t, X, Y, Z]
  ↓
[分段] → 多个 (start, end) 段
  ↓
[3D跟踪] → coords (T_seg, N, 3) + visibs (T_seg, N)
  │
  ├─ 世界→相机坐标系
  ├─ 特征提取 + 点云生成
  ├─ 滑动窗口
  │  └─ 迭代优化（相关性 + 更新器）× 6次
  └─ 相机→世界坐标系
  ↓
[Retarget] → retargeted (N, max_length, 3)
  │
  ├─ 归一化
  ├─ 鲁棒段长度（top-k）
  ├─ 累积弧长
  ├─ 同步插值
  └─ 反归一化
  ↓
[保存] → NPZ 文件（每个查询帧一个）
```

---

## 参数配置参考

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grid_size` | 20 | 查询网格密度（20×20=400点）|
| `frame_drop_rate` | 5 | 查询帧间隔 |
| `future_len` | 16 | 每段跟踪长度 |
| `num_iters` | 6 | 迭代优化次数 |
| `support_grid_size` | 16 | 支持网格密度（自动计算为 grid_size×0.8）|
| `max_num_frames` | 384 | 最大帧数限制 |
| `max_frames_per_video` | 150 | 目标帧数（用于计算stride）|

### Retarget参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interval` | 0.05 | 目标弧长间隔 |
| `max_length` | 64 | 输出最大长度 |
| `top_percent` | 0.02 | 鲁棒速度计算百分比（2%）|

---

## 性能考虑

### 内存优化

1. **分段处理**：每次只处理16帧
2. **及时释放**：每段处理后清理GPU缓存
3. **自适应采样**：长视频自动降采样

### 计算效率

1. **批量处理**：所有查询点一起推理
2. **特征复用**：窗口重叠时复用特征
3. **混合精度**：使用 bfloat16 加速

### 质量保证

1. **迭代优化**：6次迭代保证精度
2. **几何约束**：深度和相机参数约束
3. **鲁棒估计**：top-k 避免异常值

---

## 应用场景

### 机器人模仿学习

- 从演示视频提取3D轨迹
- Retarget 生成可执行的轨迹
- 适配不同机器人的执行速度

### 视觉跟踪

- 长时间3D点跟踪
- 处理遮挡和相机运动
- 输出世界坐标系轨迹

### 场景重建

- 提取场景中的运动轨迹
- 结合深度信息重建3D场景
- 分析物体运动模式

---

## 总结

TraceForge 的轨迹计算流水线是一个完整的端到端系统，从原始视频到可执行的3D轨迹：

1. **输入灵活**：支持多种输入格式和可选的先验信息
2. **几何感知**：充分利用深度和相机参数
3. **3D跟踪**：在世界坐标系中跟踪，更鲁棒
4. **自适应处理**：根据视频长度和运动速度自适应调整
5. **机器人友好**：Retarget 生成适合执行的均匀轨迹

整个流水线设计精巧，平衡了精度、效率和鲁棒性，适用于机器人模仿学习等实际应用场景。

