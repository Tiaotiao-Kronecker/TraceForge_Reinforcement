## TraceForge 轨迹计算 Pipeline 说明

本文档从「一段带深度的视频」出发，按时间顺序说明 TraceForge 是如何计算 3D 轨迹的，对应的主要代码在：

- `scripts/batch_inference/infer.py`
- `utils/video_depth_pose_utils.py`
- `utils/threed_utils.py`

数据落盘格式请结合已有文档一起阅读：

- `docs/traceforge_output_structure.md`：详细说明输出目录与 NPZ 字段。

---

## 1. 输入与预处理

### 1.1 RGB / 深度加载与尺寸

入口函数：`process_single_video(...)`（`infer.py`）

1. **确定帧采样 stride（时间下采样）**

   ```python
   if args.fps and int(args.fps) > 0:
       stride = int(args.fps)     # 用户显式指定 fps → 固定步长
   else:
       # 自动根据总帧数和 max_frames_per_video 推出 stride
       stride = ...
   ```

2. **按 stride 加载 RGB 帧**

   ```python
   video_tensor, video_mask, original_filenames = load_video_and_mask(
       video_path, args.mask_dir, stride, args.max_num_frames
   )
   # 输出: video_tensor 形状为 [T, 3, H, W]，值域 [0,1]
   ```

3. **按同样 stride 加载深度帧（若提供 `--depth_path`）**

   ```python
   depth_tensor, _, _ = load_video_and_mask(
       depth_path, None, stride, args.max_num_frames, is_depth=True
   )  # [T, H, W]，单位米 (m)
   ```

   在 `load_video_and_mask(..., is_depth=True)` 中：

   - 从 16-bit PNG 读入深度（单位毫米 mm）；
   - 转换为浮点并除以 1000，得到米：

   \[
   D_{\text{meters}}(i,j) = \frac{D_{\text{png}}(i,j)}{1000.0}
   \]

4. **对齐 RGB / 深度帧数**

   若 `len(depth_tensor) != len(video_tensor)`，会裁剪到最短长度，并同步裁剪 `original_filenames`，保证时序一一对应。

---

## 2. 深度 & 相机位姿估计（或外部几何）

TraceForge 抽象了一个「视频深度 + 位姿」模块，由 `video_depth_pose_dict` 选择不同实现：

- `depth_pose_method="vggt4"` → `VGGT4Wrapper`
- `depth_pose_method="external"` → `ExternalGeomWrapper`

调用入口（`infer.py`）：

```python
video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy = model_depth_pose(
    video_tensor,
    known_depth=depth_tensor,   # 可以为 None
    stationary_camera=False,
    replace_with_known_depth=False,
)
```

### 2.1 VGGT 模式（`VGGT4Wrapper`）

核心逻辑在 `utils/video_depth_pose_utils.py`：

1. 预处理视频成网络输入：

   ```python
   video_tensor_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H, W)
   ```

2. 前向推理得到：

   ```python
   predictions = self.model(video_tensor_processed.to(self.device))
   extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
   depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
   ```

   - `depth_map`：预测深度，单位米 (m)，形状约为 \((T, H', W')\)。
   - `extrinsic`：相机位姿（通常为 c2w 或 w2c，内部约定）。
   - `intrinsic`：像素坐标到相机坐标的内参矩阵 \(K\)。

3. 若提供 `known_depth`（实测深度）则做尺度对齐：

   - 先把 `known_depth` 插值到与预测深度同一分辨率；
   - 对每帧计算中位数比值，再取平均得到尺度因子 \(s\)：

   \[
   s = \text{mean}_t \frac{
       \operatorname{median}\big(D_{\text{known}}^{(t)}(i,j)\big)
   }{
       \operatorname{median}\big(D_{\text{pred}}^{(t)}(i,j)\big)
   }
   \]

   - 将预测深度放缩到实测尺度：

   \[
   D_{\text{aligned}}^{(t)}(i,j) = s \cdot D_{\text{pred}}^{(t)}(i,j)
   \]

   - 同时对 extrinsics 的平移向量做同样尺度放缩：

   \[
   \mathbf{t}' = s \cdot \mathbf{t}
   \]

   - 若 `replace_with_known_depth=True`，则完全用实测深度替换预测深度。

最终返回：

- `video_ten`：运行在 GPU 上的 `[T, 3, H, W]` 视频张量（归一化版本，给后续 Tapir3D 用）；
- `depth_npy`：\((T, H', W')\) 深度；
- `depth_conf`：同形状置信度；
- `extrs_npy`：\((T, 4, 4)\) 相机外参；
- `intrs_npy`：\((T, 3, 3)\) 相机内参。

### 2.2 外部几何模式（`ExternalGeomWrapper`）

当设置：

```bash
--depth_pose_method external \
--depth_path <外部深度 PNG 目录> \
--external_geom_npz external_geom_*.npz
```

`ExternalGeomWrapper` 会：

1. 从 `external_geom_npz` 中加载：

   - `intrinsics`: \((T_{\text{ext}}, 3, 3)\)
   - `extrinsics`: \((T_{\text{ext}}, 4, 4)\)

2. 直接使用 `known_depth` 作为深度：

   - 假定 `known_depth` 已是单位米 (m)，分辨率与 RGB 一致；
   - 不调用 VGGT，不做尺度对齐。

3. 对视频/深度/外部几何在时间维度上截断到相同长度：

   \[
   T_{\text{use}} = \min(T_{\text{video}}, T_{\text{depth}}, T_{\text{ext}})
   \]

最终返回的 `video_ten / depth / intrs_npy / extrs_npy` 与 VGGT 模式保持同一接口，用**完全外部的几何**驱动后续 3D 轨迹计算。

---

## 3. 查询点采样（Uniform Grid）

### 3.1 选取查询帧（`frame_drop_rate`）

在 `process_single_video(...)` 中：

```python
video_length = len(video_tensor)
query_frames = list(range(0, video_length, args.frame_drop_rate))
```

- 若 `frame_drop_rate = 1`：每一帧都作为查询帧；
- 若 `frame_drop_rate = 5`：查询帧为 `[0, 5, 10, ...]`；
- 我们会对**每个查询帧**单独发起一次 3D 轨迹跟踪。

### 3.2 每个查询帧上的 2D 网格点

函数：`create_uniform_grid_points(height, width, grid_size)`：

1. 在图像平面上均匀采样一个 \(grid\_size \times grid\_size\) 网格：

   \[
   x \in [0, W-1], \quad y \in [0, H-1]
   \]

2. 形成二维点集 \((x_k, y_k)\)，展开成长度为 \(N = grid\_size^2\) 的集合。

3. 在 TraceForge 内部，每个点记录为三元组 \((t, x, y)\)，其中：

   - \(t\)：时间索引（查询帧所在段内的位置，见下节）；
   - 初始化时 \(t = 0\)，后续按段重映射。

这些 2D 网格点就是 Tapir3D 要跟踪的「keypoints」。

---

## 4. 分段追踪：从 2D + 深度 + 相机到 3D 轨迹

### 4.1 Tracking Segment 的划分

对于每个查询帧 `frame_idx`，我们定义一个追踪段：

```python
end_frame = min(frame_idx + args.future_len, video_length)
tracking_segments.append((frame_idx, end_frame))
```

- `future_len`（默认 128）：最多向前跟踪的帧数；
- 例如：`video_length = 320`，`future_len = 128`，`frame_drop_rate = 5`：
  - 查询帧 0 → 追踪 [0, 128)
  - 查询帧 5 → 追踪 [5, 133)
  - 查询帧 10 → 追踪 [10, 138)

### 4.2 为每个段准备输入（`prepare_inputs`）

对某个查询帧对应的段 `[start_frame, end_frame)`：

1. 截取这一段的视频 / 深度 / 相机参数：

   ```python
   video_segment = video_ten[start_frame:end_frame]      # (T_seg, 3, H, W)
   depth_segment = depth_npy[start_frame:end_frame]      # (T_seg, H, W)
   intrs_segment = intrs_npy[start_frame:end_frame]      # (T_seg, 3, 3)
   extrs_segment = extrs_npy[start_frame:end_frame]      # (T_seg, 4, 4)
   ```

2. 段内查询点时间重映射

   原来的网格点中 time 维记录的是「全局帧索引」，在段内我们把它重置为 0：

   ```python
   segment_query_point = [query_point[seg_idx].copy()]
   segment_query_point[0][:, 0] = 0  # 段内从 0 开始
   ```

3. 通过 `prepare_inputs(...)` 转换为 Tapir3D 需要的 3D 查询格式：

   - **步骤 (a)：内参按分辨率做缩放**（当前代码中保持原分辨率一致）；
   - **步骤 (b)：对每一帧深度做滤波**（调用 `_filter_one_depth`），去除近/远噪声；
   - **步骤 (c)：将 2D 网格 + 深度 + 相机几何反投影到 3D 世界坐标**：

     对第 \(t\) 帧，给定某个 2D 像素 \((x, y)\)：

     - 构造像素齐次坐标：

       \[
       \mathbf{p} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
       \]

     - 将深度 \(d = D^{(t)}(x, y)\) 取出；
     - 用内参逆矩阵 \(K_t^{-1}\) 求相机坐标下的 3D 点：

       \[
       \mathbf{x}_{\text{cam}} = d \cdot K_t^{-1} \mathbf{p}
       \]

     - 用外参的相机到世界变换 \(T_{c2w}^{(t)}\) 把点变到世界坐标：

       \[
       \mathbf{x}_{\text{world}} = R_{c2w}^{(t)} \mathbf{x}_{\text{cam}} + \mathbf{t}_{c2w}^{(t)}
       \]

   - 最终 `prepare_inputs` 输出：

     ```python
     video:      (T_seg, 3, H, W) torch.float32
     depths:     (T_seg, H, W)    torch.float32
     intrinsics: (T_seg, 3, 3)    torch.float32
     extrinsics: (T_seg, 4, 4)    torch.float32
     query_point: (N, 4)          # [t, x_world, y_world, z_world]
     ```

### 4.3 Tapir3D 追踪 3D 轨迹

调用：

```python
coords_seg, visibs_seg = inference(
    model=model_3dtracker,
    video=video,
    depths=depths,
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    query_point=query_point_tensor,
    num_iters=args.num_iters,
    grid_size=support_grid_size,
    bidrectional=False,
)
```

- **输入**：
  - 视频帧序列；
  - 每帧深度图；
  - 每帧相机内外参；
  - 起始帧上的 3D 查询点（世界坐标）。

- **输出**：
  - `coords_seg`: \((T_{\text{seg}}, N, 3)\)，每一帧每个点的 3D 坐标（世界坐标系）；
  - `visibs_seg`: \((T_{\text{seg}}, N)\)，可见性（0/1 或概率）。

这些就是**原始 3D 轨迹**，还没做任何时间重采样。

---

## 5. 轨迹时间重采样（Retarget）

为了统一所有 sample 的长度（例如固定为 128 步），TraceForge 对原始 3D 轨迹做「弧长均匀重采样」，实现函数是 `retarget_trajectories(...)`。

输入：

- `trajectory`: 形状 \((N, H, D)\)，这里 \(H = T_{\text{seg}}\)，\(D \in \{2,3\}\)；
  - 对 sample NPZ 来说，通常是 `traj` 的一条轨迹集合；
- `interval`: 弧长采样间隔；
- `max_length`: 目标长度（例如 128）。

核心思想：

1. **按时间段测量「典型轨迹」的运动长度**：

   - 对每个时间段 \(t \to t+1\)，计算所有轨迹在该段的欧氏位移；
   - 取每段长度的前 `top_percent`（例如 2%）的平均作为该段代表长度，避免受噪声和静止点影响。

2. **在累计弧长上每隔 `interval` 放一个目标点**：

   - 把原始时间轴上的段长度累加成「弧长轴」；
   - 在弧长轴上等间距采样；
   - 每个采样点根据弧长落在哪个时间段里，用线性插值得到对应的轨迹点。

3. **对所有轨迹同步应用同一组 \((\text{段索引}, \alpha)\)**：

   - 所有点在同一时间步上都对应同一个段和同一个插值系数 \(\alpha\)，保持时间语义一致。

输出：

- `retargeted`: \((N, max\_length, D)\)，填充不足的时间步为 `-inf`；
- `valid_mask`: \((max\_length,)\) 布尔掩码，指示哪些时间步有效。

在 `save_structured_data(...)` 里，会对 `sample_data["traj"]` 调用这个函数，然后连同 `valid_steps` 一起写入 sample NPZ（见 `traceforge_output_structure.md`）。

---

## 6. Retarget 数学细节与例子

这一节更形式化地写出 retarget 的数学过程，并用具体数字举例说明。

### 6.1 弧长定义与段长度

设：

- 轨迹集合 \(X \in \mathbb{R}^{N \times H \times D}\)，其中：
  - \(N\)：轨迹条数（grid 上的 keypoints 数量）；
  - \(H\)：原始时间长度（segment 内的帧数）；
  - \(D \in \{2,3\}\)：每个点的维度（2D 或 3D）；
- 用 \(X_{n,t} \in \mathbb{R}^D\) 表示第 \(n\) 条轨迹在时间 \(t\) 的位置。

**段长度（per-frame displacement）**：

对每一对相邻时间步 \(t \to t+1\)，以及每条轨迹 \(n\)，先计算欧氏位移：

\[
L_{n,t} \;=\; \left\| X_{n,t+1} - X_{n,t} \right\|_2
\quad\text{for}\quad
n = 1,\dots,N,\;\; t = 0,\dots,H-2.
\]

于是我们得到一个矩阵：

\[
L \in \mathbb{R}^{N \times (H-1)}.
\]

在实现中，这一步是：

```python
diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]  # (N, H-1, D)
seglens_all = np.linalg.norm(diffs_all, axis=2)        # (N, H-1)
```

### 6.2 Top-k 段长度与「典型弧长」

为了避免少数噪声轨迹或静止轨迹影响统计，TraceForge 对每个时间段 \(t\) 使用「top k% 的平均值」来代表这一段的长度：

1. 给定超参数 `top_percent`（默认 0.02，即前 2%）；
2. 对每个时间段 \(t\)，取所有轨迹的长度 \(\{L_{n,t}\}_{n=1}^N\)，选出其中最大的前 \(k=\max(1, \lceil N \cdot \text{top\_percent} \rceil)\) 个；
3. 对这些 top-k 做平均：

\[
\hat{L}_t \;=\; \frac{1}{k} \sum_{n \in \text{TopK}(t)} L_{n,t}
\quad\text{for}\quad t = 0,\dots,H-2.
\]

这得到了一条对时间段的代表性长度序列：

\[
\hat{L} \in \mathbb{R}^{H-1}.
\]

实现中大致是：

```python
part = np.partition(seglens_all, N - k, axis=0)  # (N, H-1)
topk = part[N - k:, :]                           # (k, H-1)
robust_seglen = topk.mean(axis=0)                # (H-1,)
```

我们可以把这些长度视为在时间轴上的「局部弧长」。

### 6.3 累计弧长与等距采样点

1. 构建累计弧长序列：

\[
s_0 = 0,\qquad
s_{t+1} = s_t + \hat{L}_t
        = \sum_{j=0}^{t} \hat{L}_j,
\quad t = 0,\dots,H-2.
\]

于是得到：

\[
s \in \mathbb{R}^H,\quad s = [s_0, s_1, \dots, s_{H-1}].
\]

2. 总弧长为：

\[
S_{\text{total}} = s_{H-1} = \sum_{j=0}^{H-2} \hat{L}_j.
\]

3. 设定弧长采样间隔 `interval`，我们在 \([0, S_{\text{total}}]\) 上均匀采样：

\[
K_{\max} = \left\lfloor \frac{S_{\text{total}}}{\text{interval}} \right\rfloor,
\quad
M = \min(K_{\max} + 1,\; \text{max\_length}),
\]

并构造采样弧长：

\[
u_k = k \cdot \text{interval},\quad k = 0,\dots,M-1.
\]

最后一个采样点会被截断到总弧长内：

\[
u_{M-1} = \min\big(u_{M-1},\; S_{\text{total}}\big).
\]

### 6.4 在时间轴上的插值位置

对每个采样弧长 \(u_k\)，需要精确回答两个问题：

1. 它对应的是「原始第几帧和第几帧之间」？
2. 在那两帧之间走了多少比例（\(\alpha_k\)）？

#### 6.4.1 找到 \(u_k\) 属于哪一段 \([t_k, t_k+1]\)

累计弧长数组 `s` 满足：

- `s[0] = 0`
- `s[1] = L̂_0`
- `s[2] = L̂_0 + L̂_1`
- ...

可以把它想象成一条数轴上的刻度：

\[
0 = s_0 < s_1 < s_2 < \dots < s_{H-1}.
\]

如果 \(u_k\) 落在 `s[t]` 和 `s[t+1]` 之间，就说明它对应的物理进度在**原始第 t 帧和第 t+1 帧之间**。

代码中使用：

```python
idx_seq = np.searchsorted(s, targets, side='right') - 1
idx_seq = np.clip(idx_seq, 0, H - 2)
```

数学上就是：

\[
t_k = \max\big(0,\; \min(H-2,\; \text{searchsorted}(s, u_k,\text{right}) - 1)\big).
\]

小例子（直观理解）：

- 若 \(s = [0,1,3,6,10]\)：
  - `u_k = 0.5`：插在 0 和 1 之间 → `searchsorted` 返回 1 → `t_k = 0` → 段 \([0,1]\)；
  - `u_k = 2.0`：插在 1 和 3 之间 → 返回 2 → `t_k = 1` → 段 \([1,2]\)；
  - `u_k = 6.0`：恰好等于 6，`side='right'` 会返回 4 → `t_k = 3` → 段 \([3,4]\)。

#### 6.4.2 计算段内归一化位置 \(\alpha_k\)

在确定了 \(t_k\) 之后，这一段的代表长度是：

\[
\hat{L}_{t_k} = s_{t_k+1} - s_{t_k},
\]

段起点弧长为 \(s_{t_k}\)。于是段内的归一化位置为：

\[
\alpha_k = \frac{u_k - s_{t_k}}{\hat{L}_{t_k}},
\quad \alpha_k \in [0,1].
\]

- 当 \(u_k = s_{t_k}\) 时，\(\alpha_k = 0\)，表示完全在原始第 \(t_k\) 帧；
- 当 \(u_k = s_{t_k+1}\) 时，\(\alpha_k = 1\)，表示完全在原始第 \(t_k+1\) 帧；
- 中间情况是两帧之间的线性位置。

#### 6.4.3 对所有轨迹做线性插值

对每一条轨迹 \(n\)，原始轨迹在这两帧上的位置是：

- \(X_{n,t_k}\)：原始第 \(t_k\) 帧的 3D 点；
- \(X_{n,t_k+1}\)：原始第 \(t_k+1\) 帧的 3D 点。

retarget 后第 \(k\) 步的 3D 位置按线性插值给出：

\[
X_{n}^{\text{ret}}(k)
  = (1-\alpha_k)\, X_{n,t_k} + \alpha_k\, X_{n,t_k+1},
\quad n=1,\dots,N.
\]

实现里对应于：

```python
left = traj_norm[:, idx_seq, :]          # (N, M, D)
right = traj_norm[:, idx_seq + 1, :]     # (N, M, D)
samples_norm = left + alpha_seq[None, :, :] * (right - left)
```

**重要：**对所有轨迹 \(n\) 使用的是**同一组** \((t_k, \alpha_k)\)：

- 这意味着「retarget 后的第 k 步」在所有轨迹上都表示**同一个物理时间段上的同一相对进度**；
- 只是每条轨迹的 3D 坐标不同（对应不同 keypoint 在场景中的实际位置）。

### 6.5 统一长度与 valid_mask

最终，我们得到：

- `retargeted`：形状为 \((N, \text{max\_length}, D)\)；
  - 若 \(M < \text{max\_length}\)，后面的时间步会被填充为 \(-\infty\)（表示无效）；
- `valid_mask`：形状为 \((\text{max\_length},)\)，
  - 前 \(M\) 个位置为 `True`，其余为 `False`。

在写入 sample NPZ 时：

- `traj` 就是 `retargeted`；
- `valid_steps` 就是 `valid_mask`。

**一句话总结 retarget 的目的**：  
在保持「物理运动进度」一致的前提下，用统一的时间长度（例如 128 步）重新参数化所有轨迹，让不同 query_frame 和不同视频之间的轨迹可以在同一个标准时间轴上对齐、对比和可视化。

### 6.6 几个具体例子

#### 例 1：短轨迹（H=5），让 retarget 更长（max_length=8）

设某条轨迹在 5 个时间步上的位置（一维情况，便于直观）为：

\[
X = [0,\; 1,\; 3,\; 6,\; 10],\quad H=5.
\]

1. 段长度：

\[
\hat{L}_0 = |1-0| = 1,\\
\hat{L}_1 = |3-1| = 2,\\
\hat{L}_2 = |6-3| = 3,\\
\hat{L}_3 = |10-6| = 4.
\]

2. 累计弧长：

\[
s = [0,\; 1,\; 3,\; 6,\; 10].
\]

3. 总弧长 \(S_{\text{total}} = 10\)，设 `interval=1.5`，`max_length=8`：

\[
K_{\max} = \left\lfloor 10 / 1.5 \right\rfloor = 6,\quad M = \min(6+1, 8) = 7.
\]

采样弧长：

\[
u = [0,\; 1.5,\; 3.0,\; 4.5,\; 6.0,\; 7.5,\; 9.0].
\]

4. 例如 \(u_1 = 1.5\)：

- 找到 \(s\) 中刚好大于它的元素位置：
  - \(s = [0,1,3,\dots]\)，所以 `searchsorted(s, 1.5) = 2`；
  - 段索引 \(t_1 = 2-1 = 1\)（对应原始段 \([1,3]\)）。
- 段长度 \(\hat{L}_1 = 2\)，段起点弧长 \(s_1=1\)，插值系数：

  \[
  \alpha_1 = (1.5 - 1) / 2 = 0.25.
  \]

- 对位置做插值：

  \[
  X^{\text{ret}}(1) = (1-\alpha_1) \cdot X_1 + \alpha_1 \cdot X_2
                    = 0.75 \cdot 1 + 0.25 \cdot 3 = 1.5.
  \]

同理，可以算出其它 \(u_k\) 对应在原始轨迹上的位置，从而得到长度为 \(M=7\) 的 retarget 轨迹。若 `max_length` 更大（比如 128），则 `M` 仍受总弧长限制，后面的时间步用 `valid_mask=False` 标掉。

#### 例 2：几乎静止的轨迹

如果一条轨迹几乎不动（例如某些背景点），那么它在各段的长度 \(L_{n,t}\) 都非常接近 0。由于我们在每个段上只取「top k%」的轨迹来估计 \(\hat{L}_t\)，如果大部分轨迹都是运动的，则这些静止点不会显著影响 \(\hat{L}_t\)；反过来，如果整个场景总体运动也很小，则整条「典型弧长」也会很短，相应地 retarget 后的有效长度 \(M\) 也会比较小（后半段 `valid_steps=False`）。

#### 例 3：不同 query_frame_idx 的轨迹长度是否相同？

在实现中：

- **对所有 query_frame_idx**：
  - `retarget_trajectories` 的 `max_length` 参数是同一个（默认 128）；
  - 也就是说，**`traj` 的第二维（时间维）在 sample NPZ 中都是固定的 `T=128`**；
- 但**有效的时间步数 \(M\)** 可能随 `query_frame_idx` 而异：
  - 若某个 segment 很短（比如剩余帧数不足），则总弧长较小，`M` 可能只有 20 或 30；
  - 在这种情况下，`traj[:, M:, :]` 会被填补为 \(-\infty\)，对应的 `valid_steps[M:] = False`。

因此：

- **从数组形状上看**：所有 query_frame 的 `traj` 长度是固定的（例如 `(N, 128, 3)`）；
- **从有效时长上看**：每个 query_frame 的真实轨迹长度可以不同，需要结合 `valid_steps` 使用。

这就是为什么在可视化时，我们通常会：

```python
traj = data["traj"]          # (N, T, 3)
valid = data["valid_steps"]  # (T,)
T_eff = valid.sum()
traj_eff = traj[:, :T_eff, :]
```

来确保只播放有效的时间步。

---

## 7. 数据落盘与可视化关联

### 6.1 Sample NPZ（按查询帧存储）

对于每个查询帧 `query_frame_idx`，`save_structured_data(...)` 会写出一个：

- `samples/<video_name>_<query_frame_idx>.npz`

其结构在 `traceforge_output_structure.md` 中已经详细列出，关键字段包括：

- `traj`: \((N, T, 3)\)，世界坐标系 3D 轨迹（retarget 后）；
- `traj_2d`: \((N, T_{\text{orig}}, 2)\)，对应的 2D 投影；
- `keypoints`: \((N, 2)\)，查询帧上的 2D 网格；
- `frame_index` / `image_path` / `valid_steps` 等。

### 6.2 主 NPZ（首段原始轨迹）

同时，会为该相机写出一个主 NPZ（`images0.npz`）：

- 包含首个 segment 的原始 `coords` / `depths` / `visibs`，以及全视频的 `intrinsics` / `extrinsics` 等；
- 这部分数据与视频帧一一对应，没有做 retarget，主要用于：
  - 帧对齐的密集点云；
  - 静态/动态 3D 可视化。

### 6.3 可视化脚本如何使用这些轨迹

1. `scripts/visualization/visualize_3d_keypoint_animation.py`

   - 从 `episode_dir`（例如 `.../00000/images0`）中：
     - 读取 `samples/*.npz`：每个查询帧的 `traj`；
     - 读取主 `images0.npz`：做密集点云时用 `depths + intrinsics + extrinsics` 反投影。
   - 按时间步 \(t\) 播放：

     \[
     \text{显示点集 } \{\mathbf{x}_{k}(t)\}_{k=1}^{N}, \quad \mathbf{x}_{k}(t) \in \mathbb{R}^3
     \]

2. `scripts/visualization/visualize_single_image.py`

   - 读取单帧的 RGB / 深度 / 相机参数与一帧的轨迹数据；
   - 在 3D 视图中展示该帧对应的点云 + 轨迹分布。

---

## 8. 实际数据小结示例

这里用一个 BridgeV2 推理结果来把上面的概念「对号入座」，你可以按类似思路在本机检查任何一个 case。

### 8.1 示例路径与文件

假设我们已经对 `00000` 做过推理，输出目录为：

```text
/data2/dataset_proceseed/output_bridge_v2_full_grid80/00000/images0/
    images/           # 查询帧 RGB
    depth/            # 查询帧深度 PNG + raw npz
    samples/          # 每个 query_frame 的轨迹样本
    images0.npz       # 主 NPZ（首段原始轨迹 + 全视频相机）
```

在 Python 里可以这样查看一个 sample NPZ：

```python
import numpy as np

path = "/data2/dataset_proceseed/output_bridge_v2_full_grid80/00000/images0/samples/images0_0.npz"
data = np.load(path)

traj = data["traj"]          # (N, T, 3)
valid = data["valid_steps"]  # (T,)
keypoints = data["keypoints"]  # (N, 2)

print("traj shape:", traj.shape)
print("valid_steps sum:", valid.sum())
```

你通常会看到类似输出（假设 `grid_size=80`）：

```text
traj shape: (6400, 128, 3)
valid_steps sum: 94
```

这可以这样理解：

- **从形状看**：所有 query_frame（比如 0、5、10、15）对应 sample NPZ 里的 `traj` 时间维都是 128（`max_length`）。
- **从 `valid_steps` 看**：这条查询帧 0 的 segment 里，「典型轨迹」在弧长累计上只支持大约 94 个等距采样点：
  - 第 0～93 步是真实轨迹，`valid_steps[:94] = True`；
  - 第 94～127 步只是占位，`valid_steps[94:] = False`。

在做可视化或下游处理时，应当使用：

```python
T_eff = int(valid.sum())
traj_eff = traj[:, :T_eff, :]  # (N, T_eff, 3)
```

而不是简单遍历 0~127。

### 8.2 不同 query_frame 的对比

如果你再读取一个非零查询帧的 sample，如：

```python
path5 = "/data2/dataset_proceseed/output_bridge_v2_full_grid80/00000/images0/samples/images0_5.npz"
d5 = np.load(path5)
print("traj[5] shape:", d5["traj"].shape)
print("valid_steps[5] sum:", d5["valid_steps"].sum())
```

在多数情况下，你会看到：

- `traj[5].shape == traj[0].shape` → 依然是 `(6400, 128, 3)`；
- 但 `valid_steps[5].sum()` 可能略有不同（比如 88、92、95 等），这取决于从帧 5 开始往后的 segment 长度和运动量。

这验证了前面结论：

- **同一个 episode 内，不同 query_frame 的 sample，在数组形状上时间维统一（方便 batch 处理与存盘）**；
- **实际有效时长由 `valid_steps` 决定，可能随 query_frame 变化**。

---

## 9. 小结

综上，TraceForge 的 3D 轨迹 pipeline 可以概括为：

1. **加载视频与深度** → 得到对齐的 `[T, 3, H, W]` 与 `[T, H, W]`。
2. **深度 & 位姿估计或外部几何注入** → 得到每帧 `depths + intrinsics + extrinsics`。
3. **按 `frame_drop_rate` 选择查询帧**，在每个查询帧采样 \(N = grid\_size^2\) 个 2D 网格点。
4. **对每个查询帧定义 tracking segment**，长度最多 `future_len`。
5. **使用深度 + 相机几何，将 2D 点反投影到 3D 世界坐标**，得到初始 3D 查询点。
6. **用 Tapir3D（CoTracker 变体）在段内追踪这些 3D 点**，得到 \((T_{\text{seg}}, N, 3)\) 的原始 3D 轨迹。
7. **对轨迹做弧长均匀重采样（retarget）**，统一为固定长度（如 128），得到 sample NPZ 的 `traj + valid_steps`。
8. **将首个 segment 的原始轨迹与全视频相机参数写入主 NPZ**，供帧对齐可视化和密集点云使用。

如果你后续需要更数学化的推导（例如严格写出从像素坐标到世界坐标、再到不同相机系之间变换的矩阵公式），可以在本文件基础上再新开一节「附录：坐标系与矩阵形式」，进一步展开。 

