## TraceForge 推理流水总览（含 BridgeV2 具体示例）

本文梳理 TraceForge 从输入视频/深度到输出 3D 轨迹与可视化的完整推理流水，并以 **BridgeV2 单相机 + 外部深度** 为例，标注每一步的**数据形状（维度）**。

示例路径约定：

- 项目根：`/home/wangchen/projects/TraceForge`
- 数据：`/data1/dataset/BridgeV2/first_5_collection/00001`
  - RGB：`images0/*.jpg`
  - 深度：`depth_images0/*.png`（16-bit，mm）
- 输出：`output_00001/images0`

### 1. 顶层脚本与主要模块

- **批推入口（批量多轨迹）：**
  - `scripts/batch_inference/batch_bridge_v2.py`  
  - 对 `base_path` 下每条 BridgeV2 轨迹调用 `infer_bridge_v2.py`。
- **单轨迹 BridgeV2 推理：**
  - `scripts/batch_inference/infer_bridge_v2.py`  
  - 对一条 `traj_id`（含 `images0/1/2 + depth_images0/1/2`）调用 `infer.py` 的 `process_single_video`。
- **核心单视频推理逻辑：**
  - `scripts/batch_inference/infer.py`  
  - 封装了：
    - 帧加载与 stride 计算
    - 深度 + 位姿估计（VGGT4Track）
    - 3D 网格 query 生成
    - Tapir3D 轨迹推理
    - 结构化结果保存（`save_structured_data`）

后面所有形状说明都以 **BridgeV2 单相机 + 外部深度** 路径为例：

```bash
cd /home/wangchen/projects/TraceForge

PYTHONPATH=. conda run -n traceforge python scripts/batch_inference/infer.py \
  --video_path /data1/dataset/BridgeV2/first_5_collection/00001/images0 \
  --depth_path /data1/dataset/BridgeV2/first_5_collection/00001/depth_images0 \
  --out_dir /home/wangchen/projects/TraceForge/output_00001 \
  --frame_drop_rate 5 \
  --grid_size 80
```

输出目录：`output_00001/images0`

---

## 2. 输入加载与 stride 计算

### 2.1 `process_single_video` 入口

文件：`scripts/batch_inference/infer.py`

```python
def process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose):
    ...
```

关键参数：

- `video_path`：RGB 帧目录（如 `.../00001/images0`）
- `depth_path`：深度帧目录（如 `.../00001/depth_images0`）
- `args.frame_drop_rate`：query 帧间隔（如 5）
- `args.grid_size`：轨迹网格大小（如 80 → 每帧 80×80=6400 点）

### 2.2 stride（帧采样间隔）

```python
if args.fps and int(args.fps) > 0:
    stride = int(args.fps)
else:
    stride = 1
    if os.path.isdir(video_path):
        img_files = _collect_and_sort_frame_files(video_path, ["jpg", "jpeg", "png"])
        n_frames = len(img_files)
        target = max(1, int(getattr(args, "max_frames_per_video", 150)))
        stride = max(1, math.ceil(n_frames / target)) if n_frames > 0 else 1
```

示例：`images0` 有 350 帧，`max_frames_per_video=150` →  
`stride = ceil(350/150) = 3`，最终只加载约 `350/3 ≈ 117` 帧。

### 2.3 加载 RGB 和深度

```python
video_tensor, video_mask, original_filenames = load_video_and_mask(
    video_path, args.mask_dir, stride, args.max_num_frames
)
```

- `video_tensor` 形状：\((T, C, H, W)\)，例如：
  - \(T = 117\)（stride 后帧数）
  - \(C = 3\)
  - \(H, W\)：原图高度/宽度（如 480×640）

加载深度：

```python
depth_tensor, _, _ = load_video_and_mask(
    depth_path, None, stride, args.max_num_frames, is_depth=True
)  # [T, H, W]
...
valid_depth = (depth_tensor > 0)
depth_tensor[~valid_depth] = 0
```

- `depth_tensor` 形状：\((T, H, W)\)，与 `video_tensor` 时间、空间对齐。
- 内部转换逻辑（`is_depth=True`）：
  - 从 16-bit PNG 读入：
    - `img.convert("I;16")` → `np.array(img).astype(np.float32)`
  - 默认假设单位是 **毫米 (mm)**：
    - `depth_array = raw / 1000.0` → **米 (m)**。

---

## 3. 深度 + 相机位姿估计（VGGT4Track）

### 3.1 `model_depth_pose` 调用

```python
video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy = model_depth_pose(
    video_tensor,
    known_depth=depth_tensor,  # 外部深度，可为 None
    stationary_camera=False,
    replace_with_known_depth=False,
)
```

其中：

- `video_tensor`：\((T, 3, H, W)\)，0–255，后续内部会归一化到 [0,1] 并做 resize/crop；
- `known_depth`：\((T, H, W)\)，单位米（由 PNG 读入并除以 1000 得到）。

### 3.2 `VGGT4Wrapper.__call__` 输出形状

文件：`utils/video_depth_pose_utils.py`

```python
video_tensor_processed  # (T, 3, H_v, W_v) 经过预处理（518x518 左右）
depth_map               # (T, H_v, W_v)
extrinsic               # (T, 4, 4) C2W
intrinsic               # (T, 3, 3)
```

函数最终返回：

- `video_ten`：\((T, 3, H_v, W_v)\)，float32
- `depth_npy`：\((T, H_v, W_v)\)，float32，单位 m
- `depth_conf`：\((T, H_v, W_v)\)
- `extrs_npy`：\((T, 4, 4)\)，C2W，平移已按尺度对齐（见下）
- `intrs_npy`：\((T, 3, 3)\)

### 3.3 用外部深度做尺度校正（median scale）

当 `known_depth` 非空时：

1. 先把 `known_depth` interpolate 到预测分辨率：

   \[
   D_{\text{known}}(t,x,y)
   \in \mathbb{R}^{T \times H_v \times W_v}
   \]

2. 对每帧 \(t\) 的有效像素集合 \(\mathcal{V}_t\)（两者都 >0）：

   \[
   s_t =
   \frac{
     \operatorname{median}(D_{\text{known}}^{(t)}(x,y)\mid (x,y)\in\mathcal{V}_t)
   }{
     \operatorname{median}(D_{\text{pred}}^{(t)}(x,y)\mid (x,y)\in\mathcal{V}_t)
   }
   \]

3. 对全视频取平均：

   \[
   s = \frac{1}{T}\sum_{t=1}^T s_t
   \]

4. 用 \(s\) 统一缩放预测深度与外参平移：

   \[
   D_{\text{aligned}}^{(t)} = s \cdot D_{\text{pred}}^{(t)},\quad
   \mathbf{T}_t^\prime = s \cdot \mathbf{T}_t
   \]

5. 若 `replace_with_known_depth=True`（BridgeV2 中语义是“以实测深度为准”）：

   \[
   D_{\text{final}}^{(t)} = D_{\text{known}}^{(t)}
   \]

在当前实现中，我们通常：

- 用 `D_{\text{final}}` 作为后续 3D 推理与存盘的深度；
- 用 `\mathbf{T}_t^\prime` 作为全局尺度对齐后的相机位姿。

---

## 4. 3D 轨迹推理：query 网格与 tracking 段

### 4.1 query 网格与 tracking 段划分

在 `process_single_video` 中，基于 `frame_drop_rate` 划分查询帧：

```python
video_length = len(video_tensor)  # T'
query_frames = list(range(0, video_length, args.frame_drop_rate))
```

示例：`T' = 117`, `frame_drop_rate=5` → `query_frames = [0,5,10,15,20,25,30,35,...]`

每个 query 帧 \(t_q\) 定义一个 tracking 段：

```python
end_frame = min(frame_idx + args.future_len, video_length)
tracking_segments.append((frame_idx, end_frame))
```

例如 `future_len=128` → 对于 `t_q=10`：

- 段为 `[10, min(10+128, 117))` → `[10,117)`，长度 `L_q ≈ 107` 帧。

### 4.2 每帧 query 上的 2D 网格点

对每个 query 帧构造 \(grid\_size \times grid\_size\) 均匀网格：

```python
frame_H, frame_W = video_ten.shape[-2:]  # H_v, W_v
grid_points = create_uniform_grid_points(
    height=frame_H, width=frame_W, grid_size=args.grid_size, device="cpu"
)  # (grid_size*grid_size, 2)
```

- 示例：`grid_size=80` → 每帧查询 80×80=6400 个 2D 点。

### 4.3 query 点与深度反投影（3D query）

函数：`prepare_query_points(query_xyt, depths, intrinsics, extrinsics)`。

输入：

- `query_xyt`：包含每个 query 帧上 2D 网格的 `(t,x,y)` 列表；
- `depths`：\((T', H_v, W_v)\)；
- `intrinsics`：\((T', 3,3)\)；
- `extrinsics`：\((T', 4,4)\)。

输出：

- `query_point`：\((N_q, 3)\)，所有 query 点的 3D 坐标（通常在世界坐标或某一相机系下）。

然后打包给 3D tracker：

```python
video, depths, intrinsics, extrinsics, query_point_tensor, support_grid_size = (
    prepare_inputs(
        video_segment,
        depth_segment,
        intrs_segment,
        extrs_segment,
        segment_query_point,
        ...
    )
)
```

在 `prepare_inputs` 中还会对深度做边缘滤波 `_filter_one_depth`，保证 3D 反投影的稳定性：

- `video`：\((L_q, 3, H_v, W_v)\)
- `depths`：\((L_q, H_v, W_v)\)
- `intrinsics`：\((L_q, 3,3)\)
- `extrinsics`：\((L_q, 4,4)\)
- `query_point_tensor`：\((N_q, 3)\)

---

## 5. Tapir3D 轨迹推理与结果汇总

### 5.1 每个段的 3D 轨迹

调用 `inference`（Tapir3D 模型）：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    coords_seg, visibs_seg = inference(
        model=model_3dtracker,
        video=video,              # (L_q, 3, H_v, W_v)
        depths=depths,            # (L_q, H_v, W_v)
        intrinsics=intrinsics,    # (L_q, 3,3)
        extrinsics=extrinsics,    # (L_q, 4,4)
        query_point=query_point_tensor,  # (N_q, 3)
        num_iters=args.num_iters,
        grid_size=support_grid_size,
        bidrectional=False,
    )
```

输出：

- `coords_seg`：\((L_q, N_q, 3)\)
  - 每个 query 点（空间上一个 3D 点）在整个 tracking 段内的 3D 轨迹；
- `visibs_seg`：\((L_q, N_q)\)，可见性（0/1）。

### 5.2 汇总所有 query 段

`query_frame_results[start_frame]` 收集每段结果：

```python
query_frame_results[start_frame] = {
    "coords": coords_seg,           # (L_q, N_q, 3)
    "visibs": visibs_seg,           # (L_q, N_q)
    "video_segment": video,         # (L_q, 3, H_v, W_v)
    "depths_segment": depths,       # (L_q, H_v, W_v)
    "intrinsics_segment": intrinsics,
    "extrinsics_segment": extrinsics,
}
```

在返回 dict 之前，还会构造一个“主结果”（首段）：

```python
first_frame = min(query_frame_results.keys())
coords = query_frame_results[first_frame]["coords"]           # (L_0, N_q, 3)
visibs = query_frame_results[first_frame]["visibs"]           # (L_0, N_q)
video  = query_frame_results[first_frame]["video_segment"]    # (L_0, 3, H_v, W_v)
depths = query_frame_results[first_frame]["depths_segment"]   # (L_0, H_v, W_v)
intrinsics = query_frame_results[first_frame]["intrinsics_segment"]
extrinsics = query_frame_results[first_frame]["extrinsics_segment"]
```

返回字典中关键字段形状：

- `"coords"`：\((L_0, N_q, 3)\)
- `"visibs"`：\((L_0, N_q)\)
- `"video_tensor"`：\((L_0, 3, H_v, W_v)\)
- `"depths"`：\((L_0, H_v, W_v)\)
- `"intrinsics"` / `"extrinsics"`：\((L_0, 3,3)\), \((L_0, 4,4)\)
- `"full_intrinsics"` / `"full_extrinsics"`：\((T, 3,3)\), \((T, 4,4)\)（全视频）

---

## 6. 结构化存盘：images/、depth/、samples/ 与主 NPZ

### 6.1 `save_structured_data` 接口

```python
save_structured_data(
    video_name,           # 如 "images0"/"left"
    output_dir,           # 如 out_dir
    video_tensor,         # (L_0, 3, H_v, W_v)
    depths,               # (L_0, H_v, W_v)
    coords,               # (L_0, N_q, 3)
    visibs,               # (L_0, N_q)
    intrinsics,           # (L_0, 3,3)
    extrinsics,           # (L_0, 4,4)
    query_points_per_frame,
    horizon,
    original_filenames,
    use_all_trajectories=True,
    query_frame_results=query_frame_results,
    future_len=args.future_len,
    grid_size=args.grid_size,
)
```

在 `output_dir/video_name`（例如 `output_00001/images0`）下创建：

- `images/`
- `depth/`
- `samples/`

### 6.2 每个 query 帧的 RGB + 深度输出

对于每个 query 段（示例 `start_frame = 0`）：

- RGB：保存段内第 0 帧（query 帧）：
  - `images/{video_name}_{query_frame_idx}.png`  
    示例：`images/images0_0.png`
- 深度 PNG（可视化）：
  - `depth/{video_name}_{query_frame_idx}.png` 16-bit，经过本帧内部线性归一化到 `[0,65535]`
- 原始深度 raw：
  - `depth/{video_name}_{query_frame_idx}_raw.npz`，键名 `depth`，形状 \((H_v, W_v)\)，单位米

### 6.3 sample NPZ（每个 query 帧一份）

对每个 query 帧生成 `samples/{video_name}_{query_frame_idx}.npz`，包含：

- `traj`：\((N_q, L_q', 3)\)，经过 `retarget_trajectories` 后的 3D 轨迹（时间轴重采样到 `future_len` 步内）
- `traj_2d`：\((N_q, L_q', 2)\)，对应 2D 投影（统一固定在首帧相机系）
- `keypoints`：\((N_q, 2)\)，query 帧 2D 位置
- `frame_index`：`[query_frame_idx]`
- `image_path`：`["images/...png"]`
- `valid_steps`：\((L_q',)\) 有效时间步 mask

这些 sample NPZ 主要用于后续可视化脚本的 per-query 动画展示。

### 6.4 主 NPZ（传统可视化用）

在 `output_dir/video_name` 下还会保存一个主 NPZ（例如 `images0/images0.npz` 或 `left/left.npz`）：

- `coords`：\((L_0, N_q, 3)\) （只用首段）
- `depths`：\((L_0, H_v, W_v)\)，`float16`，单位米
- `extrinsics`：\((T, 4,4)\)，**全视频**相机位姿（C2W/C2?，视实现而定）
- `intrinsics`：\((T, 3,3)\)
- `height` / `width`：推理分辨率
- `unc_metric`：\((L_0, H_v, W_v)\)
- `visibs`：\((L_0, N_q, 1)\)
- （可选）`video`：\((L_0, H_v, W_v, 3)\)，当传入 `--save_video` 时保存

这个主 NPZ 是 `visualize_3d_keypoint_animation.py --dense_pointcloud` 的密集点云数据源。

---

## 7. 可视化：3D keypoint 动画与密集点云

### 7.1 基本接口

文件：`scripts/visualization/visualize_3d_keypoint_animation.py`

```bash
cd /home/wangchen/projects/TraceForge

PYTHONPATH=. conda run -n traceforge python scripts/visualization/visualize_3d_keypoint_animation.py \
  --episode_dir output_00001/images0 \
  --query_frame 0 \
  --keypoint_stride 5 \
  --dense_pointcloud \
  --dense_downsample 4 \
  --normalize_camera \
  --port 8080
```

关键参数：

- `--episode_dir`：指向单个 episode 下的相机目录（含主 NPZ + images/depth/samples）
- `--query_frame`：哪一个 query 帧（如 0、5、10…），不指定则自动选第一个
- `--keypoint_stride`：keypoint 子采样（每 N 个显示 1 个）
- `--dense_pointcloud`：启用密集点云模式
- `--dense_downsample`：密集点云下采样因子
- `--normalize_camera`：将轨迹变换到首帧相机坐标系
- `--port`：viser 监听端口（默认 8080）

### 7.2 sample 模式（无 `--dense_pointcloud`）

不加 `--dense_pointcloud` 时：

- 只用 `samples/*.npz` 中的 `traj/traj_2d/keypoints` 做动画；
- 背景图像来自 `images/` 中相应的 `{video_name}_{query_frame}.png`；
- 深度信息（`depth/` + `_raw.npz`）仅用于非首帧时的点云辅助，不依赖主 NPZ 的 `coords/depths`。

### 7.3 主 NPZ + 密集点云模式（`--dense_pointcloud`）

加 `--dense_pointcloud` 时，脚本会：

1. 调用 `load_main_npz_for_dense`：
   - 从主 NPZ 中读取 `coords/depths/intrinsics/extrinsics`；
   - 按时间维度 \(T = \min(T_{\text{coords}}, T_{\text{depths}}, T_{\text{intr}}, T_{\text{extr}})\) 对齐；
   - 使用 `unproject_by_depth` 将每帧深度 + 内外参反投影为密集点云：
     - 输入：`depth_batch` \((T, 1, H_v, W_v)\)，`K_batch` \((T, 3,3)\)，`c2w` \((T, 4,4)\)
     - 输出：`xyz` \((T, 3, H_v, W_v)\)
   - 对每帧点云进行下采样与有效性过滤，得到：
     - `dense_per_frame[t]`：\((M_t, 3)\)
     - `dense_colors_per_frame[t]`：\((M_t, 3)\)
     - `keypoint_traj`：\((N_q, T, 3)\)

2. 在 viser 中同时渲染：
   - keypoint 点云（带颜色）
   - 可选轨迹线（按时间播放）
   - 可选密集点云（随时间更新）
   - 各种 GUI 控件（开关 keypoint/轨迹/密集点云、调节点大小等）

### 7.4 depth_only 输出的限制

在 `--depth_only` 模式下的主 NPZ（例如 `images0.npz`）中：

- 只保留了 `extrinsics/intrinsics/depths/height/width/unc_metric`；
- **不再保存** `coords/visibs`。

因此：

- 仍然可以使用 sample 模式（不加 `--dense_pointcloud`）进行可视化；
- 若想使用密集点云模式，需确保主 NPZ 中包含 `coords`（即使用完整推理输出，而非 depth_only）。 

