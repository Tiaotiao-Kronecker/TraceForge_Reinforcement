# infer.py 详细讲解文档

## 📋 目录
1. [整体流程概览](#整体流程概览)
2. [主要函数详解](#主要函数详解)
3. [数据流和关键步骤](#数据流和关键步骤)
4. [关键参数说明](#关键参数说明)

---

## 🎯 整体流程概览

`infer.py` 是 TraceForge 项目的核心推理脚本，用于从视频中提取3D轨迹。整体流程如下：

```
输入视频/图像序列
    ↓
[1] 参数解析 (parse_args)
    ↓
[2] 模型初始化 (load_model, video_depth_pose_dict)
    ↓
[3] 批量/单视频处理判断 (batch_process)
    ↓
[4] 对每个视频:
    ├─ [4.1] 加载视频和深度 (load_video_and_mask)
    ├─ [4.2] 估计深度和位姿 (model_depth_pose)
    ├─ [4.3] 创建查询点网格 (create_uniform_grid_points)
    ├─ [4.4] 对每个查询帧:
    │   ├─ 提取视频段 (video_segment)
    │   ├─ 准备输入 (prepare_inputs)
    │   ├─ 3D轨迹跟踪 (inference)
    │   └─ 存储结果 (query_frame_results)
    ├─ [4.5] 轨迹重定向 (retarget_trajectories)
    ├─ [4.6] 保存结构化数据 (save_structured_data)
    └─ [4.7] 保存可视化NPZ
    ↓
输出: images/, depth/, samples/, *.npz
```

---

## 🔧 主要函数详解

### 1. `parse_args()` (第26-100行)

**作用**: 解析命令行参数

**关键参数**:
- `--video_path`: 视频路径（必需）
- `--depth_path`: 深度图路径（可选）
- `--checkpoint`: 模型权重路径
- `--future_len`: 每个查询帧的跟踪窗口长度（默认128帧）
- `--frame_drop_rate`: 查询帧采样间隔（默认1，每帧都查询）
- `--max_frames_per_video`: 每个视频保留的最大帧数（默认50）
- `--batch_process`: 是否批量处理
- `--grid_size`: 网格大小（代码中硬编码为20，即20x20=400个点）

---

### 2. `retarget_trajectories()` (第102-210行)

**作用**: 轨迹重定向 - 将不同长度的轨迹统一到固定长度

**算法步骤**:

#### 步骤1: 全局归一化 (第148-160行)
```python
# 使用最后一条轨迹的第一帧坐标作为归一化基准
scale_x = trajectory[-1, 0, 0]  # x方向尺度
scale_y = trajectory[-1, 0, 1]  # y方向尺度

# 归一化所有轨迹的x,y坐标
traj_norm[:, :, 0] /= scale_x
traj_norm[:, :, 1] /= scale_y

# 裁剪到[0,1]范围
np.clip(traj_norm[:, :, 0], 0.0, 1.0, ...)
np.clip(traj_norm[:, :, 1], 0.0, 1.0, ...)
# z坐标不归一化
```

**潜在问题**: 
- 对异常值敏感（依赖单个轨迹的单个点）
- 不同轨迹尺度可能不一致
- z坐标不归一化导致尺度不一致

#### 步骤2: 计算鲁棒段长度 (第162-172行)
```python
# 计算相邻帧之间的位移
diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]  # (N, H-1, D)
seglens_all = np.linalg.norm(diffs_all, axis=2)  # (N, H-1)

# 对每个时间段，取top k%的轨迹长度的均值（鲁棒统计）
k = max(1, int(np.ceil(top_percent * N)))  # top 2%
topk = part[N - k:, :]  # 取最大的k个值
robust_seglen = topk.mean(axis=0)  # (H-1,) 每个时间段的鲁棒长度
```

**为什么使用top-k均值？**
- 过滤掉异常轨迹（移动过快或过慢）
- 使用鲁棒统计量，对噪声更稳定

#### 步骤3: 构建累积弧长并放置目标点 (第179-194行)
```python
# 计算累积弧长
s = np.zeros((H,))
s[1:] = np.cumsum(robust_seglen)  # s[i] = sum(robust_seglen[0:i])

# 在累积弧长上每隔interval放置一个目标点
targets = interval * np.arange(num_samples)  # [0, interval, 2*interval, ...]

# 找到每个目标点对应的段索引和插值系数
idx_seq = np.searchsorted(s, targets, side='right') - 1  # 段索引
alpha = (targets - s[idx_seq]) / robust_seglen[idx_seq]  # 段内插值系数
```

**示例**:
```
假设 robust_seglen = [0.1, 0.2, 0.15, 0.3]
累积弧长 s = [0, 0.1, 0.3, 0.45, 0.75]
interval = 0.2
targets = [0, 0.2, 0.4, 0.6]

target=0.2 → idx_seq=1, alpha=(0.2-0.1)/0.2=0.5
target=0.4 → idx_seq=2, alpha=(0.4-0.3)/0.15=0.67
```

#### 步骤4: 同步插值 (第196-199行)
```python
# 对所有轨迹使用相同的(idx, alpha)进行插值
left = traj_norm[:, idx_seq, :]   # (N, num_samples, D)
right = traj_norm[:, idx_seq + 1, :]  # (N, num_samples, D)
samples_norm = left + alpha_seq * (right - left)  # 线性插值
```

**同步的含义**: 所有轨迹在同一时刻使用相同的插值系数，保持时间同步

#### 步骤5: 反归一化 (第201-205行)
```python
# 只对x,y反归一化，z保持插值结果
samples_out[:, :, 0] *= scale_x
samples_out[:, :, 1] *= scale_y
```

---

### 3. `save_structured_data()` (第212-393行)

**作用**: 保存结构化数据到磁盘

**输出目录结构**:
```
output_dir/
└── video_name/
    ├── images/          # RGB图像
    │   └── video_name_0.png
    ├── depth/           # 深度图
    │   ├── video_name_0.png
    │   └── video_name_0_raw.npz
    └── samples/         # 轨迹数据
        └── video_name_0.npz
```

**每个sample NPZ文件包含**:
- `image_path`: 图像路径
- `frame_index`: 帧索引
- `keypoints`: 关键点坐标 (400, 2) - 20x20网格
- `traj`: 3D轨迹 (400, T, 3) - 400条轨迹，T帧
- `traj_2d`: 2D轨迹投影 (400, T, 2)
- `valid_steps`: 有效步数掩码 (T,)

**关键步骤** (第246-391行):

1. **提取查询帧数据** (第246-260行)
   ```python
   coords_np = frame_data["coords"].cpu().numpy()  # (T, 400, 3)
   visibs_np = frame_data["visibs"].cpu().numpy()  # (T, 400)
   ```

2. **创建关键点网格** (第283-289行)
   ```python
   grid_size = 20  # 20x20 = 400个点
   y_coords = np.linspace(0, frame_h - 1, grid_size)
   x_coords = np.linspace(0, frame_w - 1, grid_size)
   xx, yy = np.meshgrid(x_coords, y_coords)
   keypoints = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (400, 2)
   ```

3. **投影到2D和3D** (第310-351行)
   ```python
   # 使用第一帧的相机参数进行投影（保持一致性）
   fixed_camera_view = camera_views_segment[0]
   tracks2d_fixed = project_tracks_3d_to_2d(...)  # (T, 400, 2)
   tracks3d_fixed = project_tracks_3d_to_3d(...)  # (T, 400, 3)
   ```

4. **轨迹重定向** (第379-381行)
   ```python
   retargeted, valid_mask = retarget_trajectories(
       sample_data["traj"], 
       max_length=args.future_len
   )
   ```

5. **保存文件** (第361-377行)
   - RGB图像: PNG格式
   - 深度图: 16位PNG + 原始NPZ
   - 样本数据: NPZ格式

---

### 4. `process_single_video()` (第396-653行)

**作用**: 处理单个视频，返回轨迹数据

**主要流程**:

#### 4.1 计算采样步长 (第400-425行)
```python
if args.fps > 0:
    stride = args.fps  # 固定步长
else:
    # 自动计算步长，使保留帧数 <= max_frames_per_video
    stride = ceil(n_frames / max_frames_per_video)
```

#### 4.2 加载视频和深度 (第427-439行)
```python
video_tensor = load_video_and_mask(video_path, ...)  # (T, C, H, W)
depth_tensor = load_video_and_mask(depth_path, ...)  # (T, H, W)
```

#### 4.3 估计深度和位姿 (第443-451行)
```python
video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy = model_depth_pose(
    video_tensor,
    known_depth=depth_tensor,  # 如果提供，用于尺度对齐
    stationary_camera=False,
    replace_with_known_depth=False,  # False = 只对齐尺度，不替换
)
```

**深度处理模式**:
- `replace_with_known_depth=False`: 估计深度，然后用已知深度对齐尺度
- `replace_with_known_depth=True`: 直接使用已知深度

#### 4.4 创建查询点网格 (第465-489行)
```python
# 根据frame_drop_rate确定查询帧
query_frames = list(range(0, video_length, args.frame_drop_rate))

for frame_idx in query_frames:
    # 创建20x20均匀网格
    grid_points = create_uniform_grid_points(
        height=frame_H, width=frame_W, 
        grid_size=20, device="cpu"
    )  # (1, 400, 3) - [t, x, y]
    
    grid_points[:, 0] = frame_idx  # 设置帧索引
```

#### 4.5 对每个查询帧进行跟踪 (第506-590行)
```python
for seg_idx, (start_frame, end_frame) in enumerate(tracking_segments):
    # 提取视频段
    video_segment = video_ten[start_frame:end_frame]
    depth_segment = depth_npy[start_frame:end_frame]
    
    # 准备输入（深度过滤、查询点转换等）
    video, depths, intrinsics, extrinsics, query_point_tensor, ... = (
        prepare_inputs(...)
    )
    
    # 3D轨迹跟踪
    coords_seg, visibs_seg = inference(
        model=model_3dtracker,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        query_point=query_point_tensor,
        num_iters=args.num_iters,
        grid_size=support_grid_size,
        bidrectional=False,  # 只向前跟踪
    )
    # coords_seg: (T, 400, 3) - T帧，400条轨迹，每轨迹3D坐标
    # visibs_seg: (T, 400) - 可见性掩码
```

**关键点**:
- 每个查询帧独立处理
- 跟踪窗口长度 = `min(future_len, video_length - start_frame)`
- 使用混合精度推理 (`torch.bfloat16`) 加速

---

### 5. `load_video_and_mask()` (第705-764行)

**作用**: 加载视频帧或图像序列

**支持格式**:
- **图像目录**: 加载所有 `.jpg`, `.png` 文件
- **视频文件**: `.mp4`, `.mov`, `.avi` 等

**处理流程**:
```python
if os.path.isdir(video_path):
    # 加载图像序列
    img_files = sorted(glob.glob(...))
    img_files = img_files[::fps]  # 采样
    
    for img_file in img_files:
        img = Image.open(img_file)
        if is_depth:
            img = img.convert("I;16")  # 16位深度图
        else:
            img = img.convert("RGB")
        video_tensor.append(torch.from_numpy(np.array(img)).float())
    
    video_tensor = torch.stack(video_tensor)  # (N, H, W, 3)
    
    if not is_depth:
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        video_tensor /= 255.0  # 归一化到[0,1]
```

---

### 6. `create_uniform_grid_points()` (第767-798行)

**作用**: 创建均匀网格查询点

**算法**:
```python
# 在图像上创建均匀分布的网格点
y_coords = np.linspace(0, height - 1, grid_size)  # [0, H-1]分成grid_size份
x_coords = np.linspace(0, width - 1, grid_size)   # [0, W-1]分成grid_size份

xx, yy = np.meshgrid(x_coords, y_coords)  # 创建网格
grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (400, 2)

# 添加时间维度
time_col = np.zeros((400, 1))
grid_points_3d = np.concatenate([time_col, grid_points], axis=1)  # (400, 3)
# 格式: [t, x, y]，其中t初始为0，后续会被设置为实际帧索引
```

**示例** (grid_size=20, 图像512x384):
```
x_coords = [0, 26.95, 53.89, ..., 511]
y_coords = [0, 20.21, 40.42, ..., 383]
→ 400个均匀分布的点
```

---

### 7. `prepare_query_points()` (第800-819行)

**作用**: 将2D图像坐标转换为3D世界坐标

**转换流程**:
```python
for query_i in query_xyt:  # query_i: (N, 3) - [t, x, y]
    t = int(query_i[0, 0])  # 帧索引
    xy = query_i[:, 1:]     # (N, 2) - 图像坐标
    
    # 1. 获取对应深度的深度值
    ji = np.round(xy).astype(int)  # 像素坐标
    d = depth_t[t, ji[..., 1], ji[..., 0]]  # 深度值
    
    # 2. 图像坐标 → 相机坐标
    xy_homo = [xy, 1]  # 齐次坐标
    local_coords = K_inv @ xy_homo.T  # 归一化相机坐标
    local_coords = local_coords * d  # 乘以深度得到3D相机坐标
    
    # 3. 相机坐标 → 世界坐标
    world_coords = c2w[:3, :3] @ local_coords + c2w[:3, 3]
    
    # 4. 组合: [t, x_world, y_world, z_world]
    final_queries.append([t, world_coords])
```

**坐标转换链**:
```
图像坐标 (x, y) 
  → 归一化相机坐标 (通过K_inv)
  → 3D相机坐标 (乘以深度d)
  → 3D世界坐标 (通过c2w变换)
```

---

### 8. `prepare_inputs()` (第822-854行)

**作用**: 准备模型输入，包括深度过滤和坐标转换

**处理步骤**:

1. **调整内参** (第836-837行)
   ```python
   # 如果图像分辨率改变，调整内参
   intrinsics[:, 0, :] *= (new_w - 1) / (old_w - 1)
   intrinsics[:, 1, :] *= (new_h - 1) / (old_h - 1)
   ```

2. **深度过滤** (第839-845行)
   ```python
   # 使用多线程并行处理每帧深度
   depths_futures = [
       executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
       for depth, intrinsic in zip(depths, intrinsics)
   ]
   depths = np.stack([future.result() for future in depths_futures])
   ```
   - `_filter_one_depth`: 过滤深度边缘，去除噪声

3. **查询点转换** (第847行)
   ```python
   query_point = prepare_query_points(query_point, depths, intrinsics, extrinsics)
   # 将2D图像坐标转换为3D世界坐标
   ```

4. **转换为Tensor** (第848-852行)
   ```python
   query_point = torch.from_numpy(query_point).float().to(device)
   video = video_ten.float().to(device).clamp(0, 1)
   depths = torch.from_numpy(depths).float().to(device)
   intrinsics = torch.from_numpy(intrinsics).float().to(device)
   extrinsics = torch.from_numpy(extrinsics).float().to(device)
   ```

---

### 9. `find_video_folders()` (第656-702行)

**作用**: 递归查找视频文件夹（用于批量处理）

**算法**:
```python
# 计算目标深度
target_depth = base_depth + scan_depth

# 遍历目录树
for root, dirs, files in os.walk(base_path):
    current_depth = root.count(os.sep)
    
    if current_depth == target_depth:
        # 检查是否有图像文件
        if any(f.endswith(('.jpg', '.png')) for f in files):
            video_folders.append(root)
    
    if current_depth > target_depth:
        dirs[:] = []  # 停止深入
```

**示例** (scan_depth=2):
```
base_path = /data/videos
target_depth = base_depth + 2

/data/videos/episode1/images/  ✓ (深度=3，匹配)
/data/videos/episode2/images/  ✓ (深度=3，匹配)
/data/videos/episode1/images/sub/  ✗ (深度=4，跳过)
```

---

## 🔄 数据流和关键步骤

### 完整数据流

```
[输入]
视频帧序列 (T帧)
    ↓
[步骤1] 加载和采样
load_video_and_mask()
→ video_tensor: (T', C, H, W)  # T' <= T (采样后)
    ↓
[步骤2] 深度和位姿估计
model_depth_pose()
→ depth_npy: (T', H, W)
→ extrs_npy: (T', 4, 4)  # 外参（世界→相机）
→ intrs_npy: (T', 3, 3)  # 内参
    ↓
[步骤3] 创建查询点
create_uniform_grid_points()
→ grid_points: (400, 3)  # [t, x, y] per frame
    ↓
[步骤4] 对每个查询帧
for each query_frame:
    [4.1] 提取视频段
    → video_segment: (future_len, C, H, W)
    
    [4.2] 准备输入
    prepare_inputs()
    → query_point_tensor: (400, 4)  # [t, x_world, y_world, z_world]
    
    [4.3] 3D轨迹跟踪
    inference()
    → coords_seg: (future_len, 400, 3)  # 3D轨迹
    → visibs_seg: (future_len, 400)     # 可见性
    ↓
[步骤5] 轨迹重定向
retarget_trajectories()
→ retargeted: (400, future_len, 3)  # 统一长度
→ valid_mask: (future_len,)          # 有效掩码
    ↓
[步骤6] 投影和保存
project_tracks_3d_to_2d()  # 投影到2D
save_structured_data()
→ samples/*.npz
→ images/*.png
→ depth/*.png
```

---

## 📊 关键参数说明

### 采样相关参数

| 参数 | 作用阶段 | 说明 | 默认值 |
|------|---------|------|--------|
| `max_frames_per_video` | 视频加载 | 控制加载多少帧 | 50 |
| `frame_drop_rate` | 查询帧选择 | 每隔N帧查询一次 | 1 |
| `future_len` | 跟踪窗口 | 每个查询帧跟踪多少帧 | 128 |
| `grid_size` | 查询点密度 | 网格大小（硬编码20） | 20 |

### 处理流程示例

假设视频有200帧：

```python
# 步骤1: 视频加载
max_frames_per_video = 50
→ stride = ceil(200/50) = 4
→ 加载帧: [0, 4, 8, 12, ..., 196] (共50帧)

# 步骤2: 查询帧选择
frame_drop_rate = 1
→ 查询帧: [0, 1, 2, 3, ..., 49] (共50个查询帧)

# 步骤3: 跟踪
future_len = 128
→ 查询帧0: 跟踪帧[0:128] (受限于50帧，实际跟踪[0:50])
→ 查询帧10: 跟踪帧[10:50] (受限于视频长度)

# 步骤4: 查询点
grid_size = 20
→ 每帧400个查询点 (20x20网格)
```

---

## 🎯 核心算法总结

### 1. 均匀网格采样
- 在图像上创建20x20均匀网格
- 每个网格点作为查询点
- 共400个查询点/帧

### 2. 3D轨迹跟踪
- 使用深度学习模型跟踪查询点的3D轨迹
- 输入: 视频段 + 深度 + 相机参数 + 查询点（3D世界坐标）
- 输出: 每条轨迹的3D坐标序列 + 可见性掩码

### 3. 轨迹重定向
- 将不同长度的轨迹统一到固定长度
- 使用弧长重采样，保持轨迹的几何特性
- 同步插值，保持时间一致性

### 4. 坐标系统
- **图像坐标**: (x, y) - 像素位置
- **相机坐标**: (X_cam, Y_cam, Z_cam) - 相对于相机
- **世界坐标**: (X_world, Y_world, Z_world) - 全局坐标系

转换链: `图像坐标 → 相机坐标 → 世界坐标`

---

## ⚠️ 注意事项

1. **内存管理**: 
   - 使用 `torch.cuda.empty_cache()` 清理GPU缓存
   - 限制 `max_num_frames` 避免OOM

2. **深度处理**:
   - 如果提供已知深度，默认只对齐尺度，不替换
   - 深度单位需要一致（米或毫米）

3. **轨迹数量**:
   - 代码中硬编码 `grid_size=20`，即400条轨迹
   - 如果修改grid_size，需要同步修改相关代码

4. **批量处理**:
   - 使用 `--batch_process` 可以处理多个视频
   - `--skip_existing` 可以跳过已处理的视频

---

## 📝 总结

`infer.py` 实现了完整的3D轨迹提取流程：
1. **输入**: 视频帧序列（可选深度图）
2. **处理**: 深度估计、位姿估计、3D轨迹跟踪
3. **输出**: 结构化数据（图像、深度、轨迹NPZ）

关键特点：
- 支持批量处理
- 使用均匀网格采样
- 轨迹重定向统一长度
- 完整的坐标转换链

