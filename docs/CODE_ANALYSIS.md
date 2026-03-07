# TraceForge 代码架构分析

> 版本：external-geom-20260305 分支
> 更新时间：2026-03-07

## 一、核心功能与数据流

### 1.1 整体流程

TraceForge 将视频 + 深度数据转换为 3D 轨迹，核心流程：

```
视频帧序列 (RGB)
深度图序列 (Depth) [可选]
几何信息 (内外参) [可选]
        ↓
[步骤1] 帧采样与加载
        - 根据 stride 采样帧
        - 深度从 16-bit PNG (mm) 转换为 float (m)
        ↓
[步骤2] 深度 + 位姿估计
        - 方式A: VGGT4Track 从 RGB 估计
        - 方式B: 使用外部提供的深度+内外参 (新)
        ↓
[步骤3] 生成 Query Points
        - 在指定帧上生成均匀网格点
        - 每隔 frame_drop_rate 帧生成一次
        ↓
[步骤4] 3D 点跟踪 (TAPIP3D)
        - 对每个 query 帧，跟踪 future_len 帧
        - 输出 3D 坐标和可见性
        ↓
[步骤5] 结果保存
        - 主 NPZ: 完整视频的深度/位姿/轨迹
        - 每帧 NPZ: 单帧轨迹数据
        - PNG: 深度可视化
```

### 1.2 关键参数

- `stride`: 帧采样间隔，自动计算为 `ceil(总帧数 / max_frames_per_video)`
- `frame_drop_rate`: query 帧间隔，默认 1（每帧都 query）
- `grid_size`: 每帧 query 点网格大小（如 80 → 6400 个点）
- `future_len`: 从 query 帧向后跟踪的帧数（默认 128）
- `num_iters`: TAPIP3D 迭代次数（默认 6）
- `depth_pose_method`: 深度位姿估计方法，'vggt4' 或 'external' (新)


## 二、深度位姿估计方法对比

### 2.1 方法选择

通过 `--depth_pose_method` 参数选择：

| 方法 | 说明 | 使用场景 |
|------|------|----------|
| `vggt4` | 使用 VGGT4Track 从 RGB 估计 | 只有 RGB 视频，或需要验证深度估计 |
| `external` | 使用外部提供的深度+内外参 | 已有精确的深度和相机标定数据 |

### 2.2 VGGT4Track 方法 (vggt4)

**作用：从单目 RGB 视频估计场景几何**

**输入：**
- RGB 视频：(T, 3, H, W)

**输出：**
- 深度图：(T, H_v, W_v)，预处理后分辨率（~518×518）
- 相机外参：(T, 4, 4)，C2W 格式
- 相机内参：(T, 3, 3)

**关键特性：**
1. 预处理 RGB：resize + center crop 到 ~518×518
2. 如果提供 `known_depth`：
   - 用 median scale 对齐预测深度
   - 可选择替换为 known_depth
   - 外参平移按 scale 缩放

**代码位置：** `utils/video_depth_pose_utils.py` - `VGGT4Wrapper`


### 2.3 外部几何方法 (external) - 新特性

**作用：完全绕过 VGGT，使用外部提供的深度和相机参数**

**输入：**
- RGB 视频：(T, 3, H, W)
- 外部深度：(T, H, W)，通过 `--depth_path` 提供
- 外部几何 NPZ：通过 `--external_geom_npz` 提供
  - `intrinsics`: (T, 3, 3)
  - `extrinsics`: (T, 4, 4)

**输出：**
- RGB：(T, 3, H, W)，简单归一化到 [0,1]，不做 resize/crop
- 深度：(T, H, W)，直接使用外部深度
- 深度置信度：(T, H, W)，简单取 (depth > 0)
- 外参：(T, 4, 4)，直接使用外部外参
- 内参：(T, 3, 3)，直接使用外部内参

**关键特性：**
1. ✅ 完全不调用 VGGT 模型
2. ✅ 保持原始分辨率，不做预处理
3. ✅ 自动对齐时间维度（取最短长度）
4. ✅ 支持静态相机模式

**使用示例：**
```bash
python scripts/batch_inference/infer.py \
  --depth_pose_method external \
  --external_geom_npz /path/to/camera_params.npz \
  --depth_path /path/to/depth/frames \
  --video_path /path/to/rgb/frames \
  --out_dir ./output
```

**代码位置：** `utils/video_depth_pose_utils.py` - `ExternalGeomWrapper`


## 三、核心模块分析

### 3.1 推理入口 (scripts/batch_inference/infer.py)

**主函数：`process_single_video()`**

关键流程：
1. 计算 stride（自适应帧采样）
2. 加载 RGB 和深度
3. 调用深度位姿估计（VGGT4 或 external）
4. 生成 query points（均匀网格）
5. 对每个 query 帧独立跟踪
6. 返回结果字典

**分段跟踪策略：**
- 每个 query 帧独立处理
- 跟踪窗口：`[query_frame, query_frame + future_len]`
- 避免长序列的显存问题

### 3.2 TAPIP3D 模型 (models/point_tracker_3d.py)

**架构：** Encoder → Correlation Feature → Point Updater

**迭代优化：**
1. 初始化 query 点的 3D 坐标
2. 迭代 num_iters 次（默认 6）
3. 每次迭代预测坐标增量并更新
4. 逐步精化跟踪结果


## 四、external-geom-20260305 分支核心特性

### 4.1 新增功能

**支持完全使用外部几何信息，绕过 VGGT**

- 新增 `ExternalGeomWrapper` 类
- 新增 `--external_geom_npz` 参数
- 新增 `depth_pose_method='external'` 选项

### 4.2 使用场景对比

| 场景 | 方法 | 优势 |
|------|------|------|
| 只有 RGB 视频 | vggt4 | 自动估计深度和位姿 |
| 有深度，无相机参数 | vggt4 + known_depth | 深度对齐，估计位姿 |
| 有深度和相机参数 | external | 完全精确，无估计误差 |

### 4.3 关键改动文件

- `utils/video_depth_pose_utils.py`: +98 行，新增 ExternalGeomWrapper
- `scripts/batch_inference/infer.py`: +12 行，添加参数和类型兼容

---

**文档生成时间：** 2026-03-05  
**分析基于：** external-geom-20260305 分支代码


## 五、自定义修改记录（2026-03-06）

### 5.1 支持 .npy 格式深度文件

**问题：** 原代码只支持 16-bit PNG 深度（单位 mm），需要转换为米

**修改：** `scripts/batch_inference/infer.py` - `load_video_and_mask()`

**改动：**
- 深度文件扩展名：添加 `"npy"` 到支持列表
- 加载逻辑：
  - `.npy` 格式：直接加载，假定单位为米（m）
  - `.png` 格式：从 16-bit PNG 加载，单位 mm，转换为米

**代码位置：** L778, L787-793


### 5.2 支持从 H5 文件读取内外参

**问题：** 原代码只支持从 NPZ 文件读取内外参，用户数据集使用 H5 格式

**修改：** `utils/video_depth_pose_utils.py` - `ExternalGeomWrapper.__init__()`

**改动：**
- 自动检测文件类型（.h5 或 .npz）
- H5 文件读取路径：
  - 内参：`observation/camera/intrinsics/{camera_name}_left`
  - 外参：`observation/camera/extrinsics/{camera_name}_left`
- 保持向后兼容 NPZ 格式

**数据格式：**
- 内参：(T, 3, 3) float64 → 转为 float32
- 外参：(T, 4, 4) float64 → 转为 float32

**代码位置：** L172-194


### 5.3 新增参数

**`--camera_name`**
- 类型：str
- 默认值：`hand_camera`
- 说明：指定 H5 文件中的相机名称
- 可选值：`hand_camera`, `varied_camera_1`, `varied_camera_2`
- 用途：与 `depth_pose_method='external'` 配合使用

**代码位置：** `scripts/batch_inference/infer.py` L46-51

### 5.4 使用示例

**处理单个相机：**
```bash
python scripts/batch_inference/infer.py \
  --depth_pose_method external \
  --external_geom_npz /path/to/trajectory_valid.h5 \
  --camera_name hand_camera \
  --depth_path /path/to/depth/hand_camera/depth \
  --video_path /path/to/rgb_stereo_valid/hand_camera/left \
  --out_dir ./output_hand_camera
```

**处理三个相机（需要分别运行）：**
```bash
# hand_camera
python scripts/batch_inference/infer.py \
  --depth_pose_method external \
  --external_geom_npz /path/to/trajectory_valid.h5 \
  --camera_name hand_camera \
  --depth_path /path/to/depth/hand_camera/depth \
  --video_path /path/to/rgb_stereo_valid/hand_camera/left \
  --out_dir ./output

# varied_camera_1
python scripts/batch_inference/infer.py \
  --depth_pose_method external \
  --external_geom_npz /path/to/trajectory_valid.h5 \
  --camera_name varied_camera_1 \
  --depth_path /path/to/depth/varied_camera_1/depth \
  --video_path /path/to/rgb_stereo_valid/varied_camera_1/left \
  --out_dir ./output

# varied_camera_2
python scripts/batch_inference/infer.py \
  --depth_pose_method external \
  --external_geom_npz /path/to/trajectory_valid.h5 \
  --camera_name varied_camera_2 \
  --depth_path /path/to/depth/varied_camera_2/depth \
  --video_path /path/to/rgb_stereo_valid/varied_camera_2/left \
  --out_dir ./output
```

---

**修改完成时间：** 2026-03-06  
**修改内容：** 支持 .npy 深度文件 + H5 内外参读取


## 六、自定义修改记录（2026-03-07）

### 6.1 DROID 批处理脚本分流：纯 VGGT 与 external 分离

**新增/调整脚本：**
- `scripts/batch_inference/batch_droid.py`：默认纯 VGGT 流程（仅 RGB），不再传入 `--depth_path` / `--external_geom_npz`
- `scripts/batch_inference/batch_droid_external.py`：external 几何直通（外部深度 + 内外参）

**关键变化：**
1. 数据集扫描逻辑调整  
   - 纯 VGGT 脚本只要求 `rgb_stereo_valid` 可用
   - external 脚本要求 RGB + depth + 外部几何（NPZ/H5）同时可用
2. DROID 相机目录兼容  
   - 允许直接传相机目录；若根目录无帧，自动回退读取 `left/`
3. 批处理稳定性增强  
   - 新增 `--task_timeout`、`--only_incomplete`
   - 每相机日志输出到 `trajectory/_logs/*.log`
   - 单 GPU 单 worker 串行，避免单卡并发过载


### 6.2 infer.py 参数语义与输出命名更新

**文件：** `scripts/batch_inference/infer.py`

**关键变化：**
1. `--video_name` 新参数  
   - 显式控制输出目录/文件名前缀
2. 输出命名优先级  
   - `--video_name` > `--camera_name`(当提供 `--external_geom_npz`) > 输入路径名
3. `--external_geom_npz` 语义扩展  
   - `depth_pose_method=external`：使用外部内外参并跳过 VGGT
   - `depth_pose_method=vggt4`：仅用外部外参替换 VGGT 外参（深度仍由 VGGT 估计）
4. 帧对齐增强  
   - 外部内外参按 stride 同步抽帧，避免与 RGB/Depth 采样错位
   - `original_filenames` 与有效帧数不一致时自动截断对齐
5. 失败治理  
   - 批量模式统计失败视频数；若存在失败，进程以非零退出码结束（`sys.exit(1)`）


### 6.3 video_depth_pose_utils.py 深度位姿封装更新

**文件：** `utils/video_depth_pose_utils.py`

**关键变化：**
1. 新增外部几何加载函数  
   - `_load_external_extrinsics()`：读取外部外参（NPZ/H5）
   - `_load_external_geom()`：读取外部内外参（NPZ/H5）
2. `VGGT4Wrapper` 支持“外部外参替换”  
   - 在 VGGT 深度估计保留的前提下，用外部外参替换模型外参
   - 自动处理帧数不一致（截断到可用长度）
3. `ExternalGeomWrapper` 重构  
   - 统一复用 `_load_external_geom()`
   - 时间维自动对齐到 `min(video, depth, geom)`
   - 输入未归一化时自动缩放到 `[0, 1]`


### 6.4 可视化帧数对齐修复

**文件：** `scripts/visualization/visualize_3d_keypoint_animation.py`

**修改：**
- 在重建稠密点云前，将 `intrinsics` / `extrinsics` 截断到 `coords` 的时间长度 `T`
- 避免相机参数帧数长于轨迹帧数时的索引错位


### 6.5 当前推荐使用方式（DROID）

**纯 VGGT（仅 RGB）：**
```bash
python scripts/batch_inference/batch_droid.py \
  --base_path /path/to/droid_raw \
  --gpu_id 0,1,2,3 \
  --grid_size 80 \
  --frame_drop_rate 5 \
  --only_incomplete
```

**external 几何直通（RGB + 深度 + 外部几何）：**
```bash
python scripts/batch_inference/batch_droid_external.py \
  --base_path /path/to/droid_raw \
  --gpu_id 0,1,2,3 \
  --grid_size 80 \
  --frame_drop_rate 5 \
  --geom_source auto \
  --only_incomplete
```
