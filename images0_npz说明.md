# images0.npz 文件说明

## 📋 文件位置

`images0.npz` 文件保存在每个 traj 的输出目录根目录下：
```
output_traj_group0/
├── traj0/
│   └── images0/
│       ├── images/
│       ├── depth/
│       ├── samples/
│       └── images0.npz    ← 这里
├── traj1/
│   └── images0/
│       └── images0.npz
└── ...
```

## 🎯 文件用途

`images0.npz` 是 **"Traditional visualization NPZ"**（传统可视化NPZ文件），用于存储整个视频的完整轨迹数据，便于进行传统的3D可视化。

## 📦 文件内容

根据 `infer.py` 第 937-954 行的代码，`images0.npz` 包含以下数据：

| 字段名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `coords` | `(T, 6400, 3)` | `float32` | 所有帧的所有轨迹的3D坐标<br/>T = 视频总帧数<br/>6400 = 轨迹数量 (80x80 grid)<br/>3 = (x, y, z) 坐标 |
| `extrinsics` | `(T, 4, 4)` | `float32` | 所有帧的相机外参矩阵（相机到世界坐标变换） |
| `intrinsics` | `(T, 3, 3)` | `float32` | 所有帧的相机内参矩阵（焦距、主点等） |
| `depths` | `(T, H, W)` | `float16` | 所有帧的深度图<br/>H, W = 图像高度和宽度 |
| `unc_metric` | `(T, H, W)` | `float16` | 所有帧的深度置信度/不确定性指标 |
| `visibs` | `(T, 6400, 1)` | `uint8` | 所有帧的所有轨迹的可见性掩码<br/>1 = 可见，0 = 不可见 |
| `height` | `()` | `int` | 图像高度 |
| `width` | `()` | `int` | 图像宽度 |
| `video` | `(T, H, W, 3)` | `uint8` | （可选）所有帧的RGB图像<br/>仅在 `--save_video` 参数启用时保存 |

## 🔄 与 samples/ 目录下NPZ文件的区别

### images0.npz（传统可视化格式）
- **位置**：`{traj_dir}/images0/images0.npz`
- **数据范围**：整个视频的所有帧
- **数据组织**：按时间维度组织 `(T, N, ...)`
- **用途**：用于传统的3D可视化工具，一次性加载整个视频的轨迹数据
- **特点**：
  - 包含完整的相机参数（每帧都有）
  - 包含完整的深度图（每帧都有）
  - 包含所有轨迹的完整时间序列

### samples/{video_name}_{frame}.npz（结构化数据格式）
- **位置**：`{traj_dir}/images0/samples/images0_{frame}.npz`
- **数据范围**：单个查询帧的轨迹数据
- **数据组织**：按轨迹维度组织 `(N, T, ...)`
- **用途**：用于训练/测试数据，每个查询帧独立存储
- **特点**：
  - 每个文件对应一个查询帧（如 frame 0, 5, 10, ...）
  - 包含该查询帧的起始关键点 `keypoints`
  - 包含该查询帧的轨迹 `traj`（经过重定向，统一长度为 `future_len`）
  - 包含2D投影轨迹 `traj_2d`
  - 包含有效性掩码 `valid_steps`

## 📊 数据对比示例

假设视频有 45 帧，使用 `frame_drop_rate=5`，查询帧为 [0, 5, 10, 15, 20, 25, 30, 35, 40]：

### images0.npz
```python
{
    'coords': (45, 6400, 3),      # 所有45帧的轨迹
    'extrinsics': (45, 4, 4),     # 所有45帧的相机外参
    'intrinsics': (45, 3, 3),     # 所有45帧的相机内参
    'depths': (45, H, W),         # 所有45帧的深度图
    'visibs': (45, 6400, 1),      # 所有45帧的可见性
    ...
}
```

### samples/images0_0.npz
```python
{
    'keypoints': (6400, 2),       # 查询帧0的起始关键点
    'traj': (6400, 128, 3),       # 查询帧0的轨迹（重定向后统一长度128）
    'traj_2d': (6400, 128, 2),     # 查询帧0的2D投影轨迹
    'valid_steps': (128,),        # 有效性掩码
    'frame_index': 0,              # 查询帧索引
    'image_path': 'images/images0_0.png',
    ...
}
```

## 💡 使用场景

### 使用 images0.npz
- 需要一次性加载整个视频的轨迹数据进行可视化
- 需要分析整个视频的相机运动
- 需要生成整个视频的3D动画
- 传统可视化工具需要的数据格式

### 使用 samples/*.npz
- 训练机器学习模型（每个查询帧作为独立样本）
- 测试/评估模型性能
- 使用 `visualize_single_image.py` 可视化单个查询帧
- 需要结构化、标准化的数据格式

## 🔍 文件大小估算

假设视频有 45 帧，图像尺寸 480x640，`grid_size=80`（6400个轨迹）：

- `coords`: 45 × 6400 × 3 × 4 bytes ≈ **3.5 MB**
- `extrinsics`: 45 × 4 × 4 × 4 bytes ≈ **2.9 KB**
- `intrinsics`: 45 × 3 × 3 × 4 bytes ≈ **1.6 KB**
- `depths`: 45 × 480 × 640 × 2 bytes ≈ **27.6 MB** (float16)
- `unc_metric`: 45 × 480 × 640 × 2 bytes ≈ **27.6 MB** (float16)
- `visibs`: 45 × 6400 × 1 × 1 byte ≈ **288 KB**

**总计**：约 **60-70 MB**（不含 `video` 字段）

如果包含 `video` 字段（`--save_video`）：
- `video`: 45 × 480 × 640 × 3 × 1 byte ≈ **41.5 MB**

**总计（含视频）**：约 **100-110 MB**

## 📝 代码位置

`images0.npz` 的生成代码位于 `infer.py` 第 937-954 行：

```python
# Always save traditional visualization NPZ in video directory root
video_dir = os.path.join(out_dir, video_name)
data_npz_load = {}
data_npz_load["coords"] = result["coords"].cpu().numpy()
data_npz_load["extrinsics"] = result["full_extrinsics"].cpu().numpy()
data_npz_load["intrinsics"] = result["full_intrinsics"].cpu().numpy()
data_npz_load["height"] = result["video_tensor"].shape[-2]
data_npz_load["width"] = result["video_tensor"].shape[-1]
data_npz_load["depths"] = result["depths"].cpu().numpy().astype(np.float16)
data_npz_load["unc_metric"] = result["depth_conf"].astype(np.float16)
data_npz_load["visibs"] = result["visibs"][..., None].cpu().numpy()
if args.save_video:
    data_npz_load["video"] = result["video_tensor"].cpu().numpy()

save_path = os.path.join(video_dir, video_name + ".npz")
np.savez(save_path, **data_npz_load)
logger.info(f"Traditional visualization NPZ saved to {save_path}")
```

