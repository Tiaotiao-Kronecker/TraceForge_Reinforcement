# TraceForge 推理输出数据结构

本文档描述 TraceForge 批量推理（`batch_infer.py` + `infer.py`）生成的数据结构，**不修改 infer 源码**，仅作为已有 10000+ 轨迹的可视化与后处理参考。

## 目录结构

每个 case（如 `00000`）的输出结构：

```
output_bridge_depth_grid80/
└── 00000/
    └── images0/                    # 视频名（主视角）
        ├── images/                 # 查询帧 RGB 图像
        │   ├── images0_0.png       # 查询帧 0
        │   ├── images0_5.png       # 查询帧 5 (frame_drop_rate=5)
        │   ├── images0_10.png
        │   └── ...
        ├── depth/                  # 深度图
        │   ├── images0_0.png       # 归一化 16-bit PNG（可视化用）
        │   ├── images0_0_raw.npz   # 原始深度值（米）
        │   └── ...
        ├── samples/                # 每个查询帧的轨迹数据
        │   ├── images0_0.npz       # 查询帧 0 的 keypoint 轨迹
        │   ├── images0_5.npz
        │   └── ...
        └── images0.npz             # 主 NPZ（全视频 coords、相机参数）
```

## 查询帧与轨迹段

- **查询帧**：`frame_drop_rate=5` 时，为 `[0, 5, 10, 15, 20, ...]`
- **每个查询帧**：从该帧开始，向前跟踪最多 `future_len`（默认 128）帧
- **物理含义**：不同查询帧对应同一段物理运动的不同采样，彼此重叠

| 查询帧 | 轨迹段 | 说明 |
|--------|--------|------|
| 0  | 帧 0 → 0+T   | 从第 0 帧开始的轨迹 |
| 5  | 帧 5 → 5+T   | 从第 5 帧开始的轨迹 |
| 10 | 帧 10 → 10+T | ... |

## Sample NPZ 结构（`samples/images0_<frame>.npz`）

每个 sample NPZ 对应**一个查询帧**的 keypoint 轨迹：

| 键 | 类型 | Shape | 说明 |
|----|------|-------|------|
| `traj` | float32 | `(N, T, 3)` | 3D 轨迹，世界坐标系。N=grid_size²，T=128（retarget 后） |
| `traj_2d` | float32 | `(N, T_orig, 2)` | 2D 投影（用于 2D 可视化） |
| `keypoints` | float32 | `(N, 2)` | 查询帧上的 2D 网格点 (x, y) |
| `frame_index` | int64 | `(1,)` | 查询帧索引 |
| `valid_steps` | bool | `(T,)` | 有效时间步掩码，无效步用 `-inf` 填充 |
| `image_path` | str | `(1,)` | 对应图像路径 |

- **N**：`grid_size × grid_size`（如 grid_size=80 → 6400）
- **T**：retarget 后固定为 128，由 `retarget_trajectories()` 做弧长重采样
- **坐标系**：`traj` 为世界坐标系下的 3D 点 `(x, y, z)`（米）

**时域与 retarget**：若原始轨迹短于 `future_len`（如 25 帧），TraceForge 会对轨迹做 `retarget_trajectories()` 插值到 128 步（弧长均匀）。因此 sample NPZ 的 `traj` 时间轴**不再与视频帧一一对应**，不能按帧数直接截断使用。

## 主 NPZ 结构（`images0.npz`）

| 键 | 类型 | Shape | 说明 |
|----|------|-------|------|
| `coords` | float | `(T_seg, N, 3)` | **首段** 3D 轨迹（未 retarget，与视频帧一一对应） |
| `extrinsics` | float | `(T_full, 4, 4)` | 全视频外参 w2c |
| `intrinsics` | float | `(T_full, 3, 3)` | 全视频内参 |
| `depths` | float16 | `(T_seg, H, W)` | **首段** 深度图 |
| `visibs` | float | `(T_seg, N, 1)` | 首段可见性 |
| `height`, `width` | int | scalar | 图像尺寸 |

**时域说明**：`coords`、`depths`、`visibs` 来自首段（query_frame=0）的**原始推理输出**，未经过 `retarget_trajectories`。`T_seg = min(视频帧数, future_len)`，与视频帧一一对应，可直接用于帧对齐的密集点云可视化。

**深度单位**：`depths` 与 `depth_raw.npz` 中的深度均为**米 (m)**。若提供已知深度（`--depth_path`），会与模型预测做中位数尺度对齐：$s = \text{median}(D_{\text{known}}) / \text{median}(D_{\text{pred}})$，$D_{\text{aligned}} = s \cdot D_{\text{pred}}$。详见 `docs/depth_scale_alignment_math.md`。

## 主 NPZ vs Sample NPZ 时域对比

| 来源 | 时域 | 与视频帧关系 | 用途 |
|------|------|--------------|------|
| **主 NPZ** `coords`/`depths` | 首段原始帧数 `T_seg` | 一一对应 | 密集点云、帧对齐可视化 |
| **Sample NPZ** `traj` | 固定 128 步（retarget 后） | 弧长均匀，不对齐 | Keypoint 动画（按步播放） |

**重要**：密集点云可视化应使用主 NPZ，因其 `coords`/`depths` 为推理原始输出，未做 retarget，与视频帧对齐。若用 sample NPZ 并按帧截断，会错误地混入 retarget 插值后的时间轴。

## 3D 可视化要点

1. **按查询帧分别可视化**：每个查询帧的 sample NPZ 对应一段从该帧开始的 3D 轨迹
2. **时间轴**：sample NPZ 的 `traj` 的 T 维为 retarget 后的时间步，用 `valid_steps` 过滤无效步；主 NPZ 的 `coords` 与视频帧一一对应
3. **Keypoint 动画**：在时间 t，显示 `traj[:, t, :]` 的 N 个 3D 点
4. **密集点云**：使用主 NPZ 的 `coords`、`depths`、`extrinsics` 做深度反投影，保证帧对齐
5. **与 SpaTrackerV2 的差异**：SpaTrackerV2 可做逐像素 3D；TraceForge 为 grid 采样，仅对 keypoint 做 3D 动画
6. **归一化**：可仿 SpaTrackerV2，用首帧外参将轨迹变换到首帧相机坐标系
