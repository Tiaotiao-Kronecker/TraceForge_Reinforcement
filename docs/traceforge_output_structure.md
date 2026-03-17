# TraceForge 输出结构

本文档描述当前维护中的 TraceForge 输出协议。

## 1. 当前默认布局：`v2`

默认输出目录：

```text
<episode_dir>/
├── scene.h5
├── scene_meta.json
├── scene_rgb.mp4
└── samples/
    ├── <video_name>_0.npz
    ├── <video_name>_15.npz
    └── ...
```

其中：

- `scene.h5`
  - `dense/depth`: 全视频深度
  - `camera/intrinsics`: 全视频内参
  - `camera/extrinsics_w2c`: 全视频 `w2c` 外参
- `scene_meta.json`
  - layout 元数据、源路径和 `future_len`
- `scene_rgb.mp4`
  - 全视频 RGB cache
- `samples/*.npz`
  - 单个 query frame 的轨迹 sample

## 2. `v2` sample NPZ 字段

当前核心字段：

| 键 | Shape | 说明 |
|----|-------|------|
| `traj_uvz` | `(N, T_seg, 3)` | query 相机坐标系下的 `(u, v, depth)` 轨迹 |
| `keypoints` | `(N, 2)` | query frame 上的 grid keypoints |
| `query_frame_index` | `(1,)` | query frame 索引 |
| `segment_frame_indices` | `(T_seg,)` | sample 时间轴对应的真实帧索引 |
| `traj_valid_mask` | `(N,)` | 最终轨迹过滤结果 |
| `traj_supervision_mask` | `(N, T_seg)` | 时域监督掩码 |
| `traj_supervision_prefix_len` | `(N,)` | 连续前缀长度 |
| `traj_supervision_count` | `(N,)` | 支撑帧计数 |
| `traj_mask_reason_bits` | `(N,)` | 过滤原因 bitmask |
| `traj_query_depth_rank` | `(N,)` | manipulator-aware 深度排序调试量 |
| `traj_motion_extent` | `(N,)` | manipulator-aware 世界位移调试量 |
| `traj_manipulator_candidate_mask` | `(N,)` | manipulator 候选轨迹 |
| `visibility` | `(N, T_seg)` 可选 | 仅在 `--save_visibility` 时保存 |

说明：

- `N = grid_size × grid_size`
- `T_seg` 是该 query frame 实际跟踪长度，不再强制 retarget 到固定步数
- 当前 sample 主时间轴与 `segment_frame_indices` 对齐

## 3. `legacy` 布局

兼容模式：

```text
<video_dir>/
├── images/
├── depth/
├── samples/
└── <video_name>.npz
```

保留原因：

- 兼容旧 checker / visualization / 数据集工具
- 兼容已有的 legacy 产物

`legacy` sample 的关键差异：

- 主轨迹字段名是 `traj`
- 会按 `future_len` padding，并配套 `valid_steps`
- 查询帧 RGB / depth 会作为独立文件保存

## 4. 坐标和时域约定

- `v2` sample 里的 `traj_uvz` 是 query-camera 坐标，不是 world 坐标
- world 轨迹需要结合 query frame 的 `intrinsics` 和 `w2c` 还原
- `traj_valid_mask` 是轨迹级保留结果
- `traj_supervision_mask` 是时域级可用帧结果

当前实现中：

- 旧 retarget helper 仍在代码里保留，但没有接入当前推理/存盘流程
- 因此当前 sample 时间轴不使用弧长重采样

## 5. 可视化建议

- 单 sample 3D 查看：
  - `scripts/visualization/visualize_single_image.py`
- episode 级导出：
  - `scripts/visualization/verify_episode_trajectory_outputs.py`
- 如果文档或旧脚本仍在讨论 retarget 后时间轴，请视为历史信息
