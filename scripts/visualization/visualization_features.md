# 可视化功能说明

## `visualize_single_image.py`

当前推荐的单 sample 3D 查看器。

### 用法

```bash
python scripts/visualization/visualize_single_image.py \
  --npz_path <episode_dir>/samples/<sample>.npz \
  --port 8080
```

可选参数：

- `--image_path`：覆盖 episode 内默认 RGB
- `--depth_path`：覆盖 episode 内默认 depth；支持 `.npz` 或带 `_raw.npz`
  配套的 `.png`

### 当前行为

- 从 sample 里读取轨迹和 `traj_valid_mask`
- 从 episode scene cache 读取 query frame 的 RGB、depth、intrinsics、extrinsics
- 把 `traj_uvz` 还原为 world coordinates 后渲染
- 只显示过滤后保留下来的轨迹

### GUI 控件

- `Point size`
- `Track width`
- `Track length`
- `Number of trajectories`
- `Number of keypoints`
- `Show point cloud`
- `Show tracks`
- `Show keypoints`
- `Show camera frustum`
- `Show world axes`

## `verify_episode_trajectory_outputs.py`

适合批量导出验证材料：

- 静态验证图
- PLY
- 可选 GIF
- `summary.json`

示例：

```bash
python scripts/visualization/verify_episode_trajectory_outputs.py \
  --episode_dir <episode_output_dir> \
  --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
  --query_frames 0,15,30 \
  --output_dir <verification_dir>
```
