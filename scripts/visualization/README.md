# 可视化脚本

本目录包含当前维护中的可视化和结果导出脚本。

## 当前脚本

- `visualize_single_image.py`
  - 读取 sample NPZ 和对应 camera output root 下的 scene artifacts
  - 按 `traj_valid_mask` 显示过滤后的 3D 轨迹
  - `--image_path` / `--depth_path` 只是可选 override
- `visualize_3d_keypoint_animation.py`
  - 逐时间步播放 keypoint 轨迹
  - 支持 `v2` 和 `legacy`
  - `--episode_dir` 实际上传入的是 camera output root，例如 `<episode>/trajectory/varied_camera_3`
- `visualize_4d_reconstruction.py`
  - 读取任意标准 TraceForge artifact camera 目录并聚合其中所有 sample NPZ 的 world-space 轨迹点
  - 支持 `All Pixels Tracking / Point Cloud / Both` 三种交互模式
  - 可同时浏览动态 dense point cloud、当前 RGB 帧和相机 frustum
  - 适用于 `sim_file_layout` 与 `xperience_raw` 等不同数据集 adapter 的统一输出
- `visualize_3d_keypoint_comparison.py`
  - 同时读取 baseline / variant 两个 episode output
  - 把 `baseline-only / overlap / variant-only` 三类轨迹叠在一个 3D 视图里
- `verify_episode_trajectory_outputs.py`
  - 对单个 episode 导出 PLY、验证图和可选 GIF
- `export_pointcloud_ply.py`
- `export_ply_from_depth.py`
- `export_droid_inference_firstframe_plys.py`
- `capture_viser_to_gif.py`

## 已验证 Viewer 路径

- 2026-04-21 已用通用 4D viewer 数据路径验证以下两类标准 artifact：
  - `sim_file_layout` smoke 输出：`frames=50`、`query_frames=[0,24,34]`
  - `xperience_raw` smoke 输出：`frames=32`、`query_frames=[0,15]`
- 验证覆盖 `SceneReader`、sample 聚合和当前帧 dense pointcloud 构建，说明 viewer 对两类 adapter 的标准输出均可读取。

## 文档

- [visualization_features.md](visualization_features.md)

## 历史调查脚本

`compare_traj_filter_results.py` 已迁移到：

- `scripts/archived/investigations/2026-03/compare_traj_filter_results.py`

旧的实现分析文档已不再维护；如果需要历史背景，请看 `docs/history/`。
