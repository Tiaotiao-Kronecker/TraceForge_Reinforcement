# 可视化脚本

本目录包含当前维护中的可视化和结果导出脚本。

## 当前脚本

- `visualize_single_image.py`
  - 读取 sample NPZ 和 episode scene cache
  - 按 `traj_valid_mask` 显示过滤后的 3D 轨迹
  - `--image_path` / `--depth_path` 只是可选 override
- `visualize_3d_keypoint_animation.py`
  - 逐时间步播放 keypoint 轨迹
  - 支持 `v2` 和 `legacy`
- `verify_episode_trajectory_outputs.py`
  - 对单个 episode 导出 PLY、验证图和可选 GIF
- `export_pointcloud_ply.py`
- `export_ply_from_depth.py`
- `export_droid_inference_firstframe_plys.py`
- `capture_viser_to_gif.py`

## 文档

- [visualization_features.md](visualization_features.md)

## 历史调查脚本

`compare_traj_filter_results.py` 已迁移到：

- `scripts/archived/investigations/2026-03/compare_traj_filter_results.py`

旧的实现分析文档已不再维护；如果需要历史背景，请看 `docs/history/`。
