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
- `visualize_3d_keypoint_comparison.py`
  - 同时读取 baseline / variant 两个 episode output
  - 把 `baseline-only / overlap / variant-only` 三类轨迹叠在一个 3D 视图里
- `visualize_xperience_sample.py`
  - 为 `xperience-10m-sample` 导出 storyboard、单帧 dashboard 和 GIF
  - 复用 `scripts/data_analysis/xperience_sample_utils.py` 的多模态对齐 loader
- `verify_episode_trajectory_outputs.py`
  - 对单个 episode 导出 PLY、验证图和可选 GIF
- `export_pointcloud_ply.py`
- `export_ply_from_depth.py`
- `export_droid_inference_firstframe_plys.py`
- `capture_viser_to_gif.py`

## 文档

- [visualization_features.md](visualization_features.md)
- [docs/xperience_sample_tooling.md](../../docs/xperience_sample_tooling.md)

## 历史调查脚本

`compare_traj_filter_results.py` 已迁移到：

- `scripts/archived/investigations/2026-03/compare_traj_filter_results.py`

旧的实现分析文档已不再维护；如果需要历史背景，请看 `docs/history/`。
