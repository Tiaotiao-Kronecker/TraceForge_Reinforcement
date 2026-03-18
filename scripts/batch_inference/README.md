# 批量推理脚本

本目录包含当前维护中的推理入口和少量校验脚本。

## 核心入口

- `infer.py`
  - 通用单视频 / 批量推理入口
  - 默认输出 `v2` layout
  - 默认 `scene_storage_mode=source_ref`，仅适用于 `depth_pose_method=external`
  - 使用默认 `vggt4` 时需要显式传 `--scene_storage_mode cache`
  - 支持 `external / external_manipulator / external_manipulator_v2 / wrist / wrist_manipulator_top95 / wrist_manipulator`
    轨迹过滤 profile
- `batch_infer.py`
  - 旧通用批处理入口，适合 legacy 数据集目录批量跑
- `batch_infer_press_one_button_demo.py`
  - `press_one_button_demo_v1` 的专用批处理入口
  - 支持 `traj_filter_profile auto`
  - `auto` 默认把 wrist-like 相机映射到 `wrist_manipulator_top95`

## 辅助脚本

- `analyze_batch_failures.py`
- `check_failed_inference.py`
- `test_inference_output_shapes.py`
- `verify_model_cache.py`
- `verify_pointcloud.py`
- `verify_traj_valid_mask.py`

## 文档

- [BATCH_INFERENCE_GUIDE.md](BATCH_INFERENCE_GUIDE.md)

如果需要查看旧实验或一次性调查，请不要从这里找，统一去
`scripts/archived/investigations/`。
