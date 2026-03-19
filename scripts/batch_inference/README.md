# 批量推理脚本

本目录包含当前维护中的推理入口和少量校验脚本。

## 核心入口

- `infer.py`
  - 通用单视频 / 批量推理入口
  - 默认输出 `v2` layout
  - 默认 `scene_storage_mode=source_ref`
  - 当前维护模式为 `depth_pose_method=external`
  - 默认 `--fps=1`、`--max_num_frames=512`
  - 未提供 `--query_frame_schedule_path` 时，使用 `--frame_drop_rate` 做 fallback query 采样
  - 支持 `external / external_manipulator / external_manipulator_v2 / wrist / wrist_manipulator_top95 / wrist_manipulator`
    轨迹过滤 profile
- `batch_infer.py`
  - 已下线的旧通用批处理入口
- `batch_infer_press_one_button_demo.py`
  - button / sim / press-one-button episode 数据集批处理入口
  - 默认 `--fps=1`、`--max_num_frames=512`
  - 默认每秒共享采样 `2~3` 个关键帧，按 episode 的 `trajectory_valid.h5` root attr `fps` 计算
  - 生成共享 schedule 到 `<episode_output>/_shared/query_frame_schedule_v1_<hash>.json`
  - 多 GPU 维护路径为 dynamic-only
  - 关键帧数量只由 `--keyframes_per_sec_min/max` 控制；固定数量时把两者设成相同值
  - 不再暴露 `--frame_drop_rate` / `--horizon` / `--max_frames_per_video`
  - 支持 `traj_filter_profile auto`
  - `auto` 默认把 wrist-like 相机映射到 `wrist_manipulator_top95`
- `batch_droid_external.py`
  - DROID external-only 批处理入口

## 辅助脚本

- `check_failed_inference.py`
- `test_inference_output_shapes.py`
- `verify_pointcloud.py`
- `verify_traj_valid_mask.py`

## 文档

- [BATCH_INFERENCE_GUIDE.md](BATCH_INFERENCE_GUIDE.md)

如果需要查看旧实验或一次性调查，请不要从这里找，统一去
`scripts/archived/investigations/`。
