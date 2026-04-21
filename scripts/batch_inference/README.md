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
  - 当前统一批处理入口
  - `--dataset_adapter file_layout` 时转发到 `batch_infer_press_one_button_demo.py`
  - `--dataset_adapter xperience` 时直接读取原始 Xperience episode（当前维护 `stereo_left`）
  - Xperience 原生路径默认输出 `v2 + adapter_ref`
- `batch_infer_press_one_button_demo.py`
  - button / sim / press-one-button episode 数据集批处理入口
  - 默认 `--fps=1`、`--max_num_frames=512`
  - 默认每秒共享采样 `2~3` 个关键帧，按 episode 的 `trajectory_valid.h5` root attr `fps` 计算
  - 如果 source frame `0` 通过 stride/cap/tail 过滤，会被强制写入 shared schedule
  - 若某个 query frame 到加载后视频末尾的剩余帧数（含自身）`<= 8`，会在采样前直接丢弃
  - 对保留下来的尾段 sample，若实际 segment 长度短于 `future_len`，`v2` sample 会在时间维用 `inf` pad 到 `future_len`
  - 生成共享 schedule 到 `<episode_output>/_shared/query_frame_schedule_v3_<hash>.json`
  - 多 GPU 维护路径为 dynamic-only
  - 关键帧数量只由 `--keyframes_per_sec_min/max` 控制；固定数量时把两者设成相同值
  - 不再暴露 `--frame_drop_rate` / `--horizon` / `--max_frames_per_video`
  - 维护态默认 `traj_filter_profile=external`
  - `auto` 仅保留为兼容别名，当前也解析为 `external`
  - wrist-oriented profile 仍可显式指定，但不再属于维护态默认路径
- `batch_droid_external.py`
  - DROID external-only 批处理入口
  - 固定输出 `v2 + source_ref`

## 辅助脚本

- `check_failed_inference.py`
- `test_inference_output_shapes.py`
- `verify_pointcloud.py`
- `verify_traj_valid_mask.py`（单个 NPZ 的历史/兼容性排查脚本，不是当前维护态 `v2` 的标准回归入口）

## 文档

- [BATCH_INFERENCE_GUIDE.md](BATCH_INFERENCE_GUIDE.md)

## 已验证 Smoke 路径

- `file_layout`：已于 2026-04-21 通过统一入口 `batch_infer.py --dataset_adapter file_layout`
  在 `/data2/yaoxuran/press_one_button_demo_v1/episode_00000` 上完成真实 smoke；
  转发到 `batch_infer_press_one_button_demo.py`、共享 query-frame schedule、`v2 + source_ref`
  保存和 artifact 完整性检查都已通过。
- `xperience`：已于 2026-04-21 在
  `/data1/dataset/xperience-10m-partial-1tb/07f3aeee-5d64-4fd2-8450-f8baf8c239fd/ep7`
  上完成真实 smoke；原始 `annotation.hdf5 + stereo_left.mp4` 读取、`v2 + adapter_ref`
  保存和 artifact 完整性检查都已通过。

如果需要查看旧实验或一次性调查，请不要从这里找，统一去
`scripts/archived/investigations/`。
