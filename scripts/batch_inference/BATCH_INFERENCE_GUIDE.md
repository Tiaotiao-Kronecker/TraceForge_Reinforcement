# 批量推理指南

本文档只覆盖当前维护的 external-only TraceForge 推理流程。

## 前提

- 在仓库根目录运行
- 已安装 `traceforge` 环境
- 已准备 TAPIP3D checkpoint
- 当前维护模式要求外部深度和外部几何同时可用

## 当前维护入口

- `scripts/batch_inference/infer.py`
  - 通用单视频 / 批量推理入口
  - 默认 `--fps=1`
  - 默认 `--max_num_frames=512`
  - 未提供共享 schedule 时，用 `--frame_drop_rate` 做 fallback query-frame 采样
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
  - button / sim / press-one-button episode 数据集批处理入口
  - 默认 `--fps=1`
  - 默认 `--max_num_frames=512`
  - 默认共享每秒采样 `2~3` 个关键帧

## 通用批量推理

```bash
python scripts/batch_inference/infer.py \
  --video_path <input_dir> \
  --depth_path <depth_dir> \
  --external_geom_npz <trajectory_valid.h5_or_geom.npz> \
  --depth_pose_method external \
  --out_dir <output_dir> \
  --scene_storage_mode source_ref \
  --fps 1 \
  --max_num_frames 512 \
  --batch_process \
  --skip_existing \
  --frame_drop_rate 5 \
  --scan_depth 2 \
  --grid_size 20
```

关键点：

- 默认输出 layout 为 `v2`
- 默认 `scene_storage_mode` 为 `source_ref`
- 当前维护模式为 external-only，必须提供 `--depth_path` 与 `--external_geom_npz`
- `--output_layout legacy` 仅用于兼容旧工具
- `future_len` 控制每个 query frame 的跟踪窗口
- `grid_size` 控制 query keypoint 密度
- 没有共享 schedule 时，`infer.py` 才会使用 `--frame_drop_rate`

如果你已经提前生成了共享关键帧 manifest，也可以直接给 `infer.py`：

```bash
python scripts/batch_inference/infer.py \
  --video_path <rgb_dir> \
  --depth_path <depth_dir> \
  --external_geom_npz <trajectory_valid.h5> \
  --depth_pose_method external \
  --camera_name varied_camera_1 \
  --query_frame_schedule_path <episode_output>/_shared/query_frame_schedule_v1_<hash>.json \
  --fps 1 \
  --max_num_frames 512
```

## Sim / Button 批处理

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --gpu_id 0,1,2,3 \
  --min_free_gpu_mem_gb 40 \
  --gpu_recovery_poll_sec 60 \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

适用场景：

- button/sim episode 数据集
- dynamic-only 多 GPU 常驻 worker 调度
- 每个 episode 提供外部深度和 `trajectory_valid.h5`
- 维护态默认值已经覆盖 `camera_names=varied_camera_1,2,3`、`depth_pose_method=external`、
  `external_geom_name=trajectory_valid.h5`、`fps=1`、`max_num_frames=512`、
  `future_len=32`、`grid_size=80`、`filter_level=standard`、`traj_filter_profile=auto`

## Press-One-Button Demo

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

补充说明：

- `batch_infer_press_one_button_demo.py` 会为每个 episode 生成一份共享 schedule：
  `<episode_output>/_shared/query_frame_schedule_v1_<hash>.json`
- 三个相机都会消费同一份 schedule，因此 query frame 的 raw 帧序号天然对齐
- schedule 里存的是 raw source frame index，`infer.py` 运行时再映射到当前
  `--fps` / `--max_num_frames` 对应的 local query frame
- 每秒关键帧数量由 `--keyframes_per_sec_min/max` 控制；当两者相等时，每秒恰好采样固定数量，
  并保证同一秒内无重复
- 真实的时间语义来自 `trajectory_valid.h5` 根属性 `fps`
- `--fps` 只是加载 stride，不是 episode 的真实帧率
- `--max_num_frames` 是 stride 之后的总帧数上限
- 维护态 batch CLI 不再暴露 `--frame_drop_rate`、`--horizon`、`--max_frames_per_video`
- `--keyframe_seed` 用于可复现的 deterministic schedule；默认 `0`
- 如果某些 `trajectory_valid.h5` 缺少根属性 `fps`，可以显式提供 `--fallback_episode_fps`
- `auto` 会把 wrist-like camera 映射到 `wrist_manipulator_top95`
- `external_manipulator`、`external_manipulator_v2`、`wrist_manipulator_top95`、`wrist_manipulator`
  需要显式指定
- `wrist_manipulator_top95` 是 wrist_manipulator 的临时去噪 profile：先走 wrist_manipulator，再按
  motion extent 只保留每个 sample 前 `95%` 的轨迹
- 推荐直接在 episode 下写 `trajectory/<camera_name>/...`
- `depth_pose_method=external` 时默认使用 `scene_storage_mode=source_ref`，直接复用源 RGB/depth/geometry

## 关键参数语义

- `trajectory_valid.h5.attrs["fps"]`
  - episode 的真实帧率
  - 只用于“每秒 x~y 个”关键帧采样
- `--fps`
  - 加载 stride
  - 例如 `--fps 2` 表示 raw 帧 `0,2,4,...`
- `--max_num_frames`
  - stride 之后最多保留多少帧
  - 默认 `512`
- `--keyframes_per_sec_min/max`
  - 每秒采样关键帧数量范围
  - 默认 `2~3`
- `--frame_drop_rate`
  - 只给没有共享 schedule 的 `infer.py` fallback 使用

## 输出结构

button/sim episode 默认就地写回：

```text
<episode_dir>/
└── trajectory/
    ├── _shared/
    │   └── query_frame_schedule_v1_<hash>.json
    ├── varied_camera_1/
    │   ├── scene_meta.json
    │   └── samples/
    ├── varied_camera_2/
    └── varied_camera_3/
```

`scene_meta.json` 会记录：

- `source_frame_indices`
- `query_frame_sampling_mode`
- `query_frame_schedule_path`
- `query_frame_indices_local`
- `query_frame_source_indices`
- `keyframes_per_sec_min`
- `keyframes_per_sec_max`

## 检查与回归

```bash
python -m unittest utils.test_keyframe_schedule_utils
python scripts/batch_inference/test_inference_output_shapes.py
python scripts/batch_inference/verify_pointcloud.py
python scripts/batch_inference/verify_traj_valid_mask.py
```

## 不在本文档范围内的内容

- 已退休的 bridge/vggt 入口
- 一次性调查命令
- 已归档的历史实验流程

这些内容如果仍需要追溯，请看 `docs/history/` 和
`scripts/archived/investigations/`。
