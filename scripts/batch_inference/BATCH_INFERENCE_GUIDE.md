# 批量推理指南

本文档只覆盖当前维护的 external-only TraceForge 推理流程。

当前维护态的轨迹过滤逻辑说明见
[`docs/maintained_traj_filter_logic.md`](../../docs/maintained_traj_filter_logic.md)。

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
- `scripts/batch_inference/batch_infer.py`
  - 统一批处理入口
  - `--dataset_adapter file_layout` 转发到现有 sim/button 维护路径
  - `--dataset_adapter xperience` 直接读取原始 Xperience 数据集并写出 `v2 + adapter_ref`

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
  --query_frame_schedule_path <episode_output>/_shared/query_frame_schedule_v3_<hash>.json \
  --fps 1 \
  --max_num_frames 512
```

## Sim / Button 批处理

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --camera_names <camera_a,camera_b,...> \
  --gpu_id 0,1,2,3 \
  --min_free_gpu_mem_gb 40 \
  --gpu_recovery_poll_sec 60 \
  --collect_profile_stats \
  --hardware_telemetry_interval_sec 30 \
  --depth_filter_workers 8 \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

默认会就地写回到 `<episode>/trajectory/<camera_name>/...`。如果显式传入
`--out_dir <output_root>`，则会改为写到
`<output_root>/<episode_name>/<camera_name>/...`。

适用场景：

- button/sim episode 数据集
- dynamic-only 多 GPU 常驻 worker 调度
- 每个 episode 提供外部深度和 `trajectory_valid.h5`
- 必须显式传入 `--camera_names`；脚本只会推理这里列出的相机名，不再假设 `varied_camera_*`
- 其他维护态默认值仍覆盖 `depth_pose_method=external`、`external_geom_name=trajectory_valid.h5`、
  `fps=1`、`max_num_frames=512`、`future_len=32`、`grid_size=80`、
  `filter_level=standard`、`traj_filter_profile=external`
- `--collect_profile_stats` 会把每个 camera task 的 `profile_stats` / `save_profile_stats`
  额外落到 `_camera_task_profiles.jsonl`
- `--hardware_telemetry_interval_sec > 0` 会周期记录 GPU/CPU/IO 指标到 `_hardware_telemetry.jsonl`
- `--depth_filter_workers` 控制 `infer.py` 里 `_DepthFilterRuntime` 的线程数，便于 CPU 侧隔离实验
- `traj_filter_profile=auto` 仅保留为兼容别名，当前也解析为 `external`
- wrist-oriented profile 仍可显式切到 `wrist_pick_place` / `wrist_pick_place_no_heatmap` /
  `wrist` / `wrist_manipulator_top95`，但这些都不再属于维护态默认路径

## Press-One-Button Demo

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path <dataset_root> \
  --camera_names <camera_a,camera_b,...> \
  --keyframes_per_sec_min 2 \
  --keyframes_per_sec_max 3 \
  --skip_existing
```

补充说明：

- `batch_infer_press_one_button_demo.py` 会为每个 episode 生成一份共享 schedule：
  `<episode_output>/_shared/query_frame_schedule_v3_<hash>.json`
- 默认只有 `--camera_names` 里的相机会消费同一份 schedule；若需要跨批次固定更大的对齐集合，显式传 `--shared_schedule_camera_names`
- schedule 里存的是 raw source frame index，`infer.py` 运行时再映射到当前
  `--fps` / `--max_num_frames` 对应的 local query frame
- query frame 如果到加载后视频末尾的剩余帧数（含自身）`<= 8`，仍会在共享 schedule 采样前丢弃
- 对保留下来的尾段 sample，若实际 segment 长度短于 `future_len`，`v2` sample 会在时间维用 `inf` pad 到 `future_len`
- 每秒关键帧数量由 `--keyframes_per_sec_min/max` 控制；当两者相等时，每秒恰好采样固定数量，
  并保证同一秒内无重复
- 如果 source frame `0` 通过了 stride/cap 过滤，它会被强制写入 shared schedule
- 因此第一秒的 query frame 数可能比名义上的 `keyframes_per_sec_min/max` 多 `1`
- 真实的时间语义来自 `trajectory_valid.h5` 根属性 `fps`
- `--fps` 只是加载 stride，不是 episode 的真实帧率
- `--max_num_frames` 是 stride 之后的总帧数上限
- 维护态 batch CLI 不再暴露 `--frame_drop_rate`、`--horizon`、`--max_frames_per_video`
- `--keyframe_seed` 用于可复现的 deterministic schedule；默认 `0`
- 如果某些 `trajectory_valid.h5` 缺少根属性 `fps`，可以显式提供 `--fallback_episode_fps`
- `auto` 当前只作为兼容别名保留，并且会解析到 `external`
- wrist-oriented profile 只在显式指定时才会启用；它们更适合作为历史调查或兼容性复跑路径
- `external_manipulator`、`external_manipulator_v2`、`wrist_manipulator_top95`、`wrist_manipulator`
  需要显式指定
- `wrist_manipulator_top95` 是 wrist_manipulator 的临时去噪 profile：先走 wrist_manipulator，再按
  motion extent 只保留每个 sample 前 `95%` 的轨迹
- 推荐直接在 episode 下写 `trajectory/<camera_name>/...`
- `depth_pose_method=external` 时默认使用 `scene_storage_mode=source_ref`，直接复用源 RGB/depth/geometry

## Xperience 原生批处理

```bash
python scripts/batch_inference/batch_infer.py \
  --dataset_adapter xperience \
  --dataset_root <xperience_root> \
  --episode_glob '*/*' \
  --camera_name stereo_left \
  --checkpoint <checkpoint_path> \
  --out_dir <output_dir> \
  --device cuda:0 \
  --fps 1 \
  --max_num_frames 512 \
  --window_size 512 \
  --scene_storage_mode adapter_ref \
  --skip_existing
```

关键点：

- 当前维护范围只包含原始 Xperience 的 `stereo_left.mp4 + annotation.hdf5`
- 原始 RGB 会在加载时对齐到 `depth/depth` 分辨率，再送入当前 TraceForge 推理链路
- 输出仍然是标准 `v2` layout，但 `scene_meta.json` 里记录的是 `source_descriptor`
  而不是缓存后的 `scene.h5`
- `adapter_ref` 依赖原始数据仍然可访问，适合大规模数据集直接推理、避免重复缓存整份 RGB/depth
- 如果你做了 resize 实验，仍然必须切回 `--scene_storage_mode cache`
- `lang.txt` 会从 `caption.config["Main Task"]` 提取并随窗口输出

## Smoke 验证

### Sim / File Layout

2026-04-21 已在真实 sim 数据集
`/data2/yaoxuran/press_one_button_demo_v1/episode_00000` 上完成 smoke，使用的是统一入口
`scripts/batch_inference/batch_infer.py --dataset_adapter file_layout`，不是绕过新入口直接调用旧脚本。

结论：

- `file_layout -> batch_infer_press_one_button_demo.py` 的转发链路正常。
- 共享 query-frame schedule 正常生成，最终保存为标准 `v2 + source_ref`。
- 输出目录通过 `is_traceforge_output_complete(...) == True` 校验。
- 该次 smoke 使用单相机 `varied_camera_1`、`future_len=16`、`grid_size=20`，实际保存了 2 个 query frame sample。
- 当 smoke 想用较小 `grid_size` 时，建议同时把 `grid_border_trim_left/right/top/bottom` 设为 `0`；否则默认 trim 可能在小网格下直接触发参数校验失败。

### Xperience / Adapter Ref

2026-04-21 已在真实 Xperience 数据集
`/data1/dataset/xperience-10m-partial-1tb/07f3aeee-5d64-4fd2-8450-f8baf8c239fd/ep7` 上完成 smoke。

结论：

- 原始 `annotation.hdf5 + stereo_left.mp4` 可直接加载并进入当前 TraceForge 推理链路。
- 输出目录保存为标准 `v2 + adapter_ref`，`scene_meta.json` 中会记录 `source_descriptor`。
- 输出目录通过 `is_traceforge_output_complete(...) == True` 校验。
- 为了让 smoke 真正进入 tracker 前向，建议把 `future_len` 设为大于 `8`，否则 short-tail 规则可能把 query frame 全部跳过。
- 在这条真实 episode 上，如果目标只是验证 tracker 真跑，建议先用 `--query_visibility_gate_mode off`；默认 `all_future_v1` 可能把 query 全部过滤掉。
- 同样地，当 smoke 使用较小 `grid_size` 时，建议把 `grid_border_trim_left/right/top/bottom` 设为 `0`。

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
    │   └── query_frame_schedule_v3_<hash>.json
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

如果显式传入了 `--out_dir <output_root>`，则把上面的 `<episode_dir>/trajectory/`
整体替换为 `<output_root>/<episode_name>/`；目录内部的 `_shared/` 和各相机子目录
结构保持不变。

同时 `<output_root>/` 根目录会额外保存：

- `_batch_run_summary.json`
- `_camera_task_metrics.jsonl`
- `_camera_task_profiles.jsonl`（仅当 `--collect_profile_stats`）
- `_hardware_telemetry.jsonl`（仅当 `--hardware_telemetry_interval_sec > 0`）

## 检查与回归

```bash
python -m unittest utils.test_keyframe_schedule_utils
python scripts/batch_inference/test_inference_output_shapes.py
python scripts/batch_inference/verify_pointcloud.py
```

`verify_traj_valid_mask.py` 目前只适合作为单个 NPZ 的历史/兼容性排查脚本，需要显式传入
`<npz_path>`，而且不属于当前维护态 `v2` 输出的标准回归检查链路。

## 不在本文档范围内的内容

- 已退休的 bridge/vggt 入口
- 一次性调查命令
- 已归档的历史实验流程

这些内容如果仍需要追溯，请看 `docs/history/` 和
`scripts/archived/investigations/`。
