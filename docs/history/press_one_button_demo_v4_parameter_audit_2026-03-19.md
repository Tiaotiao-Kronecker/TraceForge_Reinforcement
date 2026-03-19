# press_one_button_demo_v4 参数梳理记录

日期：2026-03-19

## 背景

当前在处理数据集：

- `/data2/yaoxuran/press_one_button_demo_v4`

本轮整批处理采用的标准：

- 外部几何：`--depth_pose_method external`
- 关键帧策略：每秒固定随机选 5 帧
- 轨迹长度：`--future_len 32`
- 关键点网格：`--grid_size 80`
- 多卡：`GPU 1,2,3`

后台任务信息：

- tmux session：`tf_pob_v4_20260318_234524`
- 日志：`/data1/zoyo/projects/TraceForge_Reinforcement/logs/pob_v4_full_20260318_234524.log`

## 本轮实际使用的命令

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path /data2/yaoxuran/press_one_button_demo_v4 \
  --gpu_id 1,2,3 \
  --gpu_schedule_mode dynamic \
  --min_free_gpu_mem_gb 40 \
  --gpu_recovery_poll_sec 60 \
  --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --depth_pose_method external \
  --external_geom_name trajectory_valid.h5 \
  --external_extr_mode w2c \
  --device cuda \
  --num_iters 6 \
  --fps 1 \
  --max_num_frames 512 \
  --output_layout v2 \
  --scene_storage_mode source_ref \
  --horizon 16 \
  --frame_drop_rate 15 \
  --keyframes_per_sec_min 5 \
  --keyframes_per_sec_max 5 \
  --keyframe_seed 0 \
  --fallback_episode_fps 0 \
  --future_len 32 \
  --max_frames_per_video 50 \
  --grid_size 80 \
  --filter_level standard \
  --traj_filter_profile auto \
  --min_depth 0.01 \
  --max_depth 10.0 \
  --trajectory_dirname trajectory \
  --skip_existing
```

## 真正影响样例数量的参数

这次目标是“每秒固定随机选 5 个关键帧，为了尽量多拿样例”。在当前实现里，真正决定样例数量的主要是下面三类参数：

- `--keyframes_per_sec_min 5 --keyframes_per_sec_max 5`
  - 含义：每秒固定随机抽 5 个关键帧。
  - 当前 batch 入口会为每个 episode 生成共享 schedule，并让三个相机对齐使用。

- `--fps 1`
  - 含义：载入视频时不做步长降采样，原始帧全读。
  - 当前 batch 入口要求 `--fps >= 1`，因此不会走自动 stride 模式。

- `--max_num_frames 512`
  - 含义：载入后最多保留前 512 帧。
  - 对当前数据集基本不构成限制。

对 `/data2/yaoxuran/press_one_button_demo_v4` 的快速统计结果：

- 360 个 episode 都存在
- `varied_camera_1` 的帧数范围约为 `37 ~ 63`
- median 约为 `50`

因此，本轮运行中 `--max_num_frames 512` 不会截断 episode，也不会减少样例数。

## 当前确认的冗余或失效参数

### `--max_frames_per_video 50`

当前流程下基本无效。

原因：

- 它只在 `infer.py` 中 `--fps <= 0` 时才参与自动计算 stride。
- 但 `batch_infer_press_one_button_demo.py` 已经强制要求 `--fps >= 1`。

因此：

- 在当前标准流程里，这个参数可以从运行命令中移除。
- 如果未来不再支持 `fps<=0` 的旧逻辑，也可以从代码里删掉。

### `--horizon 16`

当前实现里是死参数。

原因：

- batch 脚本仍然会把它传入保存逻辑。
- 但 `infer.py` 的 `save_structured_data(...)` 一进入函数就把 `horizon` 丢弃，没有参与任何保存或裁剪逻辑。

因此：

- 这个参数可以从运行命令中移除。
- 也可以在后续瘦身时从 parser 和调用链中删除。

### `--frame_drop_rate 15`

在当前共享关键帧方案下不生效。

原因：

- `frame_drop_rate` 只在没有 `query_frame_schedule_path` 时，作为旧的均匀 query-frame 抽样步长使用。
- 当前 batch 入口会为每个 episode 预生成共享 schedule，并显式传给每个相机。

因此：

- 这次运行里它不会影响 query frame 数量。
- 可以从当前运行命令中移除。
- 如果未来所有正式入口都统一成共享 schedule，这个参数也可以删。

## 仍然有意义，但命令里可以省略的参数

### `--output_layout v2`

- 控制输出格式。
- `v2` 是当前维护的主格式。
- 这是默认值，所以命令里可以不写。

### `--scene_storage_mode source_ref`

- 控制场景数据的存储方式。
- `source_ref` 表示不缓存 `scene.h5` / `scene_rgb.mp4`，而是在 `scene_meta.json` 里记录源数据路径。
- 对 external-only 流程是合理默认。
- 这也是默认值，所以命令里可以不写。

### `--num_iters 6`

- 这是 3D tracker 的迭代 refinement 次数。
- 会真实影响推理时间和结果质量。
- 当前不是冗余参数，只是默认值已经是 `6`，所以命令里可以省略，但不建议直接从代码删除。

## 建议保留的核心参数面

如果后续继续以当前 external-only + 共享 schedule 流程为标准，命令参数面可以优先收敛到下面这组：

```bash
python scripts/batch_inference/batch_infer_press_one_button_demo.py \
  --base_path /data2/yaoxuran/press_one_button_demo_v4 \
  --gpu_id 1,2,3 \
  --gpu_schedule_mode dynamic \
  --min_free_gpu_mem_gb 40 \
  --gpu_recovery_poll_sec 60 \
  --camera_names varied_camera_1,varied_camera_2,varied_camera_3 \
  --checkpoint ./checkpoints/tapip3d_final.pth \
  --depth_pose_method external \
  --external_geom_name trajectory_valid.h5 \
  --external_extr_mode w2c \
  --fps 1 \
  --max_num_frames 512 \
  --keyframes_per_sec_min 5 \
  --keyframes_per_sec_max 5 \
  --future_len 32 \
  --grid_size 80 \
  --trajectory_dirname trajectory \
  --skip_existing
```

## 明天跑完后的瘦身建议

建议按下面顺序做：

1. 删除死参数和旧兼容参数
   - `horizon`
   - `max_frames_per_video`
   - `frame_drop_rate`

2. 精简命令和 parser 暴露面
   - 对 `output_layout=v2`
   - 对 `scene_storage_mode=source_ref`
   - 如果确定长期固定，也可以考虑减少 CLI 暴露，改成内部默认。

3. 保留仍然有实验价值的参数
   - `num_iters`
   - `future_len`
   - `grid_size`
   - `keyframes_per_sec_min/max`
   - `max_num_frames`

4. 同步清理文档与 README
   - 避免用户看到已经无效的参数说明。

## 当前结论

当前代码里确实有比较多历史兼容参数残留。对于现在这条正式流程，最明显的冗余项是：

- `max_frames_per_video`
- `horizon`
- `frame_drop_rate`

它们不会改变本轮 `press_one_button_demo_v4` 的样例数量，也不是当前“每秒固定 5 个关键帧”方案的有效控制项。
