# process / num_iters sweep 跟进记录

日期：2026-03-21

## 背景

在 `save_structured_data()` 的 shared hot path 清理完成后，端到端主瓶颈已经重新回到
`process_single_video()`。这一轮的目标是回答两个问题：

- `num_iters` 从 `6 -> 5 -> 4` 时，`process` 是否接近线性下降
- 如果准备把维护态默认值切到 `5`，质量风险是否仍然可控

本页继续沿用当前阶段的约束：

- 只看 `support_grid_ratio=0.0`
- 保持 `warmup_runs=1`
- 同时检查定量结果和 3D 动画

## 工作负载

- episode：`episode_00105`
- cameras：
  - `varied_camera_1`，`traj_filter_profile=external`
  - `varied_camera_3`，`traj_filter_profile=wrist_manipulator_top95`
- benchmark summary：
  - `/tmp/num_iters_sweep_ep00105_sg0_warm1_20260321_1336/benchmark_summary.md`
- benchmark json：
  - `/tmp/num_iters_sweep_ep00105_sg0_warm1_20260321_1336/benchmark_results.json`

## runtime 结果

| Camera | Variant | num_iters | Process (s) | Save (s) | Total (s) | Tracker Forward (s) | Depth Filter (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `iters_6` | `6` | `57.200` | `0.392` | `57.592` | `39.449` | `11.105` |
| `varied_camera_1` | `iters_5` | `5` | `51.470` | `0.532` | `52.002` | `33.044` | `11.451` |
| `varied_camera_1` | `iters_4` | `4` | `43.514` | `0.392` | `43.906` | `26.384` | `11.281` |
| `varied_camera_3` | `iters_6` | `6` | `57.193` | `2.537` | `59.729` | `40.652` | `10.256` |
| `varied_camera_3` | `iters_5` | `5` | `48.516` | `2.104` | `50.620` | `33.988` | `9.510` |
| `varied_camera_3` | `iters_4` | `4` | `42.585` | `2.206` | `44.791` | `27.298` | `10.302` |

直接结论：

- `tracker_model_forward_seconds` 会随着 `num_iters` 下降而明显减少，幅度接近线性。
- `prepare_depth_filter_seconds` 基本不随 `num_iters` 变化，仍然稳定卡在 `~10s` 级别。
- 因此 `num_iters` sweep 解决的是 tracker forward，而不是 `prepare_depth_filter` 这块固定成本。

## 定量质量结果

相对 baseline `iters_6`：

| Camera | Variant | Process Speedup | Total Speedup | Valid Jaccard | Valid Delta | World L2 Mean | Step Delta P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `iters_5` | `1.111x` | `1.107x` | `0.9988` | `+0.6` | `4.30e-05` | `1.14e-04` |
| `varied_camera_1` | `iters_4` | `1.315x` | `1.312x` | `0.9984` | `+1.8` | `4.90e-05` | `1.11e-04` |
| `varied_camera_3` | `iters_5` | `1.179x` | `1.180x` | `0.9808` | `+0.8` | `2.46e-05` | `5.73e-05` |
| `varied_camera_3` | `iters_4` | `1.343x` | `1.334x` | `0.9631` | `+0.4` | `3.48e-05` | `7.14e-05` |

按 camera 分开看：

- `varied_camera_1` 对 `num_iters` 很稳，`4` 也仍然紧贴 baseline。
- `varied_camera_3` 对 `num_iters` 更敏感，`4` 已经开始出现更明显的偏差。

最需要盯的 sample：

- `varied_camera_3 / iters_5`
  - 最差 jaccard 出现在 frame `35`，约 `0.9167`
  - 最大 2D 偏差出现在 frame `40`，约 `1.03 px`
- `varied_camera_3 / iters_4`
  - 最差 jaccard 出现在 frame `35`，约 `0.8333`
  - 最大 2D 偏差出现在 frame `40`，约 `2.86 px`

## 结论与默认值决策

截至这轮 sweep，更稳妥的维护态选择是：

- 把默认 `num_iters` 从 `6` 切到 `5`
- 不直接把维护态默认值切到 `4`

原因：

1. `5` 已经能稳定拿到 `11%~18%` 的 `process` / `total` 降幅。
2. `5` 下 `varied_camera_1` 基本贴着 baseline，`varied_camera_3` 虽有差异，但仍明显好于 `4`。
3. `4` 虽然还能继续降时间，但当前代表 wrist case 已经出现更明显的质量风险，不适合直接作为维护态默认值。

当前代码已同步把维护态默认 `num_iters` 更新为 `5`：

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
- `utils/inference_utils.py`

## `episode_00105` 单 case 与 `v5` 聚合结果的口径说明

这页里的 `iters_5` 单 episode 结果，后续经常会被拿去和
`/tmp/v5_median3_volatility_20260321/benchmark_summary.md` 的聚合均值直比。这里需要把口径区别单独写清楚，避免把
“workload 变重”误读成“单个 query 明显变慢”。

先看工作负载本身：

- 本页 `v1` 单 case 只包含 `episode_00105`
  - loaded frames：`54`
  - query frames：`5`
  - schedule：`/tmp/num_iters_single_ep00105_sg0_warm1_n5_20260321_1538/_shared/query_frame_schedule_v1_3d534091496d.json`
- `v5 median3` 聚合均值混合了 3 个 episode
  - `episode_00006_blue`：`59` loaded frames，`4` query frames
  - `episode_00001_green`：`59` loaded frames，`5` query frames
  - `episode_00030_pink`：`61` loaded frames，`7` query frames
  - schedules：
    - `/tmp/v5_median3_volatility_20260321/episodes/episode_00006_blue/_shared/query_frame_schedule_v1_a26712912bc2.json`
    - `/tmp/v5_median3_volatility_20260321/episodes/episode_00001_green/_shared/query_frame_schedule_v1_a26712912bc2.json`
    - `/tmp/v5_median3_volatility_20260321/episodes/episode_00030_pink/_shared/query_frame_schedule_v1_038af70cf2b7.json`

因此 `v5` 汇总里的：

- `varied_camera_1 total = 59.135s`
- `varied_camera_3 total = 63.462s`

主要反映的是“聚合里混进了更长、更多 query-frame、也更重的整段 episode/camera workload”，而不是 summary 直接在表达
“每个 query-frame 都比 `episode_00105` 慢很多”。其中 `episode_00030_pink` 本身就是长 total case：

- `varied_camera_1 total = 71.937s`
- `varied_camera_3 total = 78.178s`

这两个值本身就足以把三-episode 聚合均值往上抬。

### 计时口径

`scripts/data_analysis/benchmark_depth_volatility_optimization.py` 当前的记录方式是：

1. 先整段计一次 `process_single_video()`
2. 再整段计一次 `save_structured_data()`
3. 最后额外把 `query_frame_count`、`frame_count` 记进 run record

也就是说 summary 里的 `Process / Save / Total` 都是“一个 episode 的一个 camera 整次跑完”的 wall-clock，而不是“处理一个
query frame 的平均时间”。`query_frame_count` 在当前脚本里只是额外记录，没有参与 summary 表格归一化。

### 粗略归一化对照

如果要粗略比较不同 workload，可以额外看：

- `Total / loaded frame`
- `Total / query frame`

按当前 artifact 的精确值整理如下：

| Scope | Episode | Camera | Total (s) | Loaded Frames | Query Frames | Total / Loaded (s) | Total / Query (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `v1 single` | `episode_00105` | `varied_camera_1` | `50.029` | `54` | `5` | `0.926` | `10.006` |
| `v1 single` | `episode_00105` | `varied_camera_3` | `52.956` | `54` | `5` | `0.981` | `10.591` |
| `v5 aggregate` | `3 episodes mean` | `varied_camera_1` | `59.135` | `59.667` | `5.333` | `0.991` | `11.088` |
| `v5 aggregate` | `3 episodes mean` | `varied_camera_3` | `63.462` | `59.667` | `5.333` | `1.064` | `11.899` |
| `v5 max-total case` | `episode_00030_pink` | `varied_camera_1` | `71.937` | `61` | `7` | `1.179` | `10.277` |
| `v5 max-total case` | `episode_00030_pink` | `varied_camera_3` | `78.178` | `61` | `7` | `1.282` | `11.168` |

这里要注意两点：

1. `v5 aggregate` 行是跨 episode 的“均值再相除”，只能作为粗略归一化参考，不是严格 apples-to-apples 的单 case 对照。
2. 真正更稳的判断应该优先看单 case 对单 case。按这个口径，`episode_00105` 与 `episode_00030_pink` 的 `Total / Query`
   并没有出现数量级变化；把 summary 里的聚合 total 拉长的主因，仍然是 workload 本身更长、更重。

可以固定下来的额外结论是：

- 不管看 `v1` 的 `episode_00105`，还是看 `v5` 的三-episode workload，只要按 `Total / query frame` 粗归一化，当前单卡流水的处理速度都稳定落在
  `10s` 级别，更准确地说约是 `10~12s / query frame`。
- 如果只想记一个工程判断，可以把它简化成：当前维护态单卡吞吐大致是“每个 query frame 约 `10s` 左右”。

## prepare_depth_filter 固定成本的当前判断

同一轮 `episode_00105` sweep 还给出了一个更重要的 side conclusion：

- `prepare_depth_filter_cache_miss_frames = 45`
- `prepare_depth_filter_cache_hit_frames = 49`
- `prepare_depth_filter_ray_cache_miss_frames = 1`
- `prepare_depth_filter_ray_cache_hit_frames = 44`

这说明：

1. 对当前 `5` 个 query segments 来说，重叠帧的 filtered depth 已经在 `_DepthFilterRuntime` 里被有效复用。
2. ray 构造也已经基本只做一次，不是当前 `prepare_depth_filter_seconds` 的主问题。
3. 当前剩下的 `~10s` 固定成本，主要还是每个 unique frame 内部 `_filter_one_depth(...)` 的真实计算本身。

结合代码路径：

- runtime 入口：`scripts/batch_inference/infer.py::_DepthFilterRuntime.get_filtered_depth_segment`
- per-frame 过滤：`datasets/data_ops.py::_filter_one_depth_profiled`

可以把下一轮重点收敛到 `_filter_one_depth` 内部，而不是继续怀疑 segment overlap 或 ray cache 没生效。

## 下一步

为了继续压这块固定成本，当前代码已经额外导出更细的 process profile keys：

- `prepare_depth_filter_worker_total_seconds`
- `prepare_depth_filter_ray_scale_seconds`
- `prepare_depth_filter_points_to_normals_seconds`
- `prepare_depth_filter_edge_mask_seconds`
- `prepare_depth_filter_distance_transform_seconds`
- `prepare_depth_filter_fill_seconds`
- `prepare_depth_filter_future_wait_seconds`
- `prepare_depth_filter_stack_seconds`
- `prepare_depth_filter_unique_frame_count`

因此更合适的下一轮动作是：

1. 固定 `num_iters=5`，重跑当前 e2e / process benchmark。
2. 先看 `points_to_normals`、`edge_mask`、`distance_transform` 三项谁占最大头。
3. 如果 worker 总和远大于 wall time，说明线程并行度已经在工作，下一步要优先改算法本身。
4. 如果 `future_wait_seconds` 仍然占比明显，再考虑 `max_workers` / 调度层面的 sweep。

## 3D 动画

本轮代表性 3D 动画命令已经写入：

- `/tmp/num_iters_sweep_ep00105_sg0_warm1_20260321_1336/benchmark_summary.md`

优先建议先看 `varied_camera_3` 的 `iters_6 / iters_5 / iters_4` 三组对比。
