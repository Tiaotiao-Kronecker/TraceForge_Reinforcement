# pick_place 单卡 depth_filter_workers sweep 结论

日期：2026-04-01

本文档只回答一个问题：

- 在当前 `pick_place` 维护态 workload 上，单张 H200 固定 `workers_per_gpu=1` 时，`depth_filter_workers=4/8/16` 是否还会影响总吞吐？

它是 `data_tmp/telemetry_reports/` 下自动生成报告的手写结论版，只保留对后续决策有价值的对比总表和一句话判断。

## 固定条件

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- episodes：`00000`, `00001`
- `2` 个 episode，`6` 个 camera tasks，`51` 个 query frames
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- `workers_per_gpu=1`
- `gpu_id=0`
- `hardware_telemetry_interval_sec=15`
- 主口径：`single_gpu_seconds_per_query`

## 对比总表

| depth_filter_workers | Wall(s) | Queries | SingleGPU/query | Process/query | PrepDepth/query | PrepInputs/query | Tracker/query | Save/query | GPU util mean | GPU mem mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `840.88` | `51` | `16.49` | `15.61` | `1.44` | `1.47` | `13.43` | `0.19` | `86.54%` | `17.71 GiB` |
| `8` | `810.89` | `51` | `15.90` | `15.06` | `0.88` | `0.92` | `13.43` | `0.19` | `87.35%` | `17.98 GiB` |
| `16` | `780.88` | `51` | `15.31` | `14.91` | `0.65` | `0.68` | `13.51` | `0.19` | `88.62%` | `18.86 GiB` |

## 一句话结论

- 有，CPU 侧深度过滤线程数现在仍然会影响总吞吐。
- `depth_filter_workers=4` 明显偏低，会把单卡吞吐拖慢到 `16.49 s/query/H200`。
- `depth_filter_workers=8` 是稳定基线，但 `16` 在当前 workload 上还能继续提升到 `15.31 s/query/H200`。
- 对当前单卡 H200 路径，后续实验 baseline 应切到 `workers_per_gpu=1`、`depth_filter_workers=16`。

## 解释

从 telemetry 看，变化主要发生在 CPU 预处理这一侧：

- `prepare_depth_filter_seconds/query` 从 `1.44` 降到 `0.88`、`0.65`
- `prepare_inputs_seconds/query` 从 `1.47` 降到 `0.92`、`0.68`
- `tracker_model_forward_seconds/query` 基本不动，稳定在 `13.43~13.51`
- `save_total_seconds/query` 也基本不动，稳定在 `0.19`

这说明当前瓶颈不是 tracker forward 又变快了，而是 CPU 侧深度过滤准备确实还能拖慢或放快整体 pipeline。

另一个容易误读的点是：

- `prepare_depth_filter_worker_total_seconds/query` 从 `5.23` 升到 `5.90`、`7.04`

它是所有 depth-filter worker 的累计 CPU 时间，不是 wall-facing 阶段时长；因此它升高并不和 `prepare_depth_filter_seconds/query` 下降矛盾，反而说明更高线程数是在用更多总 CPU 工时换更短的关键路径时间。

## 默认值决策

基于这轮 sweep，后续建议：

1. 单卡和多卡后续实验默认都先用 `workers_per_gpu=1`。
2. 纯性能 sweep 的 CPU 基线改为 `depth_filter_workers=16`。
3. 后续再做 `num_iters`、`support_grid_ratio`、`query_prefilter_mode` 等实验时，应基于 `workers_per_gpu=1`、`depth_filter_workers=16` 继续。

## 对应原始报告

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df4.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df8.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df16.md`
