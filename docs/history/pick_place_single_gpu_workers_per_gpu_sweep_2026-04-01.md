# pick_place 单卡 workers_per_gpu sweep 结论

日期：2026-04-01

本文档只回答一个问题：

- 在当前 `pick_place` 维护态 workload 上，单张 H200 把 `workers_per_gpu` 从 `1` 提高到 `2/3/4`，是否真的能提高吞吐？

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
- `depth_filter_workers=8`
- `gpu_id=0`
- `hardware_telemetry_interval_sec=15`
- 主口径：`single_gpu_seconds_per_query`

## 对比总表

| workers_per_gpu | Wall(s) | Queries | SingleGPU/query | Slot/query | Process/query | Save/query | GPU util mean | GPU mem mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `810.84` | `51` | `15.90` | `15.26` | `15.06` | `0.20` | `84.93%` | `17.56 GiB` |
| `2` | `811.04` | `51` | `15.90` | `30.82` | `30.62` | `0.20` | `94.35%` | `27.38 GiB` |
| `3` | `811.13` | `51` | `15.90` | `44.95` | `44.75` | `0.20` | `95.70%` | `35.85 GiB` |
| `4` | `811.27` | `51` | `15.91` | `52.30` | `52.10` | `0.20` | `96.26%` | `41.60 GiB` |

## 一句话结论

- `workers_per_gpu=1/2/3/4` 的单卡吞吐几乎完全相同，稳定在 `15.90~15.91 s/query/H200`。
- `workers_per_gpu` 越大，`slot_seconds/query` 上升越明显，说明单 task 只是被拖慢了。
- GPU util 和显存占用持续上升，但 wall clock 没有下降，表现为更强的同卡算力竞争，而不是更高的有效 throughput。
- 在当前 workload 上，`workers_per_gpu=1` 是后续实验最稳妥的 baseline。

## 解释

从 telemetry 看，随着 `workers_per_gpu` 提高：

- `tracker_model_forward_seconds/query` 从 `13.43` 升到 `28.68`、`42.48`、`49.58`
- `prepare_inputs_seconds/query` 只在 `0.92~1.01` 之间小幅波动
- `prepare_depth_filter_seconds/query` 只在 `0.89~0.96` 之间小幅波动
- `save_total_seconds/query` 始终约 `0.20`

这说明新增 worker 主要是在争抢同一张卡上的 tracker forward 算力，而不是成功隐藏了 CPU 预处理、磁盘 IO 或 save 开销。

## 默认值决策

基于这轮 sweep，后续建议：

1. 单卡和多卡后续实验默认都先用 `workers_per_gpu=1`。
2. 如果继续做“纯性能、不改语义”的 sweep，下一优先级应转向 `depth_filter_workers=4/8/16`。
3. 如果继续看并发策略，优先做 8 卡 `workers_per_gpu=1` 基线，而不是继续增加单卡 resident workers。

## 对应原始报告

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w2.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4.md`
