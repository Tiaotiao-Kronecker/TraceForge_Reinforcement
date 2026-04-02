# pick_place `num_iters=5/4/3` sweep 结论

日期：2026-04-02

本文档记录当前 `pick_place` 固定子集上的 `num_iters=5/4/3` batch sweep 结果，目标是回答两个问题：

1. 这条线能不能给到两位数收益
2. 如果能，当前质量代价是否足够小，可以直接改维护态默认值

## 固定条件

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- batch episode list：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- compare manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.json`
- episodes：`00000`, `00001`
- `2` 个 episode，`6` 个 camera tasks，`51` 个 query frames
- `gpu_id=0`
- `workers_per_gpu=1`
- `depth_filter_workers=8`
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`

## 输出与报告

baseline：

- output root：
  `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1`
- telemetry：
  `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.md`

variants：

- `iters_4`
  - output root：
    `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n4`
  - telemetry：
    `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n4.md`
  - compare：
    `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_4_20260402/comparison_summary.md`
- `iters_3`
  - output root：
    `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n3`
  - telemetry：
    `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n3.md`
  - compare：
    `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_3_20260402/comparison_summary.md`

## 速度结果

| num_iters | Wall(s) | SingleGPU/query | Process/query | Save/query | Tracker/query | PrepDepth/query | Speedup vs `5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `810.84` | `15.90` | `15.06` | `0.20` | `13.43` | `0.89` | `1.000x` |
| `4` | `661.94` | `12.98` | `12.64` | `0.10` | `10.78` | `1.08` | `1.225x` |
| `3` | `541.28` | `10.61` | `9.99` | `0.10` | `8.12` | `1.12` | `1.498x` |

直接结论：

- 这条线已经给出了明确的两位数收益。
- 收益几乎全部来自 `tracker_model_forward_seconds/query` 下降，而不是 depth filter 变快。
- `prepare_depth_filter_seconds/query` 没有跟着下降，反而从 `0.89` 小幅升到 `1.08`、`1.12`。

## 质量结果

相对 baseline `iters_5`：

| Variant | Camera | Valid Jaccard | Valid Delta | World L2 Mean | Worst Query Jaccard |
| --- | --- | ---: | ---: | ---: | ---: |
| `iters_4` | `varied_camera_1` | `0.9972` | `-2.92` | `0.000171` | n/a |
| `iters_4` | `varied_camera_2` | `0.9968` | `+0.94` | `0.000172` | n/a |
| `iters_4` | `varied_camera_3` | `0.8710` | `-38.47` | `0.001051` | `0.4830` |
| `iters_3` | `varied_camera_1` | `0.9971` | `-2.74` | `0.000180` | n/a |
| `iters_3` | `varied_camera_2` | `0.9967` | `-0.87` | `0.000178` | n/a |
| `iters_3` | `varied_camera_3` | `0.8481` | `-73.38` | `0.001199` | `0.4436` |

最需要盯的坏例子都集中在 wrist：

- `iters_4`
  - `00001 / varied_camera_3 / query_frame=16`
  - `traj_valid_mask_jaccard = 0.4830`
- `iters_3`
  - `00001 / varied_camera_3 / query_frame=16`
  - `traj_valid_mask_jaccard = 0.4436`

## 结论

可以固定下来的判断是：

1. `num_iters` 这条线已经证明存在两位数收益空间。
2. 但当前收益主要由 external camera 和 tracker forward 压缩贡献，wrist `varied_camera_3` 的质量回退太明显。
3. 因此 `iters_4`、`iters_3` 都不能直接替代当前维护态默认 `num_iters=5`。

更准确的工程表述应该是：

- `num_iters=4/3` 是当前固定 workload 上的“速度上限候选”
- 不是“可直接上线的默认值候选”

## 下一步建议

如果目标是“可上线的收益”，比继续全局下调 `num_iters` 更合理的方向是：

1. 保持 `num_iters=5` 默认不变。
2. 优先探索 profile-aware / camera-aware 的差异化收敛，而不是继续全局 `5 -> 4 -> 3`。
3. 如果继续做纯实现优化，则回到 `prepare_depth_filter` per-frame kernel。
