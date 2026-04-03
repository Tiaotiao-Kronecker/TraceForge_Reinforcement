# pick_place `num_iters` sweep 与 camera-aware 跟进结论

日期：2026-04-02

本文档记录当前 `pick_place` 固定子集上的 `num_iters=5/4/3` batch sweep 结果，以及同日追加的 camera-aware `external=4, wrist=5` 跟进验证，目标是回答三个问题：

1. 这条线能不能给到两位数收益
2. 如果能，当前质量代价是否足够小，可以直接改维护态默认值
3. 简单的 camera-aware 差异化收敛能不能保住 external 收益，同时避开 wrist 回退

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

worst-case `RGB / 2D / 3D` triptych exports：

- `iters_4`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.md`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.json`
- `iters_3`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.md`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.json`

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

## 最差样本三列轨迹可视化

已经把三组对比里 wrist `varied_camera_3` 的 `traj_valid_mask_jaccard` 最差 top-10 全部导出成之前同样的三列轨迹对比 GIF（`RGB / 2D / 3D`）：

- `iters_4`
  - summary：
    `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.md`
  - top-1：
    `00001 / varied_camera_3 / query_frame=16 / 0.482993`
- `iters_3`
  - summary：
    `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.md`
  - top-1：
    `00001 / varied_camera_3 / query_frame=16 / 0.443645`
- `ext4_wrist5`
  - summary：
    `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3_top10_jaccard/summary.md`
  - top-1：
    `00001 / varied_camera_3 / query_frame=16 / 0.099023`

## 结论

可以固定下来的判断是：

1. `num_iters` 这条线已经证明存在两位数收益空间。
2. 但当前收益主要由 external camera 和 tracker forward 压缩贡献，wrist `varied_camera_3` 的质量回退太明显。
3. 因此 `iters_4`、`iters_3` 都不能直接替代当前维护态默认 `num_iters=5`。

更准确的工程表述应该是：

- `num_iters=4/3` 是当前固定 workload 上的“速度上限候选”
- 不是“可直接上线的默认值候选”

## 追加验证：camera-aware `external=4`, `wrist=5`

上一版结论里建议优先验证 profile-aware / camera-aware 的差异化收敛。该验证已经在同一天追加完成。

输出与报告：

- 首次尝试（无效，不纳入结论）：
  - output root：
    `data_tmp/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5`
  - 说明：
    `gpu_id=0` 上已有其他进程占用约 `93 GiB` 显存，在 `00000 / varied_camera_3` 触发 OOM；该 root 只保留为 contention 记录
- clean run：
  - output root：
    `data_tmp/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5_gpu3`
  - telemetry：
    `data_tmp/telemetry_reports/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5_gpu3.md`
  - compare：
    `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3/comparison_summary.md`
  - triptych summary：
    `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3_top10_jaccard/summary.md`

固定条件（clean run）：

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- batch episode list：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- compare manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.json`
- `gpu_id=3`
- `workers_per_gpu=1`
- `depth_filter_workers=8`
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `support_grid_ratio=0.8`
- global `num_iters=5`
- per-camera override：
  `varied_camera_1:4,varied_camera_2:4,varied_camera_3:5`

速度结果：

| Variant | Wall(s) | SingleGPU/query | Process/query | Save/query | Tracker/query | PrepDepth/query | Speedup vs `5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iters_5` | `810.84` | `15.90` | `15.06` | `0.20` | `13.43` | `0.89` | `1.000x` |
| `ext4_wrist5` | `811.23` | `15.91` | `15.40` | `0.14` | `12.65` | `1.56` | `1.000x` |

按相机拆开看：

- `varied_camera_1`：`14.47 -> 13.95 s/query`
- `varied_camera_2`：`14.04 -> 13.51 s/query`
- `varied_camera_3`：`17.25 -> 19.14 s/query`

质量结果（相对 baseline `iters_5`）：

| Camera | Valid Jaccard | Valid Delta | World L2 Mean | Worst Query Jaccard |
| --- | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `0.9972` | `-2.9236` | `0.000171` | n/a |
| `varied_camera_2` | `0.9968` | `+0.9375` | `0.000172` | n/a |
| `varied_camera_3` | `0.8574` | `+39.1736` | `0.001008` | `0.0990` |

最坏样本：

- `00001 / varied_camera_3 / query_frame=16`
- `traj_valid_mask_jaccard = 0.0990`

直接结论：

- external 两路确实更快了，但收益主要被更慢的 `prepare_depth_filter` 和更差的 wrist 路径抵消。
- 更严重的是，wrist 即使保持 `num_iters=5`，质量也没有回到 baseline，反而比全局 `iters_4` 更差。
- 所以最直接的 camera-aware 配置 `external=4, wrist=5` 不是可上线配置，应视为已验证失败。

## 更新后的建议

1. `num_iters=5` 默认值继续保持，不改。
2. `depth_filter_workers=8` 的当前默认值也不改。
3. `iters_4`、`iters_3` 以及 `external=4, wrist=5` 都只保留为速度边界/失败案例参考，不作为 rollout 候选。
4. 如果还要沿 `num_iters` 这条线继续挖，前提是先定位为什么 external 改成 `4` 后，wrist 在保持 `5` 时仍会明显退化。
5. 在这个根因没解释清楚前，下一步优先级应回到 `prepare_depth_filter` per-frame kernel，而不是继续扫更简单的 camera-aware 组合。
