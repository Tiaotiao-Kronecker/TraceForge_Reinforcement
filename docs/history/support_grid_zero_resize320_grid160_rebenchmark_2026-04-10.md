# `support_grid_ratio=0.0` 在 `grid_size=160` + `resize=320x180` 下的速度/质量对比

日期：2026-04-10

## 背景

这轮实验的目的，是在新的高密度配置上回答一个更直接的问题：

- 当 `grid_size=160`
- 且 RGB / depth / 几何等处理分辨率统一降到 `320x180`
- 且其余 maintained external-only 约束不变时

如果把外层 `support_grid_ratio` 从当前基线 `0.8` 直接降到 `0.0`，能换来多少真实吞吐提升，以及轨迹质量会损失多少。

这里的关注点不是 `grid=80` 那轮 Pareto sweep，而是：

1. 在当前 `resize320 + grid160` 试验路径上，`support=0` 的真实批量速度
2. 这份速度收益是否伴随明显的轨迹质量退化
3. 在 throughput-first 的场景里，这个 trade-off 是否值得

## 固定实验条件

- datasets：
  - `wipe_the_table_gs`
  - `cup_on_coaster_gs`
  - `arrange_flowers_gs`
- episodes：
  - 每个数据集固定 `5` 个 episode
  - `00000,00001,00002,00003,00004`
- cameras：
  - `varied_camera_1,varied_camera_2`
- tracker：
  - `num_iters=3`
  - `grid_size=160`
- processing resolution：
  - `320x180`
- filtering：
  - `traj_filter_profile=external`
- support ratios：
  - baseline：`0.8`
  - variant：`0.0`
- worker layout：
  - 实际使用物理卡 `0,1,2,3,4,5,6`
  - 共 `7` 张卡

说明：

- 基线 `support=0.8` 那轮虽然 summary 里写了 `8` 卡，但实际成功任务只落在 `0..6` 七张卡上
- 本轮 `support=0.0` 直接固定为七卡运行
- 因此最终速度对比统一按“实际成功 GPU 集合”重算，而不是直接信任 `_batch_run_summary.json` 里的 `gpu_ids`
- 下面所有 `8 GPU` 数字都不是直接实测值，而是基于这轮 `7 GPU` 实测结果做的线性外推

## 运行入口与产物

- baseline telemetry：
  - `data_tmp/batch_runs/20260410_resize320_grid160_5ep`
- baseline artifacts：
  - `/DATA/disk3/tmp/traceforge_batch_artifacts_20260410_resize320_grid160_5ep`
- variant telemetry：
  - `data_tmp/batch_runs/20260410_resize320_grid160_sg00_5ep`
- variant artifacts：
  - `/DATA/disk3/tmp/traceforge_batch_artifacts_20260410_resize320_grid160_sg00_5ep`
- launcher：
  - `data_tmp/traceforge_tmux_launch_resize320_grid160_sg00_5ep_20260410.sh`
- 汇总结果：
  - `data_tmp/compare_results/20260410_resize320_grid160_sg00_vs_sg08/overall_summary.md`
  - `data_tmp/compare_results/20260410_resize320_grid160_sg00_vs_sg08/overall_summary.json`

逐数据集对比结果：

- `wipe_the_table_gs`
  - `data_tmp/compare_results/20260410_resize320_grid160_sg00_vs_sg08/wipe_the_table_gs/comparison_summary.md`
- `cup_on_coaster_gs`
  - `data_tmp/compare_results/20260410_resize320_grid160_sg00_vs_sg08/cup_on_coaster_gs/comparison_summary.md`
- `arrange_flowers_gs`
  - `data_tmp/compare_results/20260410_resize320_grid160_sg00_vs_sg08/arrange_flowers_gs/comparison_summary.md`

## 速度结果

整体统计口径：

- 总 query 数：`420`
- 实际活跃 GPU：`7`
- 双相机 raw episode 总时长：`56.2 s`
- 对比时统一按“总 query / 实际成功 GPU / 总墙钟时间”计算

### 总表

| Run | Active GPUs | Total Queries | Wall Clock (s) | Sec / Query / GPU | Queries / GPU / h | Tracker Forward / Query (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sg08` | `7` | `420` | `1744.120` | `29.069` | `123.845` | `62.806` |
| `sg00` | `7` | `420` | `1257.346` | `20.956` | `171.790` | `38.659` |

直接结论：

- `support=0.0` 相比 `0.8`，吞吐提升 `1.387x`
- 单卡单 query 耗时从 `29.07s` 降到 `20.96s`
- tracker forward / query 从 `62.81s` 降到 `38.66s`
- 说明这轮提速的主要来源，确实是 tracker forward 阶段减负，而不是保存或过滤逻辑

### 按用户历史口径换算

这次继续沿用“raw data 指双相机 episode 时长”的口径：

- `keyframes_per_sec: ~7.47`

baseline `sg08`：

- `1 keyframe -> 1 GPU x ~29.1s`
- `1s raw dual-camera data -> 1 GPU x ~217.2s`
- `1h raw dual-camera data -> 1 GPU x ~217.2h`
- `1h raw dual-camera data -> 7 GPU x ~31.0h`
- `1h raw dual-camera data -> 8 GPU x ~27.2h`

variant `sg00`：

- `1 keyframe -> 1 GPU x ~21.0s`
- `1s raw dual-camera data -> 1 GPU x ~156.6s`
- `1h raw dual-camera data -> 1 GPU x ~156.6h`
- `1h raw dual-camera data -> 7 GPU x ~22.4h`
- `1h raw dual-camera data -> 8 GPU x ~19.6h`

因此，从用户习惯的 batch 估算口径看：

- `support=0.0` 把 `1h raw dual-camera -> 8 GPU` 的处理时间，从约 `27.2h` 压到约 `19.6h`
- 相比 `support=0.8`，约再少 `7.6h / raw-hour`

## 质量结果

这里的 baseline 是 `support_grid_ratio=0.8`，variant 是 `support_grid_ratio=0.0`。

### overall

| Scope | Samples | Mask Jaccard | Valid Delta | World L2 Mean | Step Delta P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | `420` | `0.9969` | `-30.140` | `0.000142` | `0.000276` |

这个结果更适合这样理解：

- 平均每个 query 少了约 `30.1` 条有效轨迹
- 但当前 dense 点总数是 `25600`
- 所以平均有效覆盖损失约为：
  - `30.14 / 25600 ≈ 0.118%`

也就是说：

- 退化是存在的
- 但量级仍然偏小
- 更像是“小而稳定的 coverage 损失”，而不是轨迹大面积崩坏

### 分数据集

| Dataset | Samples | Mask Jaccard | Valid Delta | Valid Delta / 25600 | World L2 Mean | Step Delta P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wipe_the_table_gs` | `130` | `0.9967` | `-39.838` | `-0.156%` | `0.000131` | `0.000252` |
| `cup_on_coaster_gs` | `132` | `0.9966` | `-40.455` | `-0.158%` | `0.000152` | `0.000298` |
| `arrange_flowers_gs` | `158` | `0.9973` | `-13.544` | `-0.053%` | `0.000141` | `0.000276` |

直接现象：

- `wipe` 和 `cup` 的 valid coverage 损失更明显
- `arrange` 明显更稳
- 三个数据集上的几何误差都仍然很小，`world_l2_mean` 仍在 `1e-4` 量级

### 分相机

| Camera | Samples | Mask Jaccard | Valid Delta | Valid Delta / 25600 | World L2 Mean | Step Delta P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `210` | `0.9968` | `-31.257` | `-0.122%` | `0.000149` | `0.000309` |
| `varied_camera_2` | `210` | `0.9970` | `-29.024` | `-0.113%` | `0.000134` | `0.000242` |

### 最差样本

- 最低 Jaccard：
  - `cup_on_coaster_gs / 00002 / varied_camera_1 / q41`
  - `jaccard=0.9931`
  - `valid_delta=-85`
- 最大 Valid Delta 损失：
  - `wipe_the_table_gs / 00003 / varied_camera_2 / q93`
  - `valid_delta=-93`
  - `-93 / 25600 ≈ -0.36%`
- 最大 `world_l2_mean`：
  - `arrange_flowers_gs / 00004 / varied_camera_1 / q0`
  - `world_l2=0.000184`
- 最大 `step_delta_p95`：
  - `cup_on_coaster_gs / 00002 / varied_camera_1 / q120`
  - `step_delta_p95=0.000381`

这里最重要的不是最坏 case 的绝对值，而是它们仍然没有进入“明显几何漂移失控”的量级。

## 如何理解这轮 `Valid Delta`

这里的 `Valid Delta` 仍然是：

- `variant_valid_track_count - baseline_valid_track_count`

所以：

- `Valid Delta < 0`
  - 表示 `support=0` 最终保留下来的有效轨迹条数比 `support=0.8` 少

它反映的是：

- 最终 `traj_valid_mask` 覆盖率的变化

它不是：

- 轨迹点的几何位置误差
- 也不是 step 级的位移误差

因此，这轮结果的核心判断应是：

- `support=0` 的主要代价是 valid coverage 稳定下降
- 而不是 3D 几何位置大幅漂移

## 结论

这轮 `grid=160 + resize320` 的结论，比之前 `grid=80` 那轮更偏向 throughput-first：

- `support=0.0` 确实有价值
  - 吞吐提升约 `38.7%`
  - 折算到 `8 GPU` 时，`1h raw dual-camera` 处理时长从 `27.2h` 降到 `19.6h`
- 质量退化也确实存在
  - 但更集中体现在 valid coverage
  - overall 平均只损失约 `0.118%` 的 dense 有效轨迹
- 几何误差指标仍然很小
  - `world_l2_mean` 和 `step_delta_p95` 依旧在 `1e-4` 量级

因此，更准确的结论是：

- 如果当前目标是 batch throughput，`support=0.0` 是一个合理且收益明确的选择
- 如果更看重 `wipe/cup` 上的 valid coverage 稳定性，它不是零代价优化，但损失量级仍偏小
- 由于这次测试本身就是非默认的 `grid=160 + resize320` 试验路径，所以这轮结论更适合指导该试验配置下的吞吐决策，而不是直接外推为所有维护态默认值结论
