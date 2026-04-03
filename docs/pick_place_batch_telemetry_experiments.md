# pick_place Batch Telemetry 实验结论

日期：2026-04-01（2026-04-02 更新）

本文档作为这轮 `pick_place` 批量性能实验的持续更新结论页。

后续相关实验结论统一追加在这里，包括：

- 单卡 `workers_per_gpu` sweep
- `depth_filter_workers` sweep
- 后续 `num_iters` / `support_grid_ratio` / `query_prefilter_mode` 的语义变化实验

如果只想快速知道“当前最可靠的实验结论是什么”，优先看本文档，不要先翻历史计划文档。

## 优化路线总表

这轮提速探索分成两类：

- 不改语义：尽量不改变输出语义，只优化实现、并发和 CPU/GPU 资源利用。
- 改语义：允许改变轨迹分布、过滤逻辑或迭代强度，但必须按速度和质量一起评估。

### 已落地的同语义优化

| 项目 | 当前状态 | 当前结论 |
| --- | --- | --- |
| save 路径重写与 depth volatility hot path 收敛 | 已完成 | save 已不再是主瓶颈，shared save path 已压到很低 |
| shared save 向量化与 profile 细化 | 已完成 | external save 已接近亚秒级，wrist 剩余主要是 profile-specific filter |
| `prepare_depth_filter` cache / ray cache / 分阶段 profiling | 已完成 | 重点已从“有没有 cache”转向“cache 后每帧自身计算还重不重” |
| `pick_place` 路径复用与 fast path | 已完成 | 已有 `world_tracks` 复用、reference geometry 复用、candidate 子集 fast path |
| `manipulator_motion` motion metrics 向量化 | 已完成 | helper local microbenchmark 约 `9.2x`，且 `filter_result_manipulator_motion_seconds/query` 已从 `0.104` 降到 `0.0046`；但端到端总吞吐未改善，已降为次要方向 |

### 本轮不改语义性能实验

| 项目 | 当前状态 | 当前结论 / 默认建议 |
| --- | --- | --- |
| 单卡 `workers_per_gpu = 1/2/3/4` | 已完成 | `>1` 不提吞吐，只放大同卡竞争；默认固定 `workers_per_gpu=1` |
| 单卡 `depth_filter_workers = 4/8/16` | 已完成 | 单卡隔离时 `16` 最快，可作为单卡 override baseline |
| 两卡 `depth_filter_workers = 8/16` 小样本确认 | 已完成 | 本次 `df16` 未复现单卡收益；当前代码默认先维持 `8` |
| 8 卡 `workers_per_gpu = 1/2/4` | 未执行 | 如果继续看并发策略，应先做 8 卡 `workers_per_gpu=1` 基线，`>1` 低优先级 |

### 未完成同语义优化清单

| 项目 | 优先级 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| `prepare_depth_filter` 的 per-frame kernel 优化 | `P0` | 未开始 | 当前最该继续的同语义主线，重点看 `points_to_normals`、`edge_mask`、`distance_transform` |
| tracker 侧 batching / feature cache 复用 | `P1` | 未开始 | 收益上限最高，但风险和改动面也最大 |
| `manipulator_motion` 的 candidate 子集化 | `P2` | 暂不继续 | 向量化后 `filter_result_manipulator_motion_seconds/query` 已压到 `0.0046`，继续深挖对总吞吐预期收益很小 |

### 改语义性能-质量权衡实验

| 项目 | 当前状态 | 当前结论 / 备注 |
| --- | --- | --- |
| `num_iters: 6 -> 5` | 历史上已完成 | 已经进入维护态默认值，当前默认是 `num_iters=5` |
| `num_iters: 5 -> 4 -> 3` | 已完成 | `4` 在当前固定子集上带来约 `1.225x` wall-clock 加速、`3` 带来约 `1.498x`，但 wrist `varied_camera_3` 质量明显回退；默认值仍维持 `5` |
| camera-aware `num_iters: external=4, wrist=5` | 已完成 | clean run `810.84s -> 811.23s` 基本无收益，且 wrist `varied_camera_3` 进一步恶化到 Jaccard `0.8574`、worst query `0.0990`；当前拒绝 |
| `support_grid_ratio: 0.8 -> 0.6 -> 0.4` | 未执行 | 属于明确改语义项，当前默认仍是 `0.8` |
| `query_prefilter_mode: off -> profile_aware_static_v1` | 未执行 | 属于明确改语义项，当前默认仍是 `off` |
| `future_len: 32 -> 24` | 当前冻结 | 用户已要求本轮先不要动 |
| `grid_size: 80 -> 40` | 当前冻结 | 用户已要求本轮先不要动 |

### 当前推荐 baseline

如果目标是“单卡 isolate benchmark 的最低时间消耗”，当前推荐固定：

- `workers_per_gpu=1`
- `depth_filter_workers=16`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- `future_len=32`
- `grid_size=80`

如果目标是“当前代码默认值 / 多卡路径的稳妥配置”，当前建议固定：

- `workers_per_gpu=1`
- `depth_filter_workers=8`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- `future_len=32`
- `grid_size=80`

## 当前固定条件

单卡 isolate sweep 固定：

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- profile：`traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- `depth_filter_workers=16`

当前冻结，不在本轮前半段实验里改动：

- `future_len: 32 -> 24`
- `grid_size: 80 -> 40`

## 固定子集

单卡 `workers_per_gpu` sweep 当前统一使用：

- manifest: `scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- episodes: `00000`, `00001`

该子集对应：

- `2` 个 episode
- `6` 个 camera tasks
- `51` 个 query frames

## 速度口径

默认以“单卡 H200 归一化”作为主口径：

- `single_gpu_seconds_per_query = wall_clock * physical_gpu_count / total_queries`

单卡实验里这等价于：

- `wall_clock / total_queries`

同时保留两个辅助口径：

- `slot_seconds_per_query`
- `process_slot_seconds_per_query`

## 已完成实验

### 1. smoke telemetry 验证

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_smoke_telemetry_20260401_w1`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_smoke_telemetry_20260401_w1.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_smoke_telemetry_20260401_w1.json`

结论：

- telemetry 路径已经可用
- `profile_stats`、`save_profile_stats`、hardware telemetry 都能正常落盘
- 单卡 smoke 结果约 `16.29 s/query/H200`

### 2. 单卡 `workers_per_gpu=1`

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.json`

核心结果：

- `wall_clock_seconds = 810.84`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 15.26`
- `process/query = 15.06`
- `save/query = 0.20`
- GPU util mean `84.93%`
- GPU memory used mean `17.56 GiB`

结论：

- 单卡单 worker 已经能把 H200 打到较高利用率
- 主要瓶颈仍然是 `tracker_model_forward`
- `save` 仍然很小，不是首要优化方向

### 3. 单卡 `workers_per_gpu=2`

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w2`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w2.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w2.json`

核心结果：

- `wall_clock_seconds = 811.04`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 30.82`
- `process/query = 30.62`
- `save/query = 0.20`
- GPU util mean `94.35%`
- GPU memory used mean `27.38 GiB`

结论：

- 相比 `workers_per_gpu=1`，总吞吐几乎没有提升
- 但单 task `slot_seconds/query` 近似翻倍
- GPU 更忙了，但没有换来更高吞吐
- 当前单卡上，`workers_per_gpu=2` 更像是在放大 GPU 竞争，而不是提升总体 throughput

### 4. 单卡 `workers_per_gpu=3`

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3.json`

核心结果：

- `wall_clock_seconds = 811.13`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 44.95`
- `process/query = 44.75`
- `save/query = 0.20`
- GPU util mean `95.70%`
- GPU memory used mean `35.85 GiB`

结论：

- 相比 `workers_per_gpu=2`，总吞吐仍然没有提升
- 但单 task `slot_seconds/query` 再次显著上升
- GPU 和显存都更忙了，但 wall clock 仍然贴着 `~811s`
- `workers_per_gpu=3` 进一步证明当前瓶颈是同卡 tracker forward 竞争，不是 CPU/IO 没吃满

### 5. 单卡 `workers_per_gpu=4`

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4.json`

核心结果：

- `wall_clock_seconds = 811.27`
- `51 queries`
- `15.91 s/query/H200`
- `slot_seconds/query = 52.30`
- `process/query = 52.10`
- `save/query = 0.20`
- GPU util mean `96.26%`
- GPU memory used mean `41.60 GiB`

结论：

- `workers_per_gpu=4` 依然没有带来任何可见吞吐收益
- 继续增加 resident workers 只是在抬高单 task 时长和显存占用
- 当前单卡 H200 上，`workers_per_gpu=1` 仍然是最稳妥的后续 baseline
- 后续如果继续做 sweep，应优先转向 `depth_filter_workers` 或语义参数，而不是继续堆单卡 worker

### 6. 单卡 `depth_filter_workers=4/8/16`

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df4`
- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df8`
- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df16`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df4.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df4.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df8.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df8.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df16.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_depth_filter_sweep_20260401_df16.json`

核心结果：

| depth_filter_workers | Wall(s) | Queries | SingleGPU/query | Process/query | PrepDepth/query | PrepInputs/query | Tracker/query | Save/query | GPU util mean | GPU memory used mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `840.88` | `51` | `16.49` | `15.61` | `1.44` | `1.47` | `13.43` | `0.19` | `86.54%` | `17.71 GiB` |
| `8` | `810.89` | `51` | `15.90` | `15.06` | `0.88` | `0.92` | `13.43` | `0.19` | `87.35%` | `17.98 GiB` |
| `16` | `780.88` | `51` | `15.31` | `14.91` | `0.65` | `0.68` | `13.51` | `0.19` | `88.62%` | `18.86 GiB` |

结论：

- `depth_filter_workers=4` 明显拖慢总体吞吐，说明 CPU 侧深度过滤线程数过低会成为真实瓶颈。
- `depth_filter_workers=8` 复测后仍然稳定在 `15.90 s/query/H200`，与之前基线一致。
- `depth_filter_workers=16` 进一步把单卡吞吐拉到 `15.31 s/query/H200`，比 `8` 快约 `3.7%`。
- 变化主要来自 `prepare_depth_filter_seconds/query` 和 `prepare_inputs_seconds/query` 下降，不是 `tracker_model_forward_seconds/query` 或 `save` 变快。
- 后续单卡 baseline 应更新为 `workers_per_gpu=1`、`depth_filter_workers=16`。

### 7. `manipulator_motion` motion metrics 向量化（代码级）

代码位置：

- `utils/traj_filter_utils.py`
- `utils/test_traj_filter_utils.py`

验证：

- `/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python -m unittest utils.test_traj_filter_utils.MotionMetricHelperTests`
- `/DATA/disk2/wangchen/projects/TraceForge_Reinforcement/.venv/bin/python -m unittest utils.test_traj_filter_utils`

local microbenchmark（synthetic `4096 tracks x 32 frames x 3 masks`）：

- 旧版 reference loop：`0.3445 s/call`
- 向量化实现：`0.0373 s/call`
- helper 层加速约 `9.24x`

结论：

- `_compute_motion_metrics_for_valid_masks` 已移除按 track 的 Python 双层循环。
- 语义保持不变：仍然以“首个 valid 点”为 motion extent anchor，`step median` 与 `NaN` 行为均已用直接回归测试锁住。
- 这只是 helper 层结果，不等价于端到端 `pick_place` 吞吐已经同比例提升。
- 对应的单卡 telemetry 复测结果见下一节。

### 8. `manipulator_motion` 向量化后的单卡 telemetry 复测

输出根：

- `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_manipulator_motion_retest_20260401_df16`

报告：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_manipulator_motion_retest_20260401_df16.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_manipulator_motion_retest_20260401_df16.json`

与上一版 `depth_filter_workers=16` baseline 对比：

| 指标 | 旧 baseline | 本次复测 | 变化 |
| --- | ---: | ---: | ---: |
| `single_gpu_seconds/query` | `15.31` | `15.91` | `+0.59` |
| `process/query` | `14.91` | `15.24` | `+0.33` |
| `save/query` | `0.19` | `0.10` | `-0.09` |
| `filter_result_manipulator_motion_seconds/query` | `0.1043` | `0.0046` | `-0.0998` |
| `filter_result_total_seconds/query` | `0.1314` | `0.0323` | `-0.0991` |

补充对比：

- `prepare_depth_filter_seconds/query` 从 `0.65` 升到 `0.96`。
- `prepare_inputs_seconds/query` 从 `0.68` 升到 `1.01`。
- `tracker_model_forward_seconds/query` 从 `13.51` 微降到 `13.44`。

结论：

- 这次向量化确实把 wrist 路径里的 `manipulator_motion` 评估基本压平了，helper 改动是有效的。
- 但它只影响 save/filter 的一小段时间，量级约 `0.10 s/query`；对总吞吐的贡献上限太小。
- 在真实单卡复测里，总 wall clock 反而回到 `811.19s`，说明当前主导总耗时的仍然不是这块。
- 因此同语义主线优先级应回到 `prepare_depth_filter` per-frame kernel，而不是继续深挖 `manipulator_motion` candidate 子集化。

### 9. 两卡 `depth_filter_workers=8 vs 16` 小样本确认

固定条件：

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- episodes：`00000`, `00001`
- `gpu_id=0,1`
- `workers_per_gpu=1`
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`

输出根：

- `data_tmp/mjc_1000_multigpu_depth_filter_confirm_20260402_df8`
- `data_tmp/mjc_1000_multigpu_depth_filter_confirm_20260402_df16`

报告：

- `data_tmp/telemetry_reports/mjc_1000_multigpu_depth_filter_confirm_20260402_df8.md`
- `data_tmp/telemetry_reports/mjc_1000_multigpu_depth_filter_confirm_20260402_df8.json`
- `data_tmp/telemetry_reports/mjc_1000_multigpu_depth_filter_confirm_20260402_df16.md`
- `data_tmp/telemetry_reports/mjc_1000_multigpu_depth_filter_confirm_20260402_df16.json`

核心结果：

| depth_filter_workers | Wall(s) | Queries | SingleGPU/query | Slot/query | Process/query | Save/query | PrepDepth/query | PrepInputs/query | Tracker/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `421.37` | `51` | `16.52` | `15.53` | `15.43` | `0.11` | `1.10` | `1.15` | `13.47` |
| `16` | `452.88` | `51` | `17.76` | `15.53` | `15.42` | `0.10` | `1.08` | `1.14` | `13.48` |

结论：

- 本次两卡小样本里，`df16` 没有复现单卡 `df16 > df8` 的吞吐优势。
- `slot/query`、`process/query`、`tracker/query` 基本持平，说明 `16` 并没有把关键路径继续压短。
- `prepare_depth_filter/query` 与 `prepare_inputs/query` 只出现极小幅改善，不足以抵消更差的 wall clock。
- 因此当前不能把“单卡 isolate 最优”直接推广为“多卡默认值应改成 `16`”。
- 代码默认值先维持 `depth_filter_workers=8`；只有单卡隔离实验才继续把 `16` 当作 override baseline。

### 10. 单卡 `num_iters=5 vs 4 vs 3` batch sweep

固定条件：

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- batch episode list：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- compare manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.json`
- episodes：`00000`, `00001`
- `gpu_id=0`
- `workers_per_gpu=1`
- `depth_filter_workers=8`
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`

输出根：

- baseline `iters_5`：
  `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1`
- variant `iters_4`：
  `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n4`
- variant `iters_3`：
  `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n3`

报告：

- telemetry：
  - `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.md`
  - `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n4.md`
  - `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_num_iters_sweep_20260402_n3.md`
- quality compare：
  - `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_4_20260402/comparison_summary.md`
  - `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_3_20260402/comparison_summary.md`
- worst-case `RGB / 2D / 3D` triptych exports：
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.md`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.json`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.md`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.json`

核心速度结果：

| num_iters | Wall(s) | SingleGPU/query | Process/query | Save/query | Tracker/query | Speedup vs `5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `810.84` | `15.90` | `15.06` | `0.20` | `13.43` | `1.000x` |
| `4` | `661.94` | `12.98` | `12.64` | `0.10` | `10.78` | `1.225x` |
| `3` | `541.28` | `10.61` | `9.99` | `0.10` | `8.12` | `1.498x` |

质量对照（相对 baseline `iters_5`）：

| Variant | `varied_camera_1` Jaccard | `varied_camera_2` Jaccard | `varied_camera_3` Jaccard | `varied_camera_3` Valid Delta | Worst Wrist Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iters_4` | `0.9972` | `0.9968` | `0.8710` | `-38.47` | `0.4830` |
| `iters_3` | `0.9971` | `0.9967` | `0.8481` | `-73.38` | `0.4436` |

最差样本三列轨迹可视化已经导出并按 `traj_valid_mask_jaccard` 升序取 top-10：

- `iters_4`：`data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_4_20260402_top10_jaccard/summary.md`
- `iters_3`：`data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_3_20260402_top10_jaccard/summary.md`
- 两组 top-1 都是 `00001 / varied_camera_3 / query_frame=16`

结论：

- 这条线已经证明：`num_iters` 确实能给到两位数收益，且收益几乎全部来自 tracker forward 压缩。
- `tracker_model_forward_seconds/query` 明确随 `num_iters` 下调而下降：`13.43 -> 10.78 -> 8.12`。
- `prepare_depth_filter_seconds/query` 没有随之下降，甚至略有上升：`0.89 -> 1.08 -> 1.12`，说明这次收益不是来自 CPU depth filter。
- 两个 external 相机基本稳定，但 wrist `varied_camera_3` 质量回退明显，`iters_4` 已经低到 `0.871`，`iters_3` 进一步降到 `0.848`。
- 因此这轮 sweep 只能作为“速度上限”结论，不能直接收敛成新的维护态默认值；当前默认 `num_iters=5` 继续保持。

### 11. camera-aware `num_iters`: external=`4`, wrist=`5`

固定条件：

- 数据集：`/DATA/disk1/zoyo/mjc_1000_step1`
- batch episode list：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`
- compare manifest：`scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.json`
- episodes：`00000`, `00001`
- `workers_per_gpu=1`
- `depth_filter_workers=8`
- `traj_filter_profile=wrist_pick_place_no_heatmap`
- `future_len=32`
- `grid_size=80`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- global `num_iters=5`
- per-camera override：`varied_camera_1:4,varied_camera_2:4,varied_camera_3:5`

输出根：

- 首次尝试（无效，不纳入结论）：
  `data_tmp/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5`
- clean run：
  `data_tmp/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5_gpu3`

报告：

- telemetry：
  - `data_tmp/telemetry_reports/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5_gpu3.md`
  - `data_tmp/telemetry_reports/mjc_1000_step1_camera_aware_num_iters_20260402_ext4_wrist5_gpu3.json`
- quality compare：
  - `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3/comparison_summary.md`
  - `data_tmp/output_root_compares/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3/comparison_results.json`
- worst-case `RGB / 2D / 3D` triptych exports：
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3_top10_jaccard/summary.md`
  - `data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3_top10_jaccard/summary.json`

补充说明：

- 首次 GPU0 尝试因同卡已有进程占用约 `93 GiB` 显存，在 `00000 / varied_camera_3` 上触发 OOM；该 root 只保留为 contention 记录，不用于结论。

核心速度结果：

| Variant | Wall(s) | SingleGPU/query | Process/query | Save/query | Tracker/query | PrepDepth/query | Speedup vs `5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iters_5` | `810.84` | `15.90` | `15.06` | `0.20` | `13.43` | `0.89` | `1.000x` |
| `ext4_wrist5` | `811.23` | `15.91` | `15.40` | `0.14` | `12.65` | `1.56` | `1.000x` |

按相机看，总时间出现了明显分化：

- `varied_camera_1`：`14.47 -> 13.95 s/query`
- `varied_camera_2`：`14.04 -> 13.51 s/query`
- `varied_camera_3`：`17.25 -> 19.14 s/query`

质量对照（相对 baseline `iters_5`）：

| Camera | Valid Jaccard | Valid Delta | World L2 Mean | Worst Query Jaccard |
| --- | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `0.9972` | `-2.92` | `0.000171` | n/a |
| `varied_camera_2` | `0.9968` | `+0.94` | `0.000172` | n/a |
| `varied_camera_3` | `0.8574` | `+39.17` | `0.001008` | `0.0990` |

最差样本三列轨迹可视化同样已导出并按 `traj_valid_mask_jaccard` 升序取 top-10：

- `ext4_wrist5`：`data_tmp/query_rgb_scan/mjc_1000_step1_num_iters_5_vs_ext4_wrist5_20260402_gpu3_top10_jaccard/summary.md`
- top-1 仍是 `00001 / varied_camera_3 / query_frame=16`，但 Jaccard 已恶化到 `0.0990`

结论：

- external 两路 tracker forward 确实更快，但 `prepare_depth_filter` 也明显变慢，抵消了大部分收益。
- 更关键的是，wrist 即使保持 `num_iters=5` 也明显变差：总时长更长、质量更差，且最坏样本显著劣于全局 `iters_4`。
- 因此最直接的 camera-aware 方案 `external=4, wrist=5` 不能作为 rollout 候选；当前默认继续保持 `num_iters=5`、`depth_filter_workers=8`。

## 当前阶段性结论

截至目前，可以先确认：

1. telemetry 埋点已经足够支撑函数级和硬件级归因。
2. 当前单卡 H200 上，`workers_per_gpu=1/2/3/4` 的吞吐都几乎相同，稳定在 `15.90~15.91 s/query/H200`，所以并发 worker 不是当前优化重点。
3. CPU 侧深度过滤仍然会影响总吞吐，单卡 `depth_filter_workers=4/8/16` 分别对应约 `16.49/15.90/15.31 s/query/H200`。
4. 但这条单卡结论在本次两卡小样本上没有复现：`df16` 的 `SingleGPU/query=17.76` 反而劣于 `df8` 的 `16.52`。
5. 因此当前应区分两套 baseline：
   - 单卡 isolate：`workers_per_gpu=1`、`depth_filter_workers=16`
   - 当前代码默认 / 多卡稳妥配置：`workers_per_gpu=1`、`depth_filter_workers=8`
6. `manipulator_motion` 向量化虽然把 `filter_result_manipulator_motion_seconds/query` 从 `0.104` 压到 `0.0046`，但没有形成可见的总吞吐收益，因此不再是主优化重点。
7. `num_iters=4/3` 已经证明全局降迭代次数可以带来明确的两位数 wall-clock 收益，但 wrist `varied_camera_3` 质量退化过大，当前不能直接改默认值。
8. 最直接的 camera-aware 方案 `external=4, wrist=5` 已完成验证，但 wall-clock 仍然贴着 baseline，且 wrist 进一步恶化到 aggregate Jaccard `0.8574`、worst query `0.0990`，因此也不能作为默认值候选。
9. 如果后续还要沿 `num_iters` 深挖，前提是先解释为什么 external 改成 `4` 后，wrist 在保持 `5` 时仍会明显退化；在根因清楚前，优先级应回到 `prepare_depth_filter` per-frame kernel。

## 待补实验

当前还没补完：

- `prepare_depth_filter` per-frame kernel 优化（优先看 `points_to_normals`、`edge_mask`、`distance_transform`）
- `support_grid_ratio=0.8/0.6/0.4`
- `query_prefilter_mode=off/profile_aware_static_v1`
- 如果未来重开 `num_iters` 差异化收敛，先定位 `external=4, wrist=5` 为什么会让 wrist 回退；在根因不清楚前，不继续扫更简单的 camera-aware 组合
- 如需继续看并发策略，优先补更大样本的多卡 `depth_filter_workers=8/16` 复测，再决定是否调整默认值；单卡 `>1` 已不再是高优先级

## 更新规则

后续每完成一档实验，在本文档追加：

- 固定条件
- 输出根
- 报告路径
- 核心数字
- 一句话结论

不再把这些结论散落到新的临时笔记中。
