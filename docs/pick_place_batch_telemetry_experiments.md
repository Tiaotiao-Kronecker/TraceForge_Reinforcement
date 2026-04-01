# pick_place Batch Telemetry 实验结论

日期：2026-04-01

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

### 本轮不改语义性能实验

| 项目 | 当前状态 | 当前结论 / 默认建议 |
| --- | --- | --- |
| 单卡 `workers_per_gpu = 1/2/3/4` | 已完成 | `>1` 不提吞吐，只放大同卡竞争；默认固定 `workers_per_gpu=1` |
| 单卡 `depth_filter_workers = 4/8/16` | 已完成 | CPU 侧仍会影响总吞吐；默认固定 `depth_filter_workers=16` |
| 8 卡 `workers_per_gpu = 1/2/4` | 未执行 | 如果继续看并发策略，应先做 8 卡 `workers_per_gpu=1` 基线，`>1` 低优先级 |

### 未完成同语义优化清单

| 项目 | 优先级 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| `manipulator_motion` 的向量化 / candidate 子集化 | `P0` | 未开始 | 当前最适合先落地、再立即复测的同语义优化 |
| `prepare_depth_filter` 的 per-frame kernel 优化 | `P1` | 未开始 | 重点看 `points_to_normals`、`edge_mask`、`distance_transform` |
| tracker 侧 batching / feature cache 复用 | `P2` | 未开始 | 收益上限最高，但风险和改动面也最大 |

### 改语义性能-质量权衡实验

| 项目 | 当前状态 | 当前结论 / 备注 |
| --- | --- | --- |
| `num_iters: 6 -> 5` | 历史上已完成 | 已经进入维护态默认值，当前默认是 `num_iters=5` |
| `num_iters: 5 -> 4 -> 3` | 未执行 | 下一轮可继续做，但必须带质量对照 |
| `support_grid_ratio: 0.8 -> 0.6 -> 0.4` | 未执行 | 属于明确改语义项，当前默认仍是 `0.8` |
| `query_prefilter_mode: off -> profile_aware_static_v1` | 未执行 | 属于明确改语义项，当前默认仍是 `off` |
| `future_len: 32 -> 24` | 当前冻结 | 用户已要求本轮先不要动 |
| `grid_size: 80 -> 40` | 当前冻结 | 用户已要求本轮先不要动 |

### 当前推荐 baseline

如果目标是“当前 H200 + pick_place 维护态 workload 的最低时间消耗”，当前推荐固定：

- `workers_per_gpu=1`
- `depth_filter_workers=16`
- `num_iters=5`
- `support_grid_ratio=0.8`
- `query_prefilter_mode=off`
- `future_len=32`
- `grid_size=80`

## 当前固定条件

本轮默认固定：

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

## 当前阶段性结论

截至目前，可以先确认：

1. telemetry 埋点已经足够支撑函数级和硬件级归因。
2. 当前单卡 H200 上，`workers_per_gpu=1/2/3/4` 的吞吐都几乎相同，稳定在 `15.90~15.91 s/query/H200`，所以并发 worker 不是当前优化重点。
3. CPU 侧深度过滤仍然会影响总吞吐，`depth_filter_workers=4/8/16` 分别对应约 `16.49/15.90/15.31 s/query/H200`。
4. `depth_filter_workers` 带来的收益主要体现在 `prepare_depth_filter_seconds/query` 和 `prepare_inputs_seconds/query`，而不是 `tracker_model_forward_seconds/query` 或 `save`。
5. 当前单卡后续 baseline 应固定为 `workers_per_gpu=1`、`depth_filter_workers=16`。
6. 后续若继续做纯性能 sweep，应基于这个新 baseline 再看 `num_iters`、`support_grid_ratio`、`query_prefilter_mode`。

## 待补实验

当前还没补完：

- `num_iters=5/4/3`
- `support_grid_ratio=0.8/0.6/0.4`
- `query_prefilter_mode=off/profile_aware_static_v1`
- 如需继续看并发策略，优先把 8 卡 `workers_per_gpu=1`、`depth_filter_workers=16` 作为基线；单卡 `>1` 已不再是高优先级

## 更新规则

后续每完成一档实验，在本文档追加：

- 固定条件
- 输出根
- 报告路径
- 核心数字
- 一句话结论

不再把这些结论散落到新的临时笔记中。
