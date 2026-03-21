# 处理速度优化阶段总结

日期：2026-03-21

## 范围

这份文档回顾 2026-03-18 到 2026-03-21 这几天围绕处理速度做的主要工作，覆盖：

- button/sim episode 维护态推理流程
- 单个 `episode/camera` 的端到端 wall-clock
- `process_single_video()` 与 `save_structured_data()` 两条主链
- 已经进入维护态的优化
- 为下一轮速度优化补上的 profiling / benchmark 基础设施

详细过程仍然保留在各自的历史文档里；本页只做总览和收敛。

## 一页结论

截至当前代码状态，可以固定下来的判断是：

- 第一阶段最大的收益来自 save 路径重写。原来最重的 `compute_depth_volatility_map()` 已经不再是主瓶颈。
- 第二阶段的重点已经转移到 `process_single_video()`，其中 tracker model forward 是主瓶颈，`prepare_depth_filter` 是第二瓶颈。
- 维护态默认 `num_iters` 已经从 `6` 收敛到 `5`，代表 case 上能稳定拿到约 `11%~18%` 的 `process/total` 降幅，质量风险仍可控。
- 当前维护态单卡吞吐，按 `Total / query frame` 粗归一化后，大致是每个 query frame `10s` 左右，更准确地说约 `10~12s / query frame`。
- save 侧剩余差距已经不再主要来自 shared path，而是 wrist-specific 的 `manipulator_motion` 和 `query_depth_edge_risk`。

## 时间线

| 日期 | 主题 | 主要动作 | 已验证结果 |
| --- | --- | --- | --- |
| `2026-03-18` | 初始瓶颈定位 | 拆分端到端耗时，确认 `compute_depth_volatility_map()` 是 save 阶段主瓶颈 | 单 camera 平均总耗时 `176.46s`，其中 `save=90.35s`，而 `compute_depth_volatility_map=83.37s` |
| `2026-03-19` | 统一 depth-volatility hot path | 把 volatility 统计收敛成“只算访问到的位置 + camera 级一次性高波动 mask + temporal compare context 复用” | external save `90.974s -> 7.323s`，wrist-like save `90.661s -> 13.785s` |
| `2026-03-19` | process 主瓶颈再定位 | 拆 `process_single_video()`，确认 save 不再是主矛盾后，process 已成为主链路瓶颈 | `process` 中 tracker model forward 约占 `64.1%`，`prepare_inputs` 约占 `29.6%` |
| `2026-03-20` | process 端实验旋钮 | 增加 query prefilter、`support_grid_ratio` 和 `benchmark_inference_variants.py` | 主要是为后续 sweep 建基础设施，未直接收敛成维护态默认值 |
| `2026-03-21` | shared save hot path 清理 | 去掉不必要的整段 RGB uint8 化，向量化 query patch stats / base geometry，细化 save profiling | representative no-support case 上 external save `6.437s -> 0.337s`，wrist save `7.865s -> 1.981s` |
| `2026-03-21` | process 默认值收敛 | 做 `num_iters` sweep，并把维护态默认值从 `6` 调整到 `5` | `iters_5` 相对 `iters_6` 带来约 `11%~18%` 的 `process/total` 降幅 |
| `2026-03-21` | prepare_depth_filter 深挖 | 向量化 `depth_edge` / `normals_edge` / `points_to_normals`，并新增 `_filter_one_depth_profiled` 分阶段计时 | 当前主要价值是把 `prepare_depth_filter` 固定成本继续拆细到 `points_to_normals` / `edge_mask` / `distance_transform` |

## 已验证收益

### 1. save 路径：先砍掉 depth volatility 全图统计

2026-03-18 的 profiling 先确认了最初的大头：

- 单 camera 平均总耗时：`176.46s`
- `process_single_video()`：`86.10s`
- `save_structured_data()`：`90.35s`
- `compute_depth_volatility_map()`：`83.37s`

也就是说，当时不是写盘慢，而是 save 前做了整段全分辨率 depth 的 percentile 统计，几乎吃掉了整个 save 阶段。

2026-03-19 的统一 hot path 优化把这部分冗余大幅砍掉。代表 benchmark 上：

| Camera | Profile | Baseline Save (s) | Current Save (s) | Save Speedup | Baseline Total (s) | Current Total (s) | Total Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `90.974` | `7.323` | `12.423x` | `159.055` | `75.899` | `2.096x` |
| `varied_camera_3` | `wrist_manipulator_top95` | `90.661` | `13.785` | `6.577x` | `159.967` | `83.411` | `1.918x` |

这一步的工程含义很明确：

- `compute_depth_volatility_map()` 不再是当前维护态下的主瓶颈
- save 阶段的 CPU 统计冗余已经从“决定性瓶颈”降成了次要问题
- 端到端优化重心开始转向 `process_single_video()`

对应细节文档：

- `docs/history/depth_volatility_bottleneck_analysis_2026-03-18.md`
- `docs/history/depth_volatility_benchmark_validation_2026-03-19.md`

### 2. process 路径：主瓶颈转移到 tracker forward

在 save 热路径收敛之后，2026-03-19 继续拆了 `process_single_video()`，结果显示：

| Camera | Process Total (s) | Prepare Inputs (s) | Tracker Inference (s) | Tracker Model Forward (s) |
| --- | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `69.009` | `20.355` | `45.194` | `43.808` |
| `varied_camera_3` | `69.017` | `20.545` | `45.778` | `44.675` |
| `平均` | `69.013` | `20.450` | `45.486` | `44.241` |

按占比看：

- `tracker_model_forward_seconds / process_total_seconds = 64.1%`
- `prepare_inputs_seconds / process_total_seconds = 29.6%`

所以这轮之后，process 路径的优先级就已经很清楚了：

1. tracker 主体 forward
2. `prepare_inputs(...)`
3. 其他项都不是主要矛盾

对应细节文档：

- `docs/history/process_inference_bottleneck_analysis_2026-03-19.md`

### 3. save 路径第二轮：清理 shared hot path

2026-03-21 的工作没有再回到 old-style volatility 全图统计，而是针对新的 shared save residual 做清理，主要包括：

- `source_ref` 路径下不再无条件构造整段 `RGB uint8`
- query bundle 的 tensor-to-numpy 转换进一步下沉到真正需要的位置
- `_compute_query_depth_patch_stats()` 向量化
- `compute_traj_base_geometry()` 向量化
- save 路径各阶段 profile key 继续细化

代表性 no-support case（`episode_00105`，`support_grid_ratio=0.0`）上，结果是：

| Camera | Profile | Process (s) | Save (s) | Total (s) |
| --- | --- | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `55.632` | `0.337` | `55.969` |
| `varied_camera_3` | `wrist_manipulator_top95` | `54.769` | `1.981` | `56.750` |

相对上一轮同口径 case：

- external save：`6.437s -> 0.337s`
- wrist save：`7.865s -> 1.981s`

这一步之后，save 侧的最新判断是：

- external 的 shared save 路径已经基本压到亚秒级
- wrist 的剩余时间主要集中在 wrist-specific filter
- 如果目标是继续缩 wrist/external 差距，优先级已经不再是 shared save path

对应细节文档：

- `docs/history/save_timing_alignment_status_2026-03-21.md`

### 4. process 默认值：`num_iters 6 -> 5`

2026-03-21 针对 `process` 主链做的最重要收敛，是把 `num_iters` 从经验值变成了有 benchmark 支撑的维护态默认值决策。

`episode_00105` 的 sweep 结果是：

| Camera | Variant | Process (s) | Total (s) |
| --- | --- | ---: | ---: |
| `varied_camera_1` | `iters_6` | `57.200` | `57.592` |
| `varied_camera_1` | `iters_5` | `51.470` | `52.002` |
| `varied_camera_1` | `iters_4` | `43.514` | `43.906` |
| `varied_camera_3` | `iters_6` | `57.193` | `59.729` |
| `varied_camera_3` | `iters_5` | `48.516` | `50.620` |
| `varied_camera_3` | `iters_4` | `42.585` | `44.791` |

综合质量对照后，最终结论是：

- 维护态默认值切到 `5`
- 不直接切到 `4`

原因：

1. `5` 已经能稳定拿到约 `11%~18%` 的 `process/total` 降幅
2. `5` 的质量风险明显好于 `4`
3. `4` 在 wrist 代表 case 上已经开始出现更明显偏差

当前维护态默认值已同步到：

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
- `utils/inference_utils.py`

对应细节文档：

- `docs/history/process_num_iters_sweep_2026-03-21.md`

## 当前维护态的统一理解

把这几轮工作放到一起看，当前维护态可以用下面这组结论描述：

### 1. save 已经不再主导端到端时间

当前代表 no-support case 上：

- external：`process 55.632s`，`save 0.337s`
- wrist：`process 54.769s`，`save 1.981s`

save 已经从“和 process 一样重”降到了“只占 total 的很小一段”。优化重点不应再回到 shared save path。

### 2. process 的最核心瓶颈仍然是 tracker 主体

当前 process 路径最重的是：

- tracker model forward
- `prepare_depth_filter`

这两块才是下一轮真正还能继续产出大收益的地方。

### 3. 单卡吞吐已经可以用一个简单口径记住

`v1` 单 case 和 `v5` 三-episode workload 的对照已经说明：

- summary 里的 `Process / Save / Total` 是整段 `episode/camera` 的 wall-clock
- 它们不是 per-query 计时
- 如果按 `Total / query frame` 粗归一化，当前维护态单卡吞吐大致是：
  - `10s` 级别
  - 更精确地说约 `10~12s / query frame`

这条结论适合后续做工程估算时直接复用。

## 这几天补上的优化基础设施

除了已经验证收益并进入维护态的优化，这几天还补上了一批“为了继续压时间而必须先有”的基础设施。

这些内容本身不等于“已经收敛成新的维护态默认值”，但它们决定了后续优化能不能高效推进：

- `benchmark_depth_volatility_optimization.py`
  - 用于对比 volatility 优化前后 save / total 变化
- `benchmark_inference_variants.py`
  - 用于 sweep `query_prefilter_mode` / `support_grid_ratio` 一类 process 端旋钮
- `benchmark_num_iters_sweep.py`
  - 用于对比 `num_iters 6/5/4` 的 process、total、质量差异
- `benchmark_num_iters_manifest.py`
  - 用固定 case 集批量复测 `num_iters`
- `benchmark_wrist_filter_ablations.py`
  - 在复用同一次 tracking 结果的前提下，对 wrist save filter 做 save-only 消融
- `_filter_one_depth_profiled`
  - 把 `prepare_depth_filter` 再细拆到：
    - `ray_scale`
    - `points_to_normals`
    - `edge_mask`
    - `distance_transform`
    - `fill`
- save 路径 profile keys
  - 现在已经能区分：
    - `prepare_bundles`
    - `high_volatility_mask`
    - `filter_eval`
    - `sample_write`
    - `save_other`

当前对这些基础设施的定位应该是：

- `query_prefilter` / `support_grid_ratio`
  - 仍是 process 端的实验旋钮，当前没有收敛成新的默认值
- `traj_filter_ablation_mode`
  - 是 save-only 分析工具，不是生产默认行为
- 这些工具的主要价值，是让下一轮优化可以不再靠临时脚本和一次性 profiling

## 下一步

如果继续以“处理速度”作为第一目标，下一轮最值得做的事情仍然是：

1. 继续压 `prepare_depth_filter`
   - 优先看 `points_to_normals`、`edge_mask`、`distance_transform` 谁占最大头
   - 如果 worker 总和显著大于 wall time，优先改算法本身，而不是继续抠线程调度
2. 继续压 wrist-specific save residual
   - 优先看 `manipulator_motion`
   - 其次看 `query_depth_edge_risk`
3. 用固定 manifest 持续复测维护态吞吐
   - 保证 `num_iters=5` 这个默认值在 `v1 / v5` workload 上持续成立
4. 再决定是否推进 query prefilter / support grid 进入默认值讨论
   - 这一步需要新的质量与速度对照，不应只凭单次局部 benchmark 直接改默认值

## 相关文档

- `docs/history/depth_volatility_bottleneck_analysis_2026-03-18.md`
- `docs/history/depth_volatility_benchmark_validation_2026-03-19.md`
- `docs/history/process_inference_bottleneck_analysis_2026-03-19.md`
- `docs/history/save_timing_alignment_status_2026-03-21.md`
- `docs/history/process_num_iters_sweep_2026-03-21.md`
