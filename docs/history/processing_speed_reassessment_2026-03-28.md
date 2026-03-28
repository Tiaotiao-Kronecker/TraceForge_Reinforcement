# 处理速度现状复核与下一步计划

日期：2026-03-28

本文档用于接续 2026-03-21 的速度优化总结，但只保留当前代码状态仍然成立的判断，并补上 2026-03-28 基于真实 `pick_place` workload 的新结论。它的目标不是重复早期 profiling 过程，而是回答：

1. 旧的优化规划哪些已经完成
2. 当前真正的瓶颈在哪里
3. 下一步最值得做的优化是什么

## 背景

2026-03-18 到 2026-03-21 的几轮工作，已经完成了两个大的阶段：

- `save` 路径从“全量 depth volatility + shared CPU 统计”主导，转成次要开销
- `process_single_video()` 成为端到端主瓶颈

但那轮结论停在 2026-03-21。当前代码又增加了：

- `_DepthFilterRuntime` 的 segment 级缓存与 ray cache
- wrist / pick_place 过滤逻辑的若干同语义优化
- `episode_00648` 上的最新端到端 benchmark

因此需要重新排一次优先级。

## 当前代码里已经落地的内容

以下判断已经不是“计划”，而是已经进入当前实现：

### 1. save 共享热路径的大头已经清掉

当前 `save_structured_data()` / `save_source_ref_v2_query_results()` 只在确实需要 temporal consistency + volatility guidance 时才计算高波动 mask，见：

- `scripts/batch_inference/infer.py`
- `save_source_ref_v2_query_results()`

同时，query bundle 准备、patch stats、base geometry 和 scene-level high-volatility mask 都已经是向量化后的 shared path。

### 2. `num_iters=5` 已经成为维护态默认值

当前默认值已经体现在：

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`

因此 `num_iters 6 -> 5` 不再是待讨论方案，而是当前维护态事实。

### 3. `prepare_depth_filter` 已经有跨 segment 复用

当前 `_DepthFilterRuntime` 已经做了：

- per-frame filtered depth cache
- ray cache
- tracking segment 的 use-count 管理

这说明今天再谈 `prepare_depth_filter`，重点不该是“有没有 cache”，而该是“cache 之后每个 frame 自身的计算还重不重”。

### 4. `pick_place` 路径最近还做了几项同语义优化

最新实现里已经有：

- `world_tracks` 复用
- pick-place reference geometry 复用
- delayed rescue / contact 只在 candidate 子集上算
- 单 component nearest-distance 的 fast path

这些都说明：当前仓库已经在朝“先做同语义降本，再考虑改算法语义”的方向推进。

## 最新 benchmark 复核

本轮主要参考两组真实 workload：

- `data_tmp/e2e_profile_compare_episode_00648_20260328_run1/benchmark_summary.json`
- `data_tmp/e2e_profile_compare_episode_00648_20260328_extra_filters_run1/benchmark_summary.json`

重点看 `episode_00648 / varied_camera_3`。

### 1. wrist-like profile 下，端到端几乎完全由 process 主导

最新结果显示：

- `wrist_manipulator_top95`
  - `process ≈ 96.5% of total`
  - `save ≈ 3.5% of total`
- `wrist_pick_place_no_heatmap`
  - `process ≈ 96.6% of total`
  - `save ≈ 3.4% of total`
- `wrist_manipulator`
  - `process ≈ 96.5% of total`
  - `save ≈ 3.5% of total`
- `wrist_pick_place`
  - `process ≈ 96.2% of total`
  - `save ≈ 3.8% of total`

这比 2026-03-21 的判断更进一步：对于当前 wrist-like 维护态 workload，`save` 已经不是决定端到端耗时的主战场。

### 2. process 内部比 03-21 时更集中到 tracker forward

在这些 2026-03-28 benchmark 上：

- tracker forward 已占 `process` 的约 `86%`
- `prepare_inputs` 只占 `process` 的约 `7.5% ~ 8.0%`

这意味着旧结论里“tracker forward 第一、prepare_inputs 第二”仍然成立，但比例已经明显变化：

- tracker forward 的统治性更强了
- `prepare_inputs` 仍重要，但不再接近 `30%`

### 3. wrist save 的最大热点已经收敛到 `manipulator_motion`

对当前 wrist-like profile：

- `filter_result_manipulator_motion_seconds` 约占 `save_total_seconds` 的 `65% ~ 70%`
- `query_depth_edge_risk` 已降到 `5% ~ 6%`
- `high_volatility_mask` 已降到 `7% ~ 9%`

因此：

- 如果目标是“继续降 wrist save”，第一优先级已经非常明确，就是 `manipulator_motion`
- `query_depth_edge_risk` 和 shared save path 都不应再排在它前面

## 更新后的瓶颈排序

结合当前代码和最新 benchmark，今天更合理的优先级是：

1. tracker model forward
2. wrist save 中的 `manipulator_motion`
3. `prepare_depth_filter` 内部每帧 kernel，尤其是 `points_to_normals`
4. 其他 wrist save residual，例如 `query_depth_edge_risk`
5. shared save path

换句话说，03-21 之后如果继续按旧排序把主要精力投到 shared save 或 `query_depth_edge_risk`，收益会明显不如直接做上面前三项。

## 下一步优化规划

## P0：先优化 `manipulator_motion`

这是当前最值得先动的一项，因为它同时满足：

- 同语义
- 局部改动
- benchmark 上已经确认是 wrist save 最大热点

当前 `utils/traj_filter_utils.py::_compute_motion_metrics_for_valid_masks()` 仍保留了按 track 的 Python 层循环。下一步建议：

1. 只对 `seed_mask` / `candidate_mask` 子集计算 motion metric，再 scatter 回全量数组
2. 把 motion extent 和 step median 的计算改成向量化，去掉 per-track Python loop
3. 保持输出字段与当前语义完全一致

这是最适合先落地、再立即用现有 benchmark 脚本复测的一项。

## P1：继续压 `prepare_depth_filter`

虽然 `prepare_inputs` 的总占比已经降了，但 `prepare_depth_filter` 仍然是其中的大头，而且从细分项看：

- `points_to_normals`
- `edge_mask`
- `distance_transform`

依然明显重于其他步骤。

下一步不该再重复讨论“要不要加 cache”，而应直接看：

1. normals / edge / distance transform 是否能做 scene-level artifact cache
2. `_filter_one_depth` 核心是否值得搬到更适合的实现层
3. 是否能在不改语义的前提下减少每帧重复构造中间数组

这项仍然是同语义优化，但改动面比 P0 大。

## P2：重新设计 tracker 侧复用

真正决定端到端上限的，还是 tracker forward。

当前 `utils/inference_utils.py::inference()` 仍然是：

- 每个 query frame 单独调一次 model
- support query 只是在单次 forward 内扩 query，不涉及跨 query 复用

因此 overlapping query segments 之间仍有大量重复工作。长期最有价值的方向是：

1. 多 query frame 按 segment 长度分桶做 batched inference
2. 或者在 tracker 里做 episode-level feature cache
3. 至少复用 RGB/depth 编码，而不是每个 query 全重算

这项收益最大，但风险和实现复杂度也最大，应排在 P0/P1 之后。

## 当前不建议优先推进的方向

以下内容目前不该排到前面：

- shared save path 的继续微调
- `query_depth_edge_risk` 的进一步局部优化
- 单纯继续 sweep `support_grid_ratio`
- 把 `query_prefilter_mode` 直接推成维护态默认值

它们不是没有价值，而是当前收益排序已经落后于 P0/P1/P2。

## 推荐执行顺序

如果下一轮目标是“尽量快地产出真实收益”，建议按下面顺序推进：

1. `traj_filter_utils.py` 中 `manipulator_motion` 的同语义向量化 / 子集化
2. `prepare_depth_filter` 的 per-frame kernel 优化
3. tracker side 的 batching / feature cache 方案设计与原型

## 与旧文档的关系

本文不替代下列文档中的原始数据和实验过程，但会替代它们作为“今天继续优化该从哪里开始”的入口：

- `processing_speed_optimization_summary_2026-03-21.md`
- `process_inference_bottleneck_analysis_2026-03-19.md`
- `save_timing_alignment_status_2026-03-21.md`

而 `depth_volatility_optimization_plan_2026-03-19.md` 及其 P1/P4 实施稿属于已完成阶段的中间计划，不再建议继续作为当前入口文档保留。
