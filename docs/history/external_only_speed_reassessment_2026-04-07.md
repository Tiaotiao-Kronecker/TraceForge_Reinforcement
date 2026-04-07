# external-only 维护态速度结论复核

日期：2026-04-07

## 背景

截至当前代码状态，维护态已决定不再把 wrist 相机作为默认提供数据的一部分。代码仍保留 wrist 相关 profile，但它们不再代表维护态默认路径。

这意味着之前不少性能结论需要重新分层：

- 哪些结论仍然适用于 external-only 主线
- 哪些结论只适用于 mixed external+wrist workload
- 哪些实验需要重跑，才能支持新的维护态默认值决策

## 当前代码层面的收敛

本次收敛已经把以下默认行为改成 external-only：

- `batch_infer_press_one_button_demo.py` 默认 `camera_names=varied_camera_1,varied_camera_2`
- `batch_infer_press_one_button_demo.py` 默认 `traj_filter_profile=external`
- `repair_empty_samples_press_one_button_demo.py` 默认 `traj_filter_profile=external`
- `scripts/data_analysis/benchmark_*` 的默认相机组合改为 `varied_camera_1,varied_camera_2`
- 兼容参数 `traj_filter_profile=auto` 仍然保留，但当前只解析到 `external`

这次没有删除 wrist 代码。原因是：

- 它仍然承载历史调查与兼容性复跑价值
- 删除 wrist 实现不会直接带来运行时提速
- 当前更重要的是先把维护态默认值、文档入口和 benchmark 口径统一到 external-only

## 哪些旧结论不再适合作为主线入口

以下结论应降级为 mixed external+wrist 历史背景，不再作为维护态默认优化排序：

1. wrist save 的 `manipulator_motion` 是当前最高优先级。
2. `query_depth_edge_risk` 仍是维护态默认 save 残余的主矛盾。
3. camera-aware `external=4, wrist=5` 的失败，足以否定 external-only 的 `num_iters=4` 候选价值。

这些判断并不是“错了”，而是它们回答的是 wrist 仍在主线内时的问题。

## 哪些旧结论仍然成立

以下判断在 external-only 视角下仍然成立：

1. save 已经不是端到端主瓶颈。
2. `process_single_video()` 仍是主链路瓶颈。
3. `tracker model forward` 仍是 process 的第一热点。
4. `prepare_depth_filter` 仍是第二梯队热点，尤其是 per-frame kernel。

因此，维护态主线优先级应改写为：

1. tracker forward
2. `prepare_depth_filter`
3. `num_iters` / tracker 复用 / batching

而不是继续把 wrist-specific filter 当成默认优化主战场。

## 对 `num_iters` 结论的重述

之前 mixed workload 的结论是：

- `num_iters=4/3` 在固定子集上能带来明确加速
- 但 wrist 路径质量回退明显
- 因此不能直接替代维护态默认 `num_iters=5`

在 external-only 语境下，更准确的表述应该是：

- mixed workload 已经证明 `num_iters` 存在两位数收益空间
- 但现有证据还不足以决定 external-only 是否可以把默认值进一步从 `5` 收敛到 `4`
- 这个问题需要 external-only 复测，而不是继续引用 wrist 回退结论

## 需要补做的实验

当前最值得补做、且范围已经明显收缩的实验只有三组：

1. external-only baseline
   - 相机：`varied_camera_1,varied_camera_2`
   - profile：`external`
2. external-only `num_iters=5/4/3` sweep
   - 重点看速度变化和轨迹差异
   - 不再混入 wrist 路径
3. external-only process/save telemetry
   - 确认 `tracker_model_forward_seconds`
   - 确认 `prepare_depth_filter_*`
   - 确认 save 残余是否仍然可以忽略

## 当前建议

在 external-only 复测完成之前，当前维护态建议是：

1. 默认值先保持 `num_iters=5`
2. 默认值先保持 `depth_filter_workers=8`
3. 不再把 wrist-specific save/filter 作为默认优化主线
4. 下一轮 benchmark 和文档都统一按 external-only 口径组织

## 与旧文档的关系

下面这些文档仍然保留原始 mixed workload 调查价值，但不再适合作为 external-only 维护态默认优化入口：

- `processing_speed_optimization_summary_2026-03-21.md`
- `processing_speed_reassessment_2026-03-28.md`
- `pick_place_batch_telemetry_experiments.md`
- `pick_place_num_iters_sweep_2026-04-02.md`

如果它们与当前维护态默认值有冲突，应以当前 README、`CLAUDE.md`、批处理指南和本文为准。
