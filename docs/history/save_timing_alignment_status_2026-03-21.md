# save 时间对齐现状记录

日期：2026-03-21

## 背景

这一轮工作的目标有两个：

- 尽量降低 `save_structured_data()` 的绝对耗时
- 尽量把 wrist 和 external 两种 camera 的 save 耗时拉齐

同时，本轮分析刻意限制在 `support_grid_ratio=0.0` 的 case 上，避免 support points 混入判断。

## 当前锚定基准

当前最适合拿来做后续 profiling 对照的是：

- 结果文件：`/tmp/e2e_profile_ep00105_sg0_warm1/summary.json`
- episode：`episode_00105`
- `support_grid_ratio=0.0`
- `warmup_runs=1`
- `benchmark_runs=1`

其中 measured run 的端到端结果是：

| Camera | Profile | Process (s) | Save (s) | Total (s) |
| --- | --- | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `52.506` | `5.856` | `58.362` |
| `varied_camera_3` | `wrist_manipulator_top95` | `54.356` | `7.706` | `62.062` |

对应差值：

- process：`+1.850s`
- save：`+1.850s`
- total：`+3.700s`

## save 分解结论

同一份 measured run 里，save breakdown 是：

| Camera | prepare_bundles (s) | high_volatility_mask (s) | filter_eval (s) | sample_write (s) | save_other (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `external` | `1.689` | `0.037` | `2.964` | `0.021` | `1.146` |
| `wrist_manipulator_top95` | `1.745` | `0.149` | `4.613` | `0.024` | `1.175` |

从这里可以直接得到三个结论：

1. `high_volatility_mask_seconds` 已经不是主要矛盾。代表 case 里 wrist 比 external 只多了约 `0.11s`。
2. 当前 wrist save 额外慢下来的主因是 `filter_eval_seconds`，代表 case 上单项就多了约 `1.65s`。
3. 两种 camera 都还共享一块约 `1.15s` 的 `save_other_seconds`，说明 save 路径里还有未拆开的公共开销。

## 跨 case 当前结果

优化后的 current save 结果，三组代表 case 如下：

| Episode | External Save (s) | Wrist Save (s) | Gap (s) |
| --- | ---: | ---: | ---: |
| `episode_00097` | `7.310` | `9.429` | `2.119` |
| `episode_00105` | `6.094` | `7.773` | `1.679` |
| `episode_00113` | `6.149` | `8.514` | `2.365` |

对应产物：

- `/tmp/depth_volatility_bench_ep00097_20260321/benchmark_summary.md`
- `/tmp/depth_volatility_bench_ep00105_20260321/benchmark_summary.md`
- `/tmp/depth_volatility_bench_ep00113_20260321/benchmark_summary.md`

这说明：

- save 阶段已经从接近 `100s` 压到了 `6s~9s`
- wrist 仍然稳定比 external 慢约 `1.7s~2.4s`
- 这已经不是 depth volatility mask 本身造成的差距

## 当前判断

截至 2026-03-21，这一轮可以先固定下来的判断是：

- wrist / external 的剩余 save 差距，主嫌疑已经从 `high_volatility_mask` 转移到 `filter_eval`
- save 路径里还存在一块两边都共有的公共开销，需要进一步拆到 scene tensor 转换、bundle 准备、per-query save loop 残差
- 端到端主瓶颈依然主要在 `process_single_video()`，尤其是 tracker forward 和 prepare depth filter；但 save 侧还值得继续做，因为它同时影响总耗时和 wrist/external 对齐

## shared save 优化后复测

按同一口径重跑：

- 结果文件：`/tmp/e2e_profile_ep00105_sg0_warm1_sharedopt_20260321_1313/summary.json`
- markdown：`/tmp/e2e_profile_ep00105_sg0_warm1_sharedopt_20260321_1313/summary.md`
- episode：`episode_00105`
- `support_grid_ratio=0.0`
- `warmup_runs=1`
- `benchmark_runs=1`

measured run 的新结果变成：

| Camera | Profile | Process (s) | Save (s) | Total (s) |
| --- | --- | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `55.632` | `0.337` | `55.969` |
| `varied_camera_3` | `wrist_manipulator_top95` | `54.769` | `1.981` | `56.750` |

这里最重要的结论不是 process 波动，而是 save 绝对值已经被压到了新的量级：

- external save：`6.437s -> 0.337s`，单 case 下降约 `6.10s`
- wrist save：`7.865s -> 1.981s`，单 case 下降约 `5.88s`

shared hot path 的下降也很明确：

| Key | External Drop (s) | Wrist Drop (s) |
| --- | ---: | ---: |
| `prepare_bundle_tensor_to_numpy_seconds` | `1.571` | `1.524` |
| `filter_result_query_depth_patch_stats_seconds` | `1.744` | `1.626` |
| `filter_result_base_geometry_seconds` | `1.409` | `1.369` |
| `prepare_bundles_seconds` | `1.627` | `1.533` |
| `query_frame_save_loop_seconds` | `3.169` | `3.157` |
| `save_total_seconds` | `6.017` | `5.882` |

这说明这轮 shared save 优化是有效且足够大的：

1. `scene_video_uint8_seconds` 已经不再出现在 v2 `source_ref` 的 measured save 路径里。
2. query bundle 不再做整段 RGB uint8 化后，`prepare_bundle_tensor_to_numpy_seconds` 基本被打到接近零。
3. `_compute_query_depth_patch_stats()` 和 `compute_traj_base_geometry()` 的向量化把两项 shared filter 热点都从 `1.4s~1.8s` 级别压到了 `0.03s~0.05s`。

## 新 residual 落点

优化后，external 的 save 主要只剩：

- `prepare_bundle_temporal_context_seconds ≈ 0.169s`
- `filter_result_query_depth_patch_stats_seconds ≈ 0.045s`
- `filter_result_base_geometry_seconds ≈ 0.033s`
- `high_volatility_mask_seconds ≈ 0.029s`

wrist 的 residual 则明显变成 wrist-specific filter：

- `filter_result_manipulator_motion_seconds ≈ 0.846s`
- `filter_result_query_depth_edge_risk_seconds ≈ 0.555s`
- `filter_result_manipulator_cluster_seconds ≈ 0.059s`
- `high_volatility_mask_seconds ≈ 0.147s`
- `prepare_bundle_temporal_context_seconds ≈ 0.198s`

因此新的判断应该改成：

- 如果目标是“继续同时降低 external 和 wrist 的绝对 save 时间”，shared path 已经基本清得差不多，继续往下挖的收益会明显变小。
- 如果目标是“对齐 wrist 和 external”，下一阶段主战场已经不再是 shared save path，而是 wrist-specific 的 `manipulator_motion` 与 `query_depth_edge_risk`。
- save gap 在这个 case 上没有一起缩小，反而从约 `1.43s` 变成约 `1.64s`；原因不是 shared path 还没清掉，而是 shared path 清掉之后，剩余时间几乎都集中在 wrist 专属过滤逻辑上。

## 后续 profiling 方案

shared save 优化完成后，下一步更合适的测试方向是：

1. 把同一套新 profiling 跑到 `episode_00097 / 00105 / 00113` 三个 no-support case。
   目标：确认 external save 已经稳定落在亚秒级，而 wrist residual 是否稳定主要落在 `manipulator_motion` / `query_depth_edge_risk`。
2. 对 wrist profile 做 targeted ablation 或近似替换测试。
   目标：在不明显伤害轨迹质量的前提下，确认 `query_depth_edge_risk`、`manipulator_motion`、`cluster` 哪一层还可以继续删减或合并。

如果接下来的 profiling 结果里：

- external 仍有稳定的公共残差：优先看 `prepare_bundle_temporal_context_seconds`
- wrist gap 仍主要落在 `filter_result_manipulator_motion_seconds` / `filter_result_query_depth_edge_risk_seconds`：优先做 wrist 过滤简化，而不是继续抠 shared save path
