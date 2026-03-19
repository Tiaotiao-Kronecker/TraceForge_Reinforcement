# 深度波动优化 benchmark 验证

日期：2026-03-19

## 背景

2026-03-19 这轮实现已经不再把 `P1` 与 `P4` 作为独立可切换路径维护，而是把两者收敛为当前维护态的统一 hot path：

- camera 级一次性构建 `high_volatility_mask`
- volatility 统计只覆盖当前 camera 下所有 query 实际访问到的位置
- `evaluate_temporal_depth_consistency()` 复用预计算的 temporal compare context

对应实现见：

- `scripts/batch_inference/infer.py`
- `utils/traj_filter_utils.py`

## benchmark 设计

对比目标：

- baseline ref：`8f9060d`
- current：`feat/default-source-ref-and-wrist-top95` 当前工作树

工作负载：

- 数据集：`/data2/yaoxuran/press_one_button_demo_v1`
- episode：`episode_00000`
- cameras：
  - `varied_camera_1` 作为 external
  - `varied_camera_3` 作为 wrist-like
- 共享 query-frame schedule
- measured runs：`1`
- warmup runs：`0`
- device：`cuda:1`

benchmark 工具：

- `scripts/data_analysis/benchmark_depth_volatility_optimization.py`

本地保留产物：

- benchmark 汇总：`data_tmp/depth_volatility_benchmarks/20260319_external_wrist_keep/benchmark_summary.md`
- benchmark 原始 json：`data_tmp/depth_volatility_benchmarks/20260319_external_wrist_keep/benchmark_results.json`
- baseline/current 可视化产物：`data_tmp/depth_volatility_benchmarks/20260319_external_wrist_keep/artifacts/`

这些产物只用于本地对比，不属于仓库接口的一部分。

## 实测结果

| Camera | Profile | Baseline Save (s) | Current Save (s) | Save Speedup | Baseline Total (s) | Current Total (s) | Total Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `90.974` | `7.323` | `12.423x` | `159.055` | `75.899` | `2.096x` |
| `varied_camera_3` | `wrist_manipulator_top95` | `90.661` | `13.785` | `6.577x` | `159.967` | `83.411` | `1.918x` |

同时可以看到：

- `process_single_video()` 基本没有变快
- 提速几乎全部来自 `save_structured_data()` 热路径
- external 与 wrist-like 两类 camera 都受益，但 wrist-like 仍比 external 更慢

## 结果解释

这说明本轮 unified 方案达到了预期目标：

- 原本 save 阶段里最重的 volatility 统计冗余已经被显著削减
- 当前端到端主瓶颈已经不再是 `compute_depth_volatility_map()` 这一类 save 阶段 CPU 统计
- 下一阶段应该把分析重点转移到 `process_single_video()`，尤其是 tracker inference 主链路

从代码结构看，本轮收益主要来自三点：

1. `compute_accessed_high_volatility_mask(...)`
   - 不再对整张图做全分辨率 volatility 统计
2. `prepare_temporal_depth_consistency_context(...)`
   - 将 reprojection / compare-mask 预处理前移并复用
3. camera 级一次性 `high_volatility_mask`
   - 不再在每个 sample 内重复 threshold / dense volatility 相关工作

## 结论

截至 2026-03-19：

- 深度波动图已经不再是当前工作负载下的主瓶颈
- 当前最值得继续优化的对象是 `process_single_video()`
- 更具体地说，应继续拆分：
  - `prepare_inputs(...)`
  - `utils.inference_utils::inference(...)`
  - tracker 模型主体 forward

下一步分析文档应单独回答：

- `process` 的主要耗时是否已经集中到 3D tracking
- 其中是前处理、模型主体，还是结果回传在主导耗时
