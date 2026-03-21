# 数据分析脚本

本目录只保留当前仍有独立价值的分析脚本。

## 当前脚本

- `sample_traceforge_dataset.py`
  - 从 TraceForge case 目录中采样子集
  - 当前按 legacy case 布局工作
- `benchmark_depth_volatility_optimization.py`
  - 对比 depth volatility 优化前后在 baseline ref 与当前工作树上的运行耗时
  - 支持共享 query-frame schedule、external + wrist-like 相机成对测试、以及 save 阶段单独计时
- `benchmark_inference_variants.py`
  - 对比当前工作树内多种 inference 配置的端到端耗时与保存结果差异
  - 支持共享 query-frame schedule、process breakdown、dense 轨迹差异统计、以及可选验证图导出
- `benchmark_num_iters_sweep.py`
  - 对比 tracker `num_iters` 变体的端到端耗时与保存结果差异
  - 默认跑 `6,5,4`，并固定 `support_grid_ratio=0`，输出 process/save/profile 汇总、定量轨迹差异、可选验证图，以及可直接运行的 3D 动画命令
- `benchmark_num_iters_manifest.py`
  - 用 manifest 批量运行 `benchmark_num_iters_sweep.py`，适合维护固定测试基线 case 集
  - 当前提供 `manifests/press_one_button_demo_v5_pilot.json` 和 `manifests/press_one_button_demo_v5_median3.json`
  - 汇总 per-episode runtime、跨 episode 聚合 runtime、以及 `prepare_depth_filter_points_to_normals_seconds` 为主的波动统计
- `benchmark_wrist_filter_ablations.py`
  - 复用同一次 tracking 结果，对 wrist 过滤变体做 save-only 消融 benchmark
  - 默认只跑 `support_grid_ratio=0`，避免 support points 干扰过滤逻辑消融
  - 汇总 save 分阶段耗时、stage 级轨迹裁剪数量、与 wrist baseline 的差异，并可导出代表验证图
- `analyze_action_format.py`
- `check_action_format.py`
- `check_action_info.py`
- `analyze_dataset_structure.py`
- `analyze_first_frame_transform.py`
- `analyze_rotation_representation.py`
- `analyze_transform_relationship.py`
- `verify_transform_relationship.py`

## 文档

- [action_data_format_analysis.md](action_data_format_analysis.md)
- [docs/sample_traceforge_dataset.md](../../docs/sample_traceforge_dataset.md)

## 推荐入口

- `press_one_button_demo_v5` pilot baseline + `num_iters=5,4,3,2,1`：
  - `python scripts/data_analysis/benchmark_num_iters_manifest.py --manifest scripts/data_analysis/manifests/press_one_button_demo_v5_pilot.json --camera-names varied_camera_1,varied_camera_3 --num-iters-values 5,4,3,2,1 --baseline-num-iters 5 --support-grid-ratio 0 --warmup-runs 1 --benchmark-runs 1 --run-visual-verification`
- `press_one_button_demo_v5` median3 volatility scan：
  - `python scripts/data_analysis/benchmark_num_iters_manifest.py --manifest scripts/data_analysis/manifests/press_one_button_demo_v5_median3.json --camera-names varied_camera_1,varied_camera_3 --num-iters-values 5 --baseline-num-iters 5 --support-grid-ratio 0 --warmup-runs 1 --benchmark-runs 3`

## 历史调查脚本

以下一次性调查脚本已归档：

- `scripts/archived/investigations/2026-03/analyze_depth_volatility.py`
- `scripts/archived/investigations/2026-03/analyze_ray_artifact.py`

这些脚本保留用于追溯历史问题，不作为当前主流程的一部分。
