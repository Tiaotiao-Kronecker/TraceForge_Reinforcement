# TraceForge 历史文档

本目录保存已经完成使命、但仍有参考价值的调查记录和修正文档。

这些文档的共同特点：

- 记录的是某个时间点的调查或验证结论
- 可能引用已经归档或清理的本地产物目录
- 可能引用已经移动到 `scripts/archived/investigations/2026-03/` 的脚本
- 如果与当前代码实现不一致，应以 `docs/` 下的当前文档为准

## 历史调查与修正

- [camera_extrinsics_investigation_2026-03-12.md](camera_extrinsics_investigation_2026-03-12.md)
- [camera_extrinsics_fix_validation_2026-03-12.md](camera_extrinsics_fix_validation_2026-03-12.md)
- [camera_extrinsics_old_flow_example_2026-03-12.md](camera_extrinsics_old_flow_example_2026-03-12.md)
- [droid_h5_w2c_single_case_validation_2026-03-12.md](droid_h5_w2c_single_case_validation_2026-03-12.md)
- [branch_note_sim360_extrinsics_fix_2026-03-12.md](branch_note_sim360_extrinsics_fix_2026-03-12.md)
- [ray_artifact_investigation_2026-03-16.md](ray_artifact_investigation_2026-03-16.md)
- [repo_root_artifact_audit_2026-03-17.md](repo_root_artifact_audit_2026-03-17.md)
- [depth_volatility_benchmark_validation_2026-03-19.md](depth_volatility_benchmark_validation_2026-03-19.md)
- [process_inference_bottleneck_analysis_2026-03-19.md](process_inference_bottleneck_analysis_2026-03-19.md)
- [save_timing_alignment_status_2026-03-21.md](save_timing_alignment_status_2026-03-21.md)
- [process_num_iters_sweep_2026-03-21.md](process_num_iters_sweep_2026-03-21.md)
- [processing_speed_optimization_summary_2026-03-21.md](processing_speed_optimization_summary_2026-03-21.md)
- [processing_speed_reassessment_2026-03-28.md](processing_speed_reassessment_2026-03-28.md)
- [pick_place_batch_telemetry_rollout_status_2026-04-01.md](pick_place_batch_telemetry_rollout_status_2026-04-01.md)
- [pick_place_single_gpu_workers_per_gpu_sweep_2026-04-01.md](pick_place_single_gpu_workers_per_gpu_sweep_2026-04-01.md)
- [pick_place_single_gpu_depth_filter_workers_sweep_2026-04-01.md](pick_place_single_gpu_depth_filter_workers_sweep_2026-04-01.md)
- [manipulator_motion_rollout_status_2026-04-02.md](manipulator_motion_rollout_status_2026-04-02.md)
- [pick_place_num_iters_sweep_2026-04-02.md](pick_place_num_iters_sweep_2026-04-02.md)
- [interface_surface_simplification_plan_2026-03-28.md](interface_surface_simplification_plan_2026-03-28.md)
- [press_one_button_demo_v5_camera3_tail_empty_root_cause_2026-03-24.md](press_one_button_demo_v5_camera3_tail_empty_root_cause_2026-03-24.md)
- [pick_place_wrist_pick_place_history_2026-03-26.md](pick_place_wrist_pick_place_history_2026-03-26.md)
- [drift_patch_classification_analysis_2026-03-28.md](drift_patch_classification_analysis_2026-03-28.md)

## 当前保留的本地对比产物

以下目录因仍有明确对比意义而保留在本机，但它们不是仓库接口的一部分。
为保持仓库根目录整洁，这些输出现统一收纳在 `data_tmp/` 下：

- `data_tmp/ray_artifact_investigation/2026-03-16`
- `data_tmp/traj_filter_review/2026-03-17/wrist_vs_manipulator_round2`
- `data_tmp/traj_filter_review/2026-03-17/external_vs_external_manipulator`
- `data_tmp/traj_filter_review/2026-03-17/external_vs_external_manipulator_v2`
- `data_tmp/history_outputs/outputs_press_one_button_demo`
- `data_tmp/history_outputs/outputs_press_one_button_demo_c2w`
- `data_tmp/history_outputs/outputs_press_one_button_demo_v1_extrinsics_compare_2026-03-12`
- `data_tmp/history_outputs/outputs_droid_fixed_h5_w2c_case_2026-03-12`
- `data_tmp/history_outputs/outputs_droid_subset_fullparams_2026-03-12`
- `data_tmp/depth_volatility_benchmarks/20260319_external_wrist_keep`
- `data_tmp/mcap_v1_wrist_pick_place_10cases_20260325`
- `data_tmp/mcap_v1_wrist_pick_place_10cases_20260325_visualization`
- `data_tmp/mcap_v1_wrist_pick_place_10cases_20260326_rerun_gpu`
- `data_tmp/mcap_v1_wrist_pick_place_10cases_20260326_rerun_gpu_visualization`

## 根目录遗留项说明

- 根目录曾有一个被 git 跟踪的样例数据目录 `00001/`
- 它不属于当前主流程接口，也没有显式路径级硬依赖
- 该目录已按人工决策删除，相关判断保留在审计记录中

根目录 ignored 的零散本地产物则按清理规则直接移除，例如 rerun 日志和
采样列表文件
