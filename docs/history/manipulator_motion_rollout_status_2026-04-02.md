# manipulator_motion 优化落地状态与执行结果

日期：2026-04-02

本文档用于记录当前 `manipulator_motion` / `depth_filter_workers` 这条性能线在本地工作区的真实状态，避免和 2026-03-28 / 2026-04-01 的历史结论混淆。

## 目标

回答四个问题：

1. `manipulator_motion` 到底有没有开始做
2. 当前做到什么程度
3. 哪些结论已经进代码，哪些还只是 benchmark 建议
4. 接下来按什么顺序推进

## 当前状态

### 1. `manipulator_motion` 不是“未开始”，而是“已完成向量化并已提交”

其中最关键的一项是：

- `_compute_motion_metrics_for_valid_masks()` 已经从旧的 per-track Python loop
  改成了按 mask 批处理的向量化实现
- 对应新增了 motion metric 的回归测试
- 本地测试
  `.venv/bin/python -m unittest utils.test_traj_filter_utils`
  已通过（`61` tests）
- 代码已提交：
  `f72f205 perf(filter): vectorize manipulator motion metrics`

因此，`2026-03-28` 文档里的 `P0 manipulator_motion` 不能再简单表述为“尚未开始”，更准确的说法是：

- `P0` 已完成“motion metric 计算向量化”这一步
- benchmark 与 commit 已完成
- 但 `candidate` 子集化没有继续推进

### 2. `manipulator_motion` 没有继续推进到子集化

虽然 motion metric 核心计算已经向量化，但当前
`_apply_manipulator_aware_filter()` 仍然是把全量 `world_tracks`
送进 `_compute_motion_metrics_for_valid_masks()`。

这意味着 `2026-03-28` 文档里建议的另一半优化还没完成：

- 只对 `seed_mask` / `candidate_mask` 子集计算 motion metric
- 再 scatter 回全量数组

所以当前阶段应定义为：

- 已完成：向量化
- 未继续推进：子集化

### 3. `depth_filter_workers=16` 仍然不能直接升成代码默认

`2026-04-01` 的单卡 sweep 已得到清晰结论：

- `workers_per_gpu=1` 是当前维护态 baseline
- `depth_filter_workers=16` 在单卡 H200 workload 上优于 `8`

但当前代码默认值仍然是：

- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
  `_DEFAULT_DEPTH_FILTER_WORKERS = 8`
- `scripts/batch_inference/infer.py`
  `getattr(args, "depth_filter_workers", 8)`

本轮又补了一次两卡小样本确认：

- `df8`: `SingleGPU/query = 16.52`
- `df16`: `SingleGPU/query = 17.76`

因此这里需要明确区分：

- 已进代码默认：`workers_per_gpu=1`
- 只在单卡 isolate 上成立：`depth_filter_workers=16`
- 当前仍不应改代码默认：`depth_filter_workers=8`

### 4. telemetry 与分析基础设施已完成

2026-04-01 这轮新增的能力已经属于“可直接复用”的已完成基础设施：

- `--collect_profile_stats`
- `_camera_task_profiles.jsonl`
- `_hardware_telemetry.jsonl`
- `scripts/data_analysis/analyze_batch_run_telemetry.py`
- 单卡 `workers_per_gpu` sweep 结论
- 单卡 `depth_filter_workers` sweep 结论

因此下一步不该再回到“先补观测”，而应该直接消费这些观测能力。

## 本轮执行结果

本轮实际按下面顺序完成：

1. 核对既有 `manipulator_motion` before/after benchmark 产物
2. 提交 `traj_filter_utils` 与测试修改
3. 补跑 `depth_filter_workers 8 vs 16` 的两卡小样本确认
4. 根据结果决定是否改默认值和文档

## 每一步的判定标准

### Step 1. `manipulator_motion` before/after benchmark

已核对的既有产物：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_manipulator_motion_retest_20260401_df16.md`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_manipulator_motion_retest_20260401_df16.json`

结论：

- helper 层 microbenchmark 约 `9.24x`
- `filter_result_manipulator_motion_seconds/query` 从 `0.1043` 降到 `0.0046`
- 但端到端总吞吐没有改善

因此这一步判定为：

- benchmark 通过，可以证明优化有效
- 但它不是后续最值得继续深挖的主方向

### Step 2. 提交当前优化

已完成提交：

- `utils/traj_filter_utils.py`
- `utils/test_traj_filter_utils.py`
- commit: `f72f205`

提交信息已明确为同语义性能优化，而不是 filter 语义变更。

### Step 3. 多卡确认 `depth_filter_workers`

已完成两卡小样本确认：

- `data_tmp/mjc_1000_multigpu_depth_filter_confirm_20260402_df8`
- `data_tmp/mjc_1000_multigpu_depth_filter_confirm_20260402_df16`
- 对应报告位于 `data_tmp/telemetry_reports/`

结果：

- `df8`: `SingleGPU/query = 16.52`, `Process/query = 15.43`
- `df16`: `SingleGPU/query = 17.76`, `Process/query = 15.42`

判定：

- `df16` 没有在多卡小样本上复现单卡 isolate 的收益
- 因此这一步没有通过“升默认值”的门槛

### Step 4. 改默认值并同步文档

最终处理：

- 不改 `infer.py` / `batch_infer_press_one_button_demo.py` 默认值
- 保留代码默认 `depth_filter_workers=8`
- 在文档里明确区分：
  - 单卡 isolate baseline：`16`
  - 当前代码默认 / 多卡稳妥配置：`8`
