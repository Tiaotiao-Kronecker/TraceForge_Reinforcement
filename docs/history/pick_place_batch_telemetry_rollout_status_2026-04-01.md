# pick_place 批量 telemetry 落地状态与后续实验计划

日期：2026-04-01

本文档用于保存这轮 `pick_place` 批量性能排查的上下文，避免会话中断后丢失：

1. 基线数据来自哪里
2. 当前主口径是什么
3. 规划中的哪些项已经执行
4. 哪些项还没执行
5. 用户已经明确冻结了哪些变量

## 基线与口径

本轮基线来自两次已有结果：

- full: `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_full_wrist_pick_place_no_heatmap`
- smoke: `/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_smoke_wrist_pick_place_no_heatmap`

从 full 的 `_batch_run_summary.json` 和 `_camera_task_metrics.jsonl` 抽样得到：

- 总任务数：`3000` camera tasks
- 总 query frame 数：`25266`
- 集群 wall-clock：`48966.02s`
- 物理 GPU：`8`
- 实际并发：每张卡最大重叠 `4` 个 task，即 `4 workers/GPU`

本轮统一采用三种速度口径：

- 集群吞吐口径：`wall_clock / total_queries = 1.94s/query`
- 主口径，单卡 H200 归一化：`wall_clock * physical_gpu_count / total_queries = 15.50s/query/H200`
- 内部并发槽位口径：`sum(task_total_seconds) / total_queries = 61.69 slot-seconds/query`

smoke 的 1 卡基线为：

- `511.34s / 24 queries = 21.31s/query/H200`

这个值只作为单卡 sanity check，不作为最终结论。

## 已确认的现状判断

在开始补观测之前，已经能确认：

- 端到端瓶颈在 `process`，不是 `save`
- `varied_camera_3 / wrist_pick_place_no_heatmap` 比两个 external 相机更慢
- 共享队列和动态调度不是第一瓶颈，32 个 worker 槽位利用率已接近打满

但当时还不能严谨回答瓶颈属于 GPU、CPU 还是 IO，因为缺：

- `workers_per_gpu` / `worker_slot_count` 级别的 run summary 元数据
- `infer.py` 细分 `profile_stats` / `save_profile_stats` 的落盘
- GPU/CPU/磁盘的周期硬件采样

## 用户已确认的约束

本轮先按“先补观测，再做不改语义实验”的顺序推进。

用户已经明确要求，下面两项先不要动：

- `future_len: 32 -> 24`
- `grid_size: 80 -> 40`

因此当前阶段不做这两项 sweep，也不把它们纳入第一轮实验矩阵。

## 原始规划

### 1. 先补观测，不改语义

- 在 batch summary 中新增 `workers_per_gpu`、`worker_slot_count`、GPU 型号、主机名
- 给 batch 入口加 `--collect_profile_stats`，把 `infer.py` 已有细分时延落到 JSONL
- 增加硬件采样，记录 GPU utilization / memory / power，以及 CPU iowait / 磁盘吞吐
- 增加统一分析脚本，输出 `wall/query`、`单卡 H200/query`、按 `camera/profile/gpu/worker` 分组统计

### 2. 做不改语义的性能实验

- 单卡 H200 隔离基线：固定一个 episode 子集，扫 `workers_per_gpu = 1/2/3/4`
- 8 卡吞吐实验：同一子集上扫 `workers_per_gpu = 1/2/4`
- CPU 侧实验：把 `_DepthFilterRuntime(max_workers=8)` 暴露成参数，扫 `4/8/16`

### 3. 做改语义的性能-质量权衡实验

- `num_iters: 5 -> 4 -> 3`
- `support_grid_ratio: 0.8 -> 0.6 -> 0.4`
- `query_prefilter_mode: off -> profile_aware_static_v1`
- `future_len: 32 -> 24`
- `grid_size: 80 -> 40`

### 4. 固定语义变化的评估口径

- 同一批 episode，同一份 shared query-frame schedule
- 比较 empty sample rate
- 比较 valid trajectory count 分布
- 比较 `traj_valid_mask` 一致率 / Jaccard
- 对可对齐样本比较共同轨迹上的 3D 位置偏差
- 对 wrist / pick_place 额外比较 `traj_pick_place_*`、`traj_supervision_*`
- 抽 top-N 差异最大的 sample 做 3D 可视化复核

## 当前执行状态

以下状态基于当前未提交工作区内容。

### 已执行

#### 1. batch telemetry 元数据已补齐

已在 `scripts/batch_inference/batch_infer_press_one_button_demo.py` 中补充：

- `_camera_task_profiles.jsonl`
- `_hardware_telemetry.jsonl`
- `workers_per_gpu`
- `worker_slot_count`
- `telemetry_gpu_ids`
- `host_name`
- `gpu_info`
- `depth_filter_workers`

同时单 task record 中也写入了：

- `worker_label`
- `worker_index`
- `gpu_slot_index`
- `gpu_slot_count`

#### 2. `infer.py` 的 profile stats 已暴露到 batch 路径

已在 `scripts/batch_inference/infer.py` 中补充：

- `--collect_profile_stats`
- `--depth_filter_workers`

并将 `depth_filter_workers` 传入 `_DepthFilterRuntime(max_workers=...)`。

#### 3. batch 路径已能落盘细粒度 profile JSONL

当前 batch 路径已经会在开启 `--collect_profile_stats` 时，把：

- `profile_stats`
- `save_profile_stats`
- `per_query_save_seconds`
- `scene_finalize_overhead_seconds`

写入 `_camera_task_profiles.jsonl`。

#### 4. 硬件采样器已接入

当前 batch 路径已经支持：

- 通过 `nvidia-smi` 采样 GPU `utilization.gpu` / `utilization.memory` / `memory.used` / `memory.total` / `power.draw`
- 通过 `/proc/stat` 与 `/proc/diskstats` 估算 CPU iowait 与磁盘读写吞吐

采样结果会写入 `_hardware_telemetry.jsonl`。

#### 5. 统一分析脚本已新增

已新增 `scripts/data_analysis/analyze_batch_run_telemetry.py`，用于汇总：

- run overview
- by camera
- by camera/profile
- by GPU
- by worker
- process profile
- save profile
- hardware summary

并支持直接输出 Markdown 与 JSON 报告。

#### 6. 文档和轻量测试已补

已更新：

- `CLAUDE.md`
- `scripts/batch_inference/BATCH_INFERENCE_GUIDE.md`
- `scripts/batch_inference/test_infer_cli_surface.py`
- `scripts/batch_inference/test_press_one_button_demo_utils.py`
- `scripts/data_analysis/test_analyze_batch_run_telemetry_utils.py`

### 已部分执行但还没完成闭环

#### 1. 本地验证已经补到真实输出闭环

当前已完成：

- `.venv/bin/python -m unittest scripts.batch_inference.test_infer_cli_surface scripts.batch_inference.test_press_one_button_demo_utils scripts.data_analysis.test_analyze_batch_run_telemetry_utils`
- 共 `26` 个轻量测试通过
- 用真实 batch 输出目录重新生成并核对了 `w1/w2/w3/w4` 的 telemetry 报告
- `analyze_batch_run_telemetry.py` 已修正，只把 timing key 汇总进 `Sec/query`，不再把 frame 计数或阈值误报成秒数

仍未完成：

- 更大范围的 CLI / 集成测试矩阵
- 质量评估与可视化复核

### 尚未执行

#### 1. 单卡 H200 隔离实验

已经完成：

- `workers_per_gpu = 1/2/3/4`

当前可直接确认：

- `workers_per_gpu=1/2/3/4` 的单卡吞吐都稳定在 `15.90~15.91 s/query/H200`
- 增大 resident workers 只会把 `slot_seconds/query`、GPU util 和显存占用继续往上推
- 当前单卡后续 baseline 应固定回 `workers_per_gpu=1`

#### 2. 8 卡吞吐实验

还没有开始扫：

- `workers_per_gpu = 1/2/4`

#### 3. CPU 侧 `_DepthFilterRuntime` 实验

虽然 `depth_filter_workers` 参数已经暴露，但还没有开始扫：

- `depth_filter_workers = 4/8/16`

#### 4. 改语义实验

以下仍未开始：

- `num_iters: 5 -> 4 -> 3`
- `support_grid_ratio: 0.8 -> 0.6 -> 0.4`
- `query_prefilter_mode: off -> profile_aware_static_v1`

以下两项当前明确冻结，不进入下一轮：

- `future_len: 32 -> 24`
- `grid_size: 80 -> 40`

#### 5. 质量评估与可视化复核

以下也都尚未开始：

- empty sample rate 对比
- valid trajectory count 分布对比
- `traj_valid_mask` 一致率 / Jaccard
- 共同轨迹的 3D 偏差对比
- `traj_pick_place_*` / `traj_supervision_*` 对比
- top-N 差异 sample 的 3D 可视化复核

## 当前建议的恢复点

如果后续继续执行，本轮最合理的接续顺序是：

1. 固定单卡 baseline 为 `workers_per_gpu=1`
2. 在同一固定子集上先做 `depth_filter_workers = 4/8/16`
3. 如确实还想评估并发策略，再做 8 卡 `workers_per_gpu=1` 基线，`>1` 降为低优先级
4. 再进入 `num_iters` / `support_grid_ratio` / `query_prefilter_mode` 的语义变化实验
5. 最后补质量评估与 top-N 可视化复核

## 2026-04-01 恢复执行记录

本节记录本次会话真正已经落地的运行结果，避免后续重复确认。

### 运行环境

- 仓库内可用环境是 `.venv`
- `.venv/bin/python` 可正常导入 `numpy`、`torch`
- 在无沙箱 GPU 访问下，`torch.cuda.is_available() == True`
- 当前主机可见 `8 x NVIDIA H200`

### 已完成的 smoke run

已实际运行：

- 输出根：`/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_smoke_telemetry_20260401_w1`
- 配置：`episode=00000`、`gpu_id=0`、`workers_per_gpu=1`
- 额外观测：`--collect_profile_stats --hardware_telemetry_interval_sec 15 --depth_filter_workers 8`

结果：

- `wall_clock_seconds = 390.8515`
- `24 queries`
- `16.29 s/query/H200`
- `slot_seconds/query = 15.18`
- `process/query = 14.99`
- `save/query = 0.19`
- GPU 平均利用率约 `79.9%`
- GPU 平均显存占用约 `17.64 GiB`
- GPU 平均功耗约 `333.99 W`

对应分析报告已写到：

- `data_tmp/telemetry_reports/mjc_1000_step1_smoke_telemetry_20260401_w1.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_smoke_telemetry_20260401_w1.md`

### 已固定的单卡 sweep 子集

已新增固定子集清单：

- `scripts/data_analysis/manifests/mjc_1000_step1_single_gpu_workers_sweep_20260401.txt`

内容为：

- `00000`
- `00001`

这个子集共有：

- `2` 个 episode
- `6` 个 camera tasks
- `51` 个 query frames

### 已完成的单卡 `workers_per_gpu=1` 基线

已实际运行：

- 输出根：`/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1`
- 配置：固定子集、`gpu_id=0`、`workers_per_gpu=1`

结果：

- `wall_clock_seconds = 810.8436`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 15.26`
- `process/query = 15.06`
- `save/query = 0.20`
- GPU 平均利用率约 `84.93%`
- GPU 平均显存占用约 `17.56 GiB`
- GPU 平均功耗约 `338.18 W`

主要细分热点：

- `tracker_model_forward_seconds/query ≈ 13.43`
- `prepare_inputs_seconds/query ≈ 0.92`
- `prepare_depth_filter_seconds/query ≈ 0.89`
- `save_total_seconds/query ≈ 0.20`

对应分析报告已写到：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w1.md`

### 已完成的单卡 `workers_per_gpu=2` 对照

已实际运行：

- 输出根：`/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w2`
- 配置：固定子集、`gpu_id=0`、`workers_per_gpu=2`

结果：

- `wall_clock_seconds = 811.0395`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 30.82`
- `process/query = 30.62`
- `save/query = 0.20`
- GPU 平均利用率约 `94.35%`
- GPU 平均显存占用约 `27.38 GiB`
- GPU 平均功耗约 `337.65 W`

与 `workers_per_gpu=1` 对比可直接确认：

- 总 wall clock 基本没有下降
- 单 task 的 `slot_seconds/query` 近似翻倍
- GPU 利用率更高，但没有换来更高吞吐

这说明在当前单卡 H200 上，`workers_per_gpu=2` 已经把 GPU 推到更忙，但端到端吞吐没有提升；瓶颈更像是 tracker forward 的单卡算力竞争，而不是 CPU/IO 空闲未被吃满。

### 已完成的单卡 `workers_per_gpu=3` 对照

已实际运行：

- 输出根：`/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3`
- 配置：固定子集、`gpu_id=0`、`workers_per_gpu=3`

结果：

- `wall_clock_seconds = 811.1283`
- `51 queries`
- `15.90 s/query/H200`
- `slot_seconds/query = 44.95`
- `process/query = 44.75`
- `save/query = 0.20`
- GPU 平均利用率约 `95.70%`
- GPU 平均显存占用约 `35.85 GiB`
- GPU 平均功耗约 `342.02 W`

主要细分热点：

- `tracker_model_forward_seconds/query ≈ 42.48`
- `prepare_inputs_seconds/query ≈ 1.01`
- `prepare_depth_filter_seconds/query ≈ 0.96`
- `save_total_seconds/query ≈ 0.20`

对应分析报告已写到：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w3.md`

与 `workers_per_gpu=2` 对比可直接确认：

- 总 wall clock 依旧没有下降
- 单 task 的 `slot_seconds/query` 再次显著上升
- GPU / 显存更忙，但没有换来更高吞吐

这说明 `workers_per_gpu=3` 只是进一步放大了同卡 tracker forward 竞争。

### 已完成的单卡 `workers_per_gpu=4` 对照

已实际运行：

- 输出根：`/DATA/disk2/wangchen/projects/traceforge_runs/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4`
- 配置：固定子集、`gpu_id=0`、`workers_per_gpu=4`

结果：

- `wall_clock_seconds = 811.2676`
- `51 queries`
- `15.91 s/query/H200`
- `slot_seconds/query = 52.30`
- `process/query = 52.10`
- `save/query = 0.20`
- GPU 平均利用率约 `96.26%`
- GPU 平均显存占用约 `41.60 GiB`
- GPU 平均功耗约 `339.95 W`

主要细分热点：

- `tracker_model_forward_seconds/query ≈ 49.58`
- `prepare_inputs_seconds/query ≈ 0.99`
- `prepare_depth_filter_seconds/query ≈ 0.94`
- `save_total_seconds/query ≈ 0.20`

对应分析报告已写到：

- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4.json`
- `data_tmp/telemetry_reports/mjc_1000_step1_single_gpu_workers_sweep_20260401_w4.md`

与 `workers_per_gpu=3` 对比可直接确认：

- 总 wall clock 仍然贴着 `~811s`
- 单 task 的 `slot_seconds/query` 继续上升
- GPU 更忙、显存更高，但还是没有更高吞吐

这说明在当前单卡 H200 上，把 resident workers 堆到 `4` 仍然是纯竞争放大，不是 throughput 优化。

### 当前尚未完成的项

单卡 sweep 已经完成：

- `workers_per_gpu=1/2/3/4`

当前不再建议继续追加更高的单卡 `workers_per_gpu`。

CPU 侧 sweep 还没开始：

- `depth_filter_workers=4/8/16`

语义变化实验也还没开始：

- `num_iters=5/4/3`
- `support_grid_ratio=0.8/0.6/0.4`
- `query_prefilter_mode=off/profile_aware_static_v1`

仍然冻结：

- `future_len: 32 -> 24`
- `grid_size: 80 -> 40`

## 当前结论

截至 2026-04-01，这轮规划里“补观测”的主体实现已经写完，单卡 `workers_per_gpu=1/2/3/4` 的第一轮真实 workload sweep 也已经完成。

因此，当前最准确的状态描述是：

- 埋点与分析脚本：已执行，并已用真实输出闭环验证
- 轻量测试与真实 batch 报告：已完成
- 单卡 `workers_per_gpu` sweep：已完成
- 8 卡 / CPU sweep：未执行
- 改语义实验：未执行
- 推荐后续 baseline：`workers_per_gpu=1`
- `future_len` 与 `grid_size` 调整：当前冻结
