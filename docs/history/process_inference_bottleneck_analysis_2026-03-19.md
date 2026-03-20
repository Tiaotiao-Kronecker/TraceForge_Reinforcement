# process / inference 路径瓶颈分析

日期：2026-03-19

## 背景

2026-03-18 的上一轮分析见 `docs/history/depth_volatility_bottleneck_analysis_2026-03-18.md`，当时的结论是：

- 最大瓶颈是 `compute_depth_volatility_map()`
- tracker inference 还是第二瓶颈

2026-03-19 完成 unified depth-volatility hot path 优化后，基准验证见
`docs/history/depth_volatility_benchmark_validation_2026-03-19.md`：

- `varied_camera_1 / external`：save `90.974s -> 7.323s`
- `varied_camera_3 / wrist_manipulator_top95`：save `90.661s -> 13.785s`

这说明 save 阶段的深度波动统计已经不再是主瓶颈。本页的目标是继续回答：

- 当前主瓶颈是否已经转移到 `process_single_video()`
- 如果是，它更具体地落在 `prepare_inputs(...)`、`inference(...)`，还是 tracker 模型主体 forward

## Profiling 工作负载

当前 profiling 基于当前维护态实现，工作负载与 benchmark 保持一致：

- 数据集：`/data2/yaoxuran/press_one_button_demo_v1`
- episode：`episode_00000`
- cameras：
  - `varied_camera_1`，`traj_filter_profile=external`
  - `varied_camera_3`，`traj_filter_profile=wrist_manipulator_top95`
- shared query schedule：
  - `data_tmp/depth_volatility_benchmarks/20260319_external_wrist_keep/_shared/query_frame_schedule_v1_b4c212783657.json`
- device：`cuda:1`
- 每个 camera：
  - `frame_count = 50`
  - `query_frame_count = 5`
  - `segment_lengths_sum = 80`

原始 profiling 结果保存于：

- `data_tmp/process_breakdown_profiles/20260319_current_profile/process_breakdown_current_episode_00000.json`

## Profiling 方法

这次只拆 `process_single_video()`，不包含 `save_structured_data()`。

拆分项包括：

- `load_rgb_seconds`
- `load_depth_seconds`
- `depth_pose_wrapper_seconds`
- `prepare_inputs_seconds`
- `tracker_inference_total_seconds`
- `tracker_model_forward_seconds`
- `get_grid_queries_seconds`
- `inference_with_grid_seconds`
- `empty_cache_seconds`
- `process_other_seconds`

其中：

- `tracker_model_forward_seconds` 只包 tracker 模型 `model(...)` 主体
- `get_grid_queries_seconds` 只包 support grid query 构造
- `tracker_inference_total_seconds` 表示整个 `utils.inference_utils::inference(...)`
- `process_other_seconds` 是 `process_total_seconds` 减去上述显式统计项后的余量

## 结果

| Camera | Profile | Process Total (s) | Prepare Inputs (s) | Tracker Inference (s) | Tracker Model Forward (s) | Load RGB (s) | Load Depth (s) | Depth Wrapper (s) | Other (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `varied_camera_1` | `external` | `69.009` | `20.355` | `45.194` | `43.808` | `1.709` | `0.313` | `0.314` | `1.092` |
| `varied_camera_3` | `wrist_manipulator_top95` | `69.017` | `20.545` | `45.778` | `44.675` | `1.216` | `0.105` | `0.301` | `1.032` |
| `平均` | `-` | `69.013` | `20.450` | `45.486` | `44.241` | `1.462` | `0.209` | `0.308` | `1.062` |

平均占比：

- `tracker_inference_total_seconds / process_total_seconds = 65.9%`
- `tracker_model_forward_seconds / process_total_seconds = 64.1%`
- `tracker_model_forward_seconds / tracker_inference_total_seconds = 97.3%`
- `prepare_inputs_seconds / process_total_seconds = 29.6%`

同时还有两个直接结论：

- `get_grid_queries_seconds` 平均只有 `0.130s`，不是当前主要问题
- `load_rgb + load_depth + depth_pose_wrapper` 合计平均只有 `1.978s`，也不是当前主要问题

## 与端到端 benchmark 的对应关系

结合 `docs/history/depth_volatility_benchmark_validation_2026-03-19.md` 里的 current 总耗时：

- `varied_camera_1 / external`：total `75.899s`
- `varied_camera_3 / wrist_manipulator_top95`：total `83.411s`

再看这里的 `process_total_seconds`：

- `varied_camera_1`：`69.009s`
- `varied_camera_3`：`69.017s`

说明在当前版本下：

- external case 里，`process` 已经约占端到端总耗时的 `90.9%`
- wrist-like case 里，`process` 已经约占端到端总耗时的 `82.7%`

所以端到端主瓶颈已经明确从 save 阶段转移到了 `process_single_video()`。

## 代码路径解释

### 1. `process_single_video()` 的主耗时位置

主循环位于 `scripts/batch_inference/infer.py:1244-1372`：

- `prepare_inputs(...)` 调用在 `scripts/batch_inference/infer.py:1308-1318`
- tracker `inference(...)` 调用在 `scripts/batch_inference/infer.py:1323-1335`
- 结果 `.cpu()` 回传在 `scripts/batch_inference/infer.py:1357-1363`

profiling 结果说明：

- `.cpu()` 回传和其余零散逻辑都只落在 `process_other_seconds` 这一小块余量里
- 真正的大头是 `prepare_inputs(...)` 和 tracker inference

### 2. `prepare_inputs(...)` 是明确的第二瓶颈

实现位于 `scripts/batch_inference/infer.py:1663-1695`。

它里面最重的部分是：

- `ThreadPoolExecutor` 并行跑每帧 `_filter_one_depth(...)`
  - `scripts/batch_inference/infer.py:1681-1686`
- 然后执行 `prepare_query_points(...)`
- 最后把 `video/depth/intrinsics/extrinsics/query_point` 全部搬到 GPU
  - `scripts/batch_inference/infer.py:1688-1693`

平均 `20.450s` 的耗时说明：

- 当前每个 tracking segment 都在重复做一遍 depth filter、query-point 准备和 host-to-device 拷贝
- 即使 save 阶段已经优化完，`prepare_inputs(...)` 仍然足够重，已经成为 process 路径里的第二优化优先级

### 3. `inference(...)` 内真正重的是 tracker 模型 forward

`utils.inference_utils::inference(...)` 位于 `utils/inference_utils.py:118-186`。

这里的几段逻辑可以分开看：

- depth ROI 的 quartile 统计：
  - `utils/inference_utils.py:132-141`
- `_inference_with_grid(...)` 调用：
  - `utils/inference_utils.py:149-159`
- backward tracking 分支：
  - `utils/inference_utils.py:161-182`
- visibility sigmoid threshold：
  - `utils/inference_utils.py:184-186`

当前 workload 下，调用方固定传的是：

- `bidrectional=False`
  - `scripts/batch_inference/infer.py:1334`

所以 backward 路径根本不是当前瓶颈。

`_inference_with_grid(...)` 位于 `utils/inference_utils.py:45-77`：

- support grid query 构造在 `utils/inference_utils.py:58-61`
- tracker 模型主体调用在 `utils/inference_utils.py:65-74`

从 profiling 看：

- `get_grid_queries_seconds` 平均只有 `0.130s`
- `tracker_model_forward_seconds` 平均却有 `44.241s`

因此当前 `inference(...)` 的耗时几乎全部来自 tracker 模型主体 forward，而不是 support grid query 构造，也不是后处理。

### 4. tracker 模型内部的热点仍然是多窗口、多迭代的 3D tracking 主链

tracker 主体位于 `models/point_tracker_3d.py`。

当前最值得继续盯的路径是：

- RGB 编码：
  - `models/point_tracker_3d.py:573-578`
- `batch_unproject(...)` 与 shared context 准备：
  - `models/point_tracker_3d.py:587-604`
- streaming window 主循环：
  - `models/point_tracker_3d.py:606-707`
- 每个 window 内的 `num_iters` 迭代：
  - `models/point_tracker_3d.py:267-297`
- 每次迭代里的：
  - `corr_processor(...)`
    - `models/point_tracker_3d.py:183-188`
  - `point_updater(...)`
    - `models/point_tracker_3d.py:224-235`

而 `corr_processor` 的核心 KNN / 多层 correlation 计算位于：

- `models/corr_features/knn_feature_4d_optimized.py:543-586`

从这条调用链看，当前所谓“3D tracking 瓶颈”不是一个泛泛的概念，而是很具体地落在：

- tracker model forward
- 其内部的 multi-window streaming
- 每个 window 下的 multi-iter correlation + updater

## 结论

截至 2026-03-19，当前问题已经比较清楚：

- 是的，主瓶颈已经转移到 3D tracking
- 更精确地说，当前主瓶颈是 `process_single_video()` 内的 tracker 模型 forward
- `prepare_inputs(...)` 是第二瓶颈
- `get_grid_queries(...)`、RGB/depth 加载、depth/pose wrapper、CPU 回传都不是优先优化对象

如果只看当前维护态 workload，下一轮优化优先级应是：

1. 减少 tracker 主体工作量
   - 优先看 overlapping segment / overlapping window 之间是否能复用中间结果
   - 其次看 `num_iters`、support grid 密度、query 数是否还能继续压
2. 压缩 `prepare_inputs(...)` 的重复开销
   - 尽量避免对重叠 segment 重复执行 `_filter_one_depth(...)`
   - 尽量避免重复构造 query-point 和重复 host-to-device 拷贝
3. 如果要继续深挖 tracker 内部
   - 优先看 `corr_processor` 的 KNN / correlation 计算
   - 再看 `point_updater` 与 window 迭代次数

换句话说，下一阶段不该再把主要精力放在 volatility guidance 或写盘上，而应该直接盯 `process -> inference -> tracker model forward` 这条主链。
