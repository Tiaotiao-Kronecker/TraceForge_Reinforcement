# xperience inference-side query depth stabilization 与 tracker/local 交互排查（2026-04-10）

## 目的

在 geometry-only 控制变量实验之后，本轮继续做两件事：

1. 把 `depth_temporal_median_world_v1` 从 geometry-only 前移到真实 inference 入口
   只改 query seed 的 3D 初始化，看 downstream 4D 轨迹是否同步改善。

2. 对 `04234` 继续拆分
   把 tracker 的实际 fixed-view 漂移，与 geometry-only 本来就会有的漂移分开统计，确认它到底是不是 tracker / 局部背景交互主导。

## 本轮代码改动

### inference-side query depth stabilization

新增 CLI：

- `--query_depth_stabilization_mode`
- `--query_depth_stabilization_reproj_tol_px`
- `--query_depth_stabilization_min_support`
- `--query_depth_stabilization_min_query_depth_m`
- `--query_depth_stabilization_min_border_dist_px`

默认：

- `query_depth_stabilization_mode=off`

当前实验模式：

- `temporal_median_world_v1`

实现位置：

- [infer.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/batch_inference/infer.py)
- [batch_infer_press_one_button_demo.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/batch_inference/batch_infer_press_one_button_demo.py)

核心行为：

- 保持模型内部 dense point cloud 逻辑不变
- 只在送入 tracker 前，替换 query seed 的 `(t,X,Y,Z)` 初始化
- 替换后的 per-track 元数据也写入 sample NPZ：
  - `traj_query_depth_temporal_replace_mask`
  - `traj_query_depth_temporal_support_count`
  - `traj_query_depth_temporal_anchor_mask`
  - `traj_query_depth_temporal_delta_world_m`
  - `query_depth_stabilization_mode`

### inference-side dense depth stabilization

新增 CLI：

- `--dense_depth_stabilization_mode`
- `--dense_depth_stabilization_radius`
- `--dense_depth_stabilization_min_support`

默认：

- `dense_depth_stabilization_mode=off`

当前实验模式：

- `temporal_median_reproject_v1`

核心行为：

- 在 `prepare_inputs(...)` 之前，对整个 `depth_obs` segment 做处理
- 对每个 target frame，把时序邻域帧的 depth 先 lift 到 world，再按当前外参重投影回 target frame
- 对同一像素位置做 temporal median
- 当支持数足够时，用这个 median depth 替换 target frame 原始 depth

写入 sample NPZ 的 per-frame 元数据：

- `dense_depth_stabilization_mode`
- `dense_depth_temporal_replace_ratio`
- `dense_depth_temporal_replace_count`
- `dense_depth_temporal_support_count_median`
- `dense_depth_temporal_support_count_p95`
- `dense_depth_temporal_delta_depth_median_m`
- `dense_depth_temporal_delta_depth_p95_m`

### tracker/local interaction 诊断

新增：

- [tracker_geometry_interaction_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/tracker_geometry_interaction_diagnostics.py)
- [export_tracker_geometry_interaction_report.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/data_analysis/export_tracker_geometry_interaction_report.py)
- [test_tracker_geometry_interaction_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/test_tracker_geometry_interaction_diagnostics.py)

它的核心思想是：

- 先算 geometry-only 的 fixed-view final drift
- 再算 tracker 实际输出的 fixed-view final drift
- 两者做差，得到 `excess drift`

如果：

- geometry-only 很稳
- 但 tracker drift 很大
- 而且 excess drift 很大

那就更像 tracker / 局部背景交互问题，而不是 source geometry 本来就在晃。

## `00435` inference-side 验证

### 运行配置

对 `00435` 做了真实 inference rerun：

- `grid80`
- `notrim`
- `filter_level=none`
- `query_prefilter_mode=off`
- `query_depth_stabilization_mode=temporal_median_world_v1`

输出目录：

- `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep/trajectory_dense_none_grid80_notrim_qdepthtmw1/stereo_left`

说明：

- 这次 direct `infer.py` 没显式传 `--camera_name stereo_left`
- 初次输出目录名沿用了默认 `hand_camera`
- 后续已手动改回 `stereo_left`

### seed 替换规模

`q0`:

- replaced seeds = `5452 / 6400`
- support median = `9`
- support p95 = `16`
- `delta_world_m` p95 = `0.094 m`

`q4`:

- replaced seeds = `5324 / 6400`
- support median = `9`
- support p95 = `12`
- `delta_world_m` p95 = `0.115 m`

说明：

- 这不是只改了极少数点
- 它确实对大部分 query seed 的 3D 初始化产生了实质影响

### downstream fixed-view wobble 对比

baseline:

- 结果文件：`/tmp/scene_wobble_00435_baseline.json`

new:

- 结果文件：`/tmp/scene_wobble_00435_qdepthtmw1.json`

#### `q0`

baseline:

- `geometry_unstable = true`
- `global_final_disp_px = 6.016`
- `residual_final_p95_px = 45.476`
- `track_final_p95_px = 39.977`

`temporal_median_world_v1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.400`
- `residual_final_p95_px = 65.240`
- `track_final_p95_px = 63.751`

解释：

- 共同 wobble 明显下降
- 而且已经低到不再触发 `geometry_unstable`
- 但重尾和局部坏轨迹反而更重

#### `q4`

baseline:

- `geometry_unstable = false`
- `global_final_disp_px = 2.711`
- `residual_final_p95_px = 74.637`
- `track_final_p95_px = 73.295`

`temporal_median_world_v1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.427`
- `residual_final_p95_px = 72.817`
- `track_final_p95_px = 71.801`

解释：

- `q4` 上是轻度但一致的改善
- 但幅度不大

### 对 `00435` 的当前解释

结论不是“全部都好了”，而是更具体：

1. 只改 query seed 3D 初始化，确实能明显压低 `q0` 的共同场景级 wobble
   这说明 query seed 初始化本身就是 `00435` 问题的一部分。

2. 但 heavy-tail 并没有随之消失
   这说明 `00435` 不是“只修 query seed 就够”的问题。

3. 当前更合理的解释是：
   - query seed 初始化，确实贡献了 scene-level common drift
   - 但模型内部仍在使用原始 per-frame depth 构造 dense geometry
   - 所以局部背景 outlier / per-frame 几何不稳 / tracker-local 交互仍然存在

压缩成一句话：

- query seed stabilization 对 `00435` 的“共同 wobble”有效
- 但对“尾部坏轨迹”不够，甚至在 `q0` 上会把尾部问题暴露得更明显

## `00435` dense geometry-side rerun

### 运行配置

这次不是只改 query seed，而是叠加：

- `query_depth_stabilization_mode=temporal_median_world_v1`
- `dense_depth_stabilization_mode=temporal_median_reproject_v1`
- `dense_depth_stabilization_radius=2`
- `dense_depth_stabilization_min_support=3`

同时显式固定：

- `camera_name=stereo_left`
- `video_name=stereo_left`
- `frame_drop_rate=1`
- `future_len=32`
- `grid80`
- `notrim`
- `filter_level=none`

输出目录：

- `data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep/trajectory_dense_none_grid80_notrim_qdepthtmw1_densedepthtmr1/stereo_left`

scene wobble 结果文件：

- `/tmp/scene_wobble_00435_qdepthtmw1_densedepthtmr1.json`

### `q0`

`qdepthtmw1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.400`
- `residual_final_p95_px = 65.240`
- `track_final_p95_px = 63.751`

`qdepthtmw1 + densedepthtmr1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.264`
- `residual_final_p95_px = 61.086`
- `track_final_p95_px = 59.814`

解释：

- `q0` 上，dense geometry 侧稳定化继续改善了共同 wobble
- 同时也把 heavy-tail 往回压了一点
- 说明 dense depth 输入本身确实还在贡献坏轨迹

### `q4`

`qdepthtmw1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.427`
- `residual_final_p95_px = 72.817`
- `track_final_p95_px = 71.801`

`qdepthtmw1 + densedepthtmr1`:

- `geometry_unstable = false`
- `global_final_disp_px = 2.591`
- `residual_final_p95_px = 57.323`
- `track_final_p95_px = 56.150`

解释：

- `q4` 上，global drift 略有回升
- 但 heavy-tail 有明显下降
- 这更像 dense geometry 稳定化在压局部 outlier，而不是只改 scene-level common drift

### 全 query frame 汇总

相对 `qdepthtmw1`，新的 combined 版出现了一个关键副作用：

- `geometry_unstable` sample 数量从 `0 / 8` 回升到 `4 / 8`
- 具体变成阳性的 query frame 是：`2 / 3 / 6 / 7`

同时，sample 内部保存的 dense-depth 元数据显示：

- 每帧被替换的有效像素比例大约在 `0.83 ~ 0.98`
- 每帧 `support_count_median` 基本在 `3 ~ 5`
- `delta_depth` 的中位数只有 `2 ~ 5 mm`
- 但 `delta_depth p95` 在部分帧可以到 `0.20 m`

这说明：

- 当前 dense stabilizer 不是“只改了很少一点点像素”
- 它实际上重写了绝大多数有效深度像素
- 在当前 extrinsics 精度下，这种全帧重投影 median 过于激进

### 对 dense geometry-side 结果的当前解释

结论不是“dense depth stabilization 无效”，而是：

1. 它对 `00435` 的 heavy-tail 坏轨迹有真实帮助
   `q0/q4` 的 `track_final_p95_px` 都下降了。

2. 但它也会把当前外参/重投影误差直接写回整帧 depth
   结果是部分 query frame 的 geometry-only wobble 反而被重新放大。

3. 因此“全帧、近乎全像素替换”的 dense 版本，当前不能直接作为默认修复方案。

压缩成一句话：

- dense geometry 侧 depth stabilization 是有效方向
- 但当前这个 `temporal_median_reproject_v1` 太激进，已经暴露出“会把外参/重投影误差写回 depth”的副作用

### 下一步更合理的收缩方向

如果继续沿 dense geometry 方向走，下一版不应再做“全帧几乎全部像素替换”，而应先收紧到更可信的静态背景区域，比如：

- 只对 temporal reprojection 本身非常一致的像素替换
- 只对 far/background 区域替换
- 或只对与原始 depth 差值较小、但能稳定降低 wobble 的像素替换

也就是说：

- 方向对
- 但当前实现的作用域太大
- 下一步应该做“几何一致性约束更强的 dense stabilization”，而不是继续简单扩大 temporal median 的覆盖面

## `04234` tracker/local interaction 结果

### 运行对象

- `trajectory_dense_none_grid80_notrim/stereo_left/samples/stereo_left_0.npz`
- `trajectory_dense_none_grid80_notrim/stereo_left/samples/stereo_left_4.npz`

输出：

- `/tmp/tracker_geom_04234_q0.json`
- `/tmp/tracker_geom_04234_q4.json`

### `q0`

- `tracker_local_interaction_count = 1075`
- `geometry_limited_count = 83`

summary:

- tracker final drift median = `2.641 px`
- geometry-only final drift median = `0.055 px`
- excess final drift median = `1.432 px`

### `q4`

- `tracker_local_interaction_count = 1425`
- `geometry_limited_count = 143`

summary:

- tracker final drift median = `1.847 px`
- geometry-only final drift median = `0.095 px`
- excess final drift median = `1.420 px`

### 对 `04234` 的解释

这个结果很关键，因为它不是只看“tracker 坏不坏”，而是看：

- source geometry 本来会不会把这些点带坏
- tracker 最终到底又额外放大了多少

当前结果表明：

1. `04234` 里，geometry-only final drift 的中位数非常小
   只有 `0.055 px / 0.095 px` 量级。

2. 但 tracker final drift 中位数已经到 `1.8~2.6 px`
   说明坏轨迹并不是 source geometry 自动产生的。

3. `tracker_local_interaction_count` 远高于 `geometry_limited_count`
   `1075 vs 83`，`1425 vs 143`。

因此当前更明确支持：

- `04234` 的主问题不是上游 geometry 自己在晃
- 而是 tracker 与局部背景区域发生了大规模交互性滑移

## 当前结论

### 结论 1

`depth_temporal_median_world_v1` 前移到 inference 入口后，确实会改变 downstream 行为，而不是只在 geometry-only 指标里有效。

### 结论 2

对 `00435`，query seed 3D 初始化是 scene-level wobble 的真实组成部分。

证据：

- `q0` 的 `global_final_disp_px` 从 `6.016` 降到 `2.400`
- `geometry_unstable` 从 `true` 变成 `false`

### 结论 3

但 `00435` 的问题不只在 query seed 初始化。

证据：

- `q0` 的 `track_final_p95_px` 从 `39.977` 变成 `63.751`
- 说明尾部坏轨迹没有被一起修掉

### 结论 4

对 `04234`，当前证据已经明显偏向 tracker / 局部背景交互主导，而不是 source geometry 主导。

### 结论 5

下一步不应再把 `00435` 和 `04234` 用同一种修法处理。

- `00435`：继续往“query seed + dense geometry”两端拆
- `04234`：继续做 tracker/local background 诊断，而不是优先做 extrinsics smoothing

## 下一步建议

### `00435`

下一步最值得做的是：

1. 保留 inference-side query seed stabilization
2. 再单独实验“dense geometry 侧”的 depth stabilization

也就是继续拆：

- query seed 初始化
- per-frame dense geometry 上下文

否则只改 seed，会出现：

- common wobble 变好
- heavy-tail 仍然留在下游

### `04234`

下一步最值得做的是：

1. 导出 `tracker_local_interaction_mask` 的空间热力图
2. 叠加 query frame RGB，看这些高 excess 区域是否集中在：
   - 低纹理背景
   - 重复纹理背景
   - 远处背景
   - 反光 / 透明 / 边缘区域

## 验证

- `python3 -m py_compile scripts/batch_inference/infer.py scripts/batch_inference/batch_infer_press_one_button_demo.py scripts/data_analysis/export_tracker_geometry_interaction_report.py utils/external_wobble_diagnostics.py utils/tracker_geometry_interaction_diagnostics.py`
- `PYTHONPATH=. ../TraceForge_Reinforcement/.venv/bin/python -m unittest utils.test_external_wobble_diagnostics utils.test_tracker_geometry_interaction_diagnostics scripts.batch_inference.test_infer_cli_surface scripts.batch_inference.test_press_one_button_demo_utils`
