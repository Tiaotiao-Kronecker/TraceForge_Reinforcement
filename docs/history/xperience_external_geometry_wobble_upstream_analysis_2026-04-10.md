# xperience external depth / extrinsics 时序 wobble 上游排查（2026-04-10）

## 背景

在完成以下下游结论后，排查入口从轨迹过滤继续上移到 source geometry：

- `00190` 主要表现为边缘 bad seed / 飞线
- `00435` 表现为明显的 scene-level wobble
- `04234` 主要表现为局部背景抖动

并且在 `A2 = standard + external + external_depth_static_v1` 之后：

- `00435` 仍然被 scene wobble 指标稳定打成 `geometry_unstable`
- 当前 `traj_stereo_consistency_mask` 没有进一步筛掉任何 `traj_valid_mask`

因此本轮不再继续调 query trim / post-track 过滤，而是直接检查：

1. external extrinsics 本身是否存在明显高频时序抖动
2. external depth + extrinsics 的 source geometry 在固定 query 视角里是否已经自带 wobble

## 本轮新增工具

- [external_wobble_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/external_wobble_diagnostics.py)
- [test_external_wobble_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/test_external_wobble_diagnostics.py)
- [export_external_wobble_upstream_report.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/data_analysis/export_external_wobble_upstream_report.py)

## 诊断设计

### 1. extrinsics 时序平滑度

直接基于 `geom/geom_stereo_left_official_w2c.npz` 中的 `extrinsics`，统计：

- camera center step translation
- camera rotation step angle
- translation jerk
- rotation jerk

目的：

- 看 `00435` 是否存在比其他 case 更明显的外参高频抖动

### 2. source geometry fixed-view 自晃指标

对每个 case 的 source depth + source extrinsics，直接构造静态 query anchors：

- query frame 使用 `grid80`
- 仅保留 `query_depth > 0.2m`
- 仅保留 `border_dist >= 60 px`

然后对每个 future frame：

1. 把 query anchor 用 query depth + query extrinsics lift 到 world
2. 用当前帧 extrinsics 投影到当前帧图像
3. 在当前帧 source depth 上取该位置的 observed depth
4. 用当前帧 observed depth 再还原到 world
5. 再投回 query 视角

统计：

- `final_query_reproj_global_disp_px`
  - fixed-view 下所有 anchor 的共同中值漂移
- `final_query_reproj_drift_median_px`
  - fixed-view 下 anchor 的典型漂移
- `final_query_reproj_drift_p95_px`
  - fixed-view 下 anchor 的重尾漂移

这个指标与 4D viewer 中“固定 query 视角里点自己在晃”更一致，而且完全不依赖 tracker。

## 结果摘要

### `00190`

extrinsics:

- `step_translation_p95 = 0.01735 m`
- `rotation_jerk_p95 = 0.71681 deg`

source geometry fixed-view drift:

- `q0`: `global=0.019 px`, `median=0.724 px`, `p95=52.607 px`
- `q4`: `global=0.028 px`, `median=0.556 px`, `p95=39.301 px`

解释：

- source geometry 本身已经存在局部很差的深度/重投影 outlier
- 但共同漂移几乎没有
- 这更像“局部深度错误 + 边缘/遮挡问题”，不是整场景外参 wobble

### `00435`

extrinsics:

- `step_translation_p95 = 0.01929 m`
- `rotation_jerk_p95 = 0.57857 deg`

source geometry fixed-view drift:

- `q0`: `global=0.208 px`, `median=1.050 px`, `p95=77.848 px`
- `q4`: `global=0.145 px`, `median=0.662 px`, `p95=55.329 px`

解释：

- source geometry 在 fixed-view 下已经明显更差
- 与 `00190` 相比，不只是 p95 重尾更高，连中位数漂移都更高
- 说明 `00435` 的问题并不需要等 tracker 才产生，source depth/extrinsics 自己已经带来更广泛的 fixed-view 不稳定

### `04234`

extrinsics:

- `step_translation_p95 = 0.00695 m`
- `rotation_jerk_p95 = 1.02791 deg`

source geometry fixed-view drift:

- `q0`: `global=0.084 px`, `median=0.099 px`, `p95=0.658 px`
- `q4`: `global=0.052 px`, `median=0.074 px`, `p95=0.640 px`

解释：

- `04234` 的 source geometry 在 fixed-view 下明显最稳定
- 它的下游错误轨迹更可能来自局部背景跟踪问题，而不是上游 geometry 自身大范围自晃

## 当前判断

### 判断 1：`00435` 的上游问题是真实存在的

`00435` 的 source geometry fixed-view 漂移已经显著高于 `04234`，而且高于 `00190` 的典型漂移。

因此：

- `00435` 的 scene-level wobble 不是纯 tracker 幻觉
- 上游 geometry 本身已经给了 tracker 一个更不稳定的输入

### 判断 2：当前证据不支持“00435 的主因是 extrinsics 高频抖动”

虽然 `00435` 的相机路径更长、step translation 略大，但：

- `rotation jerk` 并不比 `00190`/`04234` 更糟
- 甚至 `04234` 的 `rotation_jerk_p95` 最高，但它的 fixed-view source geometry 反而最稳

因此当前更合理的判断是：

- 不能把 `00435` 直接归因为 extrinsics 的高频时序抖动
- 当前证据更偏向 external depth 的时序/区域性不稳定
- extrinsics 可能有贡献，但不是目前最强嫌疑项

### 判断 3：`00190` 与 `00435` 的上游问题类型不同

`00190`：

- `global drift` 很小
- `median drift` 中等
- `p95` 极高

这更像少数局部区域或边缘区域深度坏掉。

`00435`：

- `global drift` 略增
- `median drift` 明显升高
- `p95` 也很高

这更像大范围背景 source geometry 都偏不稳，而不是只有少数边缘坏点。

## 下一步建议

优先顺序：

1. 继续做 depth-focused 上游排查
   把 fixed-view reprojection drift 按 query grid 空间位置导出热力图，确认 `00435` 的不稳定区域是不是覆盖大面积背景。

2. 暂不把主要精力放在 extrinsics smoothing
   当前没有足够证据说明外参高频抖动是主因。

3. 如果 heatmap 进一步支持“广泛背景深度不稳”
   再考虑：
   - source depth 的 sample 级质量标记
   - 对 external profile 增加 sample-level geometry-unstable 标注/降权
   - 或针对背景 depth 稳定性设计更上游的 reject 策略

## 运行方式

单 case 导出命令：

```bash
PYTHONPATH=. ../TraceForge_Reinforcement/.venv/bin/python \
  scripts/data_analysis/export_external_wobble_upstream_report.py \
  --case_dir data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep
```

## 验证

- `python3 -m py_compile utils/external_wobble_diagnostics.py utils/test_external_wobble_diagnostics.py scripts/data_analysis/export_external_wobble_upstream_report.py`
- `PYTHONPATH=. ../TraceForge_Reinforcement/.venv/bin/python -m unittest utils.test_external_wobble_diagnostics`
