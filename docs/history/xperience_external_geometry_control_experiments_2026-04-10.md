# xperience external geometry 控制变量实验（depth vs extrinsics，2026-04-10）

## 目的

这一轮的目标不是再看“geometry-only 是否有问题”，而是继续把上游来源拆开：

- 固定原始 depth，只对 extrinsics 做单独处理，看 fixed-view wobble 是否明显下降
- 固定原始 extrinsics，只对 depth 对应的 query anchor world point 做 temporal stabilization，看 fixed-view wobble 是否明显下降

如果只有 depth 稳定化有效，说明更偏 `depth 主导`。

如果只有 extrinsics 平滑有效，说明更偏 `extrinsics 主导`。

如果两边都无效，但 geometry-only 本身已经稳定，再去怀疑 tracker 本身。

## 实验对象

- `stereo_left_start_00190_officialprep`
- `stereo_left_start_00435_officialprep`
- `stereo_left_start_04234_officialprep`

统一设置：

- `camera_name=stereo_left`
- `query_frames=0,4`
- `grid_size=80`
- `min_query_depth_m=0.2`
- `min_border_dist_px=60`

## 对比变体

### 1. `baseline`

原始 external depth + 原始 extrinsics。

### 2. `extrinsics_smooth_r1`

只改 extrinsics。

做法：

- 把 `w2c` 先转成 `c2w`
- 对 camera center 和 rotation 做半径 `1` 的 moving average
- rotation 再投影回合法 `SO(3)`
- 再转回 `w2c`

这是本轮唯一有物理意义的 extrinsics-only 实验。

### 3. `extrinsics_freeze_query`

只改 extrinsics，但方式是把所有帧都强行冻结到 query frame 的 `w2c`。

它的作用只是做极端 ablation：

- 用来证明 geometry-only 指标确实会随相机时序几何变化而变化
- 不能把它解释成可部署修复方案

### 4. `depth_temporal_median_world_v1`

只改 depth 侧对应的 query anchor world point，不改 extrinsics。

做法：

- 先用 baseline query anchor 作为参考 world point
- 在每个 future frame 把该 world point 投影到当前帧
- 从当前帧 depth 取 observed depth，再反投影回 world
- 只保留回到 query 视角后 reprojection error `<= 3 px` 的支持样本
- 当支持帧数 `>= 3` 时，用这些 observed world point 的 temporal median 替换原始 anchor world point

它近似回答的问题是：

- 如果只把 depth 造成的 query anchor world location 做时序稳化，geometry-only wobble 能不能下降

## 新增代码

- [export_external_wobble_control_experiments.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/data_analysis/export_external_wobble_control_experiments.py)
- [external_wobble_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/external_wobble_diagnostics.py)
- [test_external_wobble_diagnostics.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/test_external_wobble_diagnostics.py)

## 核心结果

以下只列最关键的 final fixed-view 指标：

- `final_query_reproj_drift_median_px`
- `final_query_reproj_drift_p95_px`

## 指标解释（口语版）

为了避免把不同类型的错误轨迹混在一起看，这里统一用三类 fixed-view 指标来读：

### 1. `global drift`

可以把它理解成：

- 这一整片点是不是像“整体一起平移了”

如果它大，通常更像：

- 整片背景共同 wobble
- 更接近 extrinsics 类问题，或整场景级几何不一致

### 2. `median drift`

可以把它理解成：

- 一个“典型点”大概漂了多少

如果它大，通常更像：

- 不是只有少数坏点出错
- 而是大部分点都不太稳

### 3. `p95 drift`

可以把它理解成：

- 最差那一小撮点坏到什么程度

如果它大，但 `median drift` 不大，通常更像：

- 少数局部区域特别差
- 边缘、遮挡边界、局部坏深度、局部 slip 问题

例如：

- `p95 0.640 -> 2.255`

它的意思不是“所有点都变差了 3.5 倍”，而是：

- 尾部那批最差的点，明显变得更差了
- 这通常意味着某个处理放大了重尾坏点，而不是把整场景都一起做坏

因此这三个量的口语理解可以压缩成：

- `global drift` 大：像“整片场景一起晃”
- `median drift` 大：像“多数点都在抖”
- `p95 drift` 大：像“少数坏点特别飞”

### `00190`

`q0`:

- `baseline`: `median=0.724`, `p95=52.607`
- `extrinsics_smooth_r1`: `median=0.687`, `p95=49.200`
- `depth_temporal_median_world_v1`: `median=0.540`, `p95=52.639`

`q4`:

- `baseline`: `median=0.556`, `p95=39.301`
- `extrinsics_smooth_r1`: `median=0.564`, `p95=37.229`
- `depth_temporal_median_world_v1`: `median=0.395`, `p95=39.440`

解释：

- depth 稳定化对中位漂移更 consistently 有利
- extrinsics 平滑只是在部分 p95 上有帮助，但不稳定，且对中位漂移没有形成清晰优势
- `00190` 的 heavy-tail 仍然很强，说明这里的主问题还是局部 bad depth / 边缘 outlier，不是简单的整体 extrinsics wobble

### `00435`

`q0`:

- `baseline`: `median=1.050`, `p95=77.848`
- `extrinsics_smooth_r1`: `median=1.221`, `p95=72.281`
- `depth_temporal_median_world_v1`: `median=0.847`, `p95=75.512`

`q4`:

- `baseline`: `median=0.662`, `p95=55.329`
- `extrinsics_smooth_r1`: `median=0.514`, `p95=52.899`
- `depth_temporal_median_world_v1`: `median=0.527`, `p95=55.081`

解释：

- `q0` 上，extrinsics 平滑把中位漂移做得更差，depth 稳定化则明显变好
- `q4` 上，两者都改善了中位漂移，但 depth 与 extrinsics 的幅度接近
- 合并来看，depth 方向是更稳定的改善杠杆，extrinsics 平滑并没有形成“清晰且单调”的收益

结论：

- `00435` 进一步偏向 `depth 主导`
- 但当前 `depth_temporal_median_world_v1` 还不能消掉重尾错误，说明还没有真正触到全部根因

### `04234`

`q0`:

- `baseline`: `median=0.099`, `p95=0.658`
- `extrinsics_smooth_r1`: `median=0.089`, `p95=0.581`
- `depth_temporal_median_world_v1`: `median=0.068`, `p95=0.514`

`q4`:

- `baseline`: `median=0.074`, `p95=0.640`
- `extrinsics_smooth_r1`: `median=0.111`, `p95=2.255`
- `depth_temporal_median_world_v1`: `median=0.059`, `p95=0.420`

解释：

- `04234` baseline 本来就不差
- depth 稳定化在两个 query frame 上都继续改善
- extrinsics 平滑在 `q4` 明显恶化，尤其把 `p95` 从 `0.640` 拉到 `2.255`

结论：

- 当前证据不支持把 `04234` 的问题归因为 extrinsics wobble
- 这个 case 更像“geometry 基本稳定，但 tracker / 局部背景 / 局部深度噪声交互仍可能出错”

## 如何解读 `extrinsics_freeze_query`

三个 case 里，`extrinsics_freeze_query` 都几乎把 geometry-only drift 压到接近 `0`。

这只能说明：

- fixed-view 几何误差确实和相机时序几何有关
- 诊断指标本身是灵敏的

不能说明：

- 真实部署里“只要平滑 extrinsics 就够了”
- 当前问题一定主要来自 extrinsics

原因是这个 ablation 直接取消了真实相机运动，本身不物理，也不保留任务真实几何条件。

## 当前结论

### 结论 1

在这轮有物理意义的控制变量实验里，`depth_temporal_median_world_v1` 是唯一表现出跨 case、更 consistent 收益的方向。

### 结论 2

`extrinsics_smooth_r1` 不是清晰赢家：

- 有时略有帮助
- 有时收益很弱
- 有时会明显恶化

因此当前不适合把 extrinsics smoothing 当作主修复方向。

### 结论 3

`00435` 的主嫌疑进一步收敛到 `external depth` 的时序 / 区域性不稳定，而不是 `extrinsics` 高频 wobble。

### 结论 4

`00190` 仍然主要是局部深度坏点 / 边缘 outlier 问题，depth 稳定化只改善典型点，不能解决重尾飞线。

### 结论 5

`04234` 的上游 geometry 本来就相对稳定；如果后面 4D 里仍有很多背景轨迹，更值得继续拆“tracker 本身”与“局部几何噪声交互”。

### 结论 6

这份文档对应的控制变量实验已经实际完成，不是待办计划。

也就是说，下面这组判断逻辑现在已经有了实验数据支撑：

- 固定原 depth，只改 extrinsics
- 固定原 extrinsics，只改 depth
- 看哪边更稳定、哪边更 consistently 有收益

当前结果已经表明：

- 更偏 `depth 主导`
- 不支持把 `extrinsics smoothing` 作为当前第一优先级

## 下一步建议

优先顺序：

1. 把当前 geometry-only 的 depth stabilization 思路前移到 inference 侧
   先试“query seed 3D 初始化的 depth 稳定化”，而不是继续做 extrinsics smoothing。

2. 针对 `00435` 做第一轮 inference-side depth stabilization 对比
   看 4D viewer 里的真实下游轨迹是否跟 geometry-only 结论一致。

3. 对 `00190` 单独保留 local-risk 思路
   这里需要继续针对边缘、遮挡边界、零深度邻域做局部风险控制，而不是只靠整体稳定化。

## 运行命令

单 case 导出：

```bash
PYTHONPATH=. ../TraceForge_Reinforcement/.venv/bin/python \
  scripts/data_analysis/export_external_wobble_control_experiments.py \
  --case_dir data_tmp/xperience_traceforge_attempt_20260402/motion_windows/stereo_left_start_00435_officialprep
```

## 验证

- `python3 -m py_compile utils/external_wobble_diagnostics.py utils/test_external_wobble_diagnostics.py scripts/data_analysis/export_external_wobble_control_experiments.py`
- `PYTHONPATH=. ../TraceForge_Reinforcement/.venv/bin/python -m unittest utils.test_external_wobble_diagnostics`
