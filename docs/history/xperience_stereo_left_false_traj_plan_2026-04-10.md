# xperience stereo_left 错误轨迹分析与实验计划（2026-04-10）

## 背景

本轮分析基于以下三个 xperience stereo motion-window case 的 `grid80 + notrim + filter_level=none` 结果：

- `stereo_left_start_00190_officialprep`
- `stereo_left_start_00435_officialprep`
- `stereo_left_start_04234_officialprep`

本轮目标不是直接回到历史上的 ring trim 方案，而是从第一性原理出发，区分：

- 哪些错误轨迹来自 query seed 本身不可靠
- 哪些错误轨迹来自局部背景区域的跟踪/深度不稳定
- 哪些错误轨迹来自更上游的场景级几何 wobble

## 观察到的现象

### 1. `00190`

- 人站立右前方、右后方和前方边缘区域存在明显错误轨迹。
- 错误轨迹多数表现为从图像边缘或近边缘区域发散出去的长轨迹。

### 2. `00435`

- 几乎所有 keypoint 都存在明显抖动。
- 即使不出现特别夸张的飞线，也会产生大量不必要的背景轨迹。

### 3. `04234`

- 人和桌面附近较稳定。
- 除此前景之外，多个方向上的背景 keypoint 存在较大抖动，并带来较多错误轨迹。

## 已完成的定量检查

### 共性结论

- 三个 case 中最离谱的轨迹，几乎都来自 `query depth≈0` 的 seed。
- 这些 seed 绝大多数位于图像边缘附近。
- 现有 `filter_level=none` 会同时关闭：
  - query-depth quality
  - temporal depth consistency
  - depth volatility guidance

### `00190`

- 全量轨迹固定视角 `uv step` 中位数的 p95 约为 `57.8 px`。
- 仅去掉 `query_depth<=0.05m` 且 `border<=40 px` 的 seed 后，p95 降到约 `6.8 px`。
- 极端轨迹的 seed 多数满足：
  - `query_depth=0.0`
  - `border_dist<=32 px`

结论：

- 主问题是 query seed 落在无效深度/边缘风险区域。
- 这类点在 lift 为 3D query 之后，会被放大成固定视角下的爆炸轨迹。

### `00435`

- 全量轨迹固定视角 `uv step` 中位数的 p95 约为 `242 px`。
- 去掉零深度边缘 seed 后，深度有效且居中的点仍然存在显著抖动：
  - `depth>0.2m & border>60 px` 后，p50 仍约 `2.2 px`
  - 同组 p95 仍约 `7.4 px`
- 深且居中的背景点，在固定 query 视角下仍存在约 `6 px` 的全局中值漂移。

结论：

- `00435` 不只是坏 seed 问题。
- 更像存在 scene-level wobble：
  - depth/extrinsics 时序不稳定
  - 或 tracker 在背景上集体发生固定视角漂移

### `04234`

- 全量轨迹固定视角 `uv step` 中位数的 p95 在 query frame 4 可达 `67.8 px`。
- 去掉零深度边缘 seed 后，同组 p95 降到约 `1.8 px`。
- 深且居中的点仍有局部区域抖动，但远弱于 `00435`。

结论：

- `04234` 的主问题仍然以 bad seed + 局部背景不稳定为主。
- 目前没有 `00435` 那种明显的整场景共同 wobble 证据。

## 根因分型

### 类型 A：seed 本身不可靠

典型症状：

- `query depth` 为 `0`、近 `0`、非有限值，或局部 patch 有效比例过低
- seed 落在深度跳变边缘且同时接近图像边缘
- 轨迹在 fixed-view 中表现为飞线、爆炸线、瞬时漂移很大的长轨迹

对应物理原因：

- 当前 query point 在进入 tracker 前，直接用 query-frame depth lift 为 3D 点
- 若 seed 深度本身错误，则后续所有 3D/重投影都建立在错误初值上

### 类型 B：局部背景不稳定

典型症状：

- 前景局部稳定，但远处或低纹理背景区域出现明显 jitter
- 同一局部区域内多条背景轨迹同步变差，但并非整场景一起平移

可能原因：

- 深度局部噪声
- 背景纹理弱或重复纹理导致 tracker 容易 slip
- 单视角几何验证不足

### 类型 C：scene-level wobble

典型症状：

- 深且居中的背景点，在 fixed-view 下也整体共同漂移
- 去掉边缘坏 seed 后，背景点云依然明显晃动

可能原因：

- 外参时序抖动
- 深度时序 wobble
- 用同一份 geometry 做自一致检查时，错误被彼此解释掉

## 对当前代码状态的判断

当前代码中已有以下机制，但在 `filter_level=none` 时被关闭：

- query-depth quality
- temporal depth consistency
- depth volatility guidance

它们对 `00190` / `04234` 这类 bad seed 问题足够重要，但不足以单独解决 `00435` 的场景级 wobble。

## 解决方案优先级

### 优先级 1：seed-depth 预筛选

目标：

- 在送入 3D tracker 之前剔除明显不可靠的 query seed
- 用几何可靠性替代大范围 ring trim

候选规则：

- 拒绝 `query depth<=0.05m` 或非有限值
- 拒绝 `5x5 patch` 有效深度比例 `<0.4`
- 拒绝 query depth 与 patch 中位数偏差超过 `max(0.05m, 10%)`
- 对 `depth edge risk=true` 且 `border_dist<=40 px` 的点额外拒绝

### 优先级 2：external 场景级 wobble 诊断

目标：

- 区分“单条轨迹坏”与“整段 sample 的 geometry 都在晃”

候选规则：

- 选取深且居中的背景锚点
- 统计 fixed-view 全局中值漂移
- 若漂移过大，则对该 sample 打上 geometry-unstable 诊断标签

### 优先级 3：stereo consistency

目标：

- 对单视角 slip / 深度局部错误增加跨视角约束

适用：

- 尤其针对 `04234` 这类局部背景不稳定场景

### 优先级 4：上游 geometry 修复

包括：

- 外参平滑
- 深度时序平滑
- sample 级别 geometry 质量评估

这类改动成本更高，暂不作为第一轮实验入口。

## 第一轮实验计划

### 实验名

`A1: external_depth_static_v1`

### 实验目标

验证“仅基于 query-frame 静态深度可靠性做 seed 预筛选”是否能显著减少错误轨迹，而不再依赖固定 ring trim。

### 设计原则

- 不改默认维护态行为
- 不启用 post-track trajectory filtering
- 只在 tracker 前做 seed 预筛选
- 保持 `grid80 + notrim`

### 试验参数

- `filter_level=none`
- `traj_filter_profile=external`
- `query_sampler_mode=grid`
- `query_prefilter_mode=external_depth_static_v1`
- `grid_border_trim_left/right/top/bottom=0`

### `external_depth_static_v1` 规则

对 external 系列 profile 的 dense grid seed，先做：

1. query-depth quality 检查
   基于已有 `5x5 patch`：
   - query depth 有效
   - patch 有效比例充足
   - query depth 与 patch 中位数一致

2. extra min-depth guard
   额外拒绝 `query depth<=0.05m`

3. border-adaptive depth-edge rejection
   若同时满足：
   - query depth edge risk 为真
   - `border_dist<=40 px`

   则拒绝该 seed。

### 产物命名

建议输出目录：

- `trajectory_dense_none_grid80_notrim_prefilter_extdepthv1`

### 成功判据

至少满足以下两项：

- `00190` 的边缘飞线明显减少
- `04234` 的背景错误轨迹明显减少
- 前景主体和桌面关键轨迹没有明显被过度杀伤

若 `00435` 仍明显整体晃动，则判定：

- 第一轮试验对 bad seed 有效
- 但 scene-level wobble 仍需单独处理

## 后续实验矩阵

### `A0`

- 当前 baseline：`none + notrim`

### `A1`

- `external_depth_static_v1`

### `A2`

- `A1 + standard/external`

### `A3`

- `A2 + stereo consistency`

### `A4`

- `A1/A2` 基础上增加 scene-level wobble 诊断

## 当前执行决策

先实现并验证 `A1`，因为它：

- 最贴近当前观察到的主错误类型
- 只改 query seed 入口，不动 tracker 主体
- 不会把“是否要恢复 post-track filter”这个问题和第一轮 bad-seed 试验混在一起

## 2026-04-10 追加进展

### 已落地实验

- `A1` 已实现为 `query_prefilter_mode=external_depth_static_v1`
- `A2` 已完成重跑：
  - `trajectory_dense_standard_grid80_notrim_prefilter_extdepthv1`
- 新增了 external 场景级 wobble 诊断工具：
  - [export_external_scene_wobble_report.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/scripts/data_analysis/export_external_scene_wobble_report.py)
  - [scene_wobble_utils.py](/DATA/disk2/wangchen/projects/TraceForge_Reinforcement_xperience/utils/scene_wobble_utils.py)

### A2 的新增诊断结论

对 `A2` 结果使用以下 anchor 定义做 fixed-view common-drift 诊断：

- `query_depth > 0.2m`
- `border_dist >= 60 px`
- 仅统计 `traj_valid_mask=True` 的轨迹

核心指标：

- `global_final_disp_px`
  - deep-central anchor 在 fixed-view 末帧的共同中值位移
- `geometry_unstable`
  - 当前诊断阈值为 `global_final_disp_px >= 3 px`

结果如下：

#### `00190`

- `q0`: `global_final_disp_px=1.55`
- `q4`: `global_final_disp_px=2.52`
- 均未触发 `geometry_unstable`

解释：

- `00190` 的主问题仍然更接近局部 bad seed / 边缘飞线
- 不是明显的整场景共同漂移

#### `00435`

- `q0`: `global_final_disp_px=4.00`
- `q4`: `global_final_disp_px=3.67`
- 两个 sample 都触发 `geometry_unstable`

解释：

- `00435` 的问题确实不只是坏 seed
- 即使在 `A2` 之后，deep-central background anchor 仍存在明显共同漂移
- 这与用户在 4D viewer 中看到的“几乎全场都在晃”一致

#### `04234`

- `q0`: `global_final_disp_px=0.60`
- `q4`: `global_final_disp_px=0.63`
- 均未触发 `geometry_unstable`

解释：

- `04234` 更像 bad seed + 局部背景抖动
- 没有 `00435` 那种显著的 scene-level wobble

### `00435` 的横向对比

用同一 wobble 指标比较 `A0/A1/A2`：

- `A0`
  - `q0`: `6.02`
  - `q4`: `2.71`
- `A1`
  - `q0`: `3.87`
  - `q4`: `2.43`
- `A2`
  - `q0`: `4.00`
  - `q4`: `3.67`

结论：

- `A1/A2` 对 extreme bad seed 有帮助，但没有根治 `00435` 的场景级共同漂移
- 因此 `00435` 不能再继续简单归因于 seed 入口

### 对 `A3` 的即时判断

额外检查了 `A2` 产物中的 `traj_stereo_consistency_mask`，发现当前这三个 case 上：

- `traj_valid_mask & traj_stereo_consistency_mask == traj_valid_mask`

这意味着：

- 以当前实现和当前 stereo signal，直接把 stereo consistency 接进 external profile 不会进一步过滤掉轨迹
- 所以 `A3` 在“当前代码状态”下暂时不是最优先入口

### 下一步建议

优先继续 `A4`，但把它明确拆成两类工作：

1. sample 级 geometry-unstable 诊断/标注
   先把 `00435` 这类 case 在产物层面明确标出来，避免和正常样本混在一起。

2. 上游 wobble 成因排查
   重点看 external depth / extrinsics 的时序抖动，而不是继续做 query 边界裁减。

如果要继续做过滤实验，优先级建议为：

- 不是继续加 ring trim
- 也不是直接把当前 stereo consistency 硬接进 external
- 而是先做更独立的 geometry stability 信号，再决定是否 sample 级 reject / downweight
