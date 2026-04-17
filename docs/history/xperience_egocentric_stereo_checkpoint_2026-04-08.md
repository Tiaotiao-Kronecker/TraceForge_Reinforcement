# `xperience` egocentric stereo checkpoint

日期：2026-04-08

这份 note 只保留 `xperience-10m-sample` 上一次 TraceForge 尝试的最终状态，供后续继续推进时快速接上下文。更完整的当前判断已经同步进
`docs/maintained_traj_filter_logic.md` 的 egocentric stereo 章节。

## 范围

- 数据集：`/DATA/disk0/shared/datasets/xperience-10m-sample`
- 主要产物：`data_tmp/xperience_traceforge_attempt_20260402`
- 主 case：`episode_mid_stereo_left_officialprep` / `episode_mid_stereo_right_officialprep`
- 片段：源帧 `2880..2895`
- 重点 query：`0`、`4`

## 最终结果

### 1. raw tracker 不是完全跑不起来

`trajectory_dense_egocentric_stereo_v1` 在左右目 q0/q4 上都是 `400 / 400`
保留。这说明当前 tracker 能产出 dense 轨迹，问题不是“完全无轨迹”。

### 2. 旧 `standard` 语义对 egocentric stereo 直接过严

`trajectory_dense_egocentric_stereo_v1_standard` 在左右目 q0/q4 上都变成
`0 / 400`。主要现象是：

- `temporal_fail` 直接打满
- `visibility` 极低
- 原 external / manipulator 风格过滤语义不适配当前第一人称双目片段

### 3. `semanticfix` 只能部分救回左目，右目仍几乎不可用

`trajectory_dense_egocentric_stereo_v1_standard_semanticfix` 的最终保留数为：

- `stereo_left`: q0=`67 / 400`, q4=`24 / 400`
- `stereo_right`: q0=`3 / 400`, q4=`6 / 400`

因此这轮工作到最后并没有得到“可直接作为维护态 profile 使用”的结果。

### 4. `geosanity` 说明上限并不在 tracker 完全失效

左目 `trajectory_dense_geosanity` 仍能保住：

- `stereo_left_0.npz`: `318 / 400`
- `stereo_left_4.npz`: `334 / 400`

这说明主问题更像是“当前 validity / profile 语义与 egocentric 场景不匹配”，
而不是 tracker 在这类片段上完全没有可用信号。

## 这轮真正沉淀下来的结论

### 1. 这不是优先做 fisheye 去畸变的问题

当前 `stereo_left` / `stereo_right` 可先按“已 rectified 到足以近似 pinhole”
看待。对这类数据，优先级更高的是：

- query 点怎么选
- track 完成后怎么验

### 2. 不能直接复用 `external_manipulator_v2`

egocentric manipulation 的目标不该只是“保 manipulator 主体”，而应更接近：

- 手 / 前臂附近可信轨迹
- 被交互物体轨迹
- 必要的少量上下文

因此后续应使用显式 egocentric profile，而不是继续把 external 语义硬套过来。

### 3. 当前主瓶颈是 sampling / validity mismatch，不是 motion 不够

这轮产物里，真正卡住的是：

- 极低的 supervision / visibility
- `temporal_fail` 大面积触发
- 旧 profile 对第一人称视角变化、视差、反光/低纹理区域不稳

因此“继续微调旧 standard 阈值”不是高 ROI 方向。

## 建议的下一步

如果只是先看 raw coverage / recall ceiling，推荐先做一轮小规模观察性试跑：

1. 选几个 motion 更强的窗口，而不是只盯 `2880` 这段
2. 先用 `grid_size=80`
3. 先用 `filter_level=none`
4. 只跑少量 case，先观察 raw tracked trajectories 的覆盖与伪轨迹形态

当前已有的 motion scan 候选起点可优先看：

- `190`
- `435`
- `4235`

这一步的目标是回答：

- 更密的 seed 是否能明显提升手/交互物体覆盖
- raw tracker 的“上限画面”大概长什么样
- 后续该优先补 query sampler，还是优先补 post-track validity

注意：这一步只适合看 raw 上限，不适合作为最终过滤质量结论。
