# pick_place `wrist_pick_place` 历史合并记录

日期：2026-03-26

本文合并了 2026-03-25 的初版设计记录和 2026-03-26 的 rerun 复盘，只保留仍对当前代码有参考价值的结论。当前真实实现请以 [docs/maintained_traj_filter_logic.md](/data1/wangchen/projects/TraceForge/docs/maintained_traj_filter_logic.md) 和 `utils/traj_filter_utils.py` 为准。

## 背景

`wrist_manipulator` / `wrist_manipulator_top95` 的目标是把 wrist 相机轨迹收缩到 manipulator 主体附近，但 `pick_place` 场景还需要保住被抓住并随 manipulator 一起运动的物体轨迹。因此后续单独引入了 `traj_filter_profile=wrist_pick_place`。

## 初版设计结论

`wrist_pick_place` 的核心结构是两条分支并联：

```text
wrist_seed_mask
    -> manipulator 分支
    -> object 分支

final_mask = manipulator_final_mask | pick_place_object_mask
```

其中 object 分支的设计语义是：

1. 轨迹先通过 `wrist_seed_mask`
2. sample 段内命中 `pick` heatmap
3. 与 manipulator reference 的最近世界坐标距离足够近
4. query 深度不能比 manipulator 主体远太多

这一步的目标不是进一步压成 manipulator-only，而是明确保住 grasped object。

## 10-case 初版验证

初版验证基于：

- 输出目录：`data_tmp/mcap_v1_wrist_pick_place_10cases_20260325`
- 汇总：`data_tmp/mcap_v1_wrist_pick_place_10cases_20260325_summary.json`
- 可视化：`data_tmp/mcap_v1_wrist_pick_place_10cases_20260325_visualization`

代表帧上的主要观察是：

- `valid_count` 明显低于旧 `wrist`，说明大面积背景 seed 被压下来了
- `object_count` 仍稳定保留，说明 profile 不只是“把所有点砍掉”
- 这证明 `pick_place` 需要单独 profile，而不是继续调 `wrist_manipulator_top95`

## rerun 之后保留下来的结论

2026-03-26 的 rerun 复盘表明，真正需要保留的是下面三条结论。

### 1. major-components manipulator reference 方向是对的

对双臂 / 多 component case，不能回退成“只保单个最大 component”。更合理的方向是：

- 先保留 major components
- 再在 component 内部继续收缩明显属于远处背景、桌面残留、低 motion 噪声的部分

### 2. object depth guard 需要按最近 manipulator component 判定

双臂 case 下，如果直接用全局 manipulator reference 的 query-depth 分布，会把多个 component 的深度范围混在一起，导致 object 分支被错误放宽。保留下来的修正方向是：

- object contact 仍按 component 看最近距离
- object depth guard 也改成 nearest-component aware

### 3. viewer / verification 的显示语义必须和过滤语义分开

有些 case 看起来像“轨迹闪烁”或“只剩一条臂”，后来证明主要是显示策略问题，而不是过滤本身有问题。因此保留下来的结论是：

- 需要区分 `supervision` 和 `finite` 两种显示语义
- GIF 不能简单做二次 motion 截断
- 可视化和 verification 不能再被当成过滤结论本身

## 这轮历史工作实际沉淀到代码里的内容

从这两轮文档里，真正进入维护态实现并仍值得记住的改动方向是：

- `wrist_pick_place` 作为显式 profile 保留
- manipulator 分支保留 major-components 语义，而不是回退到单 component
- object 分支采用 component-aware contact / depth 约束
- 可视化脚本支持 `render_mode={supervision,finite,hybrid}`
- verification 侧避免 PNG / GIF 使用两套不一致的截断语义

## 当前应如何使用这些历史记录

如果今天要理解 `pick_place` 过滤逻辑：

- 先看 [docs/maintained_traj_filter_logic.md](/data1/wangchen/projects/TraceForge/docs/maintained_traj_filter_logic.md)
- 再看 `utils/traj_filter_utils.py` 的当前实现

这份历史文档只回答两个问题：

1. 为什么 `pick_place` 需要单独 profile
2. 为什么后来在 major-components、component-aware depth guard、viewer 语义上做了这些修正

其余更细的逐 case 过程记录已经不再单独保留。
