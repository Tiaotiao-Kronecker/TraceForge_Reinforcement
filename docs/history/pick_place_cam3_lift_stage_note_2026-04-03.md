# pick_place cam3 抬升阶段轨迹问题记录

日期：2026-04-03

> 历史说明：本文记录 2026-04-03 对 `/home/zoyo/mcap/aao` 这批
> `pick_place` 推理结果的排查结论。若与当前代码实现不一致，以
> [docs/maintained_traj_filter_logic.md](../maintained_traj_filter_logic.md)
> 和 `utils/traj_filter_utils.py` 为准。

## 范围

本次记录只覆盖两个代表性 `cam3` case：

- `/home/zoyo/mcap/aao/00042/trajectory/varied_camera_3`，查询帧 `40`
- `/home/zoyo/mcap/aao/00047/trajectory/varied_camera_3`，查询帧 `41`

排查目标是回答两个问题：

- 原始 `cam3` 结果为什么几乎看不到夹爪和被抓杯子的 keypoint
- 当前应该停在哪个 `pick_place` 过滤版本，不继续扩大改动

## 结论摘要

1. 原始 batch 对 `cam3` 使用的是 `traj_filter_profile=auto ->
   wrist_manipulator_top95`，这不适合当前 `pick_place` 场景。
2. 切到 `wrist_pick_place_no_heatmap` 后，问题的一部分来自过滤逻辑：
   query frame 可见、但后续 `traj_uvz[..., 2]` 变负的点，会在进入
   `pick_place` 分支前死在 `depth_range_mask` 上。
3. 已在当前代码中修正这部分过滤问题：
   `wrist_pick_place` / `wrist_pick_place_no_heatmap` 现在不再把整段
   `depth_range_mask` 当成进入 pick_place 分支前的硬门槛。
4. 修正过滤后，这两个 case 的有效轨迹数明显回升，但抬升阶段在 3D
   动画里仍会看到一部分轨迹像“跟丢”。
5. 进一步对比当前 in-place 结果、`no-filter` 结果和 `filter-fix` rerun
   结果后，确认三者的 `traj_uvz` 本体逐元素一致；过滤逻辑只改变
   `traj_valid_mask`，不会把轨迹“修好”或“搞坏”。
6. 当前更像是底层 tracking / 3D 恢复在抬升阶段跟不上，表现为大量
   `z <= 0` 和明显大跳变，而不是过滤逻辑继续误杀。
   这部分问题本轮先记录，不继续处理。

## 已确认事实

### 1. no-filter 结果表明底层 tracking 不是一开始就完全坏掉

单 query 精确 rerun 的 no-filter 输出为：

- `/DATA/disk3/tmp/aao_nofilter_exact_20260403/00042/varied_camera_3`
- `/DATA/disk3/tmp/aao_nofilter_exact_20260403/00047/varied_camera_3`

在这两个输出里：

- `traj_valid_mask = 6400 / 6400`
- pick 区域上的 query 点也都还在

这说明：

- query grid 本身覆盖到了夹爪和杯子区域
- 点的 2D/时序跟踪不是一开始就全面失败
- 原始坏结果至少有一部分是过滤导致的，而不是单纯 query 采样错误

### 2. 原始过滤会把 pick 区域点提前杀掉

原始 `wrist_pick_place_no_heatmap` rerun 输出在：

- `/DATA/disk3/tmp/aao_pickplace_noheatmap_exact_20260403/00042/varied_camera_3`
- `/DATA/disk3/tmp/aao_pickplace_noheatmap_exact_20260403/00047/varied_camera_3`

排查发现：

- pick 区域原本有大量 query 点
- 这些点的 `traj_uvz[..., 2]` 在后续帧里会变成负值
- 它们会在 `depth_range_mask` 处直接失败
- 因而根本来不及进入后面的 local region / delayed-contact rescue

## 本轮代码改动

当前代码已经做的修正是：

- 只对 `wrist_pick_place`
- 和 `wrist_pick_place_no_heatmap`

把进入分支前的 base 几何门槛放宽为：

```text
wrist_pick_place_base_mask = valid_count_mask & depth_smooth_mask
```

而不是继续要求：

```text
valid_count_mask & depth_range_mask & depth_smooth_mask
```

对应代码位置：

- `utils/traj_filter_utils.py`
- `utils/test_traj_filter_utils.py`

同时补了一条回归测试，明确覆盖：

- query frame 深度正常
- 后续若干帧 `traj_uvz.z < 0`
- raw depth 仍正常
- `wrist_pick_place_no_heatmap` 不应在 base 阶段把该点提前误杀

## 过滤修正后的真实验证

新的单 query 输出在：

- `/DATA/disk3/tmp/aao_pickplace_filter_fix_20260403/00042/varied_camera_3`
- `/DATA/disk3/tmp/aao_pickplace_filter_fix_20260403/00047/varied_camera_3`

数值对比如下。

### `00042 / query 40`

- `traj_valid_mask: 1857 -> 3038`
- `traj_wrist_seed_mask: 3676 -> 6032`
- `traj_manipulator_candidate_mask: 1838 -> 3016`
- `MASK_REASON_BASE_GEOMETRY_FAIL: 2469 -> 0`

### `00047 / query 41`

- `traj_valid_mask: 1804 -> 2997`
- `traj_wrist_seed_mask: 3588 -> 5974`
- `traj_manipulator_candidate_mask: 1794 -> 2987`
- `MASK_REASON_BASE_GEOMETRY_FAIL: 2467 -> 0`

这说明本轮修正确实解决了“pick 区域点在进入 pick_place 分支前就被 base
geometry 提前杀掉”的问题。

### 3. 当前结果与 no-filter / filter-fix 的 `traj_uvz` 本体完全一致

在完成当前这批 wrist in-place 重生成后，又对以下三份 sample 做了逐元素比较：

- 当前输出：
  `/home/zoyo/mcap/aao/00042/trajectory/varied_camera_3/samples/varied_camera_3_40.npz`
- no-filter 精确 rerun：
  `/DATA/disk3/tmp/aao_nofilter_exact_20260403/00042/varied_camera_3/samples/varied_camera_3_40.npz`
- filter-fix 精确 rerun：
  `/DATA/disk3/tmp/aao_pickplace_filter_fix_20260403/00042/varied_camera_3/samples/varied_camera_3_40.npz`

以及：

- 当前输出：
  `/home/zoyo/mcap/aao/00047/trajectory/varied_camera_3/samples/varied_camera_3_41.npz`
- no-filter 精确 rerun：
  `/DATA/disk3/tmp/aao_nofilter_exact_20260403/00047/varied_camera_3/samples/varied_camera_3_41.npz`
- filter-fix 精确 rerun：
  `/DATA/disk3/tmp/aao_pickplace_filter_fix_20260403/00047/varied_camera_3/samples/varied_camera_3_41.npz`

对比结果是：

### `00042 / query 40`

- `traj_equal(current, nofilter) = True`
- `traj_equal(current, filter_fix) = True`
- `traj_uvz MAE = 0.0`
- `traj_valid_mask: current=3038, nofilter=6400, filter_fix=3038`
- 当前保留轨迹里，`z <= 0` 的时刻数仍有 `47916`
- 当前保留轨迹里，逐帧 2D 跳变 `> 40px` 有 `52116` 次，`> 80px` 有
  `39898` 次

### `00047 / query 41`

- `traj_equal(current, nofilter) = True`
- `traj_equal(current, filter_fix) = True`
- `traj_uvz MAE = 0.0`
- `traj_valid_mask: current=2997, nofilter=6400, filter_fix=2997`
- 当前保留轨迹里，`z <= 0` 的时刻数仍有 `51075`
- 当前保留轨迹里，逐帧 2D 跳变 `> 40px` 有 `55134` 次，`> 80px` 有
  `42688` 次

这说明：

- 当前看到的坏相不是“这次批量重跑重新引入的”
- 也不是“过滤把轨迹本体筛坏了”
- 当前 `pick_place` 过滤逻辑只是在同一份底层 `traj_uvz` 上决定哪些轨迹保留
- 真正没有解决的是底层 tracking / 3D 恢复在 lift 阶段失稳

## 仍未解决的问题

过滤修正后，3D 动画里仍能看到一部分轨迹在夹爪抬升阶段像“跟丢”。

当前更准确的描述不是：

- “过滤还在继续误杀”

而是：

- 底层 `traj_uvz` 的 `z` 分量在 lift 阶段仍会失真
- 同时 `UV` 也存在明显大跳变，说明并不只是 `Z` 单独坏掉
- 一旦 `traj_uvz[..., 2] <= 0` 或明显异常，3D lift 到 world 时就会断
- 因此在 Viser 里表现成 3D 轨迹断裂、飘走，或看起来像跟丢

这和“原始 depth 文件本身错误”不是一回事。当前更像是：

- 抬升 / 遮挡增强阶段，底层 tracking / 3D 恢复能力跟不上
- `UV` 和 `Z` 都会同时退化，只是 `Z <= 0` 在 3D 可视化里更显眼

## 当前决策

本轮先停在当前版本，不继续扩大修复范围。

当前建议是：

- `pick_place` 若有可用 `pick` heatmap，优先使用 `wrist_pick_place`
- `pick_place` 若没有可用 `pick` heatmap，优先使用 `wrist_pick_place_no_heatmap`
- 对当前这批 `cam3` case，先接受“过滤逻辑已修到位，但 lift 阶段 3D
  轨迹仍可能受底层 `traj_uvz.z` 失真影响”的现状

如果以后继续处理，下一步应优先考虑的是：

- 在保留下来的局部轨迹上做 `raw depth @ tracked UV` 的 `Z` 回填实验

但这不属于本轮决定落地的内容。
