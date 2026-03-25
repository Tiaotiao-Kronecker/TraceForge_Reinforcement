# press_one_button_demo_v5 `varied_camera_3` 尾段空 sample 根因分析

日期：2026-03-24

## 背景

这轮排查的直接问题是：

- `press_one_button_demo_v5` 在第一次空 sample repair 后，仍然剩余大量 `varied_camera_3` 空 sample
- 之后虽然已经把 wrist 路径的 motion gate 从旧字段切到 `traj_motion_extent_all_valid`
- 但第二次只重跑空 sample 后，空 sample 数量没有继续下降

需要回答的核心问题有两个：

1. 新的 motion 字段是否真的被用上了
2. 如果已经用上，为什么剩余空 sample 仍然没有变化

## 结论先行

结论如下：

- `wrist_manipulator_top95` 路径已经真实切到 `traj_motion_extent_all_valid`，不是“字段没替换”
- 第二次 repair 没有效果，不是因为 filter 还在读旧 motion 字段，而是因为这些剩余 tail sample 的原始 tracker 输出已经被冻结
- 对 `varied_camera_3`，当前 checkpoint 的在线 tracking 路径在 `segment_len <= 8` 时不会进入任何 tracking window
- 这会导致模型直接返回“把 query 初始 3D 坐标在整段时间上重复”的 `pred.coords`
- 后续 `traj_uvz`、`motion_extent_all_valid`、`motion gate` 只是忠实地反映了这个冻结结果
- 因此，`near-depth` / `motion` 在 filter 层面看到的是症状，不是根因；真正的根因在 `PointTracker3D` 的 short-tail online window 调度

## 数据表现

### 1. 第一次空 sample repair 结果

第一次 repair 之前与之后的统计为：

- pre empty samples: `3059`
- pre episode/camera groups: `1080`
- post empty samples: `1578`
- fixed: `1481`
- still empty: `1578`

第一次 repair 后，当前数据集剩余空 sample 分布为：

- `varied_camera_1`: `280`
- `varied_camera_2`: `280`
- `varied_camera_3`: `1018`

### 2. 第二次空 sample repair 结果

在另一个终端完成 motion 相关修改后，又对当前剩余空 sample 进行了一轮只修空 sample 的重跑：

- report root: `/data2/yaoxuran/press_one_button_demo_v5/_empty_sample_repair_reports/20260324_194716`
- backup root: `/data2/yaoxuran/press_one_button_demo_v5/_empty_sample_repair_backups/20260324_194716`

结果为：

- `fixed=0`
- `still_empty=1578`
- `failed=0`

也就是说，这一轮没有任何空 sample 被继续修复。

### 3. `varied_camera_3` 按 segment 长度的现象

对当前 `varied_camera_3` 全量 sample 统计后，得到一个非常明显的分界：

| segment_len | total | nonempty | empty |
| --- | ---: | ---: | ---: |
| 1 | 150 | 0 | 150 |
| 2 | 130 | 0 | 130 |
| 3 | 150 | 0 | 150 |
| 4 | 122 | 0 | 122 |
| 5 | 128 | 0 | 128 |
| 6 | 122 | 0 | 122 |
| 7 | 120 | 0 | 120 |
| 8 | 96 | 0 | 96 |
| 9 | 91 | 91 | 0 |
| 10 | 93 | 93 | 0 |
| 11 | 77 | 77 | 0 |
| 12 | 81 | 81 | 0 |

这说明：

- `segment_len <= 8` 的 `varied_camera_3` sample 当前是 `100%` 空
- `segment_len >= 9` 的 `varied_camera_3` sample 当前是 `100%` 非空

这已经强烈暗示根因与 short-tail tracking 调度有关，而不是普通 filter 阈值误差。

## 排查过程

### 1. 先确认 motion 字段替换是否真实生效

代码检查结果：

- `varied_camera_3` 在 `auto` 下走 `wrist_manipulator_top95`
- wrist 分支调用 `_apply_manipulator_aware_filter(...)` 时，已经明确传入 `motion_metric_mode="all_valid"`
- motion gate 判断使用的是 `traj_motion_extent_all_valid`
- top95 排序也使用 `traj_motion_extent_all_valid`

这说明 wrist 路径已经真实切换到新字段。

进一步对第二次 repair 实际处理过的 `1018` 个 `varied_camera_3` 空 sample 做旧备份 vs 新结果差分，得到：

- `traj_motion_extent_all_valid` 新字段存在：`1018 / 1018`
- `candidate_improved`: `0`
- `valid_improved`: `0`
- `candidate_became_positive`: `0`

更进一步，对这些样本中所有有限位置上的 motion 值比较：

- 新 `traj_motion_extent_all_valid` 与旧 `traj_motion_extent` 在有限位置上完全相同
- 且这些有限值全部为 `0`

所以可以排除“字段没换上”这一类问题。

### 2. 再确认 filter 层面到底卡在什么位置

对当前剩余的 `1018` 个 `varied_camera_3` 空 sample 继续拆分：

- `segment_len=1/2`：`280`
- `segment_len=3..8`：`738`

其中：

- `738` 个 `segment_len=3..8` sample 已经有非空 `traj_wrist_seed_mask`
- 但 `traj_manipulator_candidate_mask` 仍然全部为 `0`
- `traj_motion_extent_all_valid` 的 sample 级最大值在这 `738` 个样本里全部为 `0`

对这 `738` 个样本内部的 `wrist_seed` 轨迹进一步按 track 统计：

- `seed_tracks`: `3,779,212`
- `depth_rank <= 0.50`: `1,890,524`
- `depth_rank > 0.50`: `1,888,688`
- `motion_eq_0`: `3,779,212`
- `motion_gt_0`: `0`
- `motion_ge_0.03`: `0`

这说明：

- `near-depth` 不是唯一瓶颈
- 大约一半 seed 轨迹其实已经满足 `near-depth`
- 但即便这些轨迹已经通过 `near-depth`，它们的 motion 仍然全部是 `0`

因此：

- `MANIPULATOR_DEPTH_FAIL` 是真实存在的
- 但真正把 sample 打空的，是 `near-depth` 通过后也仍然全部 `motion=0`

### 3. 验证这些 sample 的源视频是否本来就静止

为了避免把问题误判成“原视频本来没动”，对几个代表空 sample 检查了源 RGB/depth 连续帧差异。

例如：

- `episode_00074_blue / varied_camera_3 / query_frame=71 / segment_len=7`
- `episode_00023_green / varied_camera_3 / query_frame=49 / segment_len=3`
- `episode_00066_green / varied_camera_3 / query_frame=57 / segment_len=7`

这些样本的源帧统计表现为：

- 相邻 RGB 帧平均绝对差约为 `2.8 ~ 3.9`
- 相邻 depth 帧平均绝对差约为 `0.012m ~ 0.017m`

也就是说：

- 源视频并不静止
- “轨迹静止”不是由源帧本身完全不变造成的

### 4. 定位到 tracker 输出端

继续做单样本 GPU 复现。

代表空样本：

- episode: `episode_00074_blue`
- camera: `varied_camera_3`
- query frame: `71`
- segment length: `7`

复现观察到：

- `_run_query_frame_core()` 产出的 `raw_coords` 形状为 `(7, 6400, 3)`
- `raw_trackwise_max_motion = 0.0`
- 所有 track 在所有帧上的 `raw_coords` 都完全相同

随后：

- `prepare_query_frame_sample_bundle()` 根据这个 `raw_coords` 生成 `traj_uvz`
- `traj_uvz_trackwise_max_motion = 0.0`
- 所有 track 在所有帧上的 `traj_uvz` 也完全相同

这说明：

- 冻结发生在 tracker 原始输出端
- 不是 `traj_uvz` 投影或后处理把原本有运动的轨迹弄成了静止

## 模型内部定位

### 1. 当前 checkpoint 的 `seq_len` 实际是 `16`

通过直接读取 checkpoint 内的 `cfg['model']['seq_len']`，以及加载后的模型对象 `model.seq_len`，当前实际使用的是：

- `seq_len = 16`

注意：

- `models/__init__.py` 里的某些 legacy fallback 路径会给出 `seq_len=12`
- 但当前 `tapip3d_final.pth` 这个 checkpoint 自带 `cfg`，实际加载值是 `16`

### 2. `PointTracker3D.streaming_forward()` 的 short-tail 逻辑

关键逻辑在 `models/point_tracker_3d.py`：

1. 如果序列长度不是 `seq_len // 2` 的整数倍，就用最后一帧复制 pad
2. `pred.coords` 初始化为：把 query 初始 3D 坐标在时间维上整段 repeat
3. 只有在

```text
for window_end in range(self.seq_len, T + 1, self.seq_len // 2)
```

里才会真正进入 tracking window
4. 最后直接返回裁回原长度后的 `pred`

对当前 `seq_len = 16`：

- `seq_len // 2 = 8`
- `segment_len = 7` 时，pad 后长度变成 `8`
- `segment_len = 8` 时，pad 后仍是 `8`
- 但 window 循环起点是 `16`

因此：

- `segment_len <= 8` 时，循环一次都不会执行
- `_wrapped_forward_window()` 不会被调用
- updater 也不会被调用
- 返回值就是“重复 query 初值”的 `pred.coords`

这与当前数据上的现象完全一致。

### 3. `len=9` 的对照复现

为了确认不是所有短序列都会冻结，又对一个 `len=9` 的非空样本做了同样的 GPU 复现：

- episode: `episode_00023_green`
- camera: `varied_camera_3`
- query frame: `43`
- segment length: `9`

这次观察到：

- `_wrapped_forward_window(start=0, end=16)` 被实际调用
- 初始 `coords_init_motion_max = 0.0`
- updater 第 1 轮就产生明显非零更新：
  - `delta_abs_max = 1.132812`
  - `delta_abs_mean = 0.071811`
- 最终 `raw_trackwise_max_motion = 0.103109`

这说明：

- 对 `segment_len >= 9` 的样本，tracking window 会真实执行
- 初始化的静止 query 坐标会被 updater 更新成有运动的轨迹

因此当前 `varied_camera_3` 上“`<=8` 全空、`>=9` 全非空”的分界，不是偶然统计相关，而是与 `seq_len=16` 下的 online short-tail 调度边界一致。

## 结论

本次问题可以分成“表象”和“根因”两层。

### filter 层面的表象

- `varied_camera_3` 剩余空 sample 在 filter 层面表现为：
  - 一部分轨迹被 `near-depth` 挡掉
  - 另一部分即使通过 `near-depth`，也会因为 `motion=0` 被 `motion gate` 挡掉
- 因此在 `traj_mask_reason_bits` 中同时会看到 `MANIPULATOR_DEPTH_FAIL` 和 `MANIPULATOR_MOTION_FAIL`

### 真正根因

- 当前 checkpoint 的 `PointTracker3D` 在线 tracking 路径在 `seq_len=16` 下，对 `segment_len <= 8` 的尾段 sample 根本不会执行任何 tracking window
- 这会直接返回“整段重复 query 初值”的静止 `pred.coords`
- 后续 `traj_uvz`、`traj_motion_extent_all_valid`、`motion gate` 只是忠实反映了这一冻结结果

因此：

- 改 motion 字段、改 motion gate 阈值，都不能解决 `segment_len <= 8` 这一类空 sample
- 对这些 sample，根因不在 `traj_filter_utils.py`
- 根因在 `models/point_tracker_3d.py` 的 short-tail online window 调度逻辑

## 后续修复方向

如果要真正修复这类 `varied_camera_3` 空 sample，优先级最高的方向是修改 `PointTracker3D.streaming_forward()` 的 short-tail 行为。可选方案包括：

1. 对 `T < seq_len` 的尾段，直接 pad 到 `seq_len` 并强制执行一个 window
2. 为 `T <= seq_len // 2` 的短尾段增加专门的 fallback tracking 路径
3. 明确区分：
   - `segment_len <= 8`：当前是“根本没进 tracking window”
   - `segment_len >= 9`：当前会进 window，问题不在同一层级

建议后续验证顺序：

- 先修 `segment_len <= 8`
- 再复查 `varied_camera_3` 剩余空 sample 是否被显著清掉
- 最后再判断是否还需要继续调 `near-depth` 或 `motion` filter 阈值
