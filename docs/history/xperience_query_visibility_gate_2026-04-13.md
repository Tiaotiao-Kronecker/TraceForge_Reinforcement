# xperience query visibility gate 试验（2026-04-13）

## 目的

验证一个非常直接的 first-pass 规则：

- 先把 query frame 的 keypoint lift 成 3D world point
- 再用每一帧相机位姿把这些 3D 点投影到后续帧
- 如果某个点在后续视角里不再处于有效可见区域，则把它标记为不可靠
- 第一轮先不改 tracker 本体，只做：
- 1. 2D 可视化
- 2. post-track 轨迹裁剪 proxy

这里的“可见”当前只使用最保守的几何口径：

- depth 有效
- 投影后在图像内
- 不在相机后方
- 满足 `min_border_dist_px`

当前没有加入 occlusion 判定，因此这是一个纯 geometric FOV gate，不是完整 visibility model。

## 新增工具

- `utils/query_visibility_gate_utils.py`
- `utils/test_query_visibility_gate_utils.py`
- `scripts/visualization/export_query_visibility_gate_2d.py`
- `scripts/data_analysis/export_query_visibility_pruned_trajectory.py`

## 2D 可视化 smoke 结果

统一对象：

- `trajectory_dense_none_grid80_notrim/stereo_left`
- `query_frames = 0,4`

### `00190`

`q0`：

- `reliable = 3766 / 6400`
- `removed = 2634 / 6400`
- `reliable_ratio = 0.588`
- `removed border median = 32.342 px`

`q4`：

- `reliable = 3721 / 6400`
- `removed = 2679 / 6400`
- `reliable_ratio = 0.581`
- `removed border median = 32.342 px`

解释：

- 这条 gate 在 `00190` 上明显会砍掉大量边界附近点
- 和 `00190 = bad seed / 边缘深度 / 飞线` 的主归因一致
- 因此它最可能对 `00190` 有真实帮助

### `00435`

`q0`：

- `reliable = 3792 / 6400`
- `removed = 2608 / 6400`
- `reliable_ratio = 0.592`
- `removed border median = 32.342 px`

`q4`：

- `reliable = 3606 / 6400`
- `removed = 2794 / 6400`
- `reliable_ratio = 0.563`
- `removed border median = 32.342 px`

解释：

- 它同样主要在删边界和后续离场点
- 但 `00435` 的主问题不是“离场后还在追”本身
- 所以这条规则可能会止掉一部分尾巴，但不会根治 scene-level wobble 主体

### `04234`

`q0`：

- `reliable = 4184 / 6400`
- `removed = 2216 / 6400`
- `reliable_ratio = 0.654`
- `removed border median = 45.278 px`

`q4`：

- `reliable = 4719 / 6400`
- `removed = 1681 / 6400`
- `reliable_ratio = 0.737`
- `removed border median = 19.405 px`

解释：

- 它会删掉一批明显会离场的点
- 但 `04234` 的主问题仍更像 tracker / local interaction
- 所以这条规则更像辅助止损，而不是主修法

## 关键观察

### 1. 这条 gate 不是纯重复已有 query-depth prefilter

如果它只是重复已有 query-frame 深度过滤，那么被删点应该大多在 `step=0` 就无效。

但当前统计显示：

- `00190 q0`: `step=0` 仅 `377`
- `00435 q0`: `step=0` 仅 `414`
- `04234 q4`: `step=0` 仅 `338`

说明它额外抓到的是：

- query 时仍可用
- 但后续某一帧已经离开视角
- tracker 仍可能继续追的点

### 2. 当前规则非常强，会删掉较大比例点

三个 case 上，保留率大约落在：

- `0.56 ~ 0.74`

也就是它不是一个“极轻量修补”，而是会显著改变 query 集。

### 3. 目前更像边界/FOV gate，而不是完整可见性判定

因为没有做 occlusion check，所以当前保留/删除的主要几何信号仍是：

- 边界距离
- 是否出画
- 是否到相机后方

因此它最有希望改善的是：

- 边界飞线
- 出视野后仍被追踪的假延续轨迹

而不是：

- 仍在画面内但 tracker 滑移的点
- `00435` 这类 in-view geometry wobble 主体
- `04234` 这类 in-view local interaction 主体

## 当前结论

到这一轮为止，最准确的判断是：

- `00190`：值得继续，最可能有真实收益
- `00435`：可作为辅助约束，但不会是主修法
- `04234`：只会局部止损，不会触及主因

## 注意事项

当前新增的 `visibility-pruned trajectory` 是 post-track proxy：

- 它只是把已有 sample 的 `traj_valid_mask` 按 visibility gate 收紧
- 方便直接做 4D 可视化对比
- 还不是“真正把这些点从 tracker 输入里剔除后重新跑”的结果

因此如果后续要严肃验证算法收益，应再做一轮：

- pre-tracker gating
- 真 rerun
- 再比较 scene wobble / tracker-geometry / 4D 可视化
