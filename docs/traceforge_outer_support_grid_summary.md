# TraceForge 外层 `support_grid_ratio` 结论整理

本文只总结当前维护态 TraceForge 外层 `support_grid_ratio` 生成的那批 support points。

不讨论两类别的 support：

- 底层旧版 SpaTrackV2 的 `support_pts_q`
- 当前 `PointTracker3D` / corr processor 内部每个 query 自己的 KNN local support neighborhood

相关代码入口：

- [`scripts/batch_inference/infer.py`](../scripts/batch_inference/infer.py)
- [`utils/inference_utils.py`](../utils/inference_utils.py)
- [`models/point_tracker_3d.py`](../models/point_tracker_3d.py)
- [`training_tapip/datatypes.py`](../training_tapip/datatypes.py)

配套示意图：

- [`traceforge_support_grid_vs_keypoints.svg`](traceforge_support_grid_vs_keypoints.svg)

## 1. 先给结论

1. 这批 support points 是在“当前 query segment 的局部首帧”上采样的，不是整段视频的绝对第 0 帧。
2. 它们和主 keypoints 一样，都属于“首帧均匀采样”，但不是同一张网格。
3. 主 keypoints 用的是贴边网格；support 用的是内缩网格，边界会留一个 `m = W / 64` 的 margin。
4. support 的候选数量是设定值，但真正进入 tracking 的数量可能更少，因为会先做一次 `depth > 0` 过滤。
5. support 有可能和主 keypoint 重合；代码里没有去重，所以即使重合，也会作为两个 query 一起送进 tracker。
6. 进入当前维护版 tracker 后，这批 support 和普通 query 没有显式类型区别；它们会参与完整 tracking，再在外层输出前被裁掉。

## 2. support 的数量怎么确定

先计算 support 网格边长：

$$
G_s = \mathrm{round}(G \cdot \rho)
$$

其中：

- `G = grid_size`
- `rho = support_grid_ratio`

对应代码：

- [`_resolve_support_grid_size()`](../scripts/batch_inference/infer.py)

如果不考虑后续过滤，候选 support 数量是：

$$
N_{supp}^{cand} = G_s^2
$$

但真正进入 tracking 的数量是：

$$
N_{supp}^{actual}
=
\sum_{p \in \mathcal{G}_{supp}}
\mathbf{1}\{d_q(\mathrm{round}(p)) > 0\}
$$

也就是：

- 先生成 `G_s x G_s` 个候选点
- 再在 query frame 深度图上取深度
- 只保留 `depth > 0` 的点

对应代码：

- [`get_grid_queries()`](../utils/inference_utils.py)

所以结论很直接：

- 设定的是候选数量
- 实际进入 tracking 的 support 数量只会更少，不会更多

## 3. support 的位置怎么确定

### 3.1 时间位置

外层构造 query segment 时，会先把主 queries 的时间改成局部时间 `t = 0`：

- [`dense_segment_query_point[:, 0] = 0`](../scripts/batch_inference/infer.py)

support 也是对同一个局部 query frame 采样，并返回 `[t=0, X, Y, Z]`：

- [`queries = torch.cat([torch.zeros_like(...), world_coords], dim=-1)`](../utils/inference_utils.py)

因此这批 support 的时间坐标固定是当前 segment 的局部首帧。

### 3.2 空间位置

support 的 2D 候选位置来自 `get_points_on_a_grid()`：

- [`get_points_on_a_grid()`](../third_party/cotracker/model_utils.py)

它不是贴着边界采样，而是留一个 margin：

$$
m = W / 64
$$

若 support grid 边长为 `G_s`，则 support 的 2D 采样位置可以写成：

$$
x_j = m + \frac{W - 2m}{G_s - 1} j,
\qquad
y_i = m + \frac{H - 2m}{G_s - 1} i
$$

其中 `i, j = 0, ..., G_s - 1`。

然后：

- 用 `round(x_j), round(y_i)` 去深度图上取深度
- 但反投影到 3D 时，使用的仍然是原始浮点 `x_j, y_i`

这一点很重要，因为它决定了“共享同一个深度像素”和“3D 点完全相同”不是一回事。

## 4. support 和主 keypoint 的网格一样吗

不一样。

主 keypoint 用的是：

$$
x \in \mathrm{linspace}(0, W-1, G), \qquad
y \in \mathrm{linspace}(0, H-1, G)
$$

对应代码：

- [`_build_grid_keypoints()`](../scripts/batch_inference/infer.py)

也就是说主 keypoint 网格会贴到图像四条边。

support 用的是内缩网格：

- 左右各留 `W / 64`
- 上下也按同一公式内缩

因此两者都属于“首帧均匀采样”，但不是同一张网格。

## 5. support 和主 keypoint 可能重合吗

可能。

但要区分两层含义：

### 5.1 2D 浮点坐标完全重合

理论上可能，但一般不多见。

原因是：

- 主 keypoint 用的是边界到边界的 `linspace`
- support 用的是带 margin 的内缩 `linspace`

两套网格公式不同，所以大多数常见参数组合下，完全相同的 `(x, y)` 并不多。

### 5.2 四舍五入后落到同一个深度像素

这种更常见。

因为 support 取深度时会使用：

$$
(\hat x, \hat y) = (\mathrm{round}(x), \mathrm{round}(y))
$$

如果某个 support 和某个主 keypoint 的浮点坐标不同，但 round 之后落在同一个像素，那么：

- 它们会读取同一个深度值
- 但由于反投影时乘的是各自不同的浮点射线方向
- 所以得到的 3D 点通常只是接近，不一定完全一样

只有当二者的 `(x, y)` 本身也完全相同，并且时间帧也相同，最终 3D 点才会完全一致。

### 5.3 代码是否去重

当前外层实现没有对 support 和主 query 做显式去重。

它只是直接拼接：

- [`query_point = torch.cat([query_point, additional_queries], dim=1)`](../utils/inference_utils.py)

所以如果真的发生重合，它们仍会以两个 query 的形式一起进入 tracker。

## 6. support 进入 tracker 后会发生什么

进入 tracker 前，总 query 集合是：

$$
Q_{all} = [Q_{main}, Q_{supp}]
$$

对应代码：

- [`_inference_with_grid()`](../utils/inference_utils.py)

当前维护版模型入口是 `PointTracker3D`，它接收的就是统一的 `query_point` 张量：

- [`PointTracker3D.streaming_forward()`](../models/point_tracker_3d.py)

因此对 tracker 来说，这批外层 support：

- 不是 metadata
- 不是只给某个模块看的 side input
- 而是一批真正的 query tokens

它们会：

- 参与 corr processor 的局部相关特征计算
- 参与 `EfficientUpdateFormer` 的时空更新
- 参与整个 point-token 集合的全局上下文建模

但最终外层返回前，会把最后拼进去的 support queries 切掉：

- [`preds = preds.query_slice(slice(0, N_total - N_supports))`](../utils/inference_utils.py)
- [`Prediction.query_slice()`](../training_tapip/datatypes.py)

所以输出 sample 里通常看不到 support tracks。

## 7. 最终可以记住的版本

最短结论可以记成下面这几句：

- `support_grid_ratio` 生成的是“当前 query frame 上另一张更稀的均匀 support 网格”。
- 它不是主 keypoint 网格的子集，也不是简单复制主 keypoint。
- 名义数量是 `round(grid_size * support_grid_ratio)^2`，但实际数量会被 `depth > 0` 过滤压缩。
- 它和主 keypoint 可能重合，当前代码不会去重。
- 进入 tracker 后，它们会作为普通 query 一起参与 tracking；只是最后输出前被裁掉。
