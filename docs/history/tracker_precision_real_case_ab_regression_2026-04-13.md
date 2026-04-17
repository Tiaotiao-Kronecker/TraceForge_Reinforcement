# tracker precision real-case A/B regression

日期：2026-04-13

## 硬件

- 实测机器不是 8x H100
- 2026-04-13 当天实际查询到的是：
  - `8 x NVIDIA H200`
  - driver `580.126.09`

## 代码前提

本轮 A/B 基于以下代码状态：

- 已新增 `tracker_precision_mode`
  - `fp32`
  - `autocast_bf16`
  - `deep_bf16`
- 已把 corr feature / update 路径里的主要强制 `float32` 障碍拆开
- 仍然保留在 `float32` 的关键部分主要是：
  - 几何投影/反投影
  - KNN 坐标搜索
  - tracker 坐标状态本身

## 回归脚本

使用：

- `scripts/minimal_tracker_inference/run_real_case_precision_ab.py`

输出：

- `/tmp/traceforge_real_case_ab_with_autocast_2026-04-13.json`

## case 与配置

case：

- episode: `/DATA/disk0/shared/press_one_button_demo_v5_test/episode_00005_blue`
- camera: `varied_camera_1`

配置：

- `max_num_frames = 32`
- `query_frames = [0, 8, 16]`
- `query_grid_size = 80`
- `support_grid_size = 0`
- `num_iters = 3`
- `warmup_runs = 1`

## 速度与显存

| mode | total wall time | mean/query frame | peak memory |
| --- | ---: | ---: | ---: |
| `fp32` | `24.87s` | `8.29s` | `8.54 GB` |
| `autocast_bf16` | `17.96s` | `5.99s` | `7.00 GB` |
| `deep_bf16` | `18.12s` | `6.04s` | `6.95 GB` |

相对 `fp32`：

- `autocast_bf16`
  - speedup `1.385x`
  - peak memory `-1.54 GB`，约 `-18.0%`
- `deep_bf16`
  - speedup `1.373x`
  - peak memory `-1.59 GB`，约 `-18.6%`

## 轨迹回归指标

以下都以 `fp32` 作为 reference，只是 regression，不是对真值的精度评估。

### `autocast_bf16` vs `fp32`

- 3D coord L2
  - mean `0.879 mm`
  - p95 `1.813 mm`
  - max `249.5 mm`
- visibility disagreement
  - `1.460%`

### `deep_bf16` vs `fp32`

- 3D coord L2
  - mean `0.891 mm`
  - p95 `1.824 mm`
  - max `249.7 mm`
- visibility disagreement
  - `1.452%`

### deep path 的附加观察

- `deep_bf16` 相比 `autocast_bf16`
  - bulk drift 基本同一量级
  - 速度没有继续明显拉开
  - peak memory 再小一点，但幅度很小

这说明本轮把主要 fp32 障碍拆掉以后，当前瓶颈已经不太在“参数是不是显式存成 bf16”，而更可能在：

- 几何/KNN 仍保留的 fp32 区域
- 非 GEMM 型热路径
- window prep / scatter / 临时张量 churn

## 结论

1. 在真实 case 上，bf16 路径已经有真实吞吐收益，不是纸面优化。
2. 当前 case 上，`autocast_bf16` 和 `deep_bf16` 的 bulk quality regression 都较小：
   - mean 约 `0.9 mm`
   - p95 约 `1.8 mm`
3. 但 tail 仍然存在稀有大漂移：
   - 3D max 接近 `0.25 m`
   - 2D reprojection max 接近 `300 px`
4. 因此目前更合适的结论不是“可以直接把 maintained 默认切到 bf16”，而是：
   - bf16 已经值得继续做真实集回归
   - 需要先定位这些大尾部 outlier 的触发条件

## 建议的下一步

1. 对 regression 脚本里 max drift 的 query/frame 做可视化定位
2. 扩到至少：
   - 多个 episode
   - 多个 camera
   - 一组更长序列
3. 如果 bulk drift 继续稳定，而 outlier 能被解释并收敛，再考虑是否切换默认 precision mode

## 更彻底 bf16 化的收益预估

下面这一节是工程预估，不是实测数据。

如果继续把当前残留的“硬编码 fp32 区域”进一步压到 bf16，例如：

- tracker 坐标状态
- 几何投影/反投影
- KNN 坐标搜索相关中间量

那么相对当前已经完成的 `autocast_bf16` / `deep_bf16`，预期额外收益大概率不会特别大。

当前较合理的区间预估是：

- 在现有 `1.37x ~ 1.39x` 基础上，再拿到额外 `0% ~ 10%` 更现实
- 比较乐观但仍然可信的情况，大约是额外 `10% ~ 15%`
- 也就是总 speedup 相对 `fp32` 可能落在 `1.45x ~ 1.60x`
- 仅靠 dtype 改写，通常不太可能稳定超过 `1.7x`

主要理由：

1. `autocast_bf16` 和 `deep_bf16` 已经非常接近，说明“把更多模型参数和 feature tensor 显式放进 bf16”本身，已经没有继续带来明显 runtime 改善。
2. 当前残留的 fp32 区域主要是几何/KNN/状态维护，这些路径通常：
   - Tensor Core 受益较弱
   - 更像索引、邻域搜索、坐标变换、scatter/gather
   - 对数值稳定性更敏感
3. 所以把这些区域也强推成 bf16，常见结果往往是：
   - 风险上涨更快
   - 收益上涨更慢

换句话说，下一阶段如果目标是继续提速，优先级通常应该是：

- 减少热路径中的同步与临时张量 churn
- 优化 window prep / scatter / gather / KNN 数据流
- 再评估 `torch.compile`、CUDA graph、算子级重写等更偏执行路径的优化

而不是把“剩余所有 fp32”机械式改成 bf16。
