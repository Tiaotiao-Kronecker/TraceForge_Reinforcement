# tracker bf16 最小环境、support 默认关闭与复杂度结论

日期：2026-04-13

分支：`feat/external-only-maintained-mode`

## 结论摘要

- 当前维护态默认继续保留 `num_iters=3`
- 当前维护态默认 `filter_level=none`
- 当前维护态默认把外层 support points 关闭：`support_grid_ratio=0.0`
- 已新增一个尽量独立的最小 tracker 推理环境，用来单独比较 `fp32` / `bf16`、统计 FLOPs，并给出 H100 roofline 下界估算

## 本轮代码落点

- `scripts/minimal_tracker_inference/`
  - `minimal_tracker_core.py`
  - `run_minimal_tracker_inference.py`
  - `README.md`
- maintained 入口默认值
  - `scripts/batch_inference/infer.py`
  - `scripts/batch_inference/batch_infer_press_one_button_demo.py`
  - `scripts/batch_inference/repair_empty_samples_press_one_button_demo.py`

这次调整后的 maintained 默认语义是：

- `filter_level=none`
- `support_grid_ratio=0.0`
- `num_iters=3`

## 为什么要先把 support points 关掉

`support_grid_ratio` 追加的是 tracker 内部的辅助 query，不是最终主输出轨迹。

把它设成 `0.0` 的作用是：

- 先简化 maintained 默认前向路径
- 降低 query 数和重复 unprojection / refinement 负载
- 让 `fp32` / `bf16` 对比更容易聚焦在 tracker 主体，而不是 support queries 带来的额外计算

如果后续要重新打开 support points，应把它当成显式实验开关，而不是默认配置。

## bf16 最小推理环境

最小环境只保留 tracker 需要的核心输入：

- `video`
- `depths`
- `intrinsics`
- `extrinsics`
- `query_points_world` 或由 query grid 反投影得到的 world-space queries

它刻意不包含这些系统层逻辑：

- episode 扫描
- shared schedule 构建
- trajectory filtering
- artifact 落盘编排
- batch worker 调度

因此它更适合回答三个问题：

1. tracker 前向本身的 wall time 有多少
2. `fp32` 和 `bf16` 的 profiler FLOPs / roofline 下界如何
3. 当前代码为什么没有接近 H100 的理论峰值

## 复杂度结论

当前最小环境里记录了两类复杂度：

- symbolic complexity
  - `window_count`
  - `query_count`
  - `support_query_count`
  - `total_query_count`
  - `total_unprojection_points`
  - `iterative_track_state_updates`
- profiler FLOPs
  - 只统计 PyTorch profiler 能识别到的算子 FLOPs，因此通常是下界

一个代表性符号例子：

- `T=32`
- `H=180`
- `W=320`
- `query_grid=80`
- `support_grid=64`
- `seq_len=16`
- `num_iters=3`

得到：

- `window_count = 3`
- `query_count = 6400`
- `support_query_count = 4096`
- `total_query_count = 10496`
- `total_unprojection_points = 7,372,800`
- `iterative_track_state_updates = 1,511,424`

`bf16` 不会改变这些算法复杂度项。它只可能改变：

- kernel 选型
- Tensor Core 利用率
- 实际可达到的吞吐上限

## H100 理论速度上限

最小环境中使用的 roofline 峰值常量为：

- H100 SXM
  - `fp32 = 67.0 TFLOPS`
  - `bf16 = 989.5 TFLOPS`
- H100 NVL
  - `fp32 = 60.0 TFLOPS`
  - `bf16 = 835.0 TFLOPS`

这里的 BF16 数字按 dense tensor core 峰值处理，来自对 NVIDIA sparsity-on 标称值减半后的近似。

因此从纯算力上看，`bf16` 的理论峰值远高于 `fp32`。但这只是 roofline 下界，不代表当前代码能自然跑到这个量级。

## 当前为什么跑不到理论 bf16 速度

当前主链路虽然已经有一层外部：

- `scripts/batch_inference/infer.py`
  - `torch.cuda.amp.autocast(dtype=torch.bfloat16)`

但它不是“真实的纯 bf16 tracker 路径”。主要原因有三类：

### 1. 显式 float32 回退

- `utils/common_utils.py`
  - `ensure_float32()` 会关闭 autocast，并把 `bf16/fp16` 张量拉回 `float32`
- `models/point_tracker_3d.py`
  - 存在显式 `float32` 区域
  - 还有 `coords` 必须保持 `float32` 的断言

### 2. 热路径里仍有大量非 GEMM 型工作

- repeated window prep
- `batch_unproject`
- clone / permute / scatter
- support-query 拼接和裁剪

这些部分不会像大矩阵乘那样稳定吃满 Tensor Core。

### 3. 现有实现更像“混合精度包裹”，不是从数据流设计出来的 bf16 path

也就是说，当前路径更接近：

- 外层尝试用 `bf16`
- 内层关键几何与状态更新再回到 `fp32`

这种结构通常会带来：

- 速度提升有限
- dtype 转换开销
- 实际 kernel 碎片化

## bf16 会不会降低轨迹质量

结论是：可能会有轻微数值漂移，但“只改配置/包一层 autocast”并不必然明显降低轨迹质量。

更具体地说：

- 算法语义没变
- query/frame/schedule/filter 逻辑没变
- 主要变化是部分算子会尝试走较低精度

在当前代码结构下，由于很多几何敏感路径仍强制保留在 `float32`，所以：

- config 级 `bf16` 更可能表现为
  - 速度收益有限
  - 轨迹有小幅漂移
- 而不是直接出现系统性崩坏

但如果后续要把几何 lift / reprojection / iterative update 等关键路径也真正改成更深的 bf16 化，就不能只靠理论判断，必须做真实 case 的 A/B 验证，重点检查：

- 长时间漂移
- depth edge 附近的抖动
- 遮挡后的恢复
- 小物体和接触边界的稳定性

因此当前更稳妥的判断是：

- “配置级 bf16” 可能带来轻微质量变化，但大概率不是主要风险点
- “真正深入到 tracker 核心的数据流 bf16 化” 需要单独质量回归

## 当前验证边界

本轮仓库内完成了：

- 代码路径核对
- 最小环境构建
- 单元测试与语法检查

本轮没有完成的，是带真实 checkpoint 的 H100 实机 benchmark，因为当前会话环境里：

- 没有可用 CUDA 设备
- 没有 `nvidia-smi`
- 当前工作目录下也没有现成 `checkpoints/`

所以这里记录的是：

- 代码级结论
- 复杂度级结论
- roofline 级理论判断

不是实机吞吐实测结论。
