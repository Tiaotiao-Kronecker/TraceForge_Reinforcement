# 深度波动图瓶颈分析

日期：2026-03-18

## 背景

当前维护中的流程：

- external-only 几何与深度
- button/sim episode 批处理推理
- 每个 episode 共享 query-frame schedule
- `grid_size=80`
- `future_len=32`

本次 profiling 目标为：

- 数据集：`/data2/yaoxuran/press_one_button_demo_v4`
- episode：`episode_00000_blue`
- 相机：`varied_camera_1`、`varied_camera_2`、`varied_camera_3`
- GPU：`GPU1`（为了避免干扰，在空闲 GPU 上重跑）

原始耗时统计文件：

- `/tmp/traceforge_timing_profile_20260318_gpu1/timing_summary_episode_00000_blue.json`

## 实际工作负载

对这个 episode：

- 每个 camera 加载帧数：`39`
- `trajectory_valid.h5` 中的 episode fps：`30`
- 每个 camera 的共享 query frame 数：`6`
- 对齐后的 raw query frame 序号：`[1, 13, 22, 31, 33, 36]`
- 深度分辨率：`720 x 1280`

## 关键结论

每个 camera 的平均耗时：

- 总耗时：`176.46s`
- `process_single_video`：`86.10s`
- `save_structured_data`：`90.35s`
- `compute_depth_volatility_map`：`83.37s`
- tracker inference：`60.42s`
- prepare_inputs：`22.31s`

对应占比：

- `compute_depth_volatility_map` 约占单 camera 总耗时的 `47%`
- `compute_depth_volatility_map` 约占 save 阶段耗时的 `92%`
- tracker inference 约占单 camera 总耗时的 `34%`

这说明当前最大的瓶颈既不是磁盘 IO，也不是 RGB/depth 加载，而是 `compute_depth_volatility_map()`。

## 为什么它会这么慢

实现位置：

- `utils/traj_filter_utils.py::compute_depth_volatility_map`

这个函数做的事情是：

1. 先在整段 raw depth video 上构造 valid mask
2. 再构造一个 `NaN` masked 的 `(T, H, W)` depth 张量
3. 对整个视频做两次 `np.nanpercentile(..., axis=0)`
   - 一次算 5% 分位数
   - 一次算 95% 分位数
4. 两者相减，得到每个像素位置的时间波动图

这次样本的输入规模是：

- `T=39`
- `H=720`
- `W=1280`
- 单帧像素数：`921,600`

所以这个函数本质上是在对 `921,600` 个像素位置分别沿时间维做两次分位数统计。

它慢的主要原因：

- percentile 比 mean/std 这类归约操作贵得多
- 它跑在 CPU 上，不走 GPU
- 它处理的是整段全分辨率 depth video
- `nanpercentile` 往往会产生较重的临时内存搬运
- 对 `(T,H,W)` 数组做 `axis=0` 的时间维统计，不是那种很便宜的连续流式归约

所以本质上不是“代码写得长”，而是“操作本身就很贵”。

## 为什么这不是写盘问题

profiling 拆分结果显示：

- 每个 camera 真正估算出来的 sample NPZ 写盘时间大约只有：`0.028s`
- `write_scene_meta`：可以忽略
- `compute_depth_volatility_map`：`83s+`

也就是说，save 阶段之所以慢，主要不是文件写入慢，而是在写入前做了一次很重的全视频 depth 统计。

## 这个 volatility map 实际用来干什么

调用链如下：

- `scripts/batch_inference/infer.py::save_structured_data`
- `utils/traj_filter_utils.py::compute_depth_volatility_map`
- `utils/traj_filter_utils.py::compute_high_volatility_mask`
- `utils/traj_filter_utils.py::evaluate_temporal_depth_consistency`

它的作用是：

- 生成一张高波动像素 mask
- 在 “stable temporal consistency” 检查中，把这些不稳定 depth 区域排除掉

所以它本质上是一个过滤辅助项，不是 tracker 主链路本身。

## 总结

这个瓶颈的结构已经很清楚了：

- 最大瓶颈：`compute_depth_volatility_map`
- 第二瓶颈：tracker inference
- 加载和真正写盘都不是大头

工程含义也很明确：

- 如果目标是尽快降低端到端耗时，`compute_depth_volatility_map` 应该是第一优先级优化对象
- 相比流水线的其他部分，它更适合做简化、近似或者缓存

## 优化方案

### P0：先停止“无条件计算”

当前代码路径在 `scripts/batch_inference/infer.py` 中，会在 `save_structured_data()` 阶段无条件调用 `compute_depth_volatility_map()`。

第一步应该先改这个：

- 先解析当前有效的 filter config
- 只有在 `use_depth_volatility_guidance=True` 时才计算这张图
- 提供一个 CLI 开关，例如 `--disable_depth_volatility_guidance`

为什么这一步最重要：

- 现在的成本是先付出去，再决定当前 run 是否真的需要 volatility guidance
- 对 profiling 和 ablation 来说，这能立刻提供一个干净的关闭开关

预期收益：

- 如果这个 workload 下直接关闭 volatility guidance，单 camera 耗时可以从约 `176s` 降到约 `93s`
- 大约是 `1.9x` 的提速

风险：

- 低
- 只有用户显式关闭 volatility guidance 时，行为才会变化

### P1：对 volatility 计算降采样

如果还需要保留 volatility guidance，这是最实用的优化方案。

思路：

- 在降采样后的 depth 上计算 volatility map，例如 `1/4` 或 `1/8` 分辨率
- 然后：
  - 要么把最终 mask 用最近邻上采样回全分辨率
  - 要么在读取 projected coordinates 时直接映射到低分辨率 mask

为什么这很适合当前场景：

- volatility map 只是 temporal consistency filtering 的一个粗粒度空间引导
- 它不是最终轨迹输出本身
- 它不需要在 `720x1280` 的全分辨率上保持精确的 per-pixel percentile fidelity

预期收益：

- `1/4` 分辨率，像素数下降约 `16x`
- `1/8` 分辨率，像素数下降约 `64x`
- 实际上，当前 `83s` 的成本大概率能降到个位数秒或者十几秒

风险：

- 低到中
- 在细小 depth 边界附近，过滤行为会有一定变化，但对这种 guidance mask 通常是可以接受的

### P2：按 camera 做缓存

这张 volatility map 只依赖于：

- 某个 camera 的 raw depth video
- 当前使用的 source frame 列表
- depth threshold / percentile 配置

所以它很适合缓存。

建议做法：

- 保存一个缓存文件，例如 `depth_volatility_map_<spec_hash>.npy`
- hash 中包含 source depth path、source frame indices、`min_depth`、`max_depth`、percentiles 等信息
- 相同输入重跑时直接复用

为什么有价值：

- 当前开发流程里，经常会对同一个 episode/camera 反复重跑，只改 schedule 或 filter 逻辑
- 每次都重新付出 `83s` 的成本，性价比很低

预期收益：

- 首次运行无收益
- 相同输入重复运行时，几乎可以把这部分成本降到 0

风险：

- 低
- 主要是缓存设计和失效规则要写清楚

### P3：把“全图 volatility”改成“只对 query 位置算”

这是结构上最优的优化，但改动也最大。

当前行为：

- 先对整张 `H x W` 图做完整 volatility map
- 后面真正使用时，其实只在轨迹重投影到的位置上取值

更合理的思路：

- 不再先算整张图
- 只对实际被 projected trajectories 命中的位置计算 volatility
- 或者直接在 temporal consistency 检查内部按 query 位置做局部时间统计

为什么这更合理：

- 当前的全图计算，大部分工作都花在了后续根本不会被访问的像素上
- 真正的消费者是 track reprojection 路径，而不是完整图像

预期收益：

- 潜在收益可能比降采样还大
- 但需要重写当前 volatility-mask 接口

风险：

- 中到高
- 改动更大，和现有行为的一致性验证也更难

### P4：改成 GPU percentile backend

这是可选项，而且优先级低于 P0/P1/P2。

可能的方向：

- 用 PyTorch / CuPy 之类的 CUDA 后端来计算 volatility

为什么我不建议先做它：

- percentile / quantile 在 GPU 上本身仍然是重操作
- 工程复杂度更高
- 在更简单的算法级优化之前，没必要先走这条路

风险：

- 中等
- backend 复杂度、显存压力、可维护性都更差

## 推荐顺序

建议的实现顺序：

1. 先加真正的开关，避免无条件计算 volatility
2. 再做降采样 volatility 计算
3. 再加 per-camera 缓存
4. 如果还不够，再考虑 query-only 的结构重写

## 对当前流水线的实际建议

对当前 button/sim 维护流程来说，更务实的选择是：

- 保留这个功能
- 但默认使用一个便宜版本
- 不要继续把当前“全分辨率、全视频、两次 percentile”的路径当成默认实现

更具体地说：

- 短期：先加开关，并且 benchmark 一次“关闭 volatility guidance”的效果
- 中期：把降采样 volatility 变成维护默认值
- 长期：如果过滤收益仍然足够大，再考虑缓存和 query-only 重写
