# 深度波动图优化方案（2026-03-19）

## 1. 背景

这份文档用于承接上一轮 profiling 记录：

- 来源文档：`/data1/zoyo/projects/TraceForge_Reinforcement/docs/history/depth_volatility_bottleneck_analysis_2026-03-18.md`

上一轮已经确认：

- 在 external-only 几何与深度的当前流水里
- `grid_size=80`
- `future_len=32`
- 每个 camera 共享 query-frame schedule

`compute_depth_volatility_map()` 是 save 阶段的主要瓶颈。

本文档的目标不是重复 2026-03-18 的 profiling 结论，而是补充两件事：

- 复核当前主仓代码，确认瓶颈是否仍然存在
- 给出可直接实现的分阶段优化方案

## 2. 当前代码复核

截至 `2026-03-19`，当前仓库中的主链路仍然会无条件计算整段 `depth_volatility_map`。

关键路径：

- `scripts/batch_inference/infer.py::save_structured_data`
- `utils/traj_filter_utils.py::compute_depth_volatility_map`
- `utils/traj_filter_utils.py::compute_high_volatility_mask`
- `utils/traj_filter_utils.py::evaluate_temporal_depth_consistency`

当前行为：

1. `save_structured_data()` 在保存每个 camera 输出前，先对整段 `full_depths` 计算一次 `depth_volatility_map`
2. 每个 query sample 再复用这张图进入 `build_traj_filter_result()`
3. `basic / standard / strict` 三个 `filter_level` 默认都开启：
   - `use_temporal_depth_consistency=True`
   - `use_depth_volatility_guidance=True`

这意味着：

- 2026-03-18 文档里的瓶颈结构在当前代码中仍然成立
- 当前 default press-one-button 流水仍会稳定付出这部分成本
- 这块成本在逻辑上属于过滤辅助项，不是 tracker 主链路本身

## 3. 当前实现里额外确认到的两个事实

### 3.1 现在仍是“无条件先算，再决定是否用”

当前保存路径并不会先解析有效 filter config，再决定是否需要 volatility guidance。

工程含义：

- 即使未来某些 run 明确不需要 volatility guidance
- 只要仍走这条保存链路
- 这张图的计算成本就会先被付出去

所以第一优先级仍然是把它改成按需计算，而不是先继续打磨 full-res percentile 的底层实现。

### 3.2 当前 percentile 是分两次独立计算的

`compute_depth_volatility_map()` 当前实现是：

- 一次 `np.nanpercentile(..., 5, axis=0)`
- 一次 `np.nanpercentile(..., 95, axis=0)`

然后再做相减。

这不是错误，但在 NumPy 里属于额外付了两遍同类成本。对于同一个输入张量，改成：

```python
depth_lo, depth_hi = np.nanpercentile(depths_nan, [5.0, 95.0], axis=0)
```

通常能拿到一笔“行为等价”的纯实现收益。

## 4. 这轮补做的 benchmark

原始 profiling 文件：

- `/tmp/traceforge_timing_profile_20260318_gpu1/timing_summary_episode_00000_blue.json`

这次在当前机器上已经不存在，因此本文档中的新增收益数字来自两部分：

- 对当前代码路径的静态复核
- 在当前仓库环境下补做的 synthetic benchmark

synthetic workload 维度与上一轮 bottleneck 文档保持同量级：

- `T=39`
- `H=720`
- `W=1280`
- `dtype=float32`

### 4.1 现实现 benchmark

直接复用当前 `compute_depth_volatility_map()` 的实测结果：

- full resolution `(39, 720, 1280)`：`76.925s`
- stride downsample `1/4` 后 `(39, 180, 320)`：`4.776s`
- stride downsample `1/8` 后 `(39, 90, 160)`：`1.194s`

这说明：

- full-res percentile 的量级与上一轮 `83s+` profiling 结论一致
- 单纯靠空间降采样就能把这块成本压到个位数秒

### 4.2 单次 percentile 的等价收益

在中等尺寸张量上对比：

- 两次独立 `nanpercentile`：`19.385s`
- 一次 `nanpercentile(..., [5,95], axis=0)`：`10.515s`

两种写法得到的 volatility 统计结果一致。

这说明：

- 即使不引入任何近似
- 仅把双 percentile 改成一次联合 percentile
- 也值得作为低风险优化先做

### 4.3 低分辨率路径的量级判断

另外补做了一个纯 NumPy block-mean downsample 的量级测试：

- `1/4` block-mean downsample：`0.647s`
- `1/8` block-mean downsample：`0.415s`

在 downsample 后张量上再做一次联合 percentile：

- `1/4` total：`3.164s`
- `1/8` total：`1.043s`

注意：

- 这里的 block-mean 结果与原 full-res volatility 数值分布不等价
- 它只是说明“先降采样，再做 percentile”在时间上非常有吸引力
- 默认主路径仍应先保持 `factor=1`，不要直接改默认行为

## 5. 优化优先级

### 5.1 P0：改成按需计算

目标：

- 只有当有效 filter config 同时满足
  - `use_temporal_depth_consistency=True`
  - `use_depth_volatility_guidance=True`
- 才计算 volatility guidance

建议接口：

- 新增 CLI：`--disable_depth_volatility_guidance`

行为：

- 只关闭 volatility guidance
- 不关闭 temporal consistency 主检查

预期收益：

- 对明确不需要 volatility guidance 的 run，能直接拿掉这部分全部耗时

风险：

- 低

### 5.2 P1：先做行为等价优化

目标：

- 不改变默认过滤语义
- 先拿最稳的实现级收益

建议改动：

- `compute_depth_volatility_map()` 改成一次联合 `np.nanpercentile(..., [low, high], axis=0)`
- `high_volatility_mask` 在每个 camera 级别只阈值化一次
- sample 级别直接复用预计算 mask，不要重复做相同 threshold

预期收益：

- 对默认 full-res 路径，能拿到一笔无近似、低风险加速

风险：

- 低

### 5.3 P2：为 volatility guidance 引入可选降采样

目标：

- volatility guidance 保留
- 但不再强制用 full-res depth video 做 percentile

建议接口：

- 新增 CLI：`--depth_volatility_downsample_factor`
- 默认值：`1`

建议首个实验档：

- `factor=4`

建议实现：

1. 对 `full_depths` 先做确定性 downsample
2. 在低分辨率上计算 volatility map
3. camera 级别阈值化成 `high_volatility_mask`
4. 用最近邻上采样回原分辨率，供 `evaluate_temporal_depth_consistency()` 直接索引

为什么优先推荐 mask 上采样，而不是在读取时做坐标映射：

- 与现有 `evaluate_temporal_depth_consistency()` 接口更兼容
- 改动面更小
- 更容易做 `factor=1` 对照验证

预期收益：

- `factor=4`：大概率把该阶段压到数秒级
- `factor=8`：还能更快，但行为漂移风险更高

风险：

- 低到中
- 主要风险在细小边缘区域的 mask 漂移

### 5.4 P3：按 camera 自动缓存

这张图只依赖于：

- source depth provenance
- source frame indices
- `min_depth / max_depth`
- `low/high_percentile`
- `downsample_factor`
- 输入分辨率

所以适合按 camera 做缓存。

建议缓存内容：

- 保存 downsample 后的 `volatility_map`
- 不直接缓存 threshold 后的 `high_volatility_mask`

这样做的原因：

- 改 `volatility_mask_percentile` 时无需重算 percentile 主体
- 同一 camera 重跑时可直接复用

建议缓存位置：

- 当前 camera 输出目录下的私有缓存子目录

建议 cache key 至少包含：

- `source_depth_path`
- `source_frame_indices`
- depth 源文件签名（如 `size + mtime_ns`）
- `min_depth`
- `max_depth`
- `low_percentile`
- `high_percentile`
- `downsample_factor`
- 原图尺寸

预期收益：

- 首次运行无收益
- 同一输入重复运行时，这部分成本接近归零

风险：

- 低

### 5.5 P4：结构性重写为“只在访问位置算 volatility”

这是结构上更优的方向，但不建议先做。

更合理的终态是：

- 不再先构造整张 `H x W` 的 volatility map
- 只在轨迹真正重投影访问到的位置上做局部统计

优势：

- 避免把大量成本花在从未被访问的像素上

问题：

- 改动大
- 需要重写当前 `depth_volatility_map -> high_volatility_mask -> temporal_consistency` 的接口关系
- 一致性验证比 P0/P1/P2/P3 更重

结论：

- 作为后续结构优化项保留
- 不放在第一阶段

## 6. 推荐实施顺序

推荐按下面顺序落地：

1. `P0`：按需计算 + `--disable_depth_volatility_guidance`
2. `P1`：单次联合 percentile + camera 级别单次 threshold
3. `P2`：`--depth_volatility_downsample_factor`，默认仍为 `1`
4. `P3`：camera 级自动缓存
5. `P4`：只对 query/访问位置做 volatility 统计

原因：

- 这条顺序先拿低风险收益
- 保证 `factor=1` 下能和当前行为逐步对齐
- 不会一上来把实现复杂度抬太高

## 7. 验收标准

第一阶段验收应至少覆盖下面几项：

#### 7.1 行为不回归

在 `--depth_volatility_downsample_factor=1` 下：

- `traj_valid_mask` 与当前实现一致
- `traj_high_volatility_hit` 与当前实现一致
- `traj_stable_depth_consistency_ratio` 与当前实现一致

#### 7.2 guidance 可显式关闭

开启：

- 走当前 volatility guidance 逻辑

关闭：

- 不再要求先算 `depth_volatility_map`
- temporal consistency 仍按 all-frame 路径工作

#### 7.3 `factor=4` 做漂移评估

建议用已有对比脚本或等价方法比较：

- keep ratio 漂移
- `traj_valid_mask` 差异比例
- `traj_high_volatility_hit` 变化

在人工验收通过前：

- 不修改默认值

#### 7.4 cache 必须可观测

同一 camera 连续运行两次时：

- 第一次 cache miss
- 第二次 cache hit
- 第二次 save 阶段明显快于第一次

## 8. 默认策略

推荐默认值：

- `--disable_depth_volatility_guidance`：默认关闭该开关，即默认仍启用 guidance
- `--depth_volatility_downsample_factor`：默认 `1`

推荐实验档：

- `factor=4`

当前不建议默认直接切到：

- `factor=8`
- GPU percentile backend
- query-only volatility 重构

## 9. 结论

截至 `2026-03-19`，当前主仓代码里：

- depth volatility guidance 仍是 save 阶段最值得优先优化的 CPU 瓶颈之一
- 这部分逻辑仍然适合先走“按需计算 + 行为等价优化 + 可选降采样 + 缓存”的路线
- 在更简单、风险更低的优化完成前，没有必要先上 GPU percentile 或更激进的接口重写

如果目标是尽快降低当前 external-only press-one-button 流水的端到端耗时，建议首先落地：

1. `--disable_depth_volatility_guidance`
2. 联合 percentile
3. `--depth_volatility_downsample_factor`
4. camera 级缓存
