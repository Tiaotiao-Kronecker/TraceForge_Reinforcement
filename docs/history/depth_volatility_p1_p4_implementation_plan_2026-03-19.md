# 深度波动优化实施计划：P1 + P4（2026-03-19）

## 1. 背景

本文档把 `docs/history/depth_volatility_optimization_plan_2026-03-19.md` 中准备优先实施的两项方案收敛为一份可直接执行的实施说明：

- `P1`：单次联合 percentile + camera 级别单次 threshold
- `P4`：只对 query/访问位置做 volatility 统计

目标是给后续基于远端分支实施优化时提供一份决策完整的落地计划，避免再次讨论接口口径。

## 2. 实施策略

推荐按两个阶段落地，先拿低风险收益，再引入结构性优化：

1. 先实现 `P1`
2. 再实现 `P4`

原因：

- `P1` 是行为等价优化，风险低，适合先作为基线收敛
- `P4` 会改变 volatility percentile 的统计集合，虽然收益更大，但需要保留旧路径做对照
- 把两者拆开后，更容易定位回归来源

## 3. P1：单次联合 percentile + camera 级别单次 threshold

### 3.1 目标

- 不改变默认过滤语义
- 保持现有 sample 输出字段不变
- 去掉当前每个 sample 内重复的 volatility threshold 成本

### 3.2 现状问题

当前链路是：

1. `save_structured_data()` 先算一次 camera 级 `depth_volatility_map`
2. `build_traj_filter_result()` 在每个 sample 内再次调用 `compute_high_volatility_mask()`
3. `compute_depth_volatility_map()` 内部仍是两次独立 `np.nanpercentile`

这意味着：

- 同一 camera 的 threshold 被重复计算
- percentile 本身也多付了一遍同类成本

### 3.3 具体改动

在 `utils/traj_filter_utils.py`：

- 将 `compute_depth_volatility_map()` 改为一次联合 `np.nanpercentile(depths_nan, [low, high], axis=0)`
- 保留 `compute_high_volatility_mask()` 作为通用工具函数，供主链路和分析脚本复用
- 将 `build_traj_filter_result()` 和 `build_traj_valid_mask()` 的 volatility 输入从 `depth_volatility_map` 改为 `high_volatility_mask`
- `evaluate_temporal_depth_consistency()` 继续消费 dense `(H, W)` 的 `high_volatility_mask`，不改判定逻辑

在 `scripts/batch_inference/infer.py`：

- 在 camera 保存阶段只做一次：
  - `full_depths -> depth_volatility_map`
  - `depth_volatility_map -> high_volatility_mask`
- 对所有 sample 直接复用这张 `high_volatility_mask`
- 不再把 `depth_volatility_map` 传进每个 sample 再单独 threshold

### 3.4 预期结果

- `full_map` 默认路径下行为与当前实现一致
- `traj_valid_mask`
- `traj_high_volatility_hit`
- `traj_stable_depth_consistency_ratio`

以上关键字段在 `factor=1`、旧逻辑等价条件下应保持不变。

## 4. P4：只对 query/访问位置做 volatility 统计

### 4.1 目标

- 不再为整张 `H x W` 图像上的所有像素构建 volatility map
- 只对当前 camera 下所有 sample 实际访问到的位置做 volatility 统计
- 降低 percentile 统计的像素规模

### 4.2 口径决策

本方案固定采用以下口径：

- percentile 统计集合：该 camera 所有 sample 在 temporal depth consistency 检查中实际参与比较的去重像素集合
- 不采用“每个 query 单独统计”的方案
- 不保持“整张图全局 percentile”的旧口径

这样做的原因：

- 同一 camera 内仍共享一套 volatility threshold，结果更稳定
- 相比 per-query 阈值，更容易解释和验证
- 相比整图口径，更符合 `P4` 的降本目标

### 4.3 上线方式

`P4` 不直接替换默认逻辑，而是新增可切换路径：

- 新增 CLI：`--depth_volatility_mode`
- 取值：`full_map`、`camera_accessed`
- 默认值：`full_map`

上线策略：

- `P1` 直接进入默认实现
- `P4` 通过 `camera_accessed` 显式启用
- 在人工验收和对比通过前，不修改默认值

### 4.4 具体改动

在 `utils/traj_filter_utils.py`：

- 将 reprojection 与 `compare_mask` 生成逻辑从 `evaluate_temporal_depth_consistency()` 中抽成共享 helper
- 该 helper 统一输出：
  - `xs_clip`
  - `ys_clip`
  - `compare_mask`
- `evaluate_temporal_depth_consistency()` 继续基于这些结果配合 dense `high_volatility_mask` 做一致性判定

在 `scripts/batch_inference/infer.py`：

- 把当前“边构建 sample、边过滤”的保存流程拆成两个阶段：
  1. 先准备每个 query 的过滤输入
  2. 再统一构建 camera 级 `high_volatility_mask`
- 对 `camera_accessed` 模式：
  1. 遍历所有 sample 的共享 helper 结果，收集该 camera 的访问去重像素
  2. 从 `full_depths[:, ys, xs]` 提取 `(T, K)` depth 序列
  3. 在这 `K` 个访问像素上做联合 percentile
  4. 在访问集合上按 percentile 做单次 threshold
  5. 将命中的访问像素回填成 dense `(H, W)` `high_volatility_mask`
- 未访问像素统一视为 `False`

### 4.5 兼容性要求

- `evaluate_temporal_depth_consistency()` 不新增新的输出字段
- sample NPZ 字段名、dtype、shape 保持不变
- `traj_high_volatility_hit` 与 `traj_volatility_exposure_ratio` 仍沿用现有定义
- 分析脚本中依赖 `compute_depth_volatility_map()` / `compute_high_volatility_mask()` 的历史入口暂不删除

## 5. 需要覆盖的入口

以下入口需要一起更新，避免 CLI 行为不一致：

- `scripts/batch_inference/infer.py`
- `scripts/batch_inference/batch_infer_press_one_button_demo.py`
- `scripts/batch_inference/infer_bridge_v2.py`

要求：

- `infer.py` 增加 `--depth_volatility_mode`
- `batch_infer_press_one_button_demo.py` 负责把该参数透传给 worker
- `infer_bridge_v2.py` 复用同一套 `args`，因此只要主入口解析了该参数即可生效

## 6. 测试与验收

### 6.1 单元测试

继续以现有 `unittest` 基线为主：

```bash
python -m unittest utils.test_traj_filter_utils -v
```

当前这组测试在本仓库环境已通过。

新增测试至少包括：

- `compute_depth_volatility_map()` 联合 percentile 与原双 percentile 结果一致
- `full_map` 模式下，camera 级预 threshold 后，现有 volatility 相关行为测试不变
- `camera_accessed` 模式下，未访问的高 volatility 像素不会影响 threshold 和轨迹结果
- 多个 query 共享一个 camera 时，`camera_accessed` 使用的是 camera 级访问去重集合，而不是 per-query 集合

### 6.2 CLI/透传测试

补充 `scripts/batch_inference/test_press_one_button_demo_utils.py` 或等价测试，确认：

- `--depth_volatility_mode` 能被 worker 命令正确透传

### 6.3 集成验收

至少做两组对比：

1. `full_map` 新实现 vs 当前实现
2. `camera_accessed` vs `full_map`

重点检查：

- `traj_valid_mask` 差异比例
- `traj_high_volatility_hit` 差异比例
- `traj_stable_depth_consistency_ratio` 差异比例
- save 阶段耗时变化

## 7. 明确不做的事

本轮不包含：

- `P2`：downsample volatility 路径
- `P3`：camera 级缓存
- 修改默认模式为 `camera_accessed`
- 修改 sample 输出 schema
- 删除历史分析脚本中的旧接口依赖

## 8. 最终落地约束

- `P1` 必须作为行为等价优化落地
- `P4` 必须先保留旧逻辑作为默认值和对照组
- `P4` 的 threshold 统计集合固定为 camera 访问去重像素集合
- 下游 temporal consistency 仍统一消费 dense `high_volatility_mask`

如果后续基于远端分支实施，优先重新确认以下 3 个入口是否仍保持相同职责：

- `save_structured_data()`
- `build_traj_filter_result()`
- `evaluate_temporal_depth_consistency()`
