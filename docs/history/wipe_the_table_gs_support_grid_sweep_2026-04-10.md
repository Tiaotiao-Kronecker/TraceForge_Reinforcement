# `wipe_the_table_gs` support-grid sweep 记录

日期：2026-04-10

## 背景

这轮实验的目标是回答一个更具体的问题：

- 在 `grid_size=80`、external-only 维护态约束不变时，外层 `support_grid_ratio` 的变化会如何影响端到端耗时与轨迹质量

用户当前更关心轨迹质量，而不是只追求极限吞吐，所以这次判断标准不是“谁最快”，而是：

1. 在两路 external 相机上是否都能稳定提速
2. 质量退化是否仍然足够小
3. 是否值得据此调整默认值，或者至少形成下一轮候选

## 固定实验条件

- manifest：
  - `scripts/data_analysis/manifests/wipe_the_table_gs_external_only_median3_20260407.json`
- dataset root：
  - `/DATA/disk1/zoyo/mcap/wipe_the_table_gs`
- episodes：
  - `00000,00001,00002`
- cameras：
  - `varied_camera_1,varied_camera_2`
- tracker：
  - `num_iters=3`
  - `grid_size=80`
- filtering：
  - `traj_filter_profile=external`
- support ratios：
  - `0.8,0.6,0.4,0.2,0.0`
- benchmark：
  - `warmup_runs=1`
  - `benchmark_runs=3`
  - `run_visual_verification=True`
- device：
  - `00000 -> cuda:0`
  - `00001 -> cuda:1`
  - `00002 -> cuda:2`

这里的 baseline 仍然按当前外层默认配置 `support_grid_ratio=0.8` 定义。

## 运行命令

逐 episode benchmark：

```bash
/usr/bin/env MPLCONFIGDIR=/tmp/matplotlib /tmp/traceforge_bench_py311/bin/python scripts/data_analysis/benchmark_inference_variants.py \
  --episode-dir /DATA/disk1/zoyo/mcap/wipe_the_table_gs/<episode> \
  --camera-names varied_camera_1,varied_camera_2 \
  --support-grid-ratios 0.8,0.6,0.4,0.2,0.0 \
  --num-iters 3 \
  --grid-size 80 \
  --traj-filter-profile external \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --run-visual-verification \
  --device cuda:<gpu> \
  --output-root /DATA/disk3/tmp/wipe_support_sweep_20260410/<episode>
```

跨 episode 聚合：

```bash
/tmp/traceforge_bench_py311/bin/python scripts/data_analysis/aggregate_inference_variants_manifest.py \
  --manifest scripts/data_analysis/manifests/wipe_the_table_gs_external_only_median3_20260407.json \
  --episode-output-root /DATA/disk3/tmp/wipe_support_sweep_20260410 \
  --output-root /DATA/disk3/tmp/wipe_support_sweep_20260410/aggregate
```

## 产物位置

- 总输出根目录：
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410`
- 聚合 JSON：
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410/aggregate/benchmark_results.json`
- 聚合 Markdown：
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410/aggregate/benchmark_summary.md`
- per-episode summaries：
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410/00000/benchmark_summary.md`
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410/00001/benchmark_summary.md`
  - `/DATA/disk3/tmp/wipe_support_sweep_20260410/00002/benchmark_summary.md`

## 聚合 runtime

### `varied_camera_1`

| Ratio | Effective Support Count | Total (s) | Total Speedup vs 0.8 | Tracker Forward (s) |
| --- | ---: | ---: | ---: | ---: |
| `0.8` | `4075.1` | `125.303` | `1.000x` | `97.260` |
| `0.6` | `2289.1` | `113.324` | `1.110x` | `80.819` |
| `0.4` | `1014.5` | `96.769` | `1.294x` | `70.441` |
| `0.2` | `255.3` | `92.899` | `1.350x` | `62.887` |
| `0.0` | `0.0` | `98.424` | `1.273x` | `59.477` |

### `varied_camera_2`

| Ratio | Effective Support Count | Total (s) | Total Speedup vs 0.8 | Tracker Forward (s) |
| --- | ---: | ---: | ---: | ---: |
| `0.8` | `3803.0` | `121.919` | `1.000x` | `95.375` |
| `0.6` | `2133.0` | `108.436` | `1.124x` | `81.484` |
| `0.4` | `948.0` | `98.650` | `1.238x` | `70.598` |
| `0.2` | `234.0` | `101.305` | `1.214x` | `63.976` |
| `0.0` | `0.0` | `91.182` | `1.339x` | `60.491` |

直接现象：

- tracker forward 会随着 support ratio 降低而明显下降
- 但总耗时不完全单调，因为 `prepare inputs` 等固定成本占比会变大
- `0.2` 在 `varied_camera_1` 最快，但在 `varied_camera_2` 反而不如 `0.4`
- `0.0` 在 `varied_camera_2` 最快，但 `varied_camera_1` 没有比 `0.2` 更好

这说明仅按速度看，也已经没有“support 越少越一定更优”的简单结论。

## 聚合质量对比

相对 baseline `0.8`：

| Camera | Ratio | Mask Jaccard | Valid Delta | World L2 Mean | Worst Query Jaccard | Worst Episode/QF |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `varied_camera_1` | `0.6` | `0.998` | `-4.067` | `0.00010` | `0.99664` | `00001/72` |
| `varied_camera_1` | `0.4` | `0.997` | `-6.658` | `0.00010` | `0.99561` | `00002/75` |
| `varied_camera_1` | `0.2` | `0.997` | `-10.512` | `0.00010` | `0.99258` | `00002/75` |
| `varied_camera_1` | `0.0` | `0.996` | `-11.652` | `0.00010` | `0.99225` | `00002/75` |
| `varied_camera_2` | `0.6` | `0.996` | `-7.985` | `0.00009` | `0.99422` | `00001/38` |
| `varied_camera_2` | `0.4` | `0.996` | `-11.152` | `0.00009` | `0.99295` | `00000/78` |
| `varied_camera_2` | `0.2` | `0.995` | `-17.270` | `0.00010` | `0.99154` | `00000/33` |
| `varied_camera_2` | `0.0` | `0.994` | `-19.779` | `0.00010` | `0.99133` | `00000/0` |

这里更值得看的是趋势而不是单个小数点：

- `0.6 -> 0.4 -> 0.2 -> 0.0` 基本呈现越来越大的 valid-point 损失
- `mask jaccard` 仍然都很高，但 `0.2/0.0` 在两路相机上都更接近系统性退化
- 最差 query 也遵循同样趋势，`0.6` 最稳，`0.4` 次之，`0.2/0.0` 更差

因此不能只用 `world_l2_mean` 很小就断言“support 不重要”。在当前 workload 下，更先掉下来的指标是 valid coverage。

## `Valid Delta` 的定义与量级解释

这里的 `Valid Delta` 指标需要单独说明，否则很容易误读成 step 级误差或几何误差。

在单个 query sample 上：

- `valid_track_count`
  - 等于 `traj_valid_mask=True` 的轨迹条数
- `valid_track_count_delta`
  - 定义为
  - `variant_valid_track_count - baseline_valid_track_count`

因此：

- `Valid Delta < 0`
  - 表示该 variant 最终保留下来的有效轨迹条数比 baseline 少
- 它反映的是最终 `traj_valid_mask` 的保留数量变化
- 它不是轨迹位置误差，也不是 step 级别误差

对当前 `traj_filter_profile=external`，最终有效掩码来自：

- `base_mask`
- `query_depth_mask`
- `temporal_mask`

所以 `Valid Delta` 更接近“有效轨迹覆盖率变化”，而不是“存活轨迹几何位置偏差”。

## 速度收益 vs `Valid Delta`

为了更直观地看 trade-off，这里把四个候选 ratio 压到一个二维视角：

- 横轴：相对 baseline `0.8` 的平均节省时间
- 纵轴：平均 `Valid Delta`

下表是跨两路相机的平均结果：

| Support Ratio | Avg Saved Seconds | Avg Total Speedup | Avg Valid Delta | Avg Valid Delta / 6400 | Avg Mask Jaccard | Saved Seconds per Lost Valid Track |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.6` | `12.731` | `1.117x` | `-6.026` | `-0.0942%` | `0.997214` | `2.113` |
| `0.4` | `25.901` | `1.266x` | `-8.905` | `-0.1391%` | `0.996673` | `2.909` |
| `0.2` | `26.509` | `1.282x` | `-13.891` | `-0.2170%` | `0.995542` | `1.908` |
| `0.0` | `28.808` | `1.306x` | `-15.715` | `-0.2455%` | `0.995186` | `1.833` |

其中最后一列不是代码内置指标，而是为了帮助决策额外计算的派生量：

- `Saved Seconds per Lost Valid Track`
  - `avg_saved_seconds / abs(avg_valid_delta)`

它可以粗略理解为：

- 每多损失 `1` 条有效轨迹，能换来多少秒总时间节省

### 这个量级到底算不算大

如果按单个 query 的 `6400` 个 dense 点来理解：

- `0.6`
  - 平均只少 `6.0` 条
  - 约 `0.09%`
- `0.4`
  - 平均少 `8.9` 条
  - 约 `0.14%`
- `0.2`
  - 平均少 `13.9` 条
  - 约 `0.22%`
- `0.0`
  - 平均少 `15.7` 条
  - 约 `0.25%`

所以从绝对数量看，这不是“轨迹大面积塌掉”的级别，而是：

- 小幅
- 稳定
- 系统性

它足以作为默认值决策信号，但还不到“肉眼一看就明显崩坏”的程度。

### 为什么 `0.4` 更像 Pareto 拐点

从二维表看，`0.4` 最接近当前 benchmark 的拐点：

- 相比 `0.6`
  - 节省时间从 `12.731s` 提高到 `25.901s`
  - `Valid Delta` 只从 `-6.026` 变到 `-8.905`
- 相比 `0.2`
  - 只少省了 `0.608s`
  - 但 `Valid Delta` 明显更好，`-8.905` 对 `-13.891`
- 相比 `0.0`
  - 少省 `2.907s`
  - 但 `Valid Delta` 明显更好，`-8.905` 对 `-15.715`

因此：

- `0.6`
  - 偏保守
  - 质量最稳，但加速有限
- `0.4`
  - 当前最像 throughput / quality 折中点
- `0.2`
  - 不够划算
  - 相比 `0.4` 几乎没多省多少时间，却多掉了一截 valid coverage
- `0.0`
  - 只适合明显 throughput-first 的场景
  - 不适合作为当前质量优先维护态默认值候选

## 结论

### 默认值判断

如果仍按“质量优先”的维护态要求决策，本轮结果还不足以支持把默认值从 `0.8` 直接切走。

更准确的结论是：

- `0.8` 仍然是最保守、质量最稳的默认配置
- `0.6` 是安全边际更大的轻量降配候选，但提速有限，只有 `~1.11x` 到 `~1.12x`
- `0.4` 是当前最值得继续跟进的 throughput/quality 折中点
- `0.2` 和 `0.0` 虽然更快，但 valid coverage 下降更明显，不适合直接做维护态默认值

### 为什么 `0.4` 是下一轮候选

`0.4` 的价值不在于“绝对最快”，而在于它同时满足：

- 两路相机都拿到了明确提速
  - `varied_camera_1`: `1.294x`
  - `varied_camera_2`: `1.238x`
- 质量退化明显小于 `0.2/0.0`
- 最差 query jaccard 仍维持在 `~0.993-0.996`

换句话说：

- 如果目标是保守维护态，继续留在 `0.8`
- 如果目标是给后续吞吐优化找一个最有希望的 follow-up 候选，优先看 `0.4`

### 当前可以固定下来的工程判断

截至这轮 benchmark，更稳的归纳是：

1. `support ratio` 降低后，主要先退化的是 valid coverage，而不是 surviving tracks 的几何位置。
2. `0.4` 已经拿到了接近 `0.2/0.0` 的大部分速度收益，但质量代价明显更低。
3. `0.2/0.0` 没有展示出足够强的额外速度收益，来证明这部分覆盖率损失是值得的。
4. 因此当前 external-only 维护态默认值仍保持 `0.8`，而 `0.4` 是最值得继续扩集验证的下一轮候选。

## 对当前代码状态的影响

这轮实验只新增了聚合脚本和测试，未改动 inference 默认值。当前默认配置不需要因为这组结果立刻调整。

如果后续要推进 `0.4`，更合理的顺序应该是：

1. 扩大 benchmark episode 集合
2. 继续人工抽查最差 query 的 3D 可视化
3. 只在确认 valid coverage 的退化仍可接受后，再讨论默认值切换

## 相关代码

- 聚合脚本：
  - `scripts/data_analysis/aggregate_inference_variants_manifest.py`
- 测试：
  - `scripts/data_analysis/test_aggregate_inference_variants_manifest_utils.py`
- 上游单集 benchmark：
  - `scripts/data_analysis/benchmark_inference_variants.py`
