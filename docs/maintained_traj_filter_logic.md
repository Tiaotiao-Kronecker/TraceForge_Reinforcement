# 维护态轨迹过滤逻辑

本文档描述当前维护态 TraceForge 的轨迹过滤逻辑，重点覆盖默认的两条生产路径：

- external 相机默认走 `traj_filter_profile=external`
- wrist-like 相机在 `traj_filter_profile=auto` 下默认走 `wrist_manipulator_top95`

实现入口以 `utils/traj_filter_utils.py` 中的 `build_traj_filter_result()` 为准。本文只解释当前真实代码行为，不记录历史实验结论，也不覆盖已归档分析脚本的独立口径。

## 1. 适用范围与默认映射

当前维护态推理链路是 external-only：

- `depth_pose_method=external`
- `filter_level=standard`
- `traj_filter_profile=auto`

`auto` 的映射规则由 `resolve_traj_filter_profile()` 决定：

- 相机名以 `camera_3` 结尾，或包含 `wrist` / `hand`，映射到 `wrist_manipulator_top95`
- 其他相机映射到 `external`

因此，当前维护态最常见的默认组合是：

| 相机类型 | 默认 profile |
| --- | --- |
| external / third-person | `external` |
| wrist-like / hand-like | `wrist_manipulator_top95` |

补充说明：

- `wrist` 仍然存在，但不是 wrist-like 相机在维护态默认 `auto` 下的结果
- `external_manipulator`、`external_manipulator_v2`、`wrist_manipulator` 都需要显式指定
- `query_prefilter_mode` 默认是 `off`，不属于默认轨迹过滤主链

## 2. 共享基础过滤框架

### 2.1 filter level 默认阈值

`filter_level` 决定共享基础过滤参数。当前维护态默认是 `standard`。

| level | min_valid_frames | boundary_margin | visibility_threshold | depth_smoothness | depth_change_threshold |
| --- | ---: | ---: | ---: | --- | ---: |
| `basic` | 3 | 50 | 0.0 | off | 0.5 |
| `standard` | 3 | 50 | 0.5 | on | 0.5 |
| `strict` | 5 | 20 | 0.6 | on | 0.3 |
| `none` | 0 | 50 | 0.0 | off | 0.5 |

所有启用过滤的 level 还共享以下默认约束：

- 深度范围：`0.01m < depth < 10.0m`
- query-depth quality 开启
- temporal depth consistency 开启
- depth volatility guidance 开启
- temporal 最低一致性比例：`0.95`
- temporal 深度容差：绝对 `0.05m`，相对 `10%`
- volatility mask percentile：`99.0`

### 2.2 base geometry

所有 profile 都会先计算一套基础几何检查：

- `valid_count_mask`
- `depth_range_mask`
- `boundary_mask`
- `visibility_mask`
- `depth_smooth_mask`

其中完整的 `base_mask` 定义为：

```text
base_mask =
    valid_count_mask
  & depth_range_mask
  & boundary_mask
  & visibility_mask
  & depth_smooth_mask
```

这里的 `visibility_mask` 和 `boundary_mask` 是否真正参与最终决策，取决于 profile 类型和是否为 tail-truncated sample，后文会分别说明。

### 2.3 query-depth quality

每条轨迹的 query keypoint 会在 query frame 上做局部深度质量检查：

- patch 半径 `2`，即 `5x5` patch
- patch 有效深度比例至少 `0.4`
- query 深度与 patch 有效深度中位数的偏差需满足：
  - 绝对误差不超过 `0.05m`
  - 或相对误差不超过 `10%`

这一步输出的主判定是 `query_depth_quality_mask`。对 wrist-like profile，还会额外计算一层 query-depth edge risk。

### 2.4 wrist-like 专用 query-depth edge risk

只对 wrist-like profile 生效：

- `wrist`
- `wrist_manipulator`
- `wrist_manipulator_top95`

这一步会先在 query depth 上做局部深度边缘检测，再结合 patch 统计拒绝高风险点：

- `depth_edge(..., rtol=0.03)`
- patch 有效深度比例仍要求至少 `0.4`
- patch depth 标准差至少 `0.003`

对应逻辑可理解为：

```text
query_depth_keep_mask = query_depth_quality_mask & (~query_depth_edge_risk_mask)
```

只有在 wrist-like 默认行为下，这层 edge risk 才会作为硬拒绝进入最终 mask。若显式使用 `traj_filter_ablation_mode=wrist_no_query_edge`，这层拒绝只保留调试量，不再进入最终判定。

### 2.5 temporal depth consistency

所有启用过滤的默认 profile 都会计算 temporal depth consistency。它会输出：

- `temporal_mask`
- `traj_depth_consistency_ratio`
- `traj_stable_depth_consistency_ratio`
- `traj_high_volatility_hit`
- `traj_volatility_exposure_ratio`
- `traj_compare_frame_count`
- `traj_stable_compare_frame_count`
- `traj_supervision_mask`
- `traj_supervision_prefix_len`
- `traj_supervision_count`

其中：

- `temporal_mask` 是 external 路径的最终硬门槛之一
- `traj_supervision_*` 是 wrist-like 路径的时域支撑依据

### 2.6 tail-truncated sample 语义

当 sample 的实际 `segment_len < future_len` 时，该 sample 会被视为 tail-truncated sample。当前实现对这类 sample 的处理是：

- `visibility` 不再作为 `base_mask` 的硬门槛
- `visibility` 也不再参与 temporal compare 的硬门槛
- 深度重投影一致性仍然正常计算并参与过滤

所以，tail-truncated sample 的“放宽”只针对 visibility，不是关闭整个 temporal depth consistency。

另外，当前维护态在 query-frame 采样阶段已经会丢掉“到视频末尾剩余帧数（含自身）`<= 8`”的 query frame；这里的 tail-truncated 主要指那些仍被保留、但真实 segment 长度短于 `future_len` 的样本。

## 3. external 默认路径

`external` 是 external / third-person 相机的维护态默认 profile。

### 3.1 代码执行顺序

在 `build_traj_filter_result()` 里，`external` 的最终逻辑非常直接：

```text
base_mask = base geometry 全通过
query_depth_mask = query-depth quality 通过
temporal_mask = temporal depth consistency 通过

final_mask = base_mask & query_depth_mask & temporal_mask
```

也就是说，`external` 不会再做任何“机械臂主体收缩”或“前缀放宽”。它的要求就是：

- 这条轨迹本身要像一个稳定的真实点
- query frame 上这个点的深度要靠谱
- 这条轨迹从头到尾都要在 3D 上自洽

### 3.2 每一步在物理上是什么意思

#### 第 1 步：base geometry

代码上它对应：

```text
base_mask =
    valid_count_mask
  & depth_range_mask
  & boundary_mask
  & visibility_mask
  & depth_smooth_mask
```

物理解释：

- `valid_count_mask`
  - 这条轨迹不能只出现一两帧
- `depth_range_mask`
  - 轨迹深度不能经常跑到明显不合理的范围
- `boundary_mask`
  - 轨迹不能大幅跑出画面
- `visibility_mask`
  - 对 external 相机来说，一个真实稳定点通常应该大部分时间都可见
- `depth_smooth_mask`
  - 深度不能一帧一帧剧烈抖动

这一层筛掉的是“看起来就不像真实稳定表面点”的轨迹。

#### 第 2 步：query-depth quality

代码上它对应 `query_depth_quality_mask`。`external` 下默认没有 wrist-like 的 edge-risk 拒绝，所以：

```text
query_depth_mask = query_depth_quality_mask
```

物理解释：

- 如果 query frame 上这个点的深度本身就是坏值、孤立值、或和周围局部 patch 明显对不上
- 那后面所有 3D 计算都会建立在错误起点上

所以这一层在问：

“这条轨迹的起点，在 query frame 上是不是落在一个可靠的 3D 表面上？”

#### 第 3 步：temporal depth consistency

代码上它对应 `temporal_mask`，由 `evaluate_temporal_depth_consistency()` 给出。

物理解释：

- 把这条轨迹在后续帧中的 3D 位置，重新和真实 depth 对照
- 如果它真的是一个稳定表面点，那么它应该持续和真实深度几何一致
- 如果只是 2D 跟踪“跟住了纹理”，但 3D 已经漂了，这一步通常会失败

这一层在问：

“这条轨迹从头到尾，是不是都还是同一个真实 3D 点？”

#### 第 4 步：最终保留

`external` 的最终保留就是三关同时通过：

```text
final_mask = base_mask & query_depth_mask & temporal_mask
```

这条路径的物理含义可以压缩成一句话：

“这必须是一条整段都稳定、整段都可信、整段都几何自洽的真实轨迹。”

### 3.3 一个直观例子

假设某条轨迹在 4 帧 sample 里的时域诊断是：

```text
traj_supervision_mask = [True, True, False, False]
```

这表示：

- 前两帧还能和真实 depth 对上
- 后两帧已经不自洽了

对 `external` 来说，即使前两帧是好的，也不会因为“前缀还不错”而保留。只要整条 temporal consistency 没通过，最终就会被过滤掉。

这正体现了 `external` 的核心风格：

- external 不奖励“前面一段是对的”
- external 只保留“整条都对”的轨迹

### 3.4 tail-truncated sample 对 external 的影响

这里有一个容易误解的细节。

如果 sample 是 tail-truncated sample，即真实 `segment_len < future_len`，`external` 会放松 `visibility` 的硬门槛，但不会放松 3D 几何一致性本身：

- `visibility` 不再直接参与 `base_mask`
- `visibility` 也不再直接参与 temporal compare
- 但 `temporal_mask` 仍然照常算

物理解释：

- 如果 sample 只是因为视频到尾巴了，后面帧不够长，不应该因为“天然没法继续看见”而被误杀
- 但如果它的 3D 重投影已经不对，那仍然会被判掉

### 3.5 主要调试信号

排查 external 样本时，优先看这些字段：

- `traj_valid_mask`
- `traj_depth_consistency_ratio`
- `traj_stable_depth_consistency_ratio`
- `traj_compare_frame_count`
- `traj_supervision_mask`
- `traj_supervision_prefix_len`
- `traj_supervision_count`
- `traj_mask_reason_bits`
- `valid_steps`

对应的 reason bits：

- `bit0`: `MASK_REASON_BASE_GEOMETRY_FAIL`
- `bit1`: `MASK_REASON_QUERY_DEPTH_FAIL`
- `bit2`: `MASK_REASON_TEMPORAL_CONSISTENCY_FAIL`
- `bit3`: `MASK_REASON_STABLE_TEMPORAL_FAIL`

## 4. wrist-like `auto` 默认路径：wrist_manipulator_top95

`wrist_manipulator_top95` 是 wrist-like 相机在维护态 `auto` 下的默认 profile。它不是直接从 external 放宽出来的，而是分成三层：

1. wrist 基础 seed
2. manipulator-aware 收缩
3. top95 motion 收缩

### 4.1 代码执行顺序

wrist-like 的 `auto` 默认结果不是单纯的 `wrist`，而是下面这条链：

```text
wrist_seed_mask =
    wrist_base_mask
  & query_depth_keep_mask
  & supervision_support_mask

traj_manipulator_candidate_mask =
    wrist_seed_mask
  & near_depth_mask
  & motion_mask

manipulator_final_mask = largest_spatial_component(traj_manipulator_candidate_mask)

final_mask = top95_motion_extent(manipulator_final_mask)
```

如果把它翻成人能理解的话，就是：

1. 先保住“前缀阶段真的像机械臂点”的轨迹
2. 再从这些点里收缩到“更像机械臂主体的一团”
3. 最后把 motion 最弱的一点尾巴削掉

### 4.2 第 1 层：wrist 基础 seed

#### 为什么 wrist 不能直接照搬 external

wrist 视角有几个天然特征：

- 机械臂/夹爪离镜头很近
- 容易快速出画
- 容易遮挡自己
- 很多真实点在后半段会因为遮挡或视角变化而坏掉

所以 wrist 不再要求“整条都稳定”，而是改成：

“从 query 开始，前面那一段必须是真的；后面可以坏，但不能坏得太早，也不能只真一两帧。”

#### wrist_base_mask 在物理上是什么意思

代码上 wrist 先把 `base_mask` 改成更宽松的：

```text
wrist_base_mask = valid_count_mask & depth_range_mask & depth_smooth_mask
```

也就是说，wrist 默认不会把下面两项当成硬门槛：

- `boundary_mask`
- `visibility_mask`

物理解释：

- wrist 视角下，真实夹爪点经常贴边、出画、被挡
- 这些现象对 wrist 来说很常见，不能像 external 那样直接当作“假轨迹”

#### query_depth_keep_mask 在物理上是什么意思

wrist-like 默认不仅要求 query depth 本身靠谱，还会额外拒绝落在危险深度边界上的点：

```text
query_depth_keep_mask = query_depth_quality_mask & (~query_depth_edge_risk_mask)
```

物理解释：

- 靠近夹爪轮廓或前景/背景分界线的深度，很容易是混合像素或边界噪声
- 这些点就算短时间能跟住，3D 也经常漂

所以 wrist-like 默认会比 external 更积极地排掉深度边界风险点。

#### supervision support 在物理上是什么意思

这是 wrist 逻辑里最重要的一步，也是最容易误解的一步。

代码不是直接拿 `temporal_mask` 当最终门槛，而是先看 `traj_supervision_mask`，再把它压缩成两个量：

- `traj_supervision_prefix_len`
  - 从第 0 帧开始，连续有多少帧是可信的
- `traj_supervision_count`
  - 整条轨迹里总共有多少帧是可信的

然后要求：

```text
required_prefix_frames = max(3, ceil(0.15 * T))
required_support_frames = max(3, ceil(0.20 * T))

supervision_support_mask =
    (traj_supervision_prefix_len >= required_prefix_frames)
  & (traj_supervision_count >= required_support_frames)
```

物理解释：

- `prefix_len` 解决的是“前缀必须真”
- `count` 解决的是“不能只真一瞬间”

也就是说，wrist 允许轨迹后面坏掉，但不允许：

- 一开始很快就坏掉
- 整体只零零散散地真几帧

#### supervision support 的例子

假设一个 4 帧 sample：

1. `traj_supervision_mask = [True, True, True, False]`
   - `prefix_len = 3`
   - `count = 3`
   - 对 4 帧 sample，要求是 `3` 和 `3`
   - 这条会通过

物理解释：

- 这像一条“前面确实落在夹爪上，后面因为遮挡或出画才失效”的真实轨迹

2. `traj_supervision_mask = [True, True, False, False]`
   - `prefix_len = 2`
   - `count = 2`
   - 这条不过

物理解释：

- 它只在 very short 的开头阶段看起来是对的，不足以说明它真的是稳定的机械臂点

把这一层合起来，就是：

```text
wrist_seed_mask =
    wrist_base_mask
  & query_depth_keep_mask
  & supervision_support_mask
```

这一步在物理上做的事是：

“先把前缀阶段真的像机械臂点的轨迹保住。”

### 4.3 第 2 层：manipulator-aware 收缩

`wrist_manipulator_top95` 不会停在 `wrist_seed_mask`，而是继续走一遍 `wrist_manipulator` 的三步收缩。

#### 第一步：near-depth gate

代码上：

```text
traj_query_depth_rank <= 0.50
```

物理解释：

- wrist 视角里，机械臂/夹爪通常是近场物体
- 越靠近相机的点，越有可能属于机械臂主体

所以这是在问：

“这些已经像 wrist 真实点的轨迹里，哪些还足够靠近镜头？”

#### 第二步：motion gate

代码上：

```text
traj_motion_extent_all_valid >= 0.03m
```

物理解释：

- 机械臂主体通常会有比较明确的 3D 位移
- 如果一个点几乎不动，它更可能是背景残留或偶然通过的伪点

这里默认用的是 `traj_motion_extent_all_valid`，不是只看 supervised prefix 的 motion。这意味着 wrist-like 默认会把“整段可用帧上的总体运动”当成更重要的主体信号。

#### 第三步：largest_spatial_component

在 near-depth 和 motion 之后，先得到候选点：

```text
traj_manipulator_candidate_mask =
    wrist_seed_mask
  & near_depth_mask
  & motion_mask
```

然后不是直接全保留，而是执行：

```text
manipulator_final_mask = largest_spatial_component(traj_manipulator_candidate_mask)
```

这里的“largest spatial component”不是 3D 连通，也不是时间连通，而是：

- 只看 query frame 上这些 candidate keypoint 的 2D 位置
- 如果两个 candidate 点在图像上足够近，就认为它们属于同一团
- 把所有 candidate 分成若干个 2D 连通簇
- 只保留最大的一团

物理解释：

- 真正的机械臂主体通常会在 query 图像上形成一块连续区域
- 零散孤点、小远团块，更像噪声、背景残留或偶然通过的伪点

#### largest_spatial_component 的例子

假设有 3 个已经通过前面两关的 candidate 点：

- A 在 `(5, 5)`
- B 在 `(11, 6)`
- C 在 `(50, 50)`

如果 A 和 B 在连通半径内，C 离它们很远，那么会分成两团：

- 团 1：`{A, B}`
- 团 2：`{C}`

这时 `largest_spatial_component` 只保留 `{A, B}`，丢掉 `C`。

物理解释：

- A/B 更像机械臂主体那一团
- C 更像空间上孤立的噪声点

#### 为什么有时又会 fallback 全保留

代码里还有一个小样本保护：

- 如果最大的团本身也太小，不足以稳定代表“主体”
- 就不再强行只保最大团
- 而是回退到“保留全部 candidate”

物理解释：

- 在小样本或极稀疏 sample 上，“最大团”这个概念本身不稳定
- 这时强行只保一团，容易把本来就不多的真点误杀

### 4.4 第 3 层：top95 motion 收缩

在默认 `wrist_manipulator_top95` 下，最后一步是：

```text
traj_pre_top95_mask = manipulator_final_mask
final_mask = top95_motion_extent(traj_pre_top95_mask)
```

也就是：

- 先得到已经很像机械臂主体的一团
- 再按 `traj_motion_extent_all_valid` 从大到小排序
- 只保留前 `95%`

物理解释：

- 这一步不是重新定义主体
- 而是在已经比较干净的主体结果里，再削掉 motion 最弱的一点尾巴

例子：

- 如果 `manipulator_final_mask` 里最终有 `20` 条轨迹
- top95 会保留其中 motion 最强的 `19` 条
- motion 最弱的 `1` 条会被当作尾部噪声裁掉

### 4.5 和纯 wrist 的关系

纯 `wrist` 会在 `wrist_seed_mask` 就停下：

```text
final_mask = wrist_seed_mask
```

它不会再经过：

- near-depth 收缩
- motion 收缩
- cluster 收缩
- top95 收缩

因此：

- `wrist` 的物理语义是“保住前缀阶段可信的 wrist 点”
- `wrist_manipulator_top95` 的物理语义是“从这些 wrist 点里进一步逼近机械臂主体，并再去一点弱噪声尾巴”

维护态 `auto` 默认采用的是后者。

### 4.6 用一句话总结 wrist-like `auto`

wrist-like `auto` 默认逻辑不是在问：

“这条轨迹是不是整段都稳定？”

而是在问：

“这条轨迹是不是在前缀阶段真实可信、落在近场、确实在动、并且属于机械臂主体那一团？”

### 4.7 主要调试信号

排查 wrist-like 默认样本时，优先看这些字段：

- `traj_valid_mask`
- `traj_supervision_mask`
- `traj_supervision_prefix_len`
- `traj_supervision_count`
- `traj_wrist_seed_mask`
- `traj_query_depth_rank`
- `traj_query_depth_edge_mask`
- `traj_query_depth_patch_valid_ratio`
- `traj_query_depth_patch_std`
- `traj_query_depth_edge_risk_mask`
- `traj_motion_extent`
- `traj_motion_extent_all_valid`
- `traj_manipulator_candidate_mask`
- `traj_manipulator_cluster_id`
- `traj_manipulator_component_size`
- `traj_manipulator_cluster_fallback_used`
- `traj_mask_reason_bits`
- `valid_steps`

对应的 reason bits：

- `bit0`: `MASK_REASON_BASE_GEOMETRY_FAIL`
- `bit1`: `MASK_REASON_QUERY_DEPTH_FAIL`
- `bit2`: `MASK_REASON_TEMPORAL_CONSISTENCY_FAIL`
- `bit4`: `MASK_REASON_MANIPULATOR_DEPTH_FAIL`
- `bit5`: `MASK_REASON_MANIPULATOR_MOTION_FAIL`
- `bit6`: `MASK_REASON_MANIPULATOR_CLUSTER_FAIL`
- `bit7`: `MASK_REASON_QUERY_DEPTH_EDGE_FAIL`

注意：wrist-like 路径里，`bit2` 对应的是 `supervision_support_mask` 不满足，而不是 external 那种“整条 temporal_mask 直接失败”。

## 5. external vs wrist-like `auto` 的最终区别

如果只记一张表，可以记这一张：

| 维度 | `external` | wrist-like `auto` (`wrist_manipulator_top95`) |
| --- | --- | --- |
| 想保留什么 | 整条都稳定、整条都几何自洽的真实 3D 点 | 前缀阶段真实可信、并且最终更像机械臂主体的一团点 |
| base 几何门槛 | 完整 `base_mask`，包含 boundary 和 visibility | `wrist_base_mask`，不把 boundary / visibility 当硬门槛 |
| query 深度 | 只看 query-depth quality | query-depth quality + query-depth edge risk 拒绝 |
| 时域判定 | 直接要求 `temporal_mask` 通过 | 不直接要求整条 temporal 通过，改看 `supervision support` |
| 对后半段坏掉的容忍 | 低 | 高，只要前缀够真、总体支撑够多 |
| 机械臂主体收缩 | 无 | near-depth + motion + largest spatial component |
| 最后一步 | 直接输出 | 再做 top95 motion 去尾 |

把它翻成一句更口语的话：

- `external` 在问：“这是不是一条从头到尾都稳定、可信、几何一致的真实轨迹？”
- wrist-like `auto` 在问：“这是不是一条前缀阶段真实可信、靠近镜头、确实在动、并且属于机械臂主体那一团的轨迹？”

## 6. 如何解读 sample 输出

当前维护态 `v2` sample 会直接写出过滤结果和主要调试量。最关键的是区分三类字段：

### 6.1 最终保留结果

- `traj_valid_mask`
- `valid_steps`
- `segment_frame_indices`

其中：

- `traj_valid_mask` 是轨迹维度上的最终保留结果
- `valid_steps` 是时间维度上的有效步前缀
- `segment_frame_indices` 只记录真实存在的帧索引，不包含 padding 位置

### 6.2 共享时域诊断

- `traj_supervision_mask`
- `traj_supervision_prefix_len`
- `traj_supervision_count`
- `traj_depth_consistency_ratio`
- `traj_stable_depth_consistency_ratio`
- `traj_compare_frame_count`
- `traj_stable_compare_frame_count`

### 6.3 wrist-like 专用调试量

- `traj_wrist_seed_mask`
- `traj_query_depth_rank`
- `traj_query_depth_edge_mask`
- `traj_query_depth_patch_valid_ratio`
- `traj_query_depth_patch_std`
- `traj_query_depth_edge_risk_mask`
- `traj_motion_extent`
- `traj_motion_step_median`
- `traj_motion_extent_all_valid`
- `traj_motion_step_median_all_valid`
- `traj_manipulator_candidate_mask`
- `traj_manipulator_cluster_id`
- `traj_manipulator_component_size`
- `traj_manipulator_cluster_fallback_used`

需要特别注意的是，下面这些中间 mask 目前只在运行时统计和 profile 计数里使用，并不会写入 sample NPZ：

- `traj_base_mask`
- `traj_query_depth_keep_mask`
- `traj_supervision_support_mask`
- `traj_near_depth_mask`
- `traj_motion_mask`
- `traj_cluster_mask`
- `traj_pre_top95_mask`

如果要定位这些中间阶段，需要看运行时 profile 统计或直接复现 `build_traj_filter_result()`。

## 7. 非默认但仍在维护的分支

### 7.1 external_manipulator

这条分支先完整通过 external seed，再做 manipulator-aware 收缩：

- seed 仍然要求 `base_mask & query_depth_mask & temporal_mask`
- motion gate 使用 supervised motion
- cluster 只保留最大连通簇

它适合“external 视角下只想看机械臂主体”的更强收缩需求，但不是维护态默认结果。

### 7.2 external_manipulator_v2

这条分支仍以 external seed 为前提，但比 `external_manipulator` 更宽松：

- `traj_query_depth_rank <= 0.70`
- `traj_motion_extent >= 0.01m`
- 保留主要连通块，而不是只保留单个最大连通块

### 7.3 query prefilter 和 ablation mode

这两项都不属于维护态默认轨迹过滤主链：

- `query_prefilter_mode=profile_aware_static_v1`
  - 在 tracking 之前做静态 query 预筛
  - 默认 `off`
  - 只有 wrist-like profile 会启用 aggressive prefilter
- `traj_filter_ablation_mode`
  - 仅用于 save-time 分析
  - 默认 `none`
  - 不能拿它代表生产默认逻辑

## 8. 新任务场景下的适配建议

本节讨论的是“当前 wrist-like `auto` 默认逻辑是否适合新任务”，不是描述已经实现的新 profile。当前结论是：

- external 相机默认 `external` 仍然基本可沿用
- wrist 相机不能再把 `auto -> wrist_manipulator_top95` 当成通用默认

### 8.1 为什么 `press` 场景下当前 wrist 默认问题不大

当前这套 wrist-like `auto` 默认逻辑，最初更接近 press 类任务：

- 主要运动主体就是机械臂和夹爪
- 被交互对象通常不需要被稳定抓住
- 任务过程中往往没有一个需要长期跟随保留的“被抓物体”

因此，当前默认的这几步：

- near-depth
- motion
- largest spatial component
- top95

虽然很偏向“机械臂主体”，但在 press 场景里通常不会造成特别大的语义损失。因为该保留的主要就是夹爪本体，而不是其他物体。

### 8.2 为什么 `pick_place` 不适合继续用 wrist `auto`

`pick_place` 的目标不只是保留夹爪，还要保留被抓住并被搬运的物体。

但当前 wrist-like `auto` 默认逻辑会系统性偏向“保 gripper body，不保 object”。

#### 哪些 stage 会出问题

| 当前 stage | 对 `pick_place` 的潜在问题 |
| --- | --- |
| query-depth edge risk | 被抓物体的边缘、遮挡边界、接触区域容易被判成高风险点 |
| near-depth | 被抓物体经常比夹爪略远，尤其在接近和接触早期不如夹爪“近” |
| motion gate | 物体在抓取前通常静止，会被当成低运动点过滤掉 |
| largest spatial component | 即使物体通过前几关，也可能和夹爪不是同一最大团 |
| top95 | 会继续去掉 motion 较弱但语义上重要的物体轨迹 |

#### 物理上为什么这是错的

当前 wrist 默认逻辑在问：

“这是不是机械臂主体的一部分？”

但 `pick_place` 真正需要回答的是：

“这是不是机械臂主体，或者是之后会被机械臂抓住并共同运动的物体？”

这两者不是一回事。对 `pick_place` 来说：

- 物体在抓取前静止是正常的
- 物体不是最大空间连通团也正常
- 物体比夹爪略远也正常

所以继续用当前 wrist `auto`，会天然漏掉被抓物体轨迹。

### 8.3 为什么 `push_pull` 不适合继续用 wrist `auto`

`push_pull` 的核心对象往往不是一个大块物体，而是门把手、拉手、抽屉边缘这类细小接触结构。

当前 wrist-like `auto` 默认逻辑在这类场景下同样有明显风险。

#### 哪些 stage 会出问题

| 当前 stage | 对 `push_pull` 的潜在问题 |
| --- | --- |
| query-depth edge risk | 把手/门边通常正好是深度边界最强的位置 |
| near-depth | 把手经常比夹爪稍远，深度秩次不稳定 |
| motion gate | 把手在接触前通常静止，接触后位移也可能不大 |
| largest spatial component | 把手是小结构，通常不会成为最大连通团 |
| top95 | 会进一步去掉 motion 较弱但关键的接触点 |

#### 物理上为什么这是错的

对 `push_pull` 来说，关键目标不是“最大、最近、最会动的一团点”，而是：

- 与夹爪发生接触的小结构
- 能代表局部 articulation 的接触区域

门把手很可能：

- 很细
- 很小
- 不够近
- 前期不怎么动
- 不会成为最大团

但它仍然是任务里最重要的对象。

### 8.4 对两类新任务的短期配置建议

如果短期内不改代码，只用现有 profile 和 ablation mode，建议这样设：

| 场景 | wrist 相机建议配置 | 原因 |
| --- | --- | --- |
| `press` | 继续可用 `auto` | 当前默认就是围绕“机械臂主体”优化的 |
| `pick_place` | 改成 `traj_filter_profile=wrist` | 停在 `wrist_seed_mask`，避免 manipulator-only 收缩杀掉物体 |
| `push_pull` | 先用 `traj_filter_profile=wrist` | 先避免 near-depth / motion / cluster / top95 把把手裁掉 |
| `push_pull` 若把手仍严重丢失 | 再试 `traj_filter_ablation_mode=wrist_no_query_edge` | 把手很容易死在 edge-risk 这一步 |
| 所有新场景 | `query_prefilter_mode=off` | 不要在 tracking 之前就按 wrist-manipulator 偏好裁 query seed |

这里要特别强调：

- `pick_place` 不建议继续使用 `wrist_manipulator`
- `pick_place` 更不建议继续使用 `wrist_manipulator_top95`
- `push_pull` 也不应把 `wrist_manipulator_top95` 当默认起点

短期内，更合理的策略是：

- 先用 `wrist` 保住“前缀阶段可信”的轨迹
- 再基于可视化结果判断缺的是被抓物体、把手，还是接触边缘

### 8.5 长期建议：新增 interaction-aware wrist profile

如果后续要把这两类任务作为维护态支持场景，建议不要继续共用一个“只收缩到机械臂主体”的 wrist 默认 profile，而是至少拆成两类。

#### 建议一：`wrist_pick_place`

目标不是“只保机械臂主体”，而是：

- 保机械臂主体
- 保被抓住并与机械臂共同运动的物体

建议的物理逻辑：

1. 先从 `wrist_seed_mask` 保住所有前缀可信的 wrist 轨迹
2. 一条分支继续做 manipulator-aware 收缩，得到 gripper body
3. 另一条分支专门寻找“在接触后与 gripper 共动”的物体轨迹
4. 最终保留 `gripper_mask OR grasped_object_mask`

其中 object 分支不应继续强依赖：

- 初始近深度
- 初始大 motion
- 最大连通团

因为对被抓物体来说，这三条都可能不成立。

#### 建议二：`wrist_push_pull`

目标不是“只保最大主体团”，而是：

- 保夹爪
- 保门把手/拉手/抽屉边缘等接触件
- 保局部 articulation 区域

建议的物理逻辑：

1. 仍从 `wrist_seed_mask` 出发
2. 保留 manipulator 主体分支
3. 单独寻找“与 manipulator 发生接触的小型结构”
4. 对这类接触结构，不再强依赖 `largest_spatial_component`
5. 对 query-depth edge risk 采用更宽松或任务特化的策略

对 `push_pull` 来说，关键对象往往不是最大团，而是“小但关键的接触件”。

### 8.6 推荐落地顺序

如果要务实推进，建议按下面顺序做，而不是一开始就发明一套很复杂的新 heuristic：

1. 在新场景上停止使用 wrist `auto`
2. 先把 wrist profile 切到 `wrist`
3. 对代表性 `pick_place` / `push_pull` case 导出可视化
4. 分别确认实际缺失的是：
   - 被抓物体
   - 门把手/拉手
   - 接触边界附近轨迹
5. 再针对任务分别新增 `wrist_pick_place` 和 `wrist_push_pull`

这比继续把所有 wrist 任务都塞进 `wrist_manipulator_top95` 更稳妥。因为：

- `press` 需要的是“机械臂主体优先”
- `pick_place` 需要的是“机械臂主体 + 被抓物体”
- `push_pull` 需要的是“机械臂主体 + 小型接触件 + articulation 区域”

三者的目标对象并不相同，不适合继续共用同一个 wrist 默认收缩逻辑。
