# 维护态轨迹过滤逻辑

本文档描述当前维护态 TraceForge 的轨迹过滤逻辑。当前维护态只有一条默认生产路径：

- 所有维护态相机默认走 `traj_filter_profile=external`

另外，代码中仍保留若干显式非默认分支，供历史调查、兼容性复跑和定向实验使用：

- `traj_filter_profile=wrist_pick_place`
- `traj_filter_profile=wrist_pick_place_no_heatmap`

实现入口以 `utils/traj_filter_utils.py` 中的 `build_traj_filter_result()` 为准。本文只解释当前真实代码行为，不记录历史实验结论，也不覆盖已归档分析脚本的独立口径。

## 1. 适用范围与默认映射

当前维护态推理链路是 external-only：

- `depth_pose_method=external`
- `filter_level=none`
- `traj_filter_profile=external`

`traj_filter_profile=auto` 仍被接受，但现在只作为兼容别名保留，并且也会解析到 `external`。

因此，当前维护态最常见的默认组合是：

| 相机类型 | 维护态默认 profile |
| --- | --- |
| all maintained cameras | `external` |

补充说明：

- `wrist`、`wrist_manipulator_top95`、`wrist_pick_place` 和 `wrist_pick_place_no_heatmap` 都仍然存在，但都不再属于维护态默认映射
- `external_manipulator`、`external_manipulator_v2`、`wrist_manipulator` 都需要显式指定
- `query_prefilter_mode` 默认是 `off`，不属于默认轨迹过滤主链
- `support_grid_ratio` 维护态默认是 `0.0`，也就是默认不追加外层 support points；它影响 tracker 前向负载，但不改变这里描述的轨迹过滤逻辑

## 2. 共享基础过滤框架

### 2.1 filter level 默认阈值

`filter_level` 决定共享基础过滤参数。当前维护态默认是 `none`。

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

这一步输出的主判定是 `query_depth_quality_mask`。对显式 wrist-like profile，还会额外计算一层 query-depth edge risk。

### 2.4 显式 wrist-like profile 专用 query-depth edge risk

只对 wrist-like profile 生效：

- `wrist`
- `wrist_pick_place`
- `wrist_pick_place_no_heatmap`
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

只有在显式 wrist-like 路径里，这层 edge risk 才会作为硬拒绝进入最终 mask。若显式使用 `traj_filter_ablation_mode=wrist_no_query_edge`，这层拒绝只保留调试量，不再进入最终判定。

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

## 4. 显式 wrist-like 路径：wrist_manipulator_top95

`wrist_manipulator_top95` 不再是维护态默认 profile。当前 external-only 维护态默认统一使用 `external`；这里只讨论用户显式指定 wrist-like profile 做历史调查或兼容性复跑时的真实代码行为。它不是直接从 external 放宽出来的，而是分成三层：

1. wrist 基础 seed
2. manipulator-aware 收缩
3. top95 motion 收缩

### 4.1 代码执行顺序

显式 wrist-like 路径不是单纯的 `wrist`，而是下面这条链：

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

显式 wrist-like 路径不仅要求 query depth 本身靠谱，还会额外拒绝落在危险深度边界上的点：

```text
query_depth_keep_mask = query_depth_quality_mask & (~query_depth_edge_risk_mask)
```

物理解释：

- 靠近夹爪轮廓或前景/背景分界线的深度，很容易是混合像素或边界噪声
- 这些点就算短时间能跟住，3D 也经常漂

所以显式 wrist-like 路径会比 external 更积极地排掉深度边界风险点。

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

这里默认用的是 `traj_motion_extent_all_valid`，不是只看 supervised prefix 的 motion。这意味着显式 wrist-like 路径会把“整段可用帧上的总体运动”当成更重要的主体信号。

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

显式使用 `wrist_manipulator_top95` 时，采用的是后者。

### 4.6 用一句话总结显式 wrist-like 路径

显式 wrist-like 路径不是在问：

“这条轨迹是不是整段都稳定？”

而是在问：

“这条轨迹是不是在前缀阶段真实可信、落在近场、确实在动、并且属于机械臂主体那一团？”

### 4.7 主要调试信号

排查显式 wrist-like 样本时，优先看这些字段：

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

## 5. external vs 显式 wrist-like path 的最终区别

如果只记一张表，可以记这一张：

| 维度 | `external` | 显式 wrist-like path (`wrist_manipulator_top95`) |
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
- 显式 wrist-like path 在问：“这是不是一条前缀阶段真实可信、靠近镜头、确实在动、并且属于机械臂主体那一团的轨迹？”

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
- `traj_base_mask`
- `traj_query_depth_quality_mask`
- `traj_query_depth_keep_mask`
- `traj_supervision_support_mask`
- `traj_query_depth_rank`
- `traj_query_depth_edge_mask`
- `traj_query_depth_patch_valid_ratio`
- `traj_query_depth_patch_std`
- `traj_query_depth_edge_risk_mask`
- `traj_query_source_bits`
- `traj_query_sampler_score`
- `traj_query_risk_bits`
- `traj_query_low_texture_score`
- `traj_query_specular_score`
- `traj_query_depth_edge_score`
- `traj_query_border_dist_px`
- `traj_motion_extent`
- `traj_motion_step_median`
- `traj_motion_extent_all_valid`
- `traj_motion_step_median_all_valid`
- `traj_manipulator_candidate_mask`
- `traj_manipulator_cluster_id`
- `traj_manipulator_component_size`
- `traj_near_depth_mask`
- `traj_motion_mask`
- `traj_cluster_mask`
- `traj_pre_top95_mask`
- `traj_manipulator_cluster_fallback_used`
- `traj_stereo_compare_frame_count`
- `traj_stereo_depth_consistency_ratio`
- `traj_stereo_patch_error`
- `traj_stereo_consistency_mask`

其中 `wrist_pick_place` 还会额外写出 object 分支调试量：

- `traj_pick_place_heatmap_hit_count`
- `traj_pick_place_heatmap_support_mask`
- `traj_pick_place_min_manipulator_distance`
- `traj_pick_place_contact_mask`
- `traj_pick_place_depth_guard_mask`
- `traj_pick_place_object_mask`

这些中间 mask 现在也会直接写进 sample NPZ，因此可以离线做 per-query diagnostic breakdown，不需要再复现 `build_traj_filter_result()`：

- `traj_base_mask`
- `traj_query_depth_quality_mask`
- `traj_query_depth_keep_mask`
- `traj_supervision_support_mask`
- `traj_near_depth_mask`
- `traj_motion_mask`
- `traj_cluster_mask`
- `traj_pre_top95_mask`

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

### 7.3 wrist_pick_place

`wrist_pick_place` 是维护态里显式支持的 pick_place wrist 相机 profile。它不是默认 `auto` 路径，必须手动指定。

它的最终目标不是“只保机械臂主体”，而是：

- 保住 wrist seed 里的 manipulator / gripper 轨迹
- 同时救回与 manipulator 接触并被 `pick` heatmap 支撑的物体轨迹

和普通 `wrist` / `wrist_manipulator` 不同，当前代码对这条分支的 base 几何门槛专门放宽了一档：

```text
wrist_pick_place_base_mask = valid_count_mask & depth_smooth_mask
```

也就是说，这里不再把整段 `depth_range_mask` 当成进入 pick_place 分支前的硬门槛。当前实现这样做的原因是：

- pick_place 的被抓物体在抬升阶段可能出现 `traj_uvz[..., 2]` 局部失真
- 但 query frame 深度和时域支撑仍然可信
- 如果仍要求整段 `depth_range_mask` 全通过，会把本来应该交给 pick_place 分支处理的局部物体点提前误杀

代码上可概括为：

```text
wrist_pick_place_base_mask & query_depth_mask & supervision_support_mask
    -> manipulator 分支
    -> pick_place object 分支

final_mask = manipulator_final_mask | pick_place_object_mask
```

其中：

- manipulator 分支沿用 `wrist_manipulator` 的 near-depth / motion 约束，但 cluster 阶段会保留多个 major components，而不是只保单个 largest component
- 在 major-components 之后，`wrist_pick_place` 还会做一层 component refinement：
  - component motion 中位数至少达到最佳 component 的 `75%`
  - 或 component query depth 中位数距离最近 component 不超过 `0.08m`
  - 其目标是继续保双臂，但压掉“整体更远且整体更静”的伪 component
- object 分支当前要求：
  - 轨迹在 sample 段内命中 `pick` heatmap
  - 轨迹与 manipulator reference 在世界坐标下足够接近
  - query 深度不能比 manipulator 分支远太多

当前实现对应的维护态阈值是：

- 最少 `2` 帧 `pick` heatmap 命中
- 与 manipulator 的最小距离不超过 `0.20m`
- query 深度相对 manipulator 的容差不超过 `0.25m`

这条分支的物理语义是：

“先用 wrist seed 保住前缀可信轨迹，再保留机械臂主体，并把与机械臂接触且被 pick cue 支撑的被抓物体一并保留下来。”

这里的 contact reference 不是把所有 manipulator 点压成一个全局 centroid，而是对每个 major component 单独建 reference，再取最小距离。这样在双臂/双夹爪近似对称时，不会因为 reference 偏到中间而把靠近第二臂的 object 误杀。

另外，`wrist_pick_place` 的 object depth guard 也不是再拿“全局 manipulator 深度分布”去比，而是：

- 先为每条 object candidate 找最近的 manipulator component
- 再只与该 component 的 query-depth upper bound 比较

这样可以避免某个更远的 component 把另一个 component 附近的 object depth guard 一起放宽。

### 7.4 wrist_pick_place_no_heatmap

`wrist_pick_place_no_heatmap` 是为“没有 per-frame `pick` heatmap”的 pick_place wrist 数据额外提供的显式 profile。它同样不是默认 `auto` 路径，必须手动指定。

它的目标不是复刻 `wrist_pick_place` 的完整 object branch，而是在尽量不增加时间花销的前提下，同时做两件事：

- 保留 query frame 局部活动区域里的近深度轨迹
- 压掉距离夹爪/物体较远的桌面伪轨迹和视野边缘伪轨迹
- 对 query frame 可见、但在接触发生前还不属于 local region 的 pre-grasp object，补一层 delayed-contact rescue

和 `wrist_pick_place` 一样，这条分支当前也不再要求整段 `depth_range_mask` 先全通过，进入分支前的 base 门槛是：

```text
wrist_pick_place_base_mask = valid_count_mask & depth_smooth_mask
```

代码上可概括为：

```text
wrist_pick_place_base_mask & query_depth_mask & supervision_support_mask
    -> near-depth + motion anchors
    -> query-frame local keep region
    -> delayed-contact rescue

final_mask = local_keep_mask | delayed_contact_rescue_mask
```

其中：

- anchor 的候选集合复用 wrist manipulator 逻辑，但只把它当作“局部活动区域估计器”，不再把 motion fail 直接当成最终拒绝条件
- motion 统计使用 all-valid motion，而不是 supervised prefix motion
- local region 直接在 query frame 上根据 anchor 的 2D 包围框构造，并做固定像素 padding
- 如果 anchor 数量太少，就回退成只做 near-depth rank gate，不额外做 region 限制
- delayed-contact rescue 只面向：
  - query frame 可见
  - 当前不在 local keep region 内
  - 后续某帧与 manipulator reference 发生接触
  - query depth 没有比最近 manipulator component 明显更远

换句话说，这条分支当前的实际结构是：

```text
local_keep_mask =
    traj_near_depth_mask
  & local_region_mask

delayed_contact_rescue_mask =
    wrist_seed_mask
  & query_visible_mask
  & (~local_keep_mask)
  & delayed_contact_mask
  & depth_guard_mask

final_mask = local_keep_mask | delayed_contact_rescue_mask
```

当前实现对应的维护态阈值是：

- query-depth rank 最多保留前 `50%`
- anchor 的 all-valid motion extent 至少 `0.03m`
- query-frame region padding：左右各 `80px`、上 `40px`、下 `220px`
- 至少需要 `8` 条 anchor 才启用 local region；否则走 rank-only fallback
- delayed-contact rescue 的 contact 距离阈值是 `0.20m`
- delayed-contact rescue 的 query depth 相对最近 manipulator component 容差是 `0.25m`

这条分支的物理语义是：

“先用 wrist seed 保住前缀可信轨迹，再用会动且更近的局部 anchor 估计夹爪活动带，保住这片局部区域里的近深度点；如果某些 query 可见的 pre-grasp object 只是在接触前不属于 local region，再用 delayed-contact rescue 把它们补回来。”

它仍然不依赖 heatmap I/O，也不试图做完整的 object semantic 分类；当前实现更接近：

- 一层低成本 local denoise
- 再加一层低成本 delayed-contact object rescue

它的目标不是彻底解决 pick/place 里的 3D 跟踪失真，而是在不显著增加 I/O 和算时的前提下，先避免把 query 可见的抓取物体在过滤前阶段提前误杀。

### 7.5 query prefilter 和 ablation mode

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

本节讨论的是“当前这套显式 wrist-like 逻辑是否适合新任务”，不是只讨论历史建议。当前结论是：

- external 相机默认 `external` 仍然基本可沿用
- wrist 相机不能再把 `auto -> wrist_manipulator_top95` 当成通用默认
- 对具备 per-frame `pick` heatmap 的 pick_place wrist 数据，当前应优先使用 `wrist_pick_place`
- 对没有 per-frame `pick` heatmap 的 pick_place wrist 数据，当前应优先使用 `wrist_pick_place_no_heatmap`

### 8.1 为什么 `press` 场景下当前 wrist 默认问题不大

当前这套显式 wrist-like 逻辑，最初更接近 press 类任务：

- 主要运动主体就是机械臂和夹爪
- 被交互对象通常不需要被稳定抓住
- 任务过程中往往没有一个需要长期跟随保留的“被抓物体”

因此，当前默认的这几步：

- near-depth
- motion
- largest spatial component
- top95

虽然很偏向“机械臂主体”，但在 press 场景里通常不会造成特别大的语义损失。因为该保留的主要就是夹爪本体，而不是其他物体。

### 8.2 为什么 `pick_place` 不适合继续用显式 wrist-like 默认链路

`pick_place` 的目标不只是保留夹爪，还要保留被抓住并被搬运的物体。

但当前这套显式 wrist-like 逻辑会系统性偏向“保 gripper body，不保 object”。

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

当前这套显式 wrist-like 逻辑在这类场景下同样有明显风险。

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

当前维护态如果只使用现有 profile 和 ablation mode，建议这样设：

| 场景 | wrist 相机建议配置 | 原因 |
| --- | --- | --- |
| `press` | 继续可用 `auto` | 当前默认就是围绕“机械臂主体”优化的 |
| `pick_place` | 优先用 `traj_filter_profile=wrist_pick_place` | 保留 manipulator 分支，并额外救回被 `pick` heatmap 支撑且与 manipulator 接触的被抓物体 |
| `pick_place` 无可用 `pick` heatmap | 优先用 `traj_filter_profile=wrist_pick_place_no_heatmap` | 从放宽后的 pick_place seed 出发，先做低成本局部区域去噪，再补一层 delayed-contact rescue |
| `push_pull` | 先用 `traj_filter_profile=wrist` | 先避免 near-depth / motion / cluster / top95 把把手裁掉 |
| `push_pull` 若把手仍严重丢失 | 再试 `traj_filter_ablation_mode=wrist_no_query_edge` | 把手很容易死在 edge-risk 这一步 |
| 所有新场景 | `query_prefilter_mode=off` | 不要在 tracking 之前就按 wrist-manipulator 偏好裁 query seed |

这里要特别强调：

- `pick_place` 现在不建议继续使用 `wrist_manipulator`
- `pick_place` 更不建议继续使用 `wrist_manipulator_top95`
- `pick_place` 在有 `pick` heatmap 时，优先使用 `wrist_pick_place`
- `pick_place` 在没有 `pick` heatmap 时，优先使用 `wrist_pick_place_no_heatmap`
- `push_pull` 也不应把 `wrist_manipulator_top95` 当默认起点

短期内，更合理的策略是：

- `pick_place` 有 `pick` heatmap 时先切到 `wrist_pick_place`
- `pick_place` 没有 `pick` heatmap 时先切到 `wrist_pick_place_no_heatmap`
- `push_pull` 先用 `wrist` 保住“前缀阶段可信”的轨迹
- 再基于可视化结果判断缺的是被抓物体、把手，还是接触边缘

### 8.5 interaction-aware wrist profile 的现状与后续建议

如果后续要把这两类任务作为维护态支持场景，建议不要继续共用一个“只收缩到机械臂主体”的 wrist 默认 profile，而是至少拆成两类。

#### 已实现：`wrist_pick_place`

目标不是“只保机械臂主体”，而是：

- 保机械臂主体
- 保被抓住并与机械臂共同运动的物体

当前实现的物理逻辑：

1. 先从 `wrist_seed_mask` 保住所有前缀可信的 wrist 轨迹
2. 一条分支继续做 manipulator-aware 收缩，得到 gripper body
3. 另一条分支专门寻找“在接触后与 gripper 共动”的物体轨迹
4. 最终保留 `gripper_mask OR grasped_object_mask`

当前 object 分支明确不再强依赖：

- 初始近深度
- 初始大 motion
- 最大连通团

因为对被抓物体来说，这三条都可能不成立。

#### 已实现：`wrist_pick_place_no_heatmap`

目标不是在没有 heatmap 时重新发明一套更重的 interaction-aware object classifier，而是：

- 继续从 `wrist_seed_mask` 出发
- 尽量保住夹爪附近、与活动区域局部相关的 pick/place 轨迹
- 用最低额外成本压掉远处桌面和视野外伪轨迹
- 对 query frame 可见、后续才接触到夹爪的 pre-grasp object，补一层轻量 rescue

当前实现的物理逻辑：

1. 先从放宽后的 pick_place seed 出发，不要求整段 `depth_range_mask` 都通过
2. 用 near-depth + all-valid motion 选出一批局部 activity anchors
3. 在 query frame 上用这些 anchors 构造一个带 padding 的局部 keep region
4. 先保留 `local_keep_mask = near_depth_mask & local_region_mask`
5. 再对 query frame 可见、但当前不在 local keep region 里的轨迹做 delayed-contact rescue
6. 如果 anchor 太少，则回退成 rank-only near-depth gate

这条分支的核心不是“识别出 object 类别”，而是：

- 先把真正发生交互的局部区域框出来
- 再把一部分 query 可见的 pre-grasp object 补回来

它比直接退回 `wrist` 更干净，也比完整依赖 heatmap/object 推断更便宜。

#### 仍建议新增：`wrist_push_pull`

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
2. `pick_place` 若有 `pick` heatmap，优先切到 `wrist_pick_place`；若没有，则优先切到 `wrist_pick_place_no_heatmap`；`push_pull` 先切到 `wrist`
3. 对代表性 `pick_place` / `push_pull` case 导出可视化
4. 分别确认实际缺失的是：
   - 被抓物体
   - 门把手/拉手
   - 接触边界附近轨迹
5. 在 `pick_place` 已有分支基础上继续调参数；并针对 `push_pull` 新增 `wrist_push_pull`

这比继续把所有 wrist 任务都塞进 `wrist_manipulator_top95` 更稳妥。因为：

- `press` 需要的是“机械臂主体优先”
- `pick_place` 需要的是“机械臂主体 + 被抓物体”
- `push_pull` 需要的是“机械臂主体 + 小型接触件 + articulation 区域”

三者的目标对象并不相同，不适合继续共用同一个 wrist 默认收缩逻辑。

## 9. Egocentric Stereo 现状与改进建议

这一节专门记录 `stereo_left` / `stereo_right` 一类第一人称双目操作视频上的当前判断，避免后续继续把它们简单等同于 `external` 视角或“只保 manipulator 主体”的场景。

### 9.1 当前判断

对于当前接入的 `stereo_left` 数据，主要问题不是“明显未矫正的 fisheye 畸变”，而是：

- 第一人称自运动带来的强全局视角变化
- 由深度差引起的明显视差
- 反光、透明、低纹理和遮挡区域上的伪稳定轨迹

结合数据检查，`stereo_left` / `stereo_right` 可以暂时按“已 rectified 到足以近似 pinhole”的输入看待，因此现阶段不建议先对它再次做额外去畸变。对这一类数据，优先级更高的问题是 query 点怎么选、track 怎么验，而不是先把相机模型做得更复杂。

### 9.2 当前模型已经处理了什么，没处理什么

当前推理链路并不是纯 2D tracker：

- query 点会先用 query 时刻的深度和相机位姿 lift 到世界坐标
- 模型前向同时接收 `rgb/depth/intrinsics/extrinsics`
- 跟踪过程中会持续把 3D 点重新投影回各帧 2D

这意味着：

- 相机自运动
- 不同深度导致的不同像素速度

并不是完全没被建模。

但当前链路仍有很强的 learned 2D tracking 成分，所以一旦局部外观本身不可靠，几何信息也不能完全救回结果。对 egocentric kitchen / tabletop 场景，最典型的失败来源通常是：

- 反光和透明区域
- 大块低纹理平面
- 接触边界和遮挡
- 交互物体本身不是大连通主体，但会短时和手共同运动

另外，当前 BA 和相机模型并没有完整显式建模真实 fisheye 畸变；内部主要仍按 `SIMPLE_PINHOLE` 工作，只保留了非常有限的简化相机模型选项。因此在已经基本 rectified 的 `stereo_left` 上，这不是第一优先级；如果将来切到更强畸变的 `fisheye_cam*`，再把相机模型问题提到更高优先级。

### 9.3 `support_grid_ratio` 不是主矛盾

`support_grid_ratio` 只决定模型前向时是否额外补一层 support queries，用来帮助跟踪；它不是最终保留下来的主输出轨迹集合。当前维护态默认已经把它设为 `0.0`，也就是先关闭 support points，再单独评估是否需要额外上下文。

因此：

- 调 `support_grid_ratio` 不是当前最关键的杆
- 直接增大 `grid_size` 确实可能提升召回
- 但如果不同时加强 prefilter / postfilter，也会同步放大伪轨迹数量

对 egocentric 数据，更高 ROI 的顺序通常是：

1. 先改 query seed 的空间分布
2. 再加更强的 post-track validity 检查
3. 最后再把 `grid_size` 往上抬

### 9.4 “内容感知撒点”指的是帧内空间选点，不是改 query frame 时间规则

这里必须明确区分两层：

- query frame scheduling：时间上选哪几帧作为 query frame
- query seed selection：在每个已选 query frame 内，具体在哪些区域种 keypoint

当前 dense query 的默认做法是：

1. 先按现有规则决定 query frame
2. 然后在每个 query frame 上按规则网格均匀撒点
3. 最后再进入跟踪和过滤

因此这里说的“内容感知撒点”，不是去改第 1 步，而是去改第 2 步。

更具体地说，egocentric manipulation 上的 query seed 更适合改成“两阶段”：

1. 先做 relevance-first 的候选区域定义与采样
2. 再为每个已采样 query 点附带 risk flags，作为后验诊断和质量解释信号

这里的第一目标应是“更多覆盖我们真正关心的部分”，而不是“优先避开所有高风险区域”。对这类数据，更合理的候选区域通常包括：

- 人手 / 前臂附近
- 机械臂或工具主体附近
- 与手或工具发生接触、邻近或短时共运动的交互物体
- 必要的少量背景上下文区域

在这个前提下，低纹理、反光、depth-edge、视野边缘等信号更适合承担下面这些角色：

- 作为逐采样点的风险标记，而不是先验一票否决
- 帮助解释后续哪些轨迹更容易失败、失败原因可能是什么
- 在需要时作为后续 validity score 或 debug ranking 的辅助特征

也就是说，风险不是 query selection 的第一准则；它更像 relevance-aware sampling 之后附带的 trackability prior / diagnostics。

现有 `build_query_prefilter_result()` 只看 query frame 静态信号，而且对 `external*` profile 基本是 no-op。要把这件事真正做起来，需要把它从“风险优先的硬过滤”扩展成“relevance-first query sampler + per-query diagnostics”，并让它能够访问：

- query RGB
- query depth
- 若可用则加入 hand / manipulator / object interaction cue

建议至少给每个 query 点保留下面这些布尔或 bit flags，便于后续追查轨迹质量问题：

- `low_texture_flag`
- `specular_or_reflection_flag`
- `depth_edge_risk_flag`
- `image_border_risk_flag`

这些 flag 不应默认直接杀掉该 query 点，而应随 sample 一起保存，供：

- 后续轨迹失败归因
- 可视化时高亮高风险 query
- 后续 validity 分支或调参分析使用

### 9.5 stereo consistency 最适合放在 track 完成后的过滤阶段

对 egocentric stereo，stereo consistency 很值得加，但最高 ROI 的插入点不是模型最前面，而是：

- 单目左视角 track 已经生成之后
- 最终 `traj_valid_mask` 决定之前

也就是作为 `build_traj_filter_result()` 里的另一条 validity 分支，与现有的几何、深度一致性、temporal consistency 并列。

建议的最小版本可以是：

1. 用左目的 `traj_uvz` 和双目几何把轨迹 lift / reproject 到右目
2. 检查右目对应位置的 depth 是否一致
3. 检查右目局部 patch 与左目重投影结果是否明显冲突
4. 把不一致程度作为新的轨迹打分或 mask reason

如果这一版有效，再考虑更重的版本，例如：

- 左右双向都跑 tracker，再做 left-right agreement
- 在 query prefilter 阶段也加入 stereo cue，提前减少明显不可靠的 seed

但实现优先级上，先做 post-track stereo filter 更稳、更便于调参。

### 9.6 Egocentric 数据不应直接复用 `external_manipulator_v2`

`external_manipulator_v2` 的目标仍然是“从 external seed 收缩到 manipulator 主体附近”，它并不天然适合第一人称人手-物体交互场景。

对 egocentric manipulation，更合理的目标应是：

- 保住手/前臂附近可信轨迹
- 同时保住被交互的物体，而不是只保操作者身体部分

因此更建议新增显式 profile，例如 `egocentric_object_interaction_v1`，其逻辑应更接近：

```text
egocentric_seed
    -> manipulator branch
    -> object-near-manipulator branch

final_mask = manipulator_mask | interaction_object_mask
```

其中 object branch 不应强依赖“最大连通团”或“只保最近深度主体”，而应更多依赖：

- 与手部或 manipulator reference 的 3D 距离
- 接触后的短时共运动
- stereo consistency
- 必要时的 delayed-contact rescue

### 9.7 mocap 最适合先做评估和调参闭环，再考虑专门训练新 tracker

对于这类数据，mocap 很适合先拿来做评估和调参，而不是直接当成 dense point tracking 的完整真值。

短期更务实的顺序是：

1. 先用 mocap 建一套 egocentric 评估集
2. 指标重点看：
   - 手附近轨迹召回
   - 接触物体轨迹召回
   - 远背景伪轨迹率
   - 短时共运动正确率
3. 基于这套指标去调：
   - query prefilter
   - egocentric interaction profile
   - stereo consistency

如果做到这一步后，仍然发现主瓶颈来自 tracker 本体对 egocentric 视角的不适应，再考虑训练或微调专门的 tracker。

换句话说：

- “训练一个专门网络”是可能的中期方向
- “先把评估闭环和过滤逻辑建立起来”是更高 ROI 的近期方向

### 9.8 推荐落地顺序

面向当前 `stereo_left` / `stereo_right`，推荐按下面顺序推进：

1. 新增 `egocentric_object_interaction_v1`，不要沿用“只保 manipulator 主体”的 external 分支语义
2. 把 query prefilter 扩展到 egocentric / external，但目标应是 relevance-first 的 query sampler：先保手、manipulator、交互物体，并为每个采样点附加 query RGB/depth 风险 flags 供后续诊断
3. 在 track 完成后加入 stereo consistency 过滤
4. 在上述三项稳定后，再增大 `grid_size`
5. 用 mocap 建评估与调参闭环
6. 若仍明显受限，再考虑训练或微调专门的 egocentric tracker

这一路线的核心原则是：

- 先减少“明显不该种的点”
- 再减少“明显不该保的轨迹”
- 最后再追求更高召回
