# 轨迹过滤方法总结（2026-03-17）

## 1. 背景与目标

这轮修改的目标不是再新增一个独立过滤器，而是在现有 `filter_level` 之上补齐一层 `camera-aware traj_filter_profile`，让不同相机视角共享同一套基础过滤框架，同时允许 wrist 相机使用更贴近机械臂运动规律的约束。

当前状态：

- 动态链路与静态链路都统一走 `utils/traj_filter_utils.py`
- sample 序列化和可视化读取都已经能消费统一的 `traj_valid_mask` 与调试字段
- `auto` profile 会把 wrist-like 相机自动映射到 `wrist`，其余相机走 `external`

## 2. 方法总览

轨迹过滤由两层配置共同决定：

- `filter_level`: `none / basic / standard / strict`
- `traj_filter_profile`: `external / external_manipulator / external_manipulator_v2 / wrist / wrist_manipulator`

其中：

- `none` 关闭过滤，直接保留全部轨迹
- `basic / standard / strict` 共享同一套基础过滤框架，只是阈值不同
- `external` 面向外部相机
- `external_manipulator` 在 `external` 基础上进一步收缩到机械臂主体轨迹
- `external_manipulator_v2` 是面向 external 相机的更宽松 manipulator 版本
- `wrist` 面向 wrist 相机，允许“短前缀但有连续支撑”的轨迹保留
- `wrist_manipulator` 在 `wrist` 基础上进一步逼近“机械夹爪/机械臂主体”轨迹

`auto` 的映射规则如下：

- 相机名以 `camera_3` 结尾，或包含 `wrist` / `hand`，映射到 `wrist`
- 其他相机映射到 `external`
- `external_manipulator`、`external_manipulator_v2` 与 `wrist_manipulator` 都需要显式指定，不参与 `auto`

## 3. 基础过滤框架

### 3.1 filter level 默认阈值

| level | min_valid_frames | boundary_margin | visibility_threshold | depth_smoothness | depth_change_threshold |
|------|------------------:|----------------:|---------------------:|------------------|-----------------------:|
| `basic` | 3 | 50 | 0.0 | off | 0.5 |
| `standard` | 3 | 50 | 0.5 | on | 0.5 |
| `strict` | 5 | 20 | 0.6 | on | 0.3 |
| `none` | 0 | 50 | 0.0 | off | 0.5 |

所有启用过滤的 level 都共享下面几条基础规则：

- 深度范围：`0.01m < depth < 10.0m`
- query-depth quality 开启
- temporal depth consistency 开启
- depth volatility guidance 开启

### 3.2 Query-depth quality

query 帧上的每个 keypoint 都会先做一次局部深度质量检查：

- patch 半径 `2`，即 `5x5` patch
- patch 内有效深度比例至少 `0.4`
- query 像素深度与 patch 有效深度中位数的偏差不超过：
  - 绝对阈值 `0.05m`
  - 或相对阈值 `10%`

这一步主要用于去掉 query 深度本身就是坏值、孤立值或平台值的轨迹种子。

### 3.3 Temporal depth consistency

时域一致性默认阈值：

- 深度绝对容差 `0.05m`
- 深度相对容差 `10%`
- 最低一致性比例 `0.95`
- volatility mask percentile `99.0`

输出调试量包括：

- `traj_depth_consistency_ratio`
- `traj_stable_depth_consistency_ratio`
- `traj_high_volatility_hit`
- `traj_volatility_exposure_ratio`
- `traj_compare_frame_count`
- `traj_stable_compare_frame_count`
- `traj_supervision_mask`
- `traj_supervision_prefix_len`
- `traj_supervision_count`

## 4. 四个 profile 的差异

### 4.1 `external`

`external` 直接使用完整基础过滤：

- `base_mask`
- `query_depth_mask`
- `temporal_mask`

最终：

```text
final_mask = base_mask & query_depth_mask & temporal_mask
```

这条路径更适合外部相机，因为外部视角下，静态背景点通常也会满足“长时可见、几何稳定、深度一致”的条件。

### 4.2 `external_manipulator`

`external_manipulator` 不会放宽 `external` 的时域要求，而是把 `external` 的最终结果作为 manipulator seed，再继续做一层机械臂主体筛选：

1. 近深度优先
   - `traj_query_depth_rank <= 0.50`
2. 最低运动幅度
   - `traj_motion_extent >= 0.03m`
3. 最大 2D 空间连通簇
   - `radius_ratio = 0.06`
   - `radius_min_px = 24`
   - `min_component_ratio = 0.005`
   - `min_component_size = 2`

最终：

```text
external_seed_mask = base_mask & query_depth_mask & temporal_mask
traj_manipulator_candidate_mask = external_seed_mask & near_depth_mask & motion_mask
final_mask = largest_spatial_component(traj_manipulator_candidate_mask)
```

这条 profile 的用途不是替代通用 `external`，而是当用户明确想看外部视角下的机械臂主体时，提供一个更强收缩版本。

### 4.3 `external_manipulator_v2`

`external_manipulator_v2` 保留了 external 链路的完整 base/query/temporal 约束，但把 external manipulator 的后处理收缩规则专门调宽了一档：

1. query 深度排名更宽松
   - `traj_query_depth_rank <= 0.70`
2. 最低运动幅度更宽松
   - `traj_motion_extent >= 0.01m`
3. 保留“主要连通块”而不是只保留单个最大连通块
   - `radius_ratio = 0.06`
   - `radius_min_px = 24`
   - `min_component_ratio = 0.002`
   - `min_component_size = 2`
   - `major_component_ratio = 0.15`

最终：

```text
external_seed_mask = base_mask & query_depth_mask & temporal_mask
traj_manipulator_candidate_mask = external_seed_mask & near_depth_mask & motion_mask
final_mask = major_spatial_components(traj_manipulator_candidate_mask)
```

这条 profile 的目标是：

- 保住 external 视角下运动幅度较小但仍属于机械臂主体的轨迹
- 当机械臂主体在 query 图像上分裂成两个主要 2D 连通块时，同时保住这些主要块
- 继续压掉明显离散的小噪声块，而不是退回到近似 `external` 的宽泛结果

### 4.4 `wrist`

`wrist` 的核心变化是，不再要求完整的 `external` 风格时域稳定，而是改为“前缀连续 + 总支撑帧数”：

- `wrist_base_mask = valid_count & depth_range & depth_smooth`
- 保留 query-depth quality
- 用 supervision 信息替代 `external` 的完整 temporal pass/fail

wrist support 的要求为：

- prefix 帧数要求：`max(3, ceil(0.15 * T))`
- support 帧数要求：`max(3, ceil(0.20 * T))`

最终 wrist seed：

```text
wrist_seed_mask = wrist_base_mask & query_depth_mask & supervision_support_mask
```

对 wrist 相机来说，这能保留“只在前缀阶段稳定出现、随后被机械臂遮挡或离开视野”的真实夹爪轨迹。

### 4.5 `wrist_manipulator`

`wrist_manipulator` 在 `wrist_seed_mask` 上继续做三步 manipulator-aware 收缩：

1. 近深度优先
   - `traj_query_depth_rank <= 0.50`
2. 最低运动幅度
   - `traj_motion_extent >= 0.03m`
3. 最大 2D 空间连通簇
   - `radius_ratio = 0.06`
   - `radius_min_px = 24`
   - `min_component_ratio = 0.005`
   - `min_component_size = 2`

最终：

```text
traj_manipulator_candidate_mask = wrist_seed_mask & near_depth_mask & motion_mask
final_mask = largest_spatial_component(traj_manipulator_candidate_mask)
```

这条 profile 的目标不是保留所有真实运动，而是更有针对性地逼近机械夹爪/机械臂主体轨迹，尽量压掉 wrist 视角下仍可能残留的静止背景伪轨迹。

## 5. 调试字段与 reason bits

这轮新增的 reason bits：

- `bit4`: `MASK_REASON_MANIPULATOR_DEPTH_FAIL`
- `bit5`: `MASK_REASON_MANIPULATOR_MOTION_FAIL`
- `bit6`: `MASK_REASON_MANIPULATOR_CLUSTER_FAIL`

这轮新增的 sample 调试字段：

- `traj_wrist_seed_mask`
- `traj_query_depth_rank`
- `traj_motion_extent`
- `traj_motion_step_median`
- `traj_manipulator_candidate_mask`
- `traj_manipulator_cluster_id`
- `traj_manipulator_component_size`
- `traj_manipulator_cluster_fallback_used`

这些字段现在已经在：

- `v2 sample NPZ`
- `legacy sample NPZ`
- `normalize_sample_data()`

之间保持一致。

## 6. 已完成验证

### 6.1 单元测试

已覆盖：

- query-depth quality
- temporal consistency 与 stable temporal fail
- `wrist_manipulator` 的 depth / motion / cluster 三段逻辑
- 小样本 fallback
- 非 manipulator profile 的默认调试字段

通过命令：

```bash
python -m py_compile utils/traj_filter_utils.py utils/traceforge_artifact_utils.py utils/test_traj_filter_utils.py scripts/batch_inference/infer.py scripts/batch_inference/batch_infer_press_one_button_demo.py
python -m unittest utils.test_traj_filter_utils utils.test_traceforge_artifact_utils
```

### 6.2 Wrist 相机 3D 对比结果

对 `varied_camera_3` 的 `query_frame=15`，已验证 4 组 before/after：

| episode | profile before | profile after | kept tracks |
|------|-----------------|---------------|------------:|
| `episode_00007` | `wrist` | `wrist_manipulator` | `4684 -> 1539` |
| `episode_00000` | `wrist` | `wrist_manipulator` | `4450 -> 1548` |
| `episode_00060` | `wrist` | `wrist_manipulator` | `4911 -> 1605` |
| `episode_00119` | `wrist` | `wrist_manipulator` | `4625 -> 1568` |

结论：

- `wrist` 已能保住被遮挡或出画前缀仍可信的机械臂轨迹
- `wrist_manipulator` 能进一步压掉 wrist 视角下的静止背景伪轨迹
- `episode_00119` 因总帧数较短，`query_frame=15` 的 segment 长度为 `31`，其余示例为 `32`

### 6.3 External 相机 3D 对比结果

对 external 相机的 `query_frame=15`，当前已验证 `external / external_manipulator / external_manipulator_v2` 三档收缩强度：

| episode | camera | external | external_manipulator | external_manipulator_v2 |
|------|--------|---------:|---------------------:|------------------------:|
| `episode_00000` | `varied_camera_1` | `6032` | `178` | `182` |
| `episode_00060` | `varied_camera_2` | `5924` | `120` | `209` |
| `episode_00119` | `varied_camera_1` | `6156` | `110` | `126` |

结论：

- `external_manipulator` 仍然是更强收缩的版本，适合只想看机械臂主体最核心部分时使用
- `external_manipulator_v2` 主要通过放宽 `motion_extent` 与保留多个主要连通块，补回 external 视角下被过度裁掉的机械臂主体轨迹
- `episode_00119` 额外暴露出一个 external 视角的小次主块，因此 `v2` 还单独放宽了 `min_component_ratio`

## 7. 已知限制

### 7.1 保留的轨迹过滤对比产物

本地仅保留当前仍有明确对比意义的目录：

- `data_tmp/traj_filter_review/2026-03-17/wrist_vs_manipulator_round2/...`
- `data_tmp/traj_filter_review/2026-03-17/external_vs_external_manipulator/...`
- `data_tmp/traj_filter_review/2026-03-17/external_vs_external_manipulator_v2/...`

### 7.2 以下脚本本轮未同步更新

相关离线分析脚本现已归档到 `scripts/archived/investigations/2026-03/`：

- `analyze_depth_volatility.py`
- `compare_traj_filter_results.py`
- `analyze_ray_artifact.py`

它们不会影响当前推理、sample 保存和 3D 可视化结果，但其字段解释和默认输入路径不再保证与当前主流程完全一致。
