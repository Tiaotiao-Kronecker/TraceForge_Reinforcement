# 3D 假射线问题调查记录（2026-03-16）

> Historical note: this is an archived investigation record. The diagnostic
> scripts mentioned here now live under
> `scripts/archived/investigations/2026-03/`, and some intermediate outputs may
> have been cleaned up after the investigation concluded.

## 1. 背景

本记录用于跟进 `2026-03-15` 晚上轨迹过滤验证后的新问题：

- 3D 动画中出现从大致视角方向发出的多条变动、间断的射线
- 靠近视角的位置射线更密
- 用户补充观察：该现象不仅出现在 `dense pointcloud`，也出现在 `keypoint / trajectory` 可视化中

本轮目标是做第一轮纯诊断，不修改默认推理逻辑、默认过滤阈值、默认输出协议。

## 2. 复现路径

基准结果目录：

- `data_tmp/traj_filter_benchmark/2026-03-15`

第一轮诊断脚本：

- `scripts/data_analysis/analyze_ray_artifact.py`

执行命令：

```bash
/home/wangchen/.conda/envs/traceforge/bin/python \
  scripts/data_analysis/analyze_ray_artifact.py
```

第一轮诊断产物：

- `data_tmp/ray_artifact_investigation/2026-03-16/summary.json`
- `data_tmp/ray_artifact_investigation/2026-03-16/summary.md`
- `data_tmp/ray_artifact_investigation/2026-03-16/dense_filter_scan.csv`
- `data_tmp/ray_artifact_investigation/2026-03-16/keypoint_scan.csv`
- `data_tmp/ray_artifact_investigation/2026-03-16/dense_only/primary_case_pointcloud_filter_comparison.png`
- `data_tmp/ray_artifact_investigation/2026-03-16/dense_only/primary_case_depth_mask_overlay.png`
- `data_tmp/ray_artifact_investigation/2026-03-16/keypoint_only/primary_case_keypoint_diagnostics.png`
- `data_tmp/ray_artifact_investigation/2026-03-16/joint_overlays/primary_case_joint_overlay_worldz.png`
- `data_tmp/ray_artifact_investigation/2026-03-16/joint_overlays/primary_case_joint_overlay_camera_depth.png`

主复现样本固定为：

- `standard/inference/episode_00000_blue/varied_camera_1/samples/varied_camera_1_15.npz`

## 3. 已确认事实

### 3.1 与轨迹过滤标准无关

在 `none_ref / basic / standard / strict` 四组结果中：

- 同一相机的 `scene.h5` 完全一致：`9 / 9`
- 同一样本的 `traj_uvz` 完全一致：`33 / 33`
- 同一样本的 `visibility` 完全一致：`33 / 33`
- 真正变化的主要是 `traj_valid_mask`

结论：

- 假射线问题早于 `basic / standard / strict` 的轨迹过滤阈值
- 不能先把问题归因给轨迹过滤标准差异

### 3.2 当前 v2 产物缺少 `depth_conf`

当前 v2 输出保存了：

- `scene.h5` 中的 depth / intrinsics / extrinsics
- sample NPZ 中的 `traj_uvz` / `traj_valid_mask` / 可选 `visibility`

但不会把 `depth_conf` 持久化到 v2 正式产物里。

这意味着：

- 当前可以先做大量纯几何诊断
- 如果后续还需要判断“坏深度像素是否本来就低置信”，则要单独补一个调试导出路径，而不是直接猜

## 4. 第一轮诊断结果

### 4.1 Dense 分支：有明确的强信号

主复现样本中，当前点云路径使用的是 `world z` 过滤；如果改成按相机深度过滤，同一帧点云会发生明显变化：

- 当前 `world z` 保留点数：`42657`
- 相机深度过滤保留点数：`57487`
- 仅被当前 `world z` 保留的点：`113`
- 仅被相机深度过滤保留的点：`14943`
- 两种过滤的点级别 mismatch ratio：`0.2614`

这不是单个样本偶发。对 `standard` tag 下全部 `33` 个 sample 做横向扫描：

- 全部样本平均 mismatch ratio：`0.1717`
- `varied_camera_1` 平均 mismatch ratio：`0.2543`
- `varied_camera_2` 平均 mismatch ratio：`0.1883`
- `varied_camera_3` 平均 mismatch ratio：`0.0724`

同一轮扫描还显示：

- `varied_camera_1` 和 `varied_camera_2` 上，当前实现会系统性丢掉大量原本满足相机深度阈值的点
- 主复现样本的深度图存在明显的 `2.0` 平台值峰值，且平台区域大面积分布在背景

当前解释：

- `dense pointcloud` 的假射线问题有充分证据说明至少部分来自点云构建路径本身
- 其中一个高优先级嫌疑点是 `build_pointcloud_from_frame()` 目前按 `world z` 做深度阈值，而不是按相机深度做过滤
- 这条分支已经有足够强的客观信号，不需要依赖肉眼判断

### 4.2 Keypoint 分支：静态 benchmark 里未复现同等强度的问题

脚本对 `standard` tag 下全部 sample 做了 `traj_valid_mask` 之后的轨迹几何扫描。

诊断指标定义为：

- `ray_angle_p90_deg`：轨迹相对相机中心的视线方向稳定性
- `radial_span`：轨迹沿视线方向的半径变化量

第一轮保守阈值：

- `ray_angle_p90_deg < 2.0`
- `radial_span > 0.2m`

结果：

- 非空样本数：`30`
- 命中可疑 ray-like 轨迹的样本数：`0`
- 主复现样本中可疑轨迹数：`0 / 6257`
- 主复现样本 `radial_span p95` 只有 `0.00067m`
- 主复现样本 `ray_angle_p90_deg p95` 只有 `0.028deg`

主复现样本里，保存下来的 filtered keypoint 轨迹更像是局部一致的小范围运动，而不是沿视线方向被拉长的长射线。

当前解释：

- 在当前 benchmark 的离线 sample 数据里，没有复现出和 dense 分支同强度的 keypoint 假射线
- 这说明“keypoint 也有假射线”的症状，目前还不能直接归因给 sample 里已保存的 `traj_uvz`
- 更可能的方向是：
  - 动画路径的显示逻辑放大了某些现象
  - 用户观察到的 keypoint 假射线来自另一批样本或另一种交互状态
  - keypoint 与 dense 不是同一个根因

## 5. 待排除的可视化干扰项

虽然这轮 benchmark 静态分析没有复现强 ray-like keypoint 轨迹，但仍有两个动画实现点需要保留为干扰项：

- `scripts/visualization/visualize_3d_keypoint_animation.py` 当前没有按逐帧 `visibility` 做显示筛选
- 该脚本在更新 keypoint 点云时，会把无效点临时置到 `(0, 0, 0)`

补充说明：

- 在本次 benchmark 的 `traj_valid_mask` 后 sample 上，这个 `(0, 0, 0)` 路径没有被大量触发
- 因此它不是当前离线 benchmark 的主因
- 但它仍然可能在其他 case 或交互使用方式下放大视觉假象

## 6. 当前结论

第一轮暂定分类：

- `B（provisional）`

含义：

- `dense pointcloud` 分支已经有明确证据指向“深度/点云构建路径”问题
- `keypoint` 分支在当前 benchmark 保存结果中没有复现同等强度的 ray-like 几何异常
- 因此两者暂时不能被当作同一个根因处理

更直接地说：

- “这就是单纯的轨迹过滤问题”这个判断可以排除
- “这就是 sample 里已保存 keypoint 轨迹本身塌成射线”这个判断，在当前 benchmark 上没有证据支持
- “dense 点云构建路径存在独立问题”这条判断有强证据支持

## 7. 已实现修复

这轮不再只修 dense 可视化，而是同时落地两条链路：

- 修 dense 点云共享 helper，解决 3D 背景里的错误深度过滤
- 修 sample `traj_valid_mask`，避免坏 query 深度轨迹进入训练数据

### 7.1 Dense 点云链路

已修改：

- 目标函数：`utils/traceforge_artifact_utils.py` 里的 `build_pointcloud_from_frame()`
- 函数签名和返回值保持不变，仍返回世界坐标系中的 `points` 和对应 `colors`
- `depth_min / depth_max` 的过滤语义从错误的 `world z` 改为正确的相机前向深度
- 过滤现在基于输入 depth 的下采样值，而不是反投影后的 `pts[:, 2]`
- 保留原有的 `np.isfinite(points)` 和 `np.isfinite(colors)` 检查

直接受影响的共享路径：

- `scripts/visualization/visualize_3d_keypoint_animation.py`
- `scripts/visualization/verify_episode_trajectory_outputs.py`
- `scripts/visualization/visualize_single_image.py`
- `scripts/visualization/export_pointcloud_ply.py`

### 7.2 Sample 轨迹链路

已新增：

- 轻量过滤模块：`utils/traj_filter_utils.py`
- `scripts/batch_inference/infer.py` 中的 `traj_valid_mask` 生成逻辑已切到该模块

当前规则：

- `none` 级别保持原语义，不增加 query-depth 过滤
- `basic / standard / strict` 都会额外叠加一层 `query-depth quality mask`
- 这层 mask 基于 `depths_segment[0]` 和 `keypoints`
- 采样方式与 `prepare_query_points()` 保持一致，使用 `round(xy)` 取 query 像素
- 以 query 像素为中心取 `5x5` patch
- query 像素本身必须满足 `min_depth < depth < max_depth`
- patch 内有效深度比例必须至少为 `40%`
- query 深度与 patch 有效深度中位数的偏差必须不超过 `max(5cm, 10%)`
- 最终 sample 中仍保存全量 `traj_uvz`，只强化 `traj_valid_mask`

这次仍然不修改：

- `traj_uvz` 本体
- 上游深度模型
- v2 sample NPZ 的字段协议

## 8. 检验与结果

### 8.1 代码级验证

已新增最小回归测试：

- `utils/test_traceforge_artifact_utils.py`
- `utils/test_traj_filter_utils.py`

覆盖点：

- `camera depth` 合法但 `world z` 非法时，dense helper 仍保留点
- `camera depth` 非法但 `world z` 看起来合法时，dense helper 过滤点
- `depth == depth_min` 和 `depth == depth_max` 都按开区间过滤
- query 深度非法、局部 patch 稀疏、局部中位数偏差过大时，query-depth mask 会过滤
- `filter_level=none` 不引入新过滤
- `filter_level=basic` 会把 query-depth mask 与原有轨迹 mask 做 `AND`

执行结果：

```bash
python -m unittest utils.test_traceforge_artifact_utils utils.test_traj_filter_utils
```

- 结果：`Ran 9 tests ... OK`

### 8.2 主复现 case 定量验证

主复现样本：

- `standard/inference/episode_00000_blue/varied_camera_1/samples/varied_camera_1_15.npz`

在当前代码下对主复现 case 重新核对，得到：

- `dense_legacy_worldz_kept = 42657`
- `dense_corrected_camera_depth_kept = 57487`
- `dense_helper_kept = 57487`
- `dense_mismatch_points = 15056`

结论：

- 修复后的共享 helper 已与 `camera-depth corrected` 口径完全对齐
- 这说明 3D 动画里 dense 背景的那部分伪射线来源，确实来自错误的 `world z` 过滤

同一主复现样本上，对已有 benchmark sample 离线重算 query-depth 质量层，得到：

- `traj_valid_mask_original_kept = 6257`
- `query_depth_quality_kept = 6364`
- `traj_valid_mask_strengthened_kept = 6233`
- `traj_additional_filtered_by_query_depth = 24`

解释：

- 在旧 benchmark sample 上，如果叠加新 query-depth 质量层，会额外去掉 `24` 条此前保留的轨迹
- 这些结果是“离线重算后的预期效果”，不会回写旧 sample 文件
- 要让磁盘上的 sample 产物真正更新，仍需要重新跑对应推理任务

### 8.3 脚本级验证

已复跑：

```bash
/home/wangchen/.conda/envs/traceforge/bin/python \
  scripts/data_analysis/analyze_ray_artifact.py
```

诊断产物已刷新到：

- `data_tmp/ray_artifact_investigation/2026-03-16/summary.json`
- `data_tmp/ray_artifact_investigation/2026-03-16/dense_filter_scan.csv`
- `data_tmp/ray_artifact_investigation/2026-03-16/keypoint_scan.csv`
- `data_tmp/ray_artifact_investigation/2026-03-16/dense_only/primary_case_pointcloud_filter_comparison.png`
- `data_tmp/ray_artifact_investigation/2026-03-16/keypoint_only/primary_case_keypoint_diagnostics.png`

该脚本本身保留了 `legacy world-z filter` 和 `camera-depth filter` 的显式对照，所以共享 helper 修复后，仍可继续用来做回归比较。

### 8.4 剩余人工 smoke check

当前已经完成代码级和定量验证，但还没有在新生成产物上做人工 3D 观察。后续可继续用主复现 case 做这三项 smoke check：

1. `verify_episode_trajectory_outputs.py`
2. `visualize_single_image.py`
3. `visualize_3d_keypoint_animation.py --dense_pointcloud`

重点观察：

- dense 背景中的结构化断裂面和射线是否消失
- keypoint / trajectory 的射线是否随着新 `traj_valid_mask` 一起减少

## 9. 当前状态

- 状态：`implemented + partially validated`
- 已确认修复：`dense pointcloud 的过滤坐标轴错误`
- 已确认补强：`sample traj_valid_mask 现在会过滤坏 query 深度轨迹`
- 尚未完成：`在重新生成的新样本上做最终人工 3D 动画验收`

## 10. 第二轮实现：时序一致性 + 高波动区域 veto

为解决“原始 3DGS 深度本身不稳，单帧 query-depth 过滤不够”的问题，这一轮继续补强 sample 侧 `traj_valid_mask`。

### 10.1 新增过滤逻辑

实现位置：

- `utils/traj_filter_utils.py`
- `scripts/batch_inference/infer.py`

新增规则对 `basic / standard / strict` 一并生效，`none` 保持原语义不变：

- 先保留原有 `base geometric mask`
- 叠加已有 `query-depth quality mask`
- 再叠加 `temporal depth consistency mask`
- 最后叠加 `high volatility veto`

其中：

- `temporal depth consistency`
  - 先把 sample 中的 `traj_uvz` 从 query camera 坐标恢复到世界坐标
  - 再逐帧重投影到该 segment 每一帧自己的相机
  - 只在同时满足以下条件的帧上比较深度：
    - 轨迹重投影有效
    - 投影点在图像内
    - 原始 raw depth 在该像素有效
    - 若 `visibs` 可用，则要求该帧可见
  - 单帧一致判据：
    - `abs(depth_proj - depth_obs) <= max(0.05m, 0.10 * depth_obs)`
  - 单轨迹保留判据：
    - `valid_compare_frames >= min(segment_len, max(3, min_valid_frames))`
    - `consistency_ratio >= 0.95`

- `high volatility veto`
  - 从整段 raw depth 视频计算：
    - `volatility_map = p95(depth) - p05(depth)`
  - 再取全图 `p99.0` 作为高波动阈值
  - 若一条轨迹在任意有效/可见重投影帧命中该高波动区域，则整条轨迹删除

### 10.2 新增 sample 调试字段

本轮 sample NPZ 额外保存了三个调试字段：

- `traj_depth_consistency_ratio`：`float16, (N,)`
- `traj_high_volatility_hit`：`bool, (N,)`
- `traj_mask_reason_bits`：`uint8, (N,)`

bit 编码：

- `bit0 = 1`：base geometric fail
- `bit1 = 2`：query-depth quality fail
- `bit2 = 4`：temporal consistency fail
- `bit3 = 8`：high volatility veto

这些字段同时写入：

- v2 sample
- legacy sample

### 10.3 新增深度波动诊断脚本

新增：

- `scripts/data_analysis/analyze_depth_volatility.py`

输出产物：

- `depth_volatility_heatmap.png`
- `depth_volatility_overlay.png`
- `depth_volatility_summary.json`
- `depth_volatility_tracks_overlay.png`

该脚本会：

- 从相机目录读取 raw depth 视频并生成时序波动热力图
- 在 RGB 上叠加高波动区域
- 若 sample 存在，则把保留/过滤/高波动命中的轨迹点叠加到 query frame 上

## 11. 第二轮验证结果

### 11.1 代码级验证

执行：

```bash
python -m unittest utils.test_traceforge_artifact_utils utils.test_traj_filter_utils
python -m py_compile \
  scripts/batch_inference/infer.py \
  utils/traj_filter_utils.py \
  utils/traceforge_artifact_utils.py \
  scripts/data_analysis/analyze_depth_volatility.py
```

结果：

- `unittest`: `Ran 14 tests ... OK`
- `py_compile`: 通过

新增覆盖点包括：

- 稳定轨迹通过时序一致性过滤
- 深度一致性比例不足时过滤
- 可比较帧数不足时过滤
- `visibility=False` 的帧不会参与时序一致性判定
- 任意一帧命中高波动区域则整条轨迹过滤
- `none` 级别完全绕过新规则

### 11.2 诊断脚本验证

已生成热力图诊断结果：

- `data_tmp/ray_artifact_investigation/2026-03-16/depth_volatility/episode_00000_blue_varied_camera_1`
- `data_tmp/ray_artifact_investigation/2026-03-16/depth_volatility/episode_00119_pink_varied_camera_1`

其中 `episode_00119_pink / varied_camera_1` 的 summary 显示：

- 高波动阈值：`1.8502m`
- 高波动像素占比：`1.0001%`
- `top20 mean volatility = 0.469m`
- `center20 mean volatility = 0.004m`
- `bottom20 mean volatility = 0.0m`

这与此前“上沿区域明显更不稳定”的人工观察一致。

### 11.3 同源 v3 真实重跑验证

为了与昨天的 `episode_00119_pink` 可视化基线保持同源，这一轮不是只做离线重算，而是直接重跑：

- 输入源：`/data1/yaoxuran/press_one_button_demo_v3/episode_00119_pink`
- 输出目录：`data_tmp/ray_artifact_investigation/2026-03-16/revalidated_temporal_volatility_fix_v3/inference/episode_00119_pink/varied_camera_1`

对 `query_frame = 15` 的 sample 做同源新旧对比，得到：

- `traj_uvz`：完全一致
- `visibility`：完全一致
- 旧 `traj_valid_mask kept = 6063`
- 新 `traj_valid_mask kept = 5789`
- 新增过滤轨迹：`274`
- 重新放回的轨迹：`0`

这说明：

- 本轮修改没有改变 tracking 结果本身
- 只改变了 sample 侧的 `traj_valid_mask`
- 且变化方向是单向更严格，不存在“只是换了随机结果”的问题

对新 sample 的调试字段统计：

- `filtered_total = 611`
- `base geometric fail = 332`
- `query-depth fail = 12`
- `temporal consistency fail = 232`
- `high volatility hit = 198`

对应的可视化诊断目录：

- `data_tmp/ray_artifact_investigation/2026-03-16/depth_volatility/episode_00119_pink_varied_camera_1_newmask`

其 `sample_summary` 中可见：

- `kept_tracks = 5789`
- `filtered_tracks = 611`
- `high_volatility_hit_tracks = 198`

### 11.4 当前结论更新

到这一轮为止，可以更明确地下结论：

- 剩余密集点云假射线的主因仍是 raw 3DGS depth 本身不稳，而不是 sample mask
- 但 sample 侧现在已经能更积极地把这类区域上的坏轨迹排除出训练数据
- 对同源 `episode_00119_pink`，新的 `traj_valid_mask` 相比旧版额外删掉了 `274` 条轨迹
- 其中高波动区域 veto 单独命中的轨迹就有 `198` 条

仍待人工验收的部分只有：

- 在 3D 动画里主观观察这些新增过滤是否足够明显
- 是否还需要进一步把 dense 背景也做“原始深度稳健化”处理

## 12. 第三轮实现：几何主导，波动只做辅助证据

第二轮的主要问题是：

- `high volatility hit` 被当成硬 veto
- 对 `episode_00119_pink / varied_camera_1 / query_frame=15` 造成了明显过删
- 新增删掉的 `274` 条里，绝大多数集中在 `top20 + center33`
- 其中 `146` 条是 `high-volatility only` 删除，深度一致性本身接近 `1.0`

这说明：

- 这些轨迹并不是“几何上自相矛盾”
- 而是“落在高波动区域”这个区域先验过强

所以第三轮把规则改回“轨迹自身几何是否可靠”。

### 12.1 规则调整

实现位置仍然是：

- `utils/traj_filter_utils.py`
- `scripts/batch_inference/infer.py`

正式过滤规则改为：

- `final_mask = base_mask & query_depth_mask & temporal_mask`

其中：

- `base_mask`：保持原样
- `query_depth_mask`：保持原样
- `temporal_mask`：
  - 先计算 `all-frame consistency`
  - 再计算排除高波动像素后的 `stable-frame consistency`
  - 若 `stable_compare_count >= required_compare_frames`
    - 优先用 `stable_consistency_ratio >= 0.95`
  - 否则
    - 回退到 `all_consistency_ratio >= 0.95`

关键变化：

- `high volatility` 不再单独删除轨迹
- 它只参与“哪些比较帧更可信”的判定

### 12.2 新增调试字段

第三轮在 sample NPZ 中继续保留原有调试字段，并新增：

- `traj_stable_depth_consistency_ratio`
- `traj_volatility_exposure_ratio`
- `traj_compare_frame_count`
- `traj_stable_compare_frame_count`

`traj_mask_reason_bits` 语义也更新为：

- `bit0`：base geometric fail
- `bit1`：query-depth fail
- `bit2`：temporal consistency fail
- `bit3`：stable-frame temporal fail

注意：

- 第三轮里不再存在“仅因为命中高波动区域就删除”的 reason bit

## 13. 第三轮验证结果

### 13.1 代码级验证

执行：

```bash
python -m unittest utils.test_traceforge_artifact_utils utils.test_traj_filter_utils
python -m py_compile \
  utils/traj_filter_utils.py \
  scripts/batch_inference/infer.py \
  utils/traceforge_artifact_utils.py
```

结果：

- `unittest`: `Ran 16 tests ... OK`
- `py_compile`: 通过

新增覆盖点包括：

- 高波动命中但几何一致的轨迹仍然保留
- stable frames 足够时，只看 stable-frame consistency
- stable frames 不足时，正确回退到 all-frame consistency
- stable-frame fail 会额外打 `bit3`
- `visibility=False` 的帧仍不会参与时序比较

### 13.2 离线重算验证

在旧 sample 上直接用第三轮新规则离线重算，得到：

- `episode_00119_pink / varied_camera_1 / frame15`
  - 旧 kept：`6063`
  - 第三轮新规则：`5924`
  - 相比第二轮 `5789`，回收了大量被过删轨迹

- `episode_00000_blue / varied_camera_1 / frame15`
  - 旧 kept：`6252`
  - 第三轮新规则：`6142`
  - 相比第二轮也明显回升

说明第三轮不是只“救当前 case”，而是同样降低了另一个 case 的过删程度。

### 13.3 同源 v3 真实重跑验证

已对同源数据重新推理：

- 输入源：`/data1/yaoxuran/press_one_button_demo_v3/episode_00119_pink`
- 输出目录：`data_tmp/ray_artifact_investigation/2026-03-16/revalidated_geometry_guided_fix_v3/inference/episode_00119_pink/varied_camera_1`

对 `query_frame = 15` 的 sample 做对比，得到：

- `traj_uvz`：完全一致
- `visibility`：完全一致
- 最旧版 kept：`6063`
- 第二轮 kept：`5789`
- 第三轮 kept：`5935`

和第二轮相比：

- 第三轮回收了 `146` 条此前被误删的轨迹
- 没有新增额外误删

和最旧版相比：

- 第三轮仍额外过滤了 `128` 条轨迹

即：

- 它保留了“比最旧版更严格”的过滤效果
- 但不再像第二轮那样把高波动区域整片打掉

第三轮新 sample 的原因分布：

- `reason_hist = [(0, 5935), (1, 230), (2, 3), (4, 13), (5, 55), (7, 7), (12, 115), (13, 40), (14, 2)]`
- `high_volatility_hit = 196`
- `stable_temporal_fail(bit3) = 157`

第三轮剩余被过滤轨迹的空间分布：

- `removed_top20 = 52.5%`
- `removed_center33 = 47.1%`

相比第二轮：

- 过滤仍主要发生在难区域
- 但不再出现“高波动区硬 veto 导致的极端集中误删”

### 13.4 当前结论更新

到第三轮为止，更合理的结论是：

- 泛化过滤规则应该以轨迹自身的几何一致性为主
- raw depth volatility 适合作为“降权/辅助证据”，不适合作为硬删除规则
- 对同源 `episode_00119_pink`，第三轮已经明显缓解第二轮的机械臂过删
- 剩余问题主要是：
  - 仍有一部分稳定难区域轨迹会被时序一致性规则过滤
  - 还需要做最终的人工 3D 观察确认这部分过滤是否合理

## 14. Remaining Tooling Gap

当前仓库里，静态可视化与 checker 已经改成 wrist-aware：

- `visualize_single_image.py`
- `verify_episode_trajectory_outputs.py`
- `checker/batch_process_result_checker.py`
- `checker/batch_process_result_checker_3d.py`

它们会先应用 `traj_valid_mask`，再按逐帧 `traj_supervision_mask` 隐掉 wrist 轨迹尾段，因此和 `visualize_3d_keypoint_animation.py` 的动态显示语义一致。

但以下分析脚本本轮仍未修改：

- `scripts/data_analysis/analyze_depth_volatility.py`
- `scripts/visualization/compare_traj_filter_results.py`
- `scripts/data_analysis/analyze_ray_artifact.py`

当前影响：

- 这些脚本仍把 wrist 相机中 `traj_valid_mask=True` 的轨迹当作“整条保留轨迹”来统计或诊断
- 它们不会按 `traj_supervision_mask` 屏蔽已被 wrist 规则裁掉的尾段
- 因此，分析统计结果可能比实际可视化中看到的 wrist 可监督时段更乐观，或在几何诊断里重新把尾段算进去

结论：

- 当前“看图验证”与“静态/动态可视化一致性”已经成立
- 但如果后续要让分析结论也完全跟 wrist 逐帧语义一致，还需要单独把上述 3 个脚本也切到 wrist-aware 读取口径
