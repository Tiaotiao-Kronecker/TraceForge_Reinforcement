# xperience stereo_left 问题结论汇总（2026-04-10）

## 目的

本文档把本轮围绕以下三个 xperience stereo motion-window case 的结论集中整理为逐条结论，避免分析结果分散在多轮对话和多份调查记录中：

- `stereo_left_start_00190_officialprep`
- `stereo_left_start_00435_officialprep`
- `stereo_left_start_04234_officialprep`

相关细节调查可参考：

- `xperience_stereo_left_false_traj_plan_2026-04-10.md`
- `xperience_external_geometry_wobble_upstream_analysis_2026-04-10.md`
- `xperience_external_geometry_control_experiments_2026-04-10.md`
- `xperience_inference_query_depth_stabilization_and_tracker_interaction_2026-04-10.md`

## 结论 1：当前底层 tracker 追踪的是 3D 点，不是纯 2D 点

当前实现中，query 不是直接以 `(t, u, v)` 送入模型，而是先结合 query frame 的 depth 和外参 lift 成 `(t, X, Y, Z)`。

因此：

- query seed 的深度和外参质量会直接影响 tracker 的初始 3D 点
- 这不是“先做 2D 跟踪，最后再补 3D”

## 结论 2：query 的 3D 初始化与每帧 dense point cloud 不是重复计算

系统中存在两类不同的 3D 几何输入：

1. sparse query point
   这是“要跟踪的目标点”。

2. per-frame dense point cloud
   这是“每一帧的 3D 场景上下文”。

因此模型里既要：

- 用 query frame depth 初始化 query 的 3D 位置
- 也要用所有帧的 depth + extrinsics 构造 dense geometry，作为后续匹配和相关特征的 3D 环境

结论：

- depth / extrinsics 不只会影响 query 初始化
- 也会影响模型内部使用的整帧 3D 场景表示

## 结论 3：错误轨迹与 depth / extrinsics 质量有直接关系

4D 轨迹的形成依赖：

- query frame depth
- 每帧 depth
- 每帧 intrinsics / extrinsics
- tracker 本身的匹配与更新

因此只要以下任一部分出错，都会在固定视角 4D 可视化中表现为错误轨迹：

- query seed depth 错
- per-frame depth 时序不稳
- extrinsics 时序不稳
- tracker 在低纹理/重复纹理区域 slip

结论：

- 看到“错误轨迹”不能直接归因给 tracker
- 必须先区分 geometry 问题和 tracker 问题

## 结论 4：可以做“geometry 主导”与“tracker 主导”的归因，但这是主导性判断，不是绝对判决

当前可采用的归因原则如下：

- 如果不跑 tracker，只用 `source depth + source extrinsics` 做 fixed-view 自一致检查，静态背景已经明显在晃，那么这部分问题一定不是 tracker 凭空制造的，而是 geometry 问题。
- 如果 geometry-only 检查基本稳定，但 tracker 输出仍明显漂移，那么更像 tracker 本身的问题，或 tracker 与局部几何噪声的交互问题。
- 如果漂移表现为整片背景共同平移，更像 extrinsics 类问题。
- 如果漂移主要集中在局部区域、深度边缘、遮挡边缘，`p95` 很高但 `global drift` 很低，更像 depth 类问题。

结论：

- 目前已经能够对这三类 case 做“哪一类原因更主导”的判断
- 并且后续控制变量实验已经完成，当前证据整体更偏向 `depth 主导`

## 结论 5：`00190` 的主问题是 bad seed / 边缘深度问题

现象：

- 人右前方、右后方和前方边缘出现明显错误轨迹
- 很多轨迹从图像边缘附近发散

定量结果：

- 最离谱的轨迹多数来自 `query_depth≈0`
- 多数 seed 位于图像边缘附近
- 仅去掉 `query_depth<=0.05m` 且 `border<=40 px` 的 seed 后，固定视角 p95 大幅下降

进一步的上游检查结果：

- fixed-view source geometry 的 `global drift` 很小
- 但 `p95` 很高

结论：

- `00190` 不是典型的整场景 wobble
- 它更像局部坏深度、边缘深度跳变、遮挡边界等问题导致的 bad seed
- 这类问题在 query lift 成 3D 之后，会被放大成固定视角飞线

## 结论 6：`00435` 的问题不只是 bad seed，而是明显的 scene-level wobble

现象：

- 几乎所有 keypoint 都明显抖动
- 即使不出现非常夸张的飞线，也会有大量不必要的背景轨迹

下游定量结果：

- 去掉零深度边缘 seed 后，深且居中的背景点仍然明显抖动
- deep-central background 在 fixed-view 下仍存在明显共同漂移

scene wobble 诊断结果：

- 在 `A2 = standard + external + external_depth_static_v1` 下，`00435` 仍被稳定打成 `geometry_unstable`

结论：

- `00435` 不能归因为“只是 seed 入口坏了”
- 它具有明显的场景级共同 wobble

## 结论 7：`04234` 的主问题更像局部背景不稳定，而不是上游大范围 geometry 自晃

现象：

- 人和桌面附近较稳定
- 其余背景区域有较多抖动和错误轨迹

下游定量结果：

- 去掉零深度边缘 seed 后，fixed-view 指标明显改善
- 没有 `00435` 那种强烈的全场共同漂移

上游结果：

- source geometry 的 fixed-view 自晃非常小

结论：

- `04234` 更像局部背景跟踪问题
- 当前更接近 tracker / 纹理歧义 / 局部深度噪声交互问题
- 不是典型的上游 geometry 大范围 wobble

## 结论 8：`A1 = external_depth_static_v1` 对 bad seed 问题有效，但不足以解决 `00435`

`A1` 的作用是：

- 在送入 tracker 前，基于 query frame 静态深度质量做 seed 预筛选

结果：

- 对 `00190` / `04234` 这类 bad seed 问题有明显帮助
- 去掉了大量 `query_depth≈0` 且位于边缘的 seed

但：

- 用户在 `00435` 上仍然看到整体抖动

结论：

- `A1` 对 bad seed 有效
- 但 `00435` 的主问题不在 seed 入口

## 结论 9：`A2 = A1 + standard/external` 也没有根治 `00435`

`A2` 在 `00190` / `04234` 上进一步改善了结果。

但对 `00435`：

- 共同漂移仍明显存在
- scene wobble 诊断仍为阳性

结论：

- `A2` 不能把 `00435` 变成一个“正常 sample”
- 因此问题必须继续上移到 source geometry 排查

## 结论 10：当前 `traj_stereo_consistency_mask` 在这三个 case 上没有实际新增过滤作用

检查结果表明，在当前实现和当前 case 上：

- `traj_valid_mask & traj_stereo_consistency_mask == traj_valid_mask`

结论：

- 直接把现有 stereo consistency 接进 external profile，不会带来实质变化
- `A3` 不是当前最优先实验方向

## 结论 11：`00435` 的 source geometry 本身已经明显不稳定，因此它不是纯 tracker 幻觉

基于 `source depth + source extrinsics` 做 fixed-view 自一致检查，不依赖 tracker：

- `00435` 的 fixed-view 中位漂移和重尾漂移都明显高于 `04234`
- 也高于 `00190` 的典型漂移

结论：

- `00435` 的 scene-level wobble 在 source geometry 层面已经存在
- tracker 只是继承并放大了这个问题
- 不能把它简单说成“tracker 自己坏了”

## 结论 12：当前证据不支持“`00435` 的主因是 extrinsics 高频时序抖动”

对 `geom/geom_stereo_left_official_w2c.npz` 的外参时序检查表明：

- `00435` 的 camera path 更长，step translation 略大
- 但 rotation jerk 并没有显著异常
- 甚至 `04234` 的 rotation jerk 指标更高，但它的 source geometry 反而最稳定

结论：

- 目前没有足够证据把 `00435` 的主因归为 extrinsics 高频抖动
- 当前更合理的主嫌疑是 external depth 的时序/区域性不稳定

## 结论 13：`00190` 与 `00435` 的 geometry 问题类型不同

`00190`：

- `global drift` 小
- `median drift` 中等
- `p95` 很高

这更像少数局部坏区域、边缘区域、遮挡边界导致的重尾错误。

`00435`：

- `global drift` 略增
- `median drift` 明显升高
- `p95` 也很高

这更像大面积背景都存在 source geometry 不稳定。

结论：

- 两者不能用同一种 trim / filter 思路处理
- `00190` 更适合 bad-seed / local-risk 处理
- `00435` 更需要上游 geometry 稳定性处理

## 结论 14：如果继续推进，最合理的主方向是 depth-focused，而不是继续裁 grid 或先做 extrinsics smoothing

基于当前证据，优先级应当是：

1. depth-focused 上游排查与修复
   尤其是 external depth 的时序稳定性和区域性质量。

2. sample-level geometry-unstable 标记
   把 `00435` 这类 sample 显式标记出来。

3. query-depth temporal stabilization
   不再只用单帧 query depth。

而不应优先继续：

- ring trim
- 更大范围边缘裁减
- 直接把当前 stereo consistency 硬接进主链
- 在证据不足时优先做 extrinsics smoothing

## 结论 15：控制变量实验进一步支持“depth 比 extrinsics 更像主导项”

geometry-only 控制变量实验比较了四种变体：

- `baseline`
- `extrinsics_smooth_r1`
- `extrinsics_freeze_query`
- `depth_temporal_median_world_v1`

其中真正有物理意义、可作为主判断依据的是：

- `extrinsics_smooth_r1`
- `depth_temporal_median_world_v1`

结果上：

- `depth_temporal_median_world_v1` 在三个 case 上都更 consistently 改善中位漂移
- `extrinsics_smooth_r1` 则是 mixed，有时略好，有时无明显优势，有时反而更差

结论：

- 当前进一步支持“depth 是更可靠的主修复杠杆”
- 不支持把 extrinsics smoothing 作为当前第一优先级

## 结论 16：`extrinsics_freeze_query` 只能作为极端 ablation，不能当成部署方案

在三个 case 上，`extrinsics_freeze_query` 都几乎把 geometry-only drift 压到接近零。

但这个结果只能说明：

- 诊断指标对时序几何变化是敏感的
- 如果强行取消真实相机运动，fixed-view 自晃当然会塌掉

不能说明：

- 当前问题主要来自真实的 extrinsics 高频 wobble
- 真实系统里应该靠“冻结外参”来修

结论：

- 这个变体只能用于验证诊断灵敏度
- 不能作为主结论证据，更不能作为部署方向

## 结论 17：`00435` 现在更明确地指向 external depth 时序/区域性不稳定

控制变量实验里：

- `00435 q0` 上，`extrinsics_smooth_r1` 把 `median` 从 `1.050` 做坏到 `1.221`
- 同一处 `depth_temporal_median_world_v1` 则把 `median` 降到 `0.847`

`q4` 上两边都略有改善，但：

- depth 方向改善幅度接近 extrinsics
- extrinsics 方向没有形成稳定单调收益

结论：

- `00435` 继续向 `external depth` 主导收敛
- 但当前 depth stabilization 仍未解决它的 heavy-tail outlier

## 结论 18：`00190` 与 `04234` 的后续方向也因此进一步分化

`00190`：

- depth stabilization 主要改善中位漂移
- 对 p95 重尾几乎无帮助

说明：

- 这里更像局部 bad-depth / 边缘 outlier 问题
- 后面要继续走 local-risk / bad-seed 方向

`04234`：

- baseline source geometry 已经较稳
- depth stabilization 还能继续变好
- extrinsics smoothing 在 `q4` 甚至显著恶化

说明：

- 这里不适合优先怀疑 extrinsics
- 更应继续拆 tracker 本身和局部背景交互问题

## 结论 19：当前最核心的未决问题

虽然现在已经能判断：

- `00435` 不是纯 tracker 问题
- 更像 external depth 主导

但仍未完成的关键判定是：

- 把 geometry-only 的 depth stabilization 前移到真实 inference 入口后，downstream 4D 轨迹是否也会同步改善
- `00435` 里 external depth 的不稳定，到底更多来自 query seed 初始化，还是每帧 dense geometry 上下文本身

结论：

- 下一步最值得做的是 inference-side depth stabilization 实验，而不是继续优先尝试 extrinsics smoothing
- 这一步会直接决定后续是走 sample reject，还是走 depth stabilization 修复

## 结论 20：inference-side query depth stabilization 已经证明“query seed 初始化”确实会影响 `00435` 的共同 wobble

对 `00435` 的真实 inference rerun 表明：

- `q0` 的 `global_final_disp_px` 从 `6.016` 降到 `2.400`
- `geometry_unstable` 从 `true` 变成 `false`

结论：

- query seed 3D 初始化不是旁枝问题
- 它确实是 `00435` scene-level wobble 的组成部分

## 结论 21：但只修 query seed 初始化，还不足以解决 `00435` 的 heavy-tail 错轨

同一轮 inference rerun 里，`00435 q0` 同时表现出：

- 共同 wobble 明显下降
- 但 `track_final_p95_px` 从 `39.977` 升到 `63.751`

结论：

- 只做 query seed stabilization，会改善 common drift
- 但不会自动解决局部背景 outlier / dense geometry / tracker-local 问题

## 结论 22：`04234` 现在已经有更直接的证据支持“tracker / 局部背景交互主导”

新的 tracker-vs-geometry 诊断显示：

- `q0`: `tracker_local_interaction_count = 1075`，`geometry_limited_count = 83`
- `q4`: `tracker_local_interaction_count = 1425`，`geometry_limited_count = 143`

同时：

- geometry-only final drift median 只有 `0.055 px / 0.095 px`
- tracker final drift median 已到 `2.641 px / 1.847 px`

结论：

- `04234` 的主要坏轨迹不是 source geometry 自动带出来的
- 更像 tracker 在局部背景区域发生了大规模滑移

## 结论 23：`00435` 的 dense geometry 侧 depth stabilization 方向是对的，但“全帧大面积替换”太激进

把 query seed stabilization 继续前移到 dense `depth_obs` 之后，`00435` 上出现了两个同时成立的事实：

- `q0/q4` 的 `track_final_p95_px` 都下降了
- 但 `geometry_unstable` 的 query frame 数量又从 `0 / 8` 回升到了 `4 / 8`

同时保存下来的 dense-depth 元数据表明：

- 每帧大约有 `83% ~ 98%` 的有效像素被替换
- 部分帧的 `delta_depth p95` 能到 `0.20 m`

结论：

- dense geometry 侧 depth stabilization 确实能压 heavy-tail
- 说明 dense point cloud 输入本身确实在贡献坏轨迹
- 但当前这版 `temporal_median_reproject_v1` 对整帧 depth 改得太多
- 它会把重投影误差/外参误差一起写回 depth，所以不能直接当默认方案

## 结论 24：`00435` 的下一步不是放弃 dense stabilization，而是把它收缩到“高可信静态背景”

基于当前 mixed result，更合理的下一步不是：

- 继续扩大 temporal median 的覆盖范围

而是：

- 只在几何一致性高的静态背景区域做 dense stabilization
- 减少对前景、人、桌面交互区和高遮挡区的全帧重写

结论：

- `00435` 仍然更像 `depth 主导`
- 只是 dense 修复要从“全帧版”收缩到“受约束版”
