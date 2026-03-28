# 代码接口面与兼容层收敛计划

日期：2026-03-28

本文档专门记录“代码的同义简化”以及“旧接口 / 旧模式兼容删减”的问题。它不讨论某一条过滤规则是否正确，而只讨论：

1. 当前维护态真正需要的接口面是什么
2. 哪些兼容分支已经主要在服务历史产物或实验分析
3. 下一步如何在不打断当前主流程的前提下收缩代码面

## 背景

最接近这类主题的旧记录，是：

- `docs/history/press_one_button_demo_v4_parameter_audit_2026-03-19.md`

那份文档已经指出：

- 有些 batch CLI 参数是死参数或旧兼容参数
- 当前正式 external-only 流水实际上已经依赖更窄的一套参数面

但那份记录主要针对 `press_one_button_demo_v4` 的批处理命令，不足以覆盖整个仓库当前的兼容层问题。因此这里补一份更通用的收敛计划。

## 当前维护态已经收敛到什么

从 `CLAUDE.md`、`docs/traceforge_output_structure.md`、`scripts/batch_inference/BATCH_INFERENCE_GUIDE.md` 和当前代码来看，真正的维护态主路径已经很明确：

- `depth_pose_method=external`
- `output_layout=v2`
- `scene_storage_mode=source_ref`
- wrist-like 相机的默认 `traj_filter_profile=auto -> wrist_manipulator_top95`

也就是说，今天仓库里仍保留的大量分支，并不都属于“维护态主流程必需”，其中一部分主要是在服务：

- legacy 输出
- 历史数据产物读取
- 分析 / ablation
- 旧批处理入口

## 当前最值得关注的兼容层

## 1. `output_layout=legacy`

这是当前最显性的兼容层之一。

它同时影响：

- `scripts/batch_inference/infer.py`
- `utils/traceforge_artifact_utils.py`
- `scripts/visualization/visualize_3d_keypoint_animation.py`
- `docs/traceforge_output_structure.md`

当前问题不是“legacy 支持有没有用”，而是：

- 维护态默认已经是 `v2 + source_ref`
- 但主代码和主可视化仍然需要持续分叉处理 `legacy`

这会带来三个成本：

1. parser 暴露面更宽
2. 可视化和 artifact reader 代码分支更多
3. 新功能默认需要同时考虑 `v2` 与 `legacy`

建议：

- 保留 legacy 读取能力
- 但逐步把 legacy 写出能力降级为“历史兼容专用入口”，而不是主推理入口的默认分支

## 2. `scene_storage_mode=cache`

当前维护态默认已经是 `source_ref`。`cache` 模式仍然有历史价值，但它和 `legacy` 一样，会放大：

- `infer.py` 的 save 分支
- `SceneReader` 的读取分支
- 文档说明面

建议：

- 继续保留 `cache` 读取兼容
- 但把“主流程默认支持两种 scene storage backend”改成“主流程默认只有 `source_ref`，`cache` 视为兼容模式”

## 3. save-time analysis / ablation 参数

当前主链上还暴露着若干分析型接口，例如：

- `traj_filter_ablation_mode`
- `query_prefilter_mode`

它们并不是错误，但它们的角色需要更清楚：

- 它们是分析 / benchmark 入口
- 不是维护态默认接口面的一部分

建议：

- 保留功能
- 但逐步从主 CLI surface 弱化，优先转移到 benchmark / analysis 专用脚本

## 4. 旧采样 / 旧调度语义的兼容参数

最典型的例子是：

- `frame_drop_rate`
- `max_frames_per_video`

这些参数在当前主路径里要么作用已经很弱，要么只在 fallback 逻辑里生效。它们会继续放大 parser 面和文档维护成本。

建议：

- 先在文档上明确“维护态主路径不依赖这些参数”
- 再逐步把它们迁到旧入口或 compatibility layer

## 5. 可视化层的 legacy 特判

`visualize_3d_keypoint_animation.py` 当前仍需处理：

- legacy 主 NPZ dense fallback
- nonzero query frame 的 static dense fallback

这些分支本身是合理的兼容逻辑，但它们会让主 viewer 同时承担：

- 当前维护态可视化
- 历史 legacy 产物可视化

建议：

- 把“维护态 viewer 行为”和“legacy fallback 行为”在代码结构上进一步隔离
- 即使不删功能，也尽量不要让两者长期混在同一主路径里

## 收敛原则

这类收敛不应一刀切删除旧功能，而应遵守下面的顺序：

1. 先把“维护态默认路径”在文档中写死
2. 再把分析型接口和兼容型接口从主入口语义上降级
3. 最后才考虑删除写出路径或 parser 暴露面

也就是说，优先级是：

- 先弱化默认暴露面
- 再隔离兼容代码
- 最后才删除真正没人再用的分支

## 建议的下一步实施顺序

## P0：先补一份“维护态接口面”文档

需要一份更明确的当前文档，直接回答：

- 维护态正式支持什么
- 哪些只是兼容模式
- 哪些只是 analysis mode

这样做的价值是，后续删接口时不会再混淆“主流程需求”和“历史兼容需求”。

## P1：收窄主 CLI surface

优先考虑下列项目：

- 在主 README / guide 中弱化 `legacy` 与 `cache`
- 把 `traj_filter_ablation_mode`、`query_prefilter_mode` 明确标成 analysis-only
- 对 `frame_drop_rate`、`max_frames_per_video` 一类参数重新标注其真实适用范围

这里先做“文档与 help 收敛”，不急着删代码。

## P2：隔离 compatibility path

重点看：

- `infer.py` 的 legacy / v2 写出分支
- `traceforge_artifact_utils.py` 的 layout / storage 分支
- `visualize_3d_keypoint_animation.py` 的 legacy fallback 分支

目标不是马上删掉，而是：

- 把维护态主路径抽得更直
- 让 compatibility path 尽量局部化

## P3：再决定是否删除旧写出路径

只有在下面两个前提都成立时，才应该真的删：

1. 当前生产与分析已经不再依赖旧写出格式
2. 历史产物仍可通过 reader / converter 使用

在那之前，优先删的是“主入口默认暴露面”，不是“所有旧功能本体”。

## 当前结论

当前仓库里，确实还缺一份通用的“同义简化 / 兼容层删减”收敛文档。最接近的旧记录只有 `press_one_button_demo_v4_parameter_audit_2026-03-19.md`，但它覆盖面不够。

因此，后续如果要推进代码面收缩，应该同时做两条线：

1. 性能优化线：继续压 tracker forward、`manipulator_motion` 和 `prepare_depth_filter`
2. 接口收敛线：逐步把 `legacy`、`cache`、analysis-only 参数和旧调度兼容从主入口语义上降级

两条线是并行关系，不应互相替代。
