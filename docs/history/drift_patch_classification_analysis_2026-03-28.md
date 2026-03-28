# drift patch 分类问题简要结论

日期：2026-03-28

这份记录只保留最终结论，不再保留中间启发式分类过程。当前代码库已移除对应的实验性 drift 诊断脚本；如果后续重新分析这类问题，应直接从这里的结论继续。

## 问题背景

围绕 `wrist_pick_place_no_heatmap` 的错误轨迹分析，早期曾尝试把错误轨迹显式分类成：

- `early_border_escape`
- `local_identity_switch`
- `tabletop carry-along`

并基于 patch / tabletop 特化规则做 2D 与 3D 可视化。

这条分析线帮助确认了一些现象，但后续证明它不适合作为维护态方案。

## 最终结论

### 1. 错误来源要先和后处理过滤逻辑分开分析

分析这类错误时，必须先看 raw tracking 输出本身，再看过滤为什么没有把它压掉。否则很容易把：

- tracker 本身的 correspondence drift
- 和 filter 误保 / 误删

混成同一个问题。

### 2. “桌面 patch 特化”不是第一性原理定义

早期规则把很多现象描述成 `tabletop carry-along`，并围绕：

- 主桌面高度
- 法向接近 world-z
- 2D patch 生长

来做判定。

这条路线的问题是，它回答的是“像不像主桌面上的错误 patch”，而不是“是不是静态表面被错误拖着走”。因此它天然会漏掉：

- 非主桌面高度的静态表面
- 被遮挡或点数很少的小 patch
- 其他不满足桌面先验、但本质相同的 coherent wrong motion

### 3. 更合理的抽象应是静态表面上的 coherent wrong motion

如果后续重做这条分析线，更合理的主问题应定义为：

- `query-time static surface membership`
- `later coherent wrong motion`

也就是：

1. 先在 query 时刻识别“这些点属于同一个静态表面或静态 surface component”
2. 再看它们后续是否整体出现了不应发生的一致错误运动

这样才能统一覆盖：

- 主桌面
- 立方体或其他静态面
- 稀疏、小型、部分遮挡的 patch

### 4. 这类错误不能只靠单点终点位移来判定

无论是边缘 drift 还是静态表面 carry-along，只看最终位移都不够。更可靠的是同时看：

- query-time 的几何归属
- prefix 时段的局部一致性
- 邻域或 surface component 的协同运动

## 当前决策

这条 drift patch 分类实验线目前不再继续保留为仓库内工具链，原因是：

- 规则过度依赖场景特化启发式
- 中间分类名容易被误当成最终正确抽象
- 它更适合当一次性调查，不适合当维护态接口

如果后续恢复这项工作，应直接按“静态表面 membership + coherent wrong motion”的框架重做，而不是恢复旧的 tabletop / patch 特化规则。
