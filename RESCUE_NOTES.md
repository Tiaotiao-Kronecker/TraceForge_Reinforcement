# RESCUE NOTES: TraceForge_Reinforcement（来自 djr-h200-1）
> 抢救时间 2026-11-22，原路径 /home/djr/TraceForge_Reinforcement，main 分支 HEAD=f44d952，仅 1 个未提交文件（CLAUDE.md）

## 说明
Tiaotiao-Kronecker/TraceForge_Reinforcement——跨本体视频 → 一致 3D 轨迹的数据集流水线（相机运动补偿 + 速度重定向），是 WorldModel_3d 的**上游轨迹数据源**：输出 (u,v,Z_c) 像素轨迹 + 相机内外参，经 `uvz_to_world()` 反投影成世界坐标轨迹，供 DataLoader-V2 按 3 相机 x 6400 点 = 19200 点采样做 flow matching 监督。近期 commit：添加深度流水说明、重构脚本、grid_size 参数支持与可视化改进。

唯一未提交文件：CLAUDE.md（/init 生成的仓库结构导览，供 AI 辅助编码用）。

## 进度
2026-03 下旬（Claude Code 会话，项目目录 /DATA/disk4/djr/TraceForge-Reinforcement）：
- 15 个子代理完成代码考古：TraceForge 保存相机参数的方式 ↔ DataLoader-V2 extrinsics_window 的映射、frame_index 与 tracking 段的对齐、coordinate transform pipeline（3D 世界坐标 → 相对 frame0 的 2D 投影）逐环节核对。
- 梳理 WorldModel_3d 训练/推理全维度流：traj (N=6400, T=10/32, 3) → 世界坐标 offset 归一化 [-1,1] → OT-CFM 加噪（x_t=(1-t)clean+t·noise，预测 velocity）→ 推理 50 步 ODE 积分。
- **发现并修复 tanh 归一化 bug**：旧流程 `(u,v,Z_c)→offset→tanh(offset*5)` 在 Z_c=0/nan 时产生 inf，tanh(inf)=1.0 可通过 isfinite 过滤，导致约 15.2% 样本带错误标签（33x loss 放大）进训练；结论为删除 tanh、直接用世界坐标 offset + 统一缩放。

## 结论
- 数据链路对齐核查完成（相机参数/帧对齐/坐标变换三处均已确认一致），tanh 标签污染 bug 已定位并有修复方案；该修复是 WorldModel_3d 离线精度达到 0.77cm 终点误差的前置条件之一。
- 本仓库本身无实验数字；后续工作重心在 WorldModel_3d 侧（见其 RESCUE NOTES）。
