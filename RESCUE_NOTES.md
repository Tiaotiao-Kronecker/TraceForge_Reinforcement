# RESCUE NOTES: TraceForge_Reinforcement(来自 djr-h200-2)
> 抢救时间 2026-11-22,原路径 /home/djr/TraceForge_Reinforcement(= /DATA/disk4/djr/TraceForge_Reinforcement),本分支为退租前工作区快照(含未提交改动+未跟踪小文件)

## 说明

- remote: `git@github.com:Tiaotiao-Kronecker/TraceForge_Reinforcement.git`,分支 main,HEAD `642cff5`("Add multi-GPU batching for xperience_raw"),**与 origin/main 同步**(main 已推齐);4 个未提交文件全部是文档:修改 `docs/README.md`(索引加两行)+ 3 个新文档(`docs/data_preprocessing_generation_flow.md`、`docs/traceforge_changes_since_initial.md`、`CLAUDE.local-before-merge-20260513-1454.md`)。注意:3 个未跟踪文档的内容**未进入本快照 tarball**,只能按会话记录重建。
- 本仓库是 TraceForge(把异构机器人/人操作视频转成统一 3D 轨迹监督信号的数据引擎)的维护态 fork:核心链路 = 外部 RGB + 深度 + 相机几何(不再用模型估计深度/位姿)→ TAPIP3D 3D 点追踪 → 质量过滤 → `v2 + source_ref` 标准化输出(`traj_uvz` 等)。与 WorldModel_T 项目配套,为闭环评测/训练提供 3D 轨迹数据(数据源如 `/DATA/disk6/gs_datasets/arrange_flowers_gs`,读取格式正是 WorldModel_3d 的 `compat_packed_v2`)。

## 进度

- 提交历史(已推送)记录了本 fork 相对初版的完整演进:`feat(xperience)` 原生 adapter 推理路径 → batch_inference 统一数据适配器与仿真 runner → smoke 运行与 viewer 验证记录 → 文档分支引用修正 → xperience_raw 多卡批处理。
- **5/13–5/14**(本机最后工作,即 4 个未提交文件的来源):两天文档整理——① 拉取远程最新后,把"数据预处理与 3D 轨迹生成全流程"(输入、原理、模型、步骤、输出,不出现程序变量名)整理成 `data_preprocessing_generation_flow.md`;② 用 git 历史对比初版(最早完整代码 `73426b4`)与当前 `642cff5`,写成 `changes_since_initial.md`,统计为 **226 files changed, +56835/−2707**;③ 据此更新 `docs/README.md` 索引。
- 同期排查了 `arrange_flowers_gs` 数据集的读取加速问题(该数据即以 compat_packed_v2 二进制格式存盘)。

## 结论

- 初版→现在的定性结论(会话原话归纳):从"单脚本、单数据布局的研究 demo"改成了"多数据源、大规模批处理、外部几何、质量过滤、标准化输出、可验证产物"的维护态数据引擎。12 条主线改动:入口重构(dispatcher+adapter)、episode 级共享查询帧计划(每秒 2–3 帧)、输出协议 legacy→v2(`traj_uvz` + valid/supervision mask,不再重复存 RGB/depth)、去掉弧长重采样改按真实帧序+inf padding、`external` 轨迹过滤 profile(几何/深度质量/时序一致性,输出过滤原因 bitmask)、查询网格 20×20→默认 80 可配、动态多 GPU 调度(显存门控、CUDA 失败重排队、telemetry)、可视化验证体系(3D 动画/PLY/对比)、性能优化(depth/ray cache、修 pointops2 硬编码 cuda:0)、仓库分层(旧路径归档)。
- 这 4 个未提交文档改动是低风险纯文档,可直接补提交推送;数据管线本身已可用,_smoke 与 viewer 验证已记录在已提交文档中_。机器上没有发现未完成代码改动。
