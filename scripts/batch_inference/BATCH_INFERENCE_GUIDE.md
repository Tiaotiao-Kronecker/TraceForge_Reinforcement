# 批量推理完整指南

本文档整合了批量推理相关的所有信息，包括问题分析、解决方案和使用指南。

## 目录

1. [快速开始](#快速开始)
2. [问题分析与解决](#问题分析与解决)
3. [压力测试](#压力测试)
4. [常见问题](#常见问题)

---

## 快速开始

### 基本使用

**重要**: 需要在 traceforge conda 环境中运行。

```bash
# 使用screen（推荐，避免网络断开）
screen -S batch_inference

# 激活conda环境
conda activate traceforge

# 运行完整推理（从scripts/batch_inference目录运行，或在项目根目录使用相对路径）
cd scripts/batch_inference
python batch_infer.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./output_bridge_depth_grid80 \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --gpu_id 0,1,2,3,4,5 \
    --max_workers 6 \
    --grid_size 80

# Detach: Ctrl+A 然后 D
# 重新连接: screen -r batch_inference
```

### 方法2: 直接使用conda环境的Python解释器

```bash
screen -S batch_inference

# 直接使用conda环境的Python解释器（从scripts/batch_inference目录运行）
cd scripts/batch_inference
/home/wangchen/.conda/envs/traceforge/bin/python batch_infer.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./output_bridge_depth_grid80 \
    --use_all_trajectories \
    --skip_existing \
    --frame_drop_rate 5 \
    --gpu_id 0,1,2,3,4,5 \
    --max_workers 6 \
    --grid_size 80
```

### 后台运行（可选）

```bash
# 方法1: 激活环境后后台运行
nohup bash -c "conda activate traceforge && python batch_infer.py ..." > batch_inference.log 2>&1 &

# 方法2: 直接使用conda环境的Python解释器
nohup /home/wangchen/.conda/envs/traceforge/bin/python batch_infer.py ... > batch_inference.log 2>&1 &

# 查看日志
tail -f batch_inference.log
```

---

## 问题分析与解决

### 问题1: SSL错误（模型下载失败）

#### 问题描述

```
requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): 
Max retries exceeded with url: /Yuxihenry/SpatialTrackerV2_Front/resolve/main/config.json 
(Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)')))")
```

#### 根本原因

1. **每个子进程都在尝试下载模型**
   - 多个子进程同时调用 `VGGT4Track.from_pretrained()` 时，都会尝试从HuggingFace下载
   - 大量并发SSL连接导致失败

2. **没有重试机制**
   - 模型加载失败后直接退出
   - 没有错误恢复机制

#### 解决方案

**方案1: 模型预加载（已实施）** ✅

在主进程中预先下载模型到本地缓存：

```python
def preload_models():
    """预先加载模型到缓存"""
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    del model  # 释放内存
```

**方案2: 添加重试机制（已实施）** ✅

在模型加载时添加重试逻辑，使用指数退避：

```python
def load_model(self, checkpoint_path="Yuxihenry/SpatialTrackerV2_Front"):
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            model = VGGT4Track.from_pretrained(checkpoint_path)
            return model
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise
```

**方案3: 强制使用本地缓存（已实施）** ✅

优先从本地缓存加载，避免网络请求：

```python
model = VGGT4Track.from_pretrained(
    checkpoint_path,
    local_files_only=True  # 强制使用本地缓存
)
```

#### 验证缓存状态

```bash
# 检查模型是否已缓存
ls -lh ~/.cache/huggingface/hub/models--Yuxihenry--SpatialTrackerV2_Front/

# 如果目录存在且有文件，说明模型已缓存
```

---

### 问题2: Conda插件冲突

#### 问题描述

```
An unexpected error has occurred. Conda has prepared the above report.
```

#### 根本原因

- Conda 25.11.1 使用了新的插件系统
- 高并发下多个进程同时调用 `conda run` 时，插件初始化发生冲突
- 导致 "An unexpected error has occurred" 错误

#### 解决方案

**直接使用Python解释器（已实施）** ✅

不使用 `conda run`，直接使用conda环境的Python解释器：

```python
# 获取conda环境的Python路径
conda_env_python = "/home/wangchen/.conda/envs/traceforge/bin/python"
if not os.path.exists(conda_env_python):
    conda_env_python = "/usr/local/miniconda3/envs/traceforge/bin/python"

cmd = [conda_env_python, infer_script, ...]

# 设置环境变量
env = os.environ.copy()
if conda_env_python and os.path.exists(conda_env_python):
    conda_env_bin = os.path.dirname(conda_env_python)
    env['PATH'] = conda_env_bin + os.pathsep + env.get('PATH', '')
```

**备选方案: 禁用Conda插件**

如果必须使用 `conda run`，添加 `--no-plugins` 选项：

```python
cmd = ["conda", "run", "--no-plugins", "-n", "traceforge", "python", infer_script, ...]
```

---

### 问题3: 并发问题

#### 已解决的问题 ✅

1. **Conda插件冲突** - 已通过直接使用Python解释器解决
2. **HuggingFace模型下载并发** - 已通过预加载模型解决
3. **打印输出混乱** - 已使用 `print_lock` 同步打印

#### 潜在问题 ⚠️

1. **文件I/O竞态条件（TOCTOU）**
   - 风险等级: 低
   - 影响: 可能产生重复写入
   - 建议: 使用原子操作或文件锁

2. **日志文件并发写入**
   - 风险等级: 中
   - 影响: 如果配置了日志文件，可能产生问题
   - 建议: 为每个进程使用独立日志文件

3. **GPU内存管理竞争**
   - 风险等级: 低
   - 影响: 每个进程使用独立GPU，不会相互影响
   - 状态: 当前实现已正确

详细分析请参考：[并发问题分析](#并发问题分析)

---

## 压力测试

### 快速测试（小规模）

```bash
cd scripts/batch_inference
python stress_test_batch_inference.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./stress_test_small \
    --gpu_id 0,1 \
    --max_trajs 10 \
    --max_workers 2 \
    --frame_drop_rate 5 \
    --check_integrity
```

### 大规模压力测试（重现累积性问题）

```bash
# 使用所有GPU，至少处理2500个traj以重现累积性问题
cd scripts/batch_inference
python stress_test_batch_inference.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./stress_test_large \
    --gpu_id 0,1,2,3,4,5 \
    --max_trajs 2500 \
    --min_trajs_for_issue 2500 \
    --max_workers 6 \
    --frame_drop_rate 5 \
    --grid_size 80 \
    --monitor_interval 60 \
    --check_integrity
```

### 性能基准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 成功率 | > 95% | 至少95%的traj处理成功 |
| 大规模成功率 | > 90% | 处理2000+ traj后的成功率 |
| GPU利用率 | > 80% | GPU应该被充分利用 |
| 平均处理时间 | < 60秒/traj | 每个traj的平均处理时间 |
| 累积性错误 | 无 | 处理2000+ traj后不应出现conda竞争等累积性错误 |

详细测试指南请参考：[压力测试指南](#压力测试指南)

---

## 推理完成后检查

### 1. 检查成功率

```bash
# 从日志中统计
grep "成功:" batch_inference.log | tail -1
grep "失败:" batch_inference.log | tail -1
```

### 2. 检查conda错误

```bash
# 检查是否有conda错误
grep -i "conda\|An unexpected error has occurred" batch_inference.log

# 如果没有输出，说明没有conda错误 ✅
```

### 3. 检查输出目录

```bash
# 统计输出目录数量
find ./output_bridge_depth_grid80 -mindepth 1 -maxdepth 1 -type d | wc -l

# 检查是否有空目录
find ./output_bridge_depth_grid80 -mindepth 1 -maxdepth 1 -type d -empty
```

---

## 常见问题

### Q: 子进程能否共享已下载的模型？

**A: 可以！** HuggingFace的缓存机制允许所有进程共享同一个缓存目录。

- 缓存位置：`~/.cache/huggingface/hub/models--Yuxihenry--SpatialTrackerV2_Front/`
- 如果模型已缓存，`from_pretrained()` 会直接从缓存读取，无需网络
- 所有子进程都可以访问同一个缓存，实现模型共享

### Q: 为什么需要预加载？

**A: 确保模型在子进程启动前就已经在缓存中。**

- 如果模型未缓存，多个子进程同时调用 `from_pretrained()` 会同时尝试下载
- 这会导致大量并发SSL连接，导致连接失败
- 预加载确保模型在主进程下载一次，子进程直接使用缓存

### Q: 如果模型已经在缓存中，还需要预加载吗？

**A: 不需要，但预加载可以验证缓存是否完整。**

- 如果模型已缓存，预加载会很快（<5秒），只是验证缓存
- 如果模型未缓存，预加载会下载模型
- 预加载可以提前发现网络问题，避免子进程失败

### Q: 如何继续已有输出目录的推理？

**A: 使用 `--skip_existing` 参数。**

```bash
python batch_infer.py \
    --base_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
    --out_dir ./output_bridge_depth_grid80 \
    --use_all_trajectories \
    --skip_existing \
    ...
```

**工作原理**:
- `--skip_existing` 会检查每个traj的输出目录是否存在
- 如果输出目录已存在且完整，会跳过该traj
- 这样不会重复处理已成功的traj

---

## 预期运行时间

- 使用6个GPU并行：约15-30小时（处理所有10500个traj）
- 如果已有2000+个traj完成，剩余约8500个，预计12-25小时
- 建议在周末或非工作时间运行

---

## 更新记录

- 2026-02-06: 初始版本，整合SSL错误分析和解决方案
- 2026-02-09: 添加conda错误分析和解决方案
- 2026-02-09: 整合并发问题分析和压力测试指南

