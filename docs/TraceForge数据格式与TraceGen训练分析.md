# TraceForge数据格式与TraceGen训练分析

## 目录
- [1. 数据结构概览](#1-数据结构概览)
- [2. 关键字段详解](#2-关键字段详解)
- [3. 主NPZ vs Sample NPZ](#3-主npz-vs-sample-npz)
- [4. 弧长域vs时间域](#4-弧长域vs时间域)
- [5. TraceGen训练数据流](#5-tracegen训练数据流)
- [6. 下游任务适配](#6-下游任务适配)

---

## 1. 数据结构概览

### 1.1 主NPZ文件结构
位置：`output/run_xxx/run_xxx.npz`

```python
{
    'coords': (21, 400, 3),      # 21帧 × 400个轨迹点 × 3D坐标
    'extrinsics': (21, 4, 4),    # 每帧的相机外参
    'intrinsics': (21, 3, 3),    # 每帧的相机内参
    'visibs': (21, 400, 1),      # 每帧每点的可见性
    'depths': (21, H, W),        # 每帧的深度图
    'unc_metric': (21, H, W),    # 深度置信度
    'height': int,               # 图像高度
    'width': int                 # 图像宽度
}
```

**用途**：可视化和场景分析，不参与训练

### 1.2 Sample NPZ文件结构
位置：`output/run_xxx/samples/run_xxx_0.npz`

```python
{
    'image_path': str,           # 查询帧图像路径
    'frame_index': int,          # 查询帧索引
    'keypoints': (400, 2),       # 查询点的初始2D位置
    'traj': (400, 128, 3),       # 3D轨迹（重采样后）
    'traj_2d': (400, 21, 2),     # 2D轨迹投影
    'valid_steps': (128,)        # 有效时间步标记
}
```

**用途**：TraceGen训练数据

---

## 2. 关键字段详解

### 2.1 valid_steps

**定义**：布尔数组，标记重采样后哪些时间步包含真实数据（非填充）

**形状**：`(128,)` - 对应`traj`的时间维度

**计算逻辑**（`infer.py`第102-210行）：

```python
# 步骤1: 计算轨迹的总弧长
diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]
seglens_all = np.linalg.norm(diffs_all, axis=2)
robust_seglen = topk.mean(axis=0)  # 使用top 2%的轨迹作为鲁棒估计
total_len = float(robust_seglen.sum())

# 步骤2: 根据采样间隔计算有效采样点数
k_max = int(np.floor(total_len / interval))  # interval=0.05
num_samples = min(k_max + 1, max_length)     # max_length=128

# 步骤3: 生成valid_mask
valid_mask = np.zeros((max_length), dtype=bool)
valid_mask[:num_samples] = True  # 前num_samples个位置为True
```

**实际含义**：
- `valid_steps[i] = True`：该时间步是通过弧长重采样得到的有效数据
- `valid_steps[i] = False`：该时间步是填充的无效数据（值为-inf）
- 有效步数取决于轨迹的总运动距离

**示例**：
```python
valid_steps[:15] = [True×9, False×6]
# 表示只有前9个时间步有效，后面是填充
```

### 2.2 visibs

**定义**：布尔数组，标记每个轨迹点在每一帧的空间可见性

**形状**：`(21, 400, 1)` - 21帧 × 400个点

**来源**：模型推理输出，预测轨迹点是否被遮挡或出界

**用途**：可视化时判断哪些点应该显示

### 2.3 visibs vs valid_steps 对比

| 特性 | visibs | valid_steps |
|------|--------|-------------|
| 形状 | (T, N, 1) | (max_length,) |
| 维度 | 时间 × 空间 | 仅时间 |
| 含义 | 空间可见性 | 时间有效性 |
| 来源 | 模型推理 | 重采样算法 |
| 问题 | 点是否可见？ | 步骤是否有效？ |
| 用途 | 可视化 | 训练过滤 |

**关系**：两者是正交的概念，没有直接关系
- `visibs`：关注轨迹点在视野中的可见性（遮挡、出界）
- `valid_steps`：关注重采样后的时间步是否包含真实数据

### 2.4 traj vs traj_2d

**traj（3D轨迹）**：
- 形状：`(400, 128, 3)`
- 坐标系：世界坐标系（米）
- 时间步：128（重采样后的均匀弧长间隔）
- 用途：TraceGen训练

**traj_2d（2D轨迹）**：
- 形状：`(400, 21, 2)`
- 坐标系：图像坐标系（像素）
- 帧数：21（原始视频帧数）
- 用途：可视化和2D分析

**生成方式**：
```python
# 1. 模型推理得到3D坐标
coords_3d = inference(...)  # (T, 6400, 3)

# 2. 投影到固定相机视角
tracks3d_fixed = project_tracks_3d_to_3d(coords_3d, camera_views)
tracks2d_fixed = project_tracks_3d_to_2d(coords_3d, camera_views)

# 3. traj: 重采样到128步
traj, valid_steps = retarget_trajectories(tracks3d_fixed, max_length=128)

# 4. traj_2d: 保持原始帧数
traj_2d = tracks2d_fixed  # (400, 21, 2)
```

---

## 3. 主NPZ vs Sample NPZ

### 3.1 数据对比

| 特性 | 主NPZ | Sample NPZ |
|------|-------|------------|
| 位置 | `run_xxx.npz` | `samples/run_xxx_0.npz` |
| 轨迹形状 | (21, 400, 3) | (400, 128, 3) |
| 时间表示 | 原始帧（时间域） | 重采样步（弧长域） |
| 相机参数 | ✓ 包含 | ✗ 不包含 |
| 可见性 | ✓ visibs | ✗ 无 |
| 有效性标记 | ✗ 无 | ✓ valid_steps |
| 用途 | 可视化 | 训练 |

### 3.2 为什么需要两种格式？

**主NPZ**：面向可视化和分析
- 保留完整的场景信息（相机参数、深度图、可见性）
- 时间域表示，便于理解和可视化
- 可以重建3D场景

**Sample NPZ**：面向训练
- 每个查询帧独立，便于数据加载和批处理
- 弧长域表示，更适合学习运动模式
- 去除冗余，提高训练效率

---

## 4. 弧长域vs时间域

### 4.1 核心区别

**时间域（主NPZ）**：
- 采样方式：固定时间间隔
- 特点：每帧之间时间相等
- 类比：视频的固定帧率（如30fps）
- 问题：静止时浪费大量帧

**弧长域（Sample NPZ）**：
- 采样方式：固定弧长间隔（interval=0.05）
- 特点：每步之间空间距离相等，时间间隔不等
- 类比：GPS轨迹的等距采样
- 优势：自动跳过静止区域

### 4.2 实际例子

```
原始视频：21帧，包含停顿

时间域表示：
帧0-10: 运动（10帧）
帧11-15: 停顿（5帧，浪费！）
帧16-20: 运动（5帧）
总计：21帧

弧长域表示：
步骤0-5: 第一段运动（6个采样点）
步骤6: 停顿点（1个采样点）
步骤7-9: 第二段运动（3个采样点）
总计：9个有效步骤（valid_steps.sum() = 9）
```

### 4.3 参数化方式对比

| 特性 | 时间域 | 弧长域 |
|------|--------|--------|
| 参数 | 时间 t | 弧长 s |
| 采样 | 均匀时间 | 均匀空间 |
| 运动快 | 帧间距离大 | 时间间隔短 |
| 运动慢 | 帧间距离小 | 时间间隔长 |
| 静止 | 多帧重复 | 单点表示 |

**结论**：这不是简单的"帧率改变"，而是从时间参数到弧长参数的转换。

---

## 5. TraceGen训练数据流

### 5.1 数据加载流程

```python
# 1. 从Sample NPZ加载（datasets.py 第526行）
npz_data = np.load(npz_path)  # samples/xxx.npz
keypoints = npz_data['keypoints']
trajectories = npz_data['traj']  # (400, 128, 3)
valid_steps = npz_data['valid_steps']  # (128,)

# 2. 计算trajectory_mask（第542-543行）
trajectory_mask = trajectories != -np.inf  # 动态计算
# valid_steps[t]=False → traj[:, t, :] = -inf → trajectory_mask[:, t, :] = False

# 3. 过滤低质量样本（trainer.py 第297行）
valid_mask = (batch['valid_steps'] > 3) & \
             (batch['movement_bool'].sum(dim=1) > 0) & \
             (batch['gt_mask'].sum(dim=1) > 0)
# 只保留valid_steps>3且有运动的样本

# 4. 计算损失（trajectory_loss.py 第78行）
diffusion_loss = F.mse_loss(
    noise_pred * traj_mask,
    noise_target * traj_mask
)
# 使用trajectory_mask过滤无效点
```

### 5.2 关键发现

**TraceGen训练时**：
- ✓ 使用Sample NPZ（`samples/`目录）
- ✓ 使用`valid_steps`过滤样本（阈值>3）
- ✓ 使用`trajectory_mask`（从traj计算）过滤损失
- ✗ 不使用主NPZ
- ✗ 不使用`visibs`

**数据流总结**：
```
Sample NPZ → traj + valid_steps
           ↓
    trajectory_mask = traj != -inf
           ↓
    过滤: valid_steps > 3
           ↓
    损失计算: 使用trajectory_mask
```

### 5.3 相对动作表示

TraceGen使用`absolute_action: false`配置：

```python
# datasets.py 第604-606行
if not self.absolute_action:
    # 转换为相对位移
    gt_trajectory[:, 1:, :] = gt_trajectory[:, 1:, :] - gt_trajectory[:, :-1, :]
```

**含义**：
- 训练时学习相对位移（每步相对于上一步的增量）
- 而非绝对位置
- 更适合机器人控制的动作空间

---

## 6. 下游任务适配

### 6.1 弧长域表示的核心优势

**解耦"运动几何"和"运动时序"**：
- 运动几何：路径的形状（去哪里、怎么走）
- 运动时序：运动的速度（何时到达）

弧长域专注于运动几何，时序可以灵活调整。

### 6.2 便利性1：运动不变性

**问题场景**：识别"挥手"动作

**时间域的困境**：
```python
# 快速挥手（0.5秒，15帧）
fast_wave = [(x1,y1), (x2,y2), ..., (x15,y15)]

# 慢速挥手（2秒，60帧）
slow_wave = [(x1,y1), (x2,y2), ..., (x60,y60)]

# 问题：同样的动作，表示完全不同
# 模型需要学习多个速度变体
```

**弧长域的优势**：
```python
# 快速或慢速挥手，路径相同
wave_motion = [起点, 轨迹点1, ..., 终点]  # 9个有效点

# 优势：只需学习一个运动模式
# 速度信息可以单独处理（如果需要）
```

### 6.3 便利性2：数据效率

**时间域的浪费**：
```
机器人从A移动到B，中间停顿1秒

时间域（30fps）：
- 前10帧：运动
- 中间30帧：停顿（浪费！）
- 后10帧：运动
总计：50帧，其中30帧冗余

弧长域：
- 前5步：运动
- 1步：停顿点
- 后5步：运动
总计：11步，数据压缩率 22%
```

### 6.4 便利性3：灵活的速度适配

**场景**：机器人抓取任务

```python
# TraceGen生成弧长域轨迹
arc_traj = model.generate(image, text)
waypoints = arc_traj[:valid_steps.sum()]  # 9个路径点

# 适配不同机器人：

# 快速机器人（1.5秒完成）
fast_robot.execute(waypoints, duration=1.5)

# 慢速机器人（5秒完成，精细操作）
slow_robot.execute(waypoints, duration=5.0)

# 遇到障碍（时间变长，路径不变）
robot.execute(waypoints, duration=8.0)

# 一个模型 → 适配所有速度
```

### 6.5 便利性4：任意帧率生成

**场景**：视频生成

```python
# TraceGen输出（弧长域）
arc_traj = model.generate(...)  # 9个有效步骤

# 生成不同帧率的视频：

# 30fps视频（3秒 = 90帧）
video_30fps = interpolate_to_time(arc_traj, fps=30, duration=3.0)

# 60fps视频（3秒 = 180帧）
video_60fps = interpolate_to_time(arc_traj, fps=60, duration=3.0)

# 慢动作视频（6秒 = 180帧）
slow_motion = interpolate_to_time(arc_traj, fps=30, duration=6.0)

# 同一个模型输出 → 满足所有帧率需求
```

### 6.6 三种下游适配方案

**方案A：直接使用（路径规划）**
```python
# 适用于：路径跟踪控制器
waypoints = tracegen_output['traj'][:valid_steps.sum()]
robot.follow_path(waypoints)
# 控制器自动处理速度规划
```

**方案B：时间重采样（固定帧率控制）**
```python
# 适用于：需要固定时间步的控制器
from scipy.interpolate import interp1d

arc_traj = tracegen_output['traj'][:valid_steps.sum()]
# 插值到固定帧率
time_traj = interpolate_uniform_time(arc_traj, fps=30, duration=3.0)
robot.execute_timed_trajectory(time_traj)
```

**方案C：速度规划（动力学约束）**
```python
# 适用于：有速度/加速度限制的机器人
positions = tracegen_output['traj'][:valid_steps.sum()]

# 根据动力学约束规划速度
velocities = plan_velocity_profile(
    positions,
    max_velocity=1.0,
    max_acceleration=0.5
)

# 生成时间戳
timestamps = compute_timestamps(positions, velocities)
robot.execute_with_timing(positions, timestamps)
```

### 6.7 核心优势总结

| 优势 | 说明 | 价值 |
|------|------|------|
| 运动不变性 | 同样动作，不同速度 → 相同表示 | 减少训练数据需求 |
| 数据效率 | 去除静止帧冗余 | 训练更快、更高效 |
| 一次训练多种部署 | 一个模型适配所有速度/帧率 | 降低开发成本 |
| 更好的泛化 | 学习运动本质而非表象 | 提高模型性能 |
| 下游灵活性 | 可根据需求自由调整时序 | 适配更多应用场景 |

---

## 7. 总结

### 7.1 数据格式设计理念

TraceForge采用双格式设计：
- **主NPZ**：时间域，面向可视化和分析
- **Sample NPZ**：弧长域，面向训练和部署

这种设计兼顾了可解释性和训练效率。

### 7.2 TraceGen的关键创新

1. **弧长域表示**：解耦运动几何和运动时序
2. **相对动作**：学习增量而非绝对位置
3. **有效性过滤**：使用`valid_steps`确保数据质量
4. **灵活适配**：一个模型满足多种下游需求

### 7.3 实践建议

**训练TraceGen时**：
- 确保`valid_steps > 3`（过滤太短的轨迹）
- 使用`trajectory_mask`过滤损失计算
- 采用相对动作表示（`absolute_action: false`）

**部署到下游任务时**：
- 路径规划：直接使用弧长域waypoints
- 时序控制：插值到所需帧率
- 动力学约束：结合速度规划算法

**数据准备时**：
- 主NPZ用于调试和可视化
- Sample NPZ用于训练
- 确保`valid_steps`正确标记有效数据

---

## 8. 参考代码位置

### TraceForge (推理)
- `infer.py` 第102-210行：`retarget_trajectories`函数
- `infer.py` 第379-381行：生成`valid_steps`
- `infer.py` 第937-954行：保存主NPZ
- `infer.py` 第212-393行：保存Sample NPZ

### TraceGen (训练)
- `dataio/datasets.py` 第526-543行：加载Sample NPZ
- `dataio/datasets.py` 第604-606行：相对动作转换
- `trainer/trainer.py` 第297行：使用`valid_steps`过滤
- `losses/trajectory_loss.py` 第78行：使用`trajectory_mask`计算损失

---

*文档生成时间：2026-03-14*
