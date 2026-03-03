# BridgeData V2 Action数据格式分析

## 数据集信息

- **数据集**: BridgeData V2
- **项目主页**: https://rail-berkeley.github.io/bridgedata/
- **机器人**: WidowX 250 6DOF机械臂
- **控制频率**: 5 Hz
- **控制方式**: VR遥操作
- **平均轨迹长度**: 38个时间步

## 数据文件结构

每个数据目录（如 `00000/`）包含以下文件：

```
00000/
├── lang.txt                    # 任务描述/指令（如："pick up pot"）
├── agent_data.pkl              # Agent数据（需要ROS依赖才能读取）
├── policy_out.pkl              # 策略输出（包含action信息）✅
├── obs_dict.pkl                # 观察数据字典
├── images0/                    # RGB图像（主视角）
├── images1/                    # RGB图像（随机视角1）
├── images2/                    # RGB图像（随机视角2）
├── depth_images0/              # 深度图像（主视角）
├── depth_images1/              # 深度图像（随机视角1）
└── depth_images2/              # 深度图像（随机视角2）
```

## policy_out.pkl 结构

### 整体结构

```python
policy_out = [
    {  # 第0帧
        'actions': <numpy数组>,              # action数据
        'new_robot_transform': <4x4矩阵>,    # 新的机器人变换
        'delta_robot_transform': <4x4矩阵>   # 机器人变换增量
    },
    {  # 第1帧
        ...
    },
    ...
]  # 长度 = 帧数（如24帧）
```

### 字段说明

#### 1. `actions` - Action数据（主要字段）

- **类型**: `numpy.ndarray`
- **shape**: `(7,)`（1D数组，7个元素）
- **dtype**: `float64`
- **维度**: 1D

**格式**: `[dx, dy, dz, dqx, dqy, dqz, gripper]`

##### 详细说明

**前3维 - 位置增量 (`dx, dy, dz`)**:
- **单位**: 米
- **值范围**: 约 `[-0.03, 0.13]` 米
- **含义**: 相对于当前末端执行器位置的移动量
- **示例**:
  - 帧 0: `[-3.376e-10, -8.845e-11, -1.900e-10]`（几乎为0，初始状态）
  - 帧 1: `[0.0158, -0.0317, -0.0168]`（移动约1-3厘米）

**中间3维 - 旋转增量 (`dqx, dqy, dqz`)**:
- **表示方式**: 6D笛卡尔末端执行器运动的一部分（论文：6D Cartesian end-effector motion）
- **论文描述**: "relative changes in pose"（姿态的相对变化）
- **⚠️ 重要说明**: 论文**未明确说明**旋转的具体表示方式（轴角、欧拉角等）
- **数据分析结果**: 
  - 通过对比`action[3:6]`和`delta_robot_transform`旋转矩阵的轴角表示
  - 匹配率约95.7%，最大差异约0.01弧度
  - **最可能是轴角（axis-angle / rotation vector）表示**
- **值范围**: 约 `[-0.17, 0.12]` 弧度
- **含义**: 相对于当前末端执行器姿态的旋转量（相对变化）
- **示例**:
  - 帧 0: `[2.325e-07, 3.780e-07, 6.539e-07]`（几乎为0）
  - 帧 1: `[-0.0571, 0.0414, -0.0888]`（旋转约2-5度）

**第7维 - Gripper状态 (`gripper`)**:
- **类型**: 离散维度（论文：discrete dimension）
- **值**: 
  - `1.0` = gripper打开
  - `0.0` = gripper关闭
- **论文描述**: "discrete dimension to control the opening and closing of the gripper"
- **统计**: 在24帧轨迹中，通常有10帧打开，14帧关闭

#### 2. `new_robot_transform` - 新的机器人变换

- **类型**: `numpy.ndarray`
- **shape**: `(4, 4)`
- **dtype**: `float64`
- **格式**: 4x4齐次变换矩阵
- **含义**: 表示机器人末端执行器的新位置和姿态

```python
transform = [
    [R11, R12, R13, x],  # 旋转矩阵 + 位置
    [R21, R22, R23, y],
    [R31, R32, R33, z],
    [0,   0,   0,   1 ]
]
```

#### 3. `delta_robot_transform` - 机器人变换增量

- **类型**: `numpy.ndarray`
- **shape**: `(4, 4)`
- **dtype**: `float64`
- **格式**: 4x4齐次变换矩阵（增量）
- **含义**: 表示机器人末端执行器的位置和姿态增量
- **⚠️ 重要说明**: 
  - 旋转部分：与`new_robot_transform`的关系为 `new[t][:3,:3] ≈ delta[t][:3,:3] @ new[t-1][:3,:3]`（旋转匹配度很高）
  - 位置部分：`delta_robot_transform`中的位置增量可能不在世界坐标系中，不能直接用于计算`new_robot_transform`的位置
  - 建议：直接使用`delta_robot_transform`，而不是从`actions`推导

## Action格式验证

### 三个选项的可行性分析

#### 选项1: `[x, y, z, qx, qy, qz, qw]`（3D位置 + 四元数旋转）

❌ **不太可能**

**理由**:
1. 第7维只有 `0.0` 和 `1.0`，不符合四元数 `qw` 的连续值特征
2. 如果是位置+四元数，第7维应该是 `qw`，但实际是二进制状态
3. 四元数需要归一化，但第7维是二进制值

#### 选项2: `[joint1, joint2, ..., joint7]`（7个关节角度）

❌ **不可能**

**理由**:
1. WidowX 250只有6个关节（6DOF）
2. 不应该有7个关节角度
3. 关节角度范围通常较大（如 `[-π, π]`），但当前值较小

#### 选项3: `[dx, dy, dz, dqx, dqy, dqz, gripper]`（增量控制 + gripper）

✅ **已确认！**（论文验证）

**论文原文**（BridgeData V2: A Dataset for Robot Learning at Scale, CoRL 2023）:
> "The 7D action space of the robot consists of continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose, as well as a discrete dimension to control the opening and closing of the gripper."

**支持证据**:
1. ✓ **论文确认**: 7维action空间 = 连续6维笛卡尔末端执行器运动（相对变化）+ 离散gripper控制
2. ✓ 符合6DOF机器人 + gripper的结构
3. ✓ 前6维值较小，符合增量控制特征
4. ✓ 第7维是二进制（0/1），符合gripper状态
5. ✓ 与 `delta_robot_transform` 字段关联

### 数值统计（基于00000样本）

- **轨迹长度**: 24帧
- **位置增量范围**: `[-0.031708, 0.134845]` 米
- **位置增量平均值**: `0.006729` 米
- **旋转增量范围**: `[-0.169958, 0.124645]` 弧度
- **旋转增量平均值**: `-0.013810` 弧度
- **Gripper状态**: 10帧打开（1.0），14帧关闭（0.0）

## 读取代码示例

### Python代码

```python
import pickle
import numpy as np

# 读取policy_out.pkl
data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
with open(f"{data_dir}/policy_out.pkl", 'rb') as f:
    policy_out = pickle.load(f)

# 提取所有actions
actions = [frame['actions'] for frame in policy_out]
actions_array = np.array(actions)  # shape: (24, 7)

# 访问第t帧的action
action_t = actions_array[t]  # shape: (7,)

# 解析action
dx, dy, dz = action_t[:3]      # 位置增量（米）
dqx, dqy, dqz = action_t[3:6]  # 旋转增量（弧度）
gripper = action_t[6]           # gripper状态（1.0=打开, 0.0=关闭）

# 读取任务描述
with open(f"{data_dir}/lang.txt", 'r') as f:
    task_description = f.read().strip()  # 如："pick up pot"
```

### 完整示例

```python
import pickle
import numpy as np
import os

def load_action_data(data_dir):
    """加载action数据"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    lang_path = os.path.join(data_dir, "lang.txt")
    
    # 读取policy_out
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    # 提取actions
    actions = np.array([frame['actions'] for frame in policy_out])
    
    # 读取任务描述
    with open(lang_path, 'r') as f:
        task_description = f.read().strip()
    
    return {
        'actions': actions,  # shape: (T, 7)
        'task_description': task_description,
        'num_frames': len(policy_out),
        'transforms': [frame['new_robot_transform'] for frame in policy_out],
        'delta_transforms': [frame['delta_robot_transform'] for frame in policy_out]
    }

# 使用示例
data = load_action_data("/usr/data/dataset/opt/dataset_temp/bridge_depth/00000")
print(f"任务: {data['task_description']}")
print(f"帧数: {data['num_frames']}")
print(f"Actions shape: {data['actions'].shape}")
print(f"第一帧action: {data['actions'][0]}")
```

## 数据格式总结

### Action格式

```
action = [dx, dy, dz, dqx, dqy, dqz, gripper]
         |---位置增量---|  |---旋转增量---|  |状态|
```

### 完整数据结构

```python
policy_out = [
    {
        'actions': np.ndarray,              # shape=(7,), dtype=float64
        'new_robot_transform': np.ndarray,  # shape=(4,4), 齐次变换矩阵
        'delta_robot_transform': np.ndarray  # shape=(4,4), 增量变换矩阵
    },
    ...
]  # 长度 = 帧数
```

## 注意事项

1. **增量控制**: action是增量控制（相对变化），不是绝对位置/姿态（论文：relative changes in pose）
2. **坐标系**: 位置和旋转都是相对于当前末端执行器状态
3. **Gripper**: 是离散维度（论文：discrete dimension），不是连续值
4. **时间同步**: action与图像帧同步（控制频率5 Hz）
5. **论文确认**: 7维 = 连续6维笛卡尔末端执行器运动 + 离散gripper控制
6. **⚠️ 旋转表示方式**: 论文未明确说明旋转的具体表示方式，基于数据分析最可能是**轴角（axis-angle）**表示

## 参考

- BridgeData V2项目主页: https://rail-berkeley.github.io/bridgedata/
- 数据集论文: BridgeData V2: A Dataset for Robot Learning at Scale (CoRL 2023)
  - **论文确认**: "The 7D action space of the robot consists of continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose, as well as a discrete dimension to control the opening and closing of the gripper."

## 旋转表示方式分析

### 论文描述

论文原文：
> "The 7D action space of the robot consists of continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose, as well as a discrete dimension to control the opening and closing of the gripper."

**关键点**:
- ✅ 确认了7维action空间
- ✅ 确认了6维连续笛卡尔末端执行器运动
- ✅ 确认了相对变化（relative changes）
- ✅ 确认了离散gripper控制
- ❌ **但未明确说明旋转的具体表示方式**

### 数据分析结果

通过对比`action[3:6]`和`delta_robot_transform`旋转矩阵，我们发现：

1. **轴角表示验证**:
   - 从`delta_robot_transform`的旋转矩阵提取轴角表示
   - 与`action[3:6]`对比，匹配率约**95.7%**
   - 最大差异约**0.01弧度**（非常小）

2. **结论**:
   - **最可能是轴角（axis-angle / rotation vector）表示**
   - 轴角表示：旋转向量 `[rx, ry, rz]`，其中向量的方向是旋转轴，向量的长度是旋转角度（弧度）

3. **其他可能性**:
   - 欧拉角（Euler angles）表示：在小角度旋转时与轴角接近，但匹配度较低
   - 其他表示方式：可能性较小

### 建议

由于论文未明确说明，建议：
1. 查看BridgeData V2的代码实现（如果有开源）
2. 查看相关论文的补充材料或附录
3. 通过实验验证（将action应用到机器人，看效果）

## Transform之间的关系分析

### 问题

能否从当前帧的`actions`和`delta_robot_transform`，结合上一帧的`new_robot_transform`，计算得到当前帧的`new_robot_transform`？

### 验证结果

#### 关系1: `new_robot_transform[t] = new_robot_transform[t-1] @ delta_robot_transform[t]`

❌ **不成立**
- 平均误差: 0.110627
- 最大误差: 0.269049

#### 关系2: `new_robot_transform[t] = delta_robot_transform[t] @ new_robot_transform[t-1]`

⚠️ **部分成立**
- 平均误差: 0.007759（较小）
- 最大误差: 0.078425
- **旋转部分匹配度很高**（误差 < 0.000001）
- **位置部分不匹配**（可能在不同坐标系中）

#### 关系3: 从`actions`构建`delta_robot_transform`

❌ **不成立**
- 平均误差: 0.023618
- 最大误差: 0.056853
- 虽然`actions`的旋转部分与`delta_robot_transform`的旋转矩阵轴角表示匹配（95.7%），但构建完整变换矩阵时存在误差

### 结论

1. **旋转部分**: `delta_robot_transform`的旋转部分与`new_robot_transform`的关系为：
   ```
   new_robot_transform[t][:3,:3] ≈ delta_robot_transform[t][:3,:3] @ new_robot_transform[t-1][:3,:3]
   ```
   匹配度非常高（误差 < 0.000001）

2. **位置部分**: `delta_robot_transform`中的位置增量可能不在世界坐标系中，不能直接用于计算`new_robot_transform`的位置

3. **从actions推导**: 虽然`actions`的旋转部分与`delta_robot_transform`的轴角表示匹配，但无法准确构建完整的`delta_robot_transform`

### 建议

- **直接使用`delta_robot_transform`**: 不要从`actions`推导，直接使用数据中提供的`delta_robot_transform`
- **旋转部分**: 可以使用关系2计算旋转部分
- **位置部分**: 直接使用`new_robot_transform`中的位置，不要从`delta_robot_transform`推导
- **查看源码**: 如需准确理解关系，建议查看BridgeData V2的代码实现

## 数据集完整结构分析

### 文件列表

每个数据目录（如 `00000/`）包含以下文件：

```
00000/
├── lang.txt                    # 任务描述/指令
├── agent_data.pkl              # Agent数据（需要ROS依赖）
├── policy_out.pkl              # 策略输出（已详细分析）✅
├── obs_dict.pkl                # 观察数据字典 ✅
├── images0/                    # RGB图像（主视角）
├── images1/                    # RGB图像（随机视角1）
├── images2/                    # RGB图像（随机视角2）
├── depth_images0/              # 深度图像（主视角）
├── depth_images1/              # 深度图像（随机视角1）
└── depth_images2/              # 深度图像（随机视角2）
```

### 1. lang.txt - 任务描述

- **格式**: 纯文本文件
- **内容**: 自然语言任务指令
- **示例**: `"pick up pot"`, `"put the carrot on the plate"`
- **含义**: 描述机器人要执行的任务，用于语言条件学习

### 2. policy_out.pkl - 策略输出（已详细分析）

详见前面的章节。

### 3. obs_dict.pkl - 观察数据字典

包含机器人的状态信息和传感器数据：

#### 字段说明

| 字段名 | 类型 | Shape | 含义 |
|--------|------|-------|------|
| `qpos` | `np.ndarray` | `(T, 6)` | 关节位置（6个关节角度，单位：弧度） |
| `qvel` | `np.ndarray` | `(T, 6)` | 关节速度（6个关节角速度，单位：弧度/秒） |
| `joint_effort` | `np.ndarray` | `(T, 6)` | 关节力矩/力（6个关节力矩） |
| `state` | `np.ndarray` | `(T, 7)` | 机器人状态（可能是6个关节角度 + gripper状态） |
| `full_state` | `np.ndarray` | `(T, 7)` | 完整状态（与state可能相同） |
| `desired_state` | `np.ndarray` | `(T, 7)` | 期望状态（目标状态） |
| `eef_transform` | `np.ndarray` | `(T, 4, 4)` | 末端执行器变换（4x4齐次变换矩阵） |
| `high_bound` | `np.ndarray` | `(T, 5)` | 关节上限（前5个关节的角度上限，单位：弧度） |
| `low_bound` | `np.ndarray` | `(T, 5)` | 关节下限（前5个关节的角度下限，单位：弧度） |
| `time_stamp` | `list` | `(T,)` | 时间戳列表 |
| `env_done` | `list` | `(T,)` | 环境是否完成的标志（布尔值） |
| `task_stage` | `list` | `(T,)` | 任务阶段（可能是整数或字符串） |

**注意**: `T` 表示轨迹长度（帧数），通常为24-38帧。

#### 关键字段详解

**`qpos` (关节位置)**:
- 6个关节的角度值（弧度）
- 值范围: 约 `[-0.7, 1.6]` 弧度
- 对应WidowX 250的6个关节

**`qvel` (关节速度)**:
- 6个关节的角速度（弧度/秒）
- 值范围: 约 `[-2.7, 1.3]` 弧度/秒

**`eef_transform` (末端执行器变换)**:
- 4x4齐次变换矩阵
- 与`policy_out.pkl`中的`new_robot_transform`可能相同
- 表示末端执行器在世界坐标系中的位置和姿态

**`state` / `full_state`**:
- 7维状态向量
- 可能是 `[qpos[0], qpos[1], ..., qpos[5], gripper]`
- 值范围: 约 `[-0.5, 1.0]`

### 4. agent_data.pkl - Agent数据

- **状态**: 需要ROS依赖才能读取（包含`sensor_msgs`）
- **错误**: `No module named 'sensor_msgs'`
- **含义**: 可能包含ROS消息格式的传感器数据
- **建议**: 如果需要访问，需要安装ROS相关依赖

### 5. 图像数据

#### RGB图像 (images0/1/2)

- **格式**: JPEG
- **数量**: 每视角约25张图像
- **命名**: `im_0.jpg`, `im_1.jpg`, ..., `im_9.jpg`
- **视角说明**:
  - `images0`: 主视角（over-the-shoulder，固定相机）
  - `images1`: 随机视角1（数据收集时每50轨迹随机调整）
  - `images2`: 随机视角2（数据收集时每50轨迹随机调整）

#### 深度图像 (depth_images0/1/2)

- **格式**: 16位PNG
- **数量**: 每视角约25张图像
- **命名**: `im_0.png`, `im_1.png`, ..., `im_9.png`
- **单位**: 可能是毫米或厘米（需要根据实际情况确认）
- **对齐**: 与对应RGB图像对齐（相同视角、相同时间戳）

### 数据同步

- **帧数**: 所有数据（`policy_out.pkl`、`obs_dict.pkl`、图像）的帧数应该一致
- **时间戳**: `obs_dict.pkl`中的`time_stamp`提供了时间信息
- **控制频率**: 5 Hz（每帧间隔约0.2秒）

### 数据使用建议

1. **训练模型**: 使用`images0`（主视角）和`policy_out.pkl`中的`actions`
2. **多视角学习**: 使用`images0/1/2`进行多视角训练
3. **状态信息**: 使用`obs_dict.pkl`中的`qpos`、`qvel`等状态信息
4. **深度信息**: 使用`depth_images0`进行深度感知任务
5. **语言条件**: 使用`lang.txt`进行语言条件学习

## 更新日期

2026-02-06

