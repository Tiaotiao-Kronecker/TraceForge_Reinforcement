# 深度尺度对齐的数学原理

## 问题背景

在3D跟踪中，我们有两个深度来源：
1. **模型估计的深度** `d_pred`：由VGGT4模型从RGB图像估计得到（单位：米，但尺度可能不准确）
2. **读取的真实深度** `d_known`：从深度传感器或深度图像读取（单位：米，尺度准确）

两者可能存在**尺度差异**，需要进行对齐。

## 数学假设

假设模型估计的深度和真实深度之间存在**线性尺度关系**：

```
d_known ≈ s × d_pred
```

其中：
- `d_known`: 真实深度（已知，尺度正确）
- `d_pred`: 模型估计深度（未知尺度）
- `s`: 尺度因子（待求解）

## 单帧对齐公式

### 步骤1：计算有效像素

```python
valid_mask = (d_known > 0) & (d_pred > 0)
```

只考虑两个深度都有效的像素（深度 > 0）。

### 步骤2：计算中位数

```python
median_known = median(d_known[valid_mask])
median_pred = median(d_pred[valid_mask])
```

### 步骤3：计算尺度因子

```python
scale = median_known / median_pred
```

**数学公式：**

```
s = median(d_known[valid]) / median(d_pred[valid])
```

### 步骤4：对齐深度

```python
d_aligned = d_pred × scale
```

**数学公式：**

```
d_aligned = d_pred × s
```

## 视频多帧对齐公式

对于视频（多帧），对每一帧计算尺度，然后取平均：

```python
# 对每一帧计算尺度
for t in range(T):
    scale_t = median(d_known[t][valid]) / median(d_pred[t][valid])
    scales.append(scale_t)

# 取所有帧的平均尺度
scale = mean(scales)

# 使用平均尺度对齐所有帧
d_aligned = d_pred × scale
```

**数学公式：**

```
scale_t = median(d_known[t][valid]) / median(d_pred[t][valid])  (对每一帧 t)

scale = (1/T) × Σ(scale_t)  (所有帧的平均)

d_aligned[t] = d_pred[t] × scale  (对所有帧 t)
```

## 为什么使用中位数？

### 1. 鲁棒性（Robustness）

中位数对异常值不敏感。例如：

```
d_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0]  # 最后一个异常值
d_known = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

使用均值：
  scale = mean(d_known) / mean(d_pred) = 2.75 / 1.45 ≈ 1.90  ❌ 受异常值影响

使用中位数：
  scale = median(d_known) / median(d_pred) = 2.75 / 0.55 = 5.00  ✓ 不受异常值影响
```

### 2. 代表性

中位数能更好地代表"典型"深度值，不受极值影响。

### 3. 稳定性

深度图中可能有：
- 无效区域（深度 = 0）
- 遮挡区域
- 传感器噪声

中位数能更好地处理这些情况。

## 完整示例

### 输入数据

```python
# 模型估计的深度（尺度偏小，比如单位理解错误）
d_pred = [
    [0.1, 0.2, 0.3],
    [0.2, 0.4, 0.5],
    [0.3, 0.5, 0.6]
]

# 真实深度（正确尺度）
d_known = [
    [0.5, 1.0, 1.5],
    [1.0, 2.0, 2.5],
    [1.5, 2.5, 3.0]
]
```

### 计算过程

```python
# 步骤1: 有效像素
valid_mask = (d_known > 0) & (d_pred > 0)  # 全部有效

# 步骤2: 中位数
median_known = median([0.5, 1.0, 1.5, 1.0, 2.0, 2.5, 1.5, 2.5, 3.0]) = 2.0
median_pred = median([0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.3, 0.5, 0.6]) = 0.4

# 步骤3: 尺度因子
scale = 2.0 / 0.4 = 5.0

# 步骤4: 对齐
d_aligned = d_pred × 5.0 = [
    [0.5, 1.0, 1.5],
    [1.0, 2.0, 2.5],
    [1.5, 2.5, 3.0]
]
```

### 验证

对齐后的深度 `d_aligned` 应该接近真实深度 `d_known`：

```
误差 = mean(|d_aligned - d_known|) ≈ 0
```

## 代码实现

```python
def align_depth_scale(pred_depth, known_depth):
    """
    单帧深度尺度对齐
    """
    valid_mask = (known_depth > 0) & (pred_depth > 0)
    scale = np.median(known_depth[valid_mask]) / np.median(pred_depth[valid_mask])
    return scale

def align_video_depth_scale(pred_depth, known_depth):
    """
    视频深度尺度对齐
    """
    scales = []
    for t in range(pred_depth.shape[0]):
        scale_t = align_depth_scale(pred_depth[t], known_depth[t])
        scales.append(scale_t)
    scale = np.array(scales).mean()
    aligned_depth = pred_depth * scale
    return aligned_depth, scale
```

## 总结

深度尺度对齐的核心思想：
1. **假设线性关系**：`d_known ≈ s × d_pred`
2. **使用中位数估计尺度**：`s = median(d_known) / median(d_pred)`
3. **对齐深度**：`d_aligned = d_pred × s`

这种方法简单、鲁棒，适用于大多数深度对齐场景。

