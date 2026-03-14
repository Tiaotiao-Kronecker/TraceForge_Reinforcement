# `grid_size` 和 `support_grid_size` 的区别

## 一、概述

在 `infer.py` 中，有两个不同的 `grid_size` 参数，它们用于不同的目的：

1. **`grid_size`** (值为20)：用于创建**查询点（query points）**
2. **`support_grid_size`** (值为16)：用于创建**支持点（support points）**

---

## 二、详细分析

### 2.1 `grid_size` (查询点网格)

**位置**：`infer.py` 第493行

```python
grid_points = create_uniform_grid_points(
    height=frame_H, width=frame_W, grid_size=20, device="cpu"
)
```

**作用**：
- 创建**用户想要跟踪的点**
- 生成 `20×20 = 400` 个均匀分布的查询点
- 这些点会被保存到NPZ文件中，作为轨迹的起始点

**函数定义**（第780-811行）：
```python
def create_uniform_grid_points(height, width, grid_size=20, device="cuda"):
    """Create uniform grid points across the image.
    
    Args:
        grid_size (int): Grid size (20x20)
    
    Returns:
        torch.Tensor: Grid points [1, grid_size*grid_size, 3]
    """
    # 创建均匀网格
    y_coords = np.linspace(0, height - 1, grid_size)
    x_coords = np.linspace(0, width - 1, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    # 返回 20×20 = 400 个点
```

**特点**：
- **固定值**：硬编码为20
- **输出数量**：400个查询点
- **用途**：这些是**最终保存的轨迹点**

---

### 2.2 `support_grid_size` (支持点网格)

**位置**：`infer.py` 第547行和第564行

```python
# 第547行：传递给prepare_inputs
support_grid_size=16,

# 第564行：传递给inference函数作为grid_size参数
coords_seg, visibs_seg = inference(
    ...
    grid_size=support_grid_size,  # 16
    ...
)
```

**作用**：
- 在推理过程中，模型会**自动添加额外的支持点**
- 这些支持点用于**提高跟踪质量**，但**不会保存到最终结果中**

**在 `utils/inference_utils.py` 中的使用**（第46-77行）：

```python
def _inference_with_grid(
    *,
    model: torch.nn.Module,
    ...
    grid_size: int = 8,  # 这是support_grid_size
    **kwargs,
):
    if grid_size != 0:
        # 生成额外的支持点
        additional_queries = get_grid_queries(
            grid_size, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics
        )
        # 将支持点添加到查询点中
        query_point = torch.cat([query_point, additional_queries], dim=1)
        N_supports = additional_queries.shape[1]  # 16×16 = 256个支持点
    else:
        N_supports = 0
    
    # 模型推理（使用查询点+支持点）
    preds, train_data_list = model(...)
    
    # 只返回查询点的结果，丢弃支持点的结果
    N_total = preds.coords.shape[2]
    preds = preds.query_slice(slice(0, N_total - N_supports))
    return preds, train_data_list
```

**特点**：
- **固定值**：硬编码为16
- **输出数量**：`16×16 = 256` 个支持点
- **用途**：**辅助跟踪**，提高模型性能，但**不保存到结果中**

---

## 三、关键区别总结

| 特性 | `grid_size` (查询点) | `support_grid_size` (支持点) |
|------|---------------------|------------------------------|
| **值** | 20 | 16 |
| **生成点数** | 20×20 = 400 | 16×16 = 256 |
| **用途** | 用户想要跟踪的点 | 模型内部辅助点 |
| **是否保存** | ✅ 保存到NPZ文件 | ❌ 不保存，推理后丢弃 |
| **位置** | `create_uniform_grid_points()` | `_inference_with_grid()` |
| **作用** | 定义轨迹的起始点 | 提高跟踪质量 |

---

## 四、工作流程

```
1. 创建查询点（grid_size=20）
   ↓
   生成 400 个查询点（用户想要跟踪的点）
   ↓
2. 准备推理输入
   ↓
   将查询点转换为世界坐标
   ↓
3. 模型推理（support_grid_size=16）
   ↓
   - 自动生成 256 个支持点
   - 将查询点(400) + 支持点(256) = 656 个点一起输入模型
   - 模型使用支持点辅助跟踪查询点
   ↓
4. 返回结果
   ↓
   - 只返回 400 个查询点的轨迹
   - 丢弃 256 个支持点的结果
   ↓
5. 保存到NPZ
   ↓
   只保存 400 个查询点的轨迹
```

---

## 五、为什么需要支持点？

**支持点的作用**：
1. **提供上下文**：支持点帮助模型理解场景的3D结构
2. **提高精度**：更多的点可以帮助模型更好地估计相机运动和深度
3. **稳定跟踪**：支持点提供额外的约束，使跟踪更稳定

**为什么不保存支持点**：
- 支持点只是**辅助工具**，不是用户关心的结果
- 保存所有点会**增加存储空间**（656个点 vs 400个点）
- 用户只需要**查询点的轨迹**

---

## 六、代码位置总结

### `grid_size` (查询点)
- **定义**：`infer.py` 第493行
- **函数**：`create_uniform_grid_points()` (第780行)
- **值**：20（硬编码）
- **输出**：400个查询点

### `support_grid_size` (支持点)
- **定义**：`infer.py` 第547行
- **传递**：`infer.py` 第564行 → `inference()` → `_inference_with_grid()`
- **函数**：`utils/inference_utils.py` 第46行
- **值**：16（硬编码）
- **输出**：256个支持点（不保存）

---

## 七、总结

- **`grid_size=20`**：创建**400个查询点**，这些是**用户想要跟踪的点**，会**保存到结果中**
- **`support_grid_size=16`**：创建**256个支持点**，这些是**模型内部使用的辅助点**，用于**提高跟踪质量**，但**不会保存**

两者配合工作：
- 查询点定义**要跟踪什么**
- 支持点帮助模型**更好地跟踪**

