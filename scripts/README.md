# 脚本目录

本目录包含项目中的所有脚本，按功能分类组织。

## 目录结构

```
scripts/
├── README.md                    # 本文件
├── batch_inference/              # 批量推理相关脚本和文档
│   ├── README.md
│   ├── batch_infer.py
│   ├── infer.py
│   ├── stress_test_batch_inference.py
│   ├── run_large_scale_stress_test.sh
│   ├── check_batch_inference_results.sh
│   ├── analyze_batch_failures.py
│   ├── verify_model_cache.py
│   ├── test_model_sharing.py
│   └── BATCH_INFERENCE_GUIDE.md
├── data_analysis/                # 数据分析相关脚本和文档
│   ├── README.md
│   ├── analyze_action_format.py
│   ├── analyze_dataset_structure.py
│   ├── analyze_rotation_representation.py
│   ├── analyze_transform_relationship.py
│   ├── check_action_format.py
│   ├── check_action_info.py
│   ├── verify_transform_relationship.py
│   └── action_data_format_analysis.md
├── visualization/                # 可视化相关脚本和文档
│   ├── README.md
│   ├── visualize_single_image.py
│   └── visualization_features.md
└── archived/                     # 归档脚本（已完成或不再使用）
    ├── find_widowx_urdf.py
    └── check_agent_data_for_urdf.py
```

## 核心脚本（保留在根目录）

以下脚本保留在项目根目录，因为它们是核心功能或环境设置：

- **setup_env.sh** - 环境设置脚本（应在根目录）

## 使用说明

每个子目录都有独立的README.md文件，说明该目录下的脚本和文档。

## BridgeV2 批量推理的深度处理与存储约定

针对 BridgeV2 数据集，`batch_inference/batch_bridge_v2.py` 会调用 `infer_bridge_v2.py` → `infer.py` 完成每条轨迹三相机的推理。深度相关的**处理时序**与**存储约定**如下。

### 深度处理时序（pipeline 概览）

1. **从 16-bit PNG 读取实测深度（mm → m）**
   - 数据集提供的深度图为 16-bit PNG，像素值 \(P(u,v)\in\{0,\dots,65535\}\)，单位约为毫米 (mm)。
   - 读入时转换为米：
     \[
     D_{\text{orig}}^{(t)}(u,v) = \frac{P^{(t)}(u,v)}{1000}
     \]
     在代码中对应：
     - `infer.py` 中 `load_video_and_mask(..., is_depth=True)`（读取 `depth_images*`）  
     - `/scripts/batch_inference/infer.py` 的 L775–L783：`np.array(img).astype(np.float32) / 1000.0`

2. **VGGT 对 RGB 预处理，确定工作分辨率 / 视野**
   - RGB 通过 `preprocess_image` 被 resize + center crop 到固定大小（默认约 518×518），决定网络的空间分辨率 \((H_1, W_1)\) 与有效视野。
   - 对应代码：
     - `/models/SpaTrackV2/models/vggt4track/utils/load_fn.py` 的 `preprocess_image`（L148–L201）

3. **在 VGGT 内，将实测深度插值到网络分辨率**
   - `infer.py` 中调用深度 + 位姿网络：
     - `/scripts/batch_inference/infer.py` L472–L480：`model_depth_pose(video_tensor, known_depth=depth_tensor, ...)`
   - 在 `VGGT4Wrapper.__call__` 中，如果提供了 `known_depth`，会先插值到网络输出分辨率 \((H_1, W_1)\)：
     \[
     D_{\text{known}}^{(t)}(x,y)
     = \sum_{i,j} w_{ij}(x,y)\, D_{\text{orig}}^{(t)}(u_{ij}, v_{ij})
     \]
     对应代码：
     - `/utils/video_depth_pose_utils.py` L134–L137：`torch.nn.functional.interpolate(..., size=depth_npy.shape[1:])`

4. **用实测深度对 VGGT 预测深度做尺度对齐（median scale）**
   - VGGT 预测深度记为 \(D_{\text{pred}}^{(t)}(x,y)\)，实测插值后深度为 \(D_{\text{known}}^{(t)}(x,y)\)。
   - 对每一帧 \(t\) 的有效像素集合 \(\mathcal{V}_t\)（两者都 > 0）：
     \[
     s_t = \frac{
       \operatorname{median}\left(D_{\text{known}}^{(t)}(x,y)\;\middle|\;(x,y)\in\mathcal{V}_t\right)
     }{
       \operatorname{median}\left(D_{\text{pred}}^{(t)}(x,y)\;\middle|\;(x,y)\in\mathcal{V}_t\right)
     }
     \]
   - 对所有帧取平均：
     \[
     s = \frac{1}{T}\sum_{t=1}^T s_t
     \]
   - 用该尺度对 VGGT 预测深度和外参平移进行缩放：
     \[
     D_{\text{aligned}}^{(t)}(x,y) = s \cdot D_{\text{pred}}^{(t)}(x,y),\quad
     \mathbf{T}'_t = s \cdot \mathbf{T}_t
     \]
   - 对应代码：
     - `/utils/video_depth_pose_utils.py` L23–L39：`align_video_depth_scale`
     - `/utils/video_depth_pose_utils.py` L134–141：调用对齐函数
     - `/utils/video_depth_pose_utils.py` L146：`extrs_npy[:, :3, 3] *= scale`

5. **可选：用实测深度替换对齐后的网络深度**
   - 若 `replace_with_known_depth=True`（BridgeV2 语义上是“以实测深度为准”），则最终用于三维跟踪与存盘的深度为：
     \[
     D_{\text{final}}^{(t)}(x,y) = D_{\text{known}}^{(t)}(x,y)
     \]
   - 对应代码：
     - `/utils/video_depth_pose_utils.py` L142–144：`depth_npy = known_depth`

6. **后续三维跟踪与存盘阶段**
   - `infer.py` 中的 `prepare_inputs` 对 `D_{\text{final}}` 做边缘滤波 / 填洞，不再改变分辨率：
     - `/scripts/batch_inference/infer.py` L885–L917：`_filter_one_depth(...)`
   - 三维跟踪直接在该分辨率下运行，最终：
     - 主 NPZ 中 `depths` 保存 \(D_{\text{final}}\)（转为 `float16`，单位米）
     - 每个 query 帧：
       - `*_raw.npz` 保存该帧的 \(D_{\text{final}}\)（`float32`，单位米）
       - PNG 深度图保存的是 \(D_{\text{final}}\) 在线性拉伸到 `[0, 65535]` 后的可视化版本

### 深度存储约定（按输出文件组织）

针对 BridgeV2 数据集，`batch_inference/batch_bridge_v2.py` 会在每个相机输出目录（如 `out_dir/{traj_id}/images0`）下写出深度相关结果：

- **主 NPZ 中的深度 (`imagesX.npz` 里的 `depths`)**
  - 记第 \(t\) 帧、像素 \((x,y)\) 的深度为 \(D_{\text{raw}}^{(t)}(x,y)\)，单位为米 (m)。
  - 主 NPZ 中保存的 `depths[t, y, x]` 即为 \(D_{\text{raw}}^{(t)}(x,y)\) 的浮点值（存盘时仅做 `float16` 压缩，不改变物理含义）。

- **每个 query 帧的可视化深度 PNG**
  - 对于某个 query 帧，原始深度图为 \(D_{\text{raw}}(x,y)\)（来自上述 `depths` 中对应帧）。
  - 只在 \(D_{\text{raw}}(x,y) > 0\) 的有效像素集合 \(\mathcal{V}\) 上统计最小/最大值：
    \[
    d_{\min} = \min_{(x,y)\in\mathcal{V}} D_{\text{raw}}(x,y), \quad
    d_{\max} = \max_{(x,y)\in\mathcal{V}} D_{\text{raw}}(x,y)
    \]
  - 若 \(|\mathcal{V}| > 0\) 且 \(d_{\max} > d_{\min}\)，则 PNG 中的 16-bit 深度值为：
    \[
    D_{\text{png}}(x,y)
    = \operatorname{clip}\left(
      \frac{D_{\text{raw}}(x,y) - d_{\min}}{d_{\max} - d_{\min}} \cdot 65535,\;
      0,\;65535
    \right)
    \]
    之后量化为 `uint16` 并以 `mode="I;16"` 保存为 `depth/{video_name}_{query_idx}.png`。
  - 若所有有效深度相同 (\(d_{\max} = d_{\min}\))，则整图设为常数 32767；若没有任何有效深度（\(|\mathcal{V}| = 0\)），整图为 0。
  - **注意**：这是逐帧自适应对比度拉伸，仅用于人眼可视化，不能从 PNG 可靠恢复米单位深度。

- **每个 query 帧的 raw 深度 (`*_raw.npz`)**
  - 同一 query 帧的原始深度 \(D_{\text{raw}}(x,y)\) 会以：
    \[
    \texttt{depth\_raw\_npz}[x,y] = D_{\text{raw}}(x,y)
    \]
    的形式保存在 `depth/{video_name}_{query_idx}_raw.npz` 中（键名为 `depth`）。
  - 这是用于后续几何重建 / 密集点云反投影的高精度深度来源，与主 NPZ 中 `depths` 对应帧一致。

## 快速链接

- [批量推理脚本](batch_inference/)
- [数据分析脚本](data_analysis/)
- [可视化脚本](visualization/)

