# 3D Keypoint 动画可视化脚本实现逻辑

本文档详细分析 `visualize_3d_keypoint_animation.py` 的实现逻辑，便于维护和扩展。

---

## 1. 入口与参数

- **入口**：`main()` → 解析 `--episode_dir`（必填）、`--query_frame`、`--keypoint_stride`、`--dense_pointcloud`、`--dense_downsample`、`--normalize_camera`、`--port`。
- **目录约定**：`episode_dir` 为「单个相机的输出目录」，其**目录名**即 `video_name`（如 `images0`、`varied_camera_1`）。脚本期望该目录下存在：
  - `samples/<video_name>_<frame>.npz`：每个查询帧的轨迹样本；
  - `images/<video_name>_<frame>.png`：查询帧 RGB；
  - `depth/<video_name>_<frame>_raw.npz`：可选，用于非首帧密集点云；
  - `<video_name>.npz`：主 NPZ（coords/depths/intrinsics/extrinsics），用于密集点云与坐标转换。

---

## 2. 数据加载的两条分支

脚本根据「是否启用密集点云 + 是否首帧」选择**两种数据源**，二者互斥。

### 2.1 分支 A：首帧 + `--dense_pointcloud` + 主 NPZ 存在

**条件**：`use_dense_mode = (args.dense_pointcloud and main_npz.exists() and (query_frame is None or query_frame == 0))`。

**流程**：

1. 调用 `load_main_npz_for_dense(main_npz, downsample)`：
   - 从主 NPZ 读取 `coords` (T, N, 3)、`depths` (T, H, W)、`intrinsics`、`extrinsics`（w2c）。
   - 时间维度不一致时，取 `T = min(T_coords, T_depths, T_intr, T_extr)` 对齐。
   - RGB：优先用 NPZ 内 `video`；否则从 `images/` 下 `video_name_*.png` 按帧索引加载并归一化到 [0,1]。
   - Keypoint 轨迹：`keypoint_traj = coords.transpose(1,0,2)` → (N, T, 3)。
   - 密集点云：对每帧 t，用 `unproject_by_depth(depths[t], K[t], c2w[t])` 得到 (H, W, 3) 世界坐标，再按 `downsample` 下采样，过滤无效深度（z≤0 或 z≥10 或非有限），得到 `dense_per_frame[t]` 和对应 RGB → `dense_colors_per_frame[t]`。
2. 轨迹即主 NPZ 的 coords，与**视频帧一一对应**；`valid_steps` 设为全 True，`n_valid = T`。
3. Keypoint 采样：`indices = arange(0, n_total, stride)`，若 `n_total > 500` 且 stride>1 则强制 stride=1；`traj_sub = keypoint_traj[indices]`，`traj_full = keypoint_traj` 保留全量供 stride 切换。

**特点**：密集点云**每帧更新**（多帧），keypoint 轨迹与视频帧对齐，适合「首帧查询 + 要看到完整场景点云动画」的场景。

### 2.2 分支 B：非首帧，或未开密集点云，或主 NPZ 不存在

**流程**：

1. 扫描 `samples_dir` 下 `video_name_*.npz`，解析出 `(frame_idx, npz_path)` 列表，按 frame 排序。
2. 根据 `--query_frame` 选择要加载的 sample（未匹配则用第一个）。
3. `load_sample_npz(npz_path)`：
   - 读取 `traj` (N, T, 3)、`keypoints` (N, 2)、`frame_index`。
   - `valid_steps`：若 NPZ 有则直接用，否则用 `~np.all(np.isinf(traj), axis=(0,2))` 推断（某时间步全部 inf 则视为无效）。
4. **坐标格式兼容**：若检测到 traj 为「像素+深度」格式（数值范围像 u,v,z），且主 NPZ 存在，则用主 NPZ 该帧的 K、w2c 调用 `traj_pixel_depth_to_world` 将 traj 转为世界坐标。
5. 将无效时间步置为 nan：`traj_full[:, valid_steps, :] = traj[:, valid_steps, :]`，其余保持 nan。
6. Keypoint 采样与分支 A 类似（stride，n_total>500 时 stride=1），得到 `traj_sub`、`traj_full`、`n_show`、`n_valid`。
7. **非首帧密集点云**（可选）：若 `--dense_pointcloud` 且主 NPZ 存在，调用 `load_dense_for_query_frame(episode_dir, video_name, frame_idx, main_npz, downsample)`：
   - 读取 `depth/<video_name>_<frame_idx>_raw.npz` 的 depth 与主 NPZ 该帧的 K、w2c；
   - `unproject_by_depth` 得到单帧点云，下采样+有效掩码；
   - RGB 从 `images/<video_name>_<frame_idx>.png` 取。
   - 返回单帧点云与颜色，放入 `dense_per_frame = [dense_pts]`（**只有一帧**，静态）。

**特点**：轨迹来自 sample NPZ（可能经过 retarget，时间步与视频帧未必对齐）；非首帧时密集点云仅一帧静态显示。

---

## 3. 可选：归一化到首帧相机

若 `--normalize_camera` 且主 NPZ 存在：

- 取首帧（或当前 segment 起始帧）的 w2c：`extrs[seg_start]`。
- Keypoint 轨迹：`traj_sub = normalize_to_first_frame(traj_sub, w2c)`，即对每个点做 `w2c @ [x,y,z,1]^T`，得到首帧相机坐标系下的坐标。
- 密集点云：对 `dense_per_frame[t]` 的每个点同样用该 w2c 变换到首帧相机系。

这样所有几何都在「首帧相机」下表达，便于与 SpaTrackerV2 等对齐视角。

---

## 4. 颜色与运动量

- **Keypoint 颜色**：`colors = get_track_colors(traj_sub[:, 0, :])`。根据首帧（或第 0 时间步）的 3D 位置做归一化，再按到原点距离排序映射到 turbo colormap，得到 (n_show, 3) RGB [0,1]。每条轨迹一种颜色，全程不变。
- **运动量排序**：`motion_order, _ = compute_motion_rank(traj_sub)`。对每条轨迹计算弧长（相邻帧位移之和，仅 finite 段参与），按运动量降序排列索引。用于「仅显示动态轨迹」时只画前 K 条（K 由「动态比例」滑块决定）。

---

## 5. Viser 场景与句柄

- 创建 `ViserServer(port)`，`set_up_direction("-y")`。
- **Keypoint 点云**：`add_point_cloud("keypoints", points=pts_init, colors=colors, point_size=0.03, ...)`。`pts_init` 为第 0 时间步位置，invalid 点填 0。后续通过 `point_cloud_handle.points = ...` 和 `point_cloud_handle.point_size = gui_point_size.value` 更新。
- **密集点云**：仅当 `dense_per_frame is not None` 时创建 `add_point_cloud("dense_pointcloud", ...)`，初始为第 0 帧点云；更新时根据当前时间步 t 选 `dense_per_frame[dense_idx]`（首帧多帧时 dense_idx=t，非首帧单帧时 dense_idx=0）。

---

## 6. GUI 控件与回调

- **时间**：`gui_time` 滑块 [0, n_valid-1]，`gui_playing` 勾选、`gui_fps` 滑块控制自动播放。
- **Keypoint**：`gui_keypoint_stride`（采样步长）、只读的当前数/总数、`gui_point_size`（点大小）、`gui_show_keypoints`（显隐）。
- **密集点云**：`gui_dense_point_size`、`gui_show_dense`（仅在有密集点云时创建）。
- **轨迹线**：`gui_show_trails`、`gui_trail_full`（完整 0→末帧 或 0→当前帧）、`gui_trail_line_width`、`gui_trail_dynamic_only`、`gui_trail_dynamic_ratio`。

**关键回调**：

- `gui_time.on_update` → `update_display()` + `update_trails()`。
- `gui_keypoint_stride.on_update` → `apply_keypoint_stride()`：按新 stride 重算 `traj_sub`、`colors`、`motion_order`，**重建 keypoint 点云**（因点数可能变化），再 `update_display` 与 `update_trails`。
- `gui_point_size.on_update` → 仅更新 `point_cloud_handle.point_size`。
- 其余勾选/滑块 → 更新显隐或线宽或调用 `update_trails()`。

---

## 7. 每帧显示更新（update_display）

1. 当前时间步 `t = int(gui_time.value)`，裁剪到 [0, n_valid-1]。
2. **Keypoint 位置**：`pts = get_points_at_time(t)`：
   - 取 `traj_sub[:, t, :]`，将 `~np.isfinite(pts).any(axis=1)` 的行置为 0，避免 Viser 收到 inf/nan。
3. 写回 `point_cloud_handle.points`、`point_cloud_handle.colors`、`point_cloud_handle.point_size`、`point_cloud_handle.visible`。
4. **密集点云**：若有，选 `dense_idx = 0 if len(dense_per_frame)==1 else t`，更新 dense 点云的 points/colors/point_size/visible。

---

## 8. 轨迹线（update_trails）

- 若未勾选「显示轨迹线」，清空已有 trail 句柄并 return。
- `t_end`：勾选「完整轨迹」则为 n_valid-1，否则为当前 t。
- **参与绘制的轨迹**：若「仅动态轨迹」勾选，取运动量最大的前 `n_dynamic = max(10, n_show * gui_trail_dynamic_ratio)` 条；否则全部。
- 对每条轨迹 i、每一段 [j, j+1]：若 `p0=traj_sub[i,j,:]` 与 `p1=traj_sub[i,j+1,:]` 均 `np.isfinite`，则加入线段列表；否则跳过（**整段 inf 的轨迹不会产生任何线段**）。
- 用 `add_line_segments` 一次性添加当前所有线段，线宽由滑块控制；旧线段先 `remove()` 再清空 `trail_handles`，避免堆积。

---

## 9. 无效/全 inf 轨迹的处理汇总

| 环节 | 处理方式 |
|------|----------|
| **valid_steps** | NPZ 无该键时用「某时间步是否全部 inf」推断；有则直接用。 |
| **get_points_at_time** | 当前时间步某轨迹若含 inf/nan，该点被置为 (0,0,0)，仍会画出一个点（在原点）。 |
| **轨迹线** | 仅当 p0、p1 都 finite 才画段；整条轨迹全 inf 则不会画任何线段。 |
| **运动量** | 非 finite 的位移不计入弧长，全 inf 轨迹运动量为 0，排在「动态轨迹」末尾。 |

---

## 10. 动画循环

主线程在 `main()` 末尾：

```python
while True:
    if gui_playing.value and n_valid > 1:
        new_t = (int(gui_time.value) + 1) % n_valid
        gui_time.value = new_t
    time.sleep(1.0 / max(1, gui_fps.value))
```

通过不断改写 `gui_time.value` 触发 `on_update`，从而驱动 `update_display` 和 `update_trails`，实现动画。Ctrl+C 退出。

---

## 11. 小结

- **双数据源**：首帧+密集点云时用主 NPZ（帧对齐轨迹+多帧密集点云）；否则用 sample NPZ（单查询帧轨迹，可选单帧密集点云）。
- **坐标**：支持世界坐标或首帧相机坐标（`--normalize_camera`）；支持 sample 内 (u,v,z) 转世界坐标。
- **显示层**：Keypoint 点云（可调 stride/点大小/显隐）+ 可选密集点云（可调点大小/显隐）+ 可选轨迹线（可限动态比例、线宽、完整/当前段）。
- **鲁棒性**：valid_steps 缺失时从 inf 推断；渲染时 inf 点置零、轨迹线只画 finite 段，避免崩溃与错误线段。
