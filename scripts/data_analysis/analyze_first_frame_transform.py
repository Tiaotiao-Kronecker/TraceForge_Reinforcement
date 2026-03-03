#!/usr/bin/env python3
"""
分析 output_bridge_depth_grid80 对应样本中 policy_out 首帧的 new_robot_transform 规律

用法:
    python analyze_first_frame_transform.py \
        --output_dir /home/wangchen/projects/TraceForge/output_bridge_depth_grid80 \
        --dataset_path /usr/data/dataset/opt/dataset_temp/bridge_depth \
        [--max_samples 500]  # 限制分析样本数，默认全部
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_policy_out(policy_path):
    """加载 policy_out.pkl"""
    with open(policy_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='/home/wangchen/projects/TraceForge/output_bridge_depth_grid80',
                        help='推理输出目录')
    parser.add_argument('--dataset_path', type=str,
                        default='/usr/data/dataset/opt/dataset_temp/bridge_depth',
                        help='数据集根路径（含 policy_out.pkl）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大分析样本数，默认全部')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset_path)

    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return

    # 获取推理完成的所有样本 ID
    traj_dirs = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    print(f"📁 推理 output 中共有 {len(traj_dirs)} 个样本")

    transforms = []
    missing = []
    errors = []

    for i, traj_id in enumerate(traj_dirs):
        if args.max_samples and i >= args.max_samples:
            break
        policy_path = dataset_path / traj_id / 'policy_out.pkl'
        if not policy_path.exists():
            missing.append(traj_id)
            continue
        try:
            policy_out = load_policy_out(policy_path)
            if not policy_out or len(policy_out) == 0:
                errors.append((traj_id, 'policy_out 为空'))
                continue
            T0 = policy_out[0].get('new_robot_transform')
            if T0 is None:
                errors.append((traj_id, '无 new_robot_transform'))
                continue
            T0 = np.asarray(T0)
            if T0.shape != (4, 4):
                errors.append((traj_id, f'形状异常: {T0.shape}'))
                continue
            transforms.append({'traj_id': traj_id, 'T': T0})
        except Exception as e:
            errors.append((traj_id, str(e)))

    n_ok = len(transforms)
    print(f"✅ 成功加载 {n_ok} 个样本的首帧 new_robot_transform")
    if missing:
        print(f"⚠️  缺失 policy_out.pkl: {len(missing)} 个 (如 {missing[:5]}...)")
    if errors:
        print(f"⚠️  加载异常: {len(errors)} 个")
        for tid, msg in errors[:5]:
            print(f"   {tid}: {msg}")

    if n_ok == 0:
        print("无有效样本可分析")
        return

    # 收集数据
    R_all = np.array([t['T'][:3, :3] for t in transforms])
    p_all = np.array([t['T'][:3, 3] for t in transforms])

    I4 = np.eye(4)
    identity_count = 0
    for t in transforms:
        if np.allclose(t['T'], I4, atol=1e-6):
            identity_count += 1

    # 旋转部分：是否为单位阵
    R_identity_count = sum(1 for R in R_all if np.allclose(R, np.eye(3), atol=1e-6))
    R_det = np.array([np.linalg.det(R) for R in R_all])

    print("\n" + "=" * 60)
    print("首帧 new_robot_transform 规律分析")
    print("=" * 60)

    # 1. 是否为单位阵
    print(f"\n1. 完整 4x4 为单位阵的样本数: {identity_count} / {n_ok} ({100*identity_count/n_ok:.2f}%)")

    # 2. 旋转部分
    print(f"\n2. 旋转矩阵 R[:3,:3] 为单位阵的样本数: {R_identity_count} / {n_ok} ({100*R_identity_count/n_ok:.2f}%)")
    print(f"   det(R) 统计: min={R_det.min():.6f}, max={R_det.max():.6f}, mean={R_det.mean():.6f} (应为 1.0)")

    # 3. 位置部分 - 基础统计
    print(f"\n3. 位置向量 p[:3] (x,y,z) 统计 (米):")
    print(f"   x: min={p_all[:,0].min():.4f}, max={p_all[:,0].max():.4f}, mean={p_all[:,0].mean():.4f}, std={p_all[:,0].std():.4f}")
    print(f"   y: min={p_all[:,1].min():.4f}, max={p_all[:,1].max():.4f}, mean={p_all[:,1].mean():.4f}, std={p_all[:,1].std():.4f}")
    print(f"   z: min={p_all[:,2].min():.4f}, max={p_all[:,2].max():.4f}, mean={p_all[:,2].mean():.4f}, std={p_all[:,2].std():.4f}")

    # 3b. 位置 x,y,z 单独分析：是否离散、分位数、分布
    print(f"\n3b. 位置 x,y,z 单独规律分析:")
    for dim, name in enumerate(['x', 'y', 'z']):
        vals = p_all[:, dim]
        # 唯一值数量（不同精度）
        n_unique_6 = len(np.unique(np.round(vals, 6)))
        n_unique_4 = len(np.unique(np.round(vals, 4)))
        n_unique_3 = len(np.unique(np.round(vals, 3)))
        n_unique_2 = len(np.unique(np.round(vals, 2)))
        print(f"   {name}: 唯一值数 (精度6位/4位/3位/2位) = {n_unique_6}/{n_unique_4}/{n_unique_3}/{n_unique_2}")
        # 分位数
        p25, p50, p75 = np.percentile(vals, [25, 50, 75])
        print(f"   {name}: 25%/50%/75% 分位 = {p25:.4f} / {p50:.4f} / {p75:.4f}")
        # 是否非负
        n_nonneg = np.sum(vals >= 0)
        print(f"   {name}: 非负样本占比 = {n_nonneg/n_ok*100:.2f}%")

    # 3c. 位置 x,y,z 离散化聚类（是否来自固定场景）
    print(f"\n3c. 位置离散化聚类 (四舍五入到 2cm 后的唯一值):")
    for prec in [0.02, 0.01, 0.005]:  # 2cm, 1cm, 0.5cm
        rounded = np.round(p_all / prec) * prec
        unique_pos = set(tuple(p) for p in rounded)
        print(f"   精度 {prec*100:.1f}cm: {len(unique_pos)} 个唯一位置（共 {n_ok} 样本）")

    # 3d. x,y,z 相关性
    cov = np.cov(p_all.T)
    corr = np.corrcoef(p_all.T)
    print(f"\n3d. 位置 x,y,z 相关系数矩阵:")
    print(f"       x      y      z")
    for i, n in enumerate(['x','y','z']):
        print(f"  {n}  {corr[i,0]:.4f}  {corr[i,1]:.4f}  {corr[i,2]:.4f}")

    # 4. 位置是否一致
    p_mean = p_all.mean(axis=0)
    p_std = p_all.std(axis=0)
    print(f"\n4. 位置均值和标准差:")
    print(f"   mean = {p_mean}")
    print(f"   std  = {p_std}")
    p_is_constant = np.all(p_std < 1e-6)
    print(f"   位置是否近似常数 (std < 1e-6): {p_is_constant}")

    # 5. 旋转是否一致（除单位阵外）
    if R_identity_count < n_ok:
        non_identity_R = R_all[[i for i in range(n_ok) if not np.allclose(R_all[i], np.eye(3), atol=1e-6)]]
        R_diff_from_I = np.linalg.norm(non_identity_R - np.eye(3), axis=(1, 2))
        print(f"\n5. 非单位旋转的 Frobenius 范数 ||R-I||: min={R_diff_from_I.min():.6f}, max={R_diff_from_I.max():.6f}")
    else:
        print("\n5. 所有样本旋转均为单位阵")

    # 6. 典型示例
    print("\n6. 典型示例 (前 3 个):")
    for t in transforms[:3]:
        print(f"   traj {t['traj_id']}:")
        print(f"      R:\n{t['T'][:3,:3]}")
        print(f"      p = {t['T'][:3, 3]}")
        print(f"      bottom = {t['T'][3, :]}")

    # 7. 聚类/唯一性
    unique_positions = set()
    for p in p_all:
        unique_positions.add(tuple(np.round(p, 6)))
    print(f"\n7. 首帧位置唯一值数量: {len(unique_positions)}")
    if len(unique_positions) <= 10:
        print(f"   唯一位置: {list(unique_positions)[:10]}")

    # 8. 相邻样本相似度（是否来自同一场景）
    if n_ok >= 2:
        consecutive_pos_diff = np.linalg.norm(np.diff(p_all, axis=0), axis=1)
        consecutive_R_diff = np.array([
            np.linalg.norm(R_all[i+1] - R_all[i], 'fro') for i in range(n_ok-1)
        ])
        print(f"\n8. 相邻样本 (traj_id 连续) 首帧差异:")
        print(f"   位置差 ||p[i+1]-p[i]||: min={consecutive_pos_diff.min():.6f}, max={consecutive_pos_diff.max():.6f}, mean={consecutive_pos_diff.mean():.6f}")
        print(f"   旋转差 ||R[i+1]-R[i]||_F: min={consecutive_R_diff.min():.6f}, max={consecutive_R_diff.max():.6f}, mean={consecutive_R_diff.mean():.6f}")
        # 相邻样本相似（可能同场景）的比例
        pos_similar = np.sum(consecutive_pos_diff < 0.01) / (n_ok - 1) * 100
        print(f"   相邻位置差 < 1cm 的比例: {pos_similar:.2f}%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
