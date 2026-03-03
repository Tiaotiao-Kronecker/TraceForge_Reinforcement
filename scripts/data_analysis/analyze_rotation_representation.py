#!/usr/bin/env python3
"""
分析action中6D旋转的具体表示方式
"""
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys

def analyze_rotation_representation(data_dir):
    """分析旋转表示方式"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    all_actions = np.array([f['actions'] for f in policy_out])
    all_deltas = np.array([f['delta_robot_transform'] for f in policy_out])
    
    print("=" * 70)
    print("分析6D旋转的具体表示方式")
    print("=" * 70)
    
    print(f"\n【论文描述】")
    print(f"\"continuous 6D Cartesian end-effector motion, corresponding to relative changes in pose\"")
    print(f"→ 6维笛卡尔末端执行器运动，对应姿态的相对变化")
    print(f"→ 前3维确定是位置增量 (dx, dy, dz)")
    print(f"→ 后3维是旋转增量，但表示方式未明确说明")
    
    print(f"\n【可能的旋转表示方式】")
    print(f"1. 轴角（axis-angle / rotation vector）")
    print(f"2. 欧拉角（Euler angles: roll, pitch, yaw）")
    print(f"3. 其他表示方式")
    
    print(f"\n【验证方法】")
    print(f"对比action的旋转部分和delta_robot_transform的旋转矩阵")
    print(f"看哪种表示方式能最好地匹配")
    
    # 方法1: 从delta_robot_transform提取轴角，看是否匹配action
    print(f"\n【方法1: 从delta_robot_transform提取轴角】")
    matches_axis_angle = 0
    total_frames = 0
    
    for i in range(len(policy_out)):
        delta_transform = all_deltas[i]
        action_rot = all_actions[i, 3:6]
        delta_rot_matrix = delta_transform[:3, :3]
        
        # 跳过单位矩阵（几乎无旋转）
        if np.allclose(delta_rot_matrix, np.eye(3), atol=0.001):
            continue
        
        total_frames += 1
        
        try:
            # 从旋转矩阵提取轴角
            rot_from_matrix = Rotation.from_matrix(delta_rot_matrix)
            axis_angle_from_matrix = rot_from_matrix.as_rotvec()
            
            # 检查是否匹配
            if np.allclose(action_rot, axis_angle_from_matrix, atol=0.01):
                matches_axis_angle += 1
        except:
            pass
    
    if total_frames > 0:
        match_rate = matches_axis_angle / total_frames * 100
        print(f"  匹配率: {matches_axis_angle}/{total_frames} ({match_rate:.1f}%)")
        if match_rate > 80:
            print(f"  → 很可能是轴角表示！")
        else:
            print(f"  → 可能不是轴角表示")
    
    # 方法2: 将action旋转部分转换为旋转矩阵，与delta_robot_transform对比
    print(f"\n【方法2: 将action旋转转换为旋转矩阵】")
    errors_axis_angle = []
    errors_euler_xyz = []
    errors_euler_zyx = []
    
    for i in range(len(policy_out)):
        action_rot = all_actions[i, 3:6]
        delta_rot_matrix = all_deltas[i][:3, :3]
        
        # 跳过单位矩阵
        if np.allclose(delta_rot_matrix, np.eye(3), atol=0.001):
            continue
        
        try:
            # 轴角
            rot_from_axis_angle = Rotation.from_rotvec(action_rot)
            rot_matrix_aa = rot_from_axis_angle.as_matrix()
            error_aa = np.abs(delta_rot_matrix - rot_matrix_aa).max()
            errors_axis_angle.append(error_aa)
            
            # 欧拉角 xyz
            rot_from_euler_xyz = Rotation.from_euler('xyz', action_rot)
            rot_matrix_euler_xyz = rot_from_euler_xyz.as_matrix()
            error_euler_xyz = np.abs(delta_rot_matrix - rot_matrix_euler_xyz).max()
            errors_euler_xyz.append(error_euler_xyz)
            
            # 欧拉角 zyx
            rot_from_euler_zyx = Rotation.from_euler('zyx', action_rot)
            rot_matrix_euler_zyx = rot_from_euler_zyx.as_matrix()
            error_euler_zyx = np.abs(delta_rot_matrix - rot_matrix_euler_zyx).max()
            errors_euler_zyx.append(error_euler_zyx)
        except:
            pass
    
    print(f"\n  误差统计（与delta_robot_transform的旋转矩阵对比）:")
    if errors_axis_angle:
        print(f"  轴角表示:")
        print(f"    平均误差: {np.mean(errors_axis_angle):.6f}")
        print(f"    最大误差: {np.max(errors_axis_angle):.6f}")
        print(f"    最小误差: {np.min(errors_axis_angle):.6f}")
    
    if errors_euler_xyz:
        print(f"  欧拉角(xyz)表示:")
        print(f"    平均误差: {np.mean(errors_euler_xyz):.6f}")
        print(f"    最大误差: {np.max(errors_euler_xyz):.6f}")
        print(f"    最小误差: {np.min(errors_euler_xyz):.6f}")
    
    if errors_euler_zyx:
        print(f"  欧拉角(zyx)表示:")
        print(f"    平均误差: {np.mean(errors_euler_zyx):.6f}")
        print(f"    最大误差: {np.max(errors_euler_zyx):.6f}")
        print(f"    最小误差: {np.min(errors_euler_zyx):.6f}")
    
    # 方法3: 详细查看前几帧
    print(f"\n【方法3: 详细查看前5帧】")
    for i in range(min(5, len(policy_out))):
        action_rot = all_actions[i, 3:6]
        delta_transform = all_deltas[i]
        delta_rot_matrix = delta_transform[:3, :3]
        
        print(f"\n帧 {i}:")
        print(f"  action旋转部分: {action_rot}")
        
        if np.allclose(delta_rot_matrix, np.eye(3), atol=0.001):
            print(f"  → 几乎无旋转（单位矩阵）")
            continue
        
        try:
            # 从旋转矩阵提取轴角
            rot_from_matrix = Rotation.from_matrix(delta_rot_matrix)
            axis_angle_from_matrix = rot_from_matrix.as_rotvec()
            
            print(f"  从delta_robot_transform提取的轴角: {axis_angle_from_matrix}")
            print(f"  差异: {np.abs(action_rot - axis_angle_from_matrix)}")
            print(f"  最大差异: {np.abs(action_rot - axis_angle_from_matrix).max():.6f}")
            
            if np.allclose(action_rot, axis_angle_from_matrix, atol=0.01):
                print(f"  ✓ 匹配轴角表示")
            else:
                print(f"  ✗ 不匹配轴角表示")
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n【结论】")
    print(f"论文只说了\"6D Cartesian end-effector motion\"和\"relative changes in pose\"")
    print(f"但没有明确说明旋转的具体表示方式。")
    print(f"")
    print(f"基于数据分析:")
    if errors_axis_angle and errors_euler_xyz:
        if np.mean(errors_axis_angle) < np.mean(errors_euler_xyz) and np.mean(errors_axis_angle) < np.mean(errors_euler_zyx):
            print(f"→ 轴角表示误差最小，最可能是轴角（axis-angle）表示")
        elif np.mean(errors_euler_xyz) < np.mean(errors_axis_angle):
            print(f"→ 欧拉角(xyz)表示误差最小，可能是欧拉角表示")
        else:
            print(f"→ 需要进一步分析，可能不是简单的轴角或欧拉角")
    
    print(f"\n【建议】")
    print(f"由于论文未明确说明，建议:")
    print(f"1. 查看BridgeData V2的代码实现（如果有开源）")
    print(f"2. 查看相关论文的补充材料或附录")
    print(f"3. 通过实验验证（将action应用到机器人，看效果）")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    analyze_rotation_representation(data_dir)

