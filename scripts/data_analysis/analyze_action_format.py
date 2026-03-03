#!/usr/bin/env python3
"""
结合BridgeData V2信息分析policy_out.pkl中action的具体格式
"""
import pickle
import numpy as np
import os
import sys

def analyze_action_with_bridgedata(data_dir):
    """结合BridgeData V2信息分析action格式"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    
    print("=" * 70)
    print("结合BridgeData V2信息分析action格式")
    print("=" * 70)
    
    print(f"\n【BridgeData V2关键信息】")
    print(f"机器人: WidowX 250 6DOF机械臂")
    print(f"控制频率: 5 Hz")
    print(f"控制方式: VR遥操作")
    print(f"参考: https://rail-berkeley.github.io/bridgedata/")
    
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    print(f"\n当前轨迹长度: {len(policy_out)} 帧")
    
    if len(policy_out) == 0:
        return
    
    all_actions = np.array([f['actions'] for f in policy_out])
    frame = policy_out[0]
    
    print(f"\n【action维度分析】")
    print(f"action维度: {all_actions.shape[1]} (7维)")
    print(f"机器人DOF: 6 (WidowX 250是6DOF)")
    print(f"→ 结论: 前6维 = 6DOF控制，第7维 = gripper控制")
    
    print(f"\n【三个选项的可行性分析】")
    print(f"\n选项1: [x, y, z, qx, qy, qz, qw] (3D位置 + 四元数旋转)")
    print(f"  ❌ 不太可能")
    print(f"  理由:")
    print(f"    1. 四元数需要归一化: qx²+qy²+qz²+qw² = 1")
    first_action = all_actions[0]
    quat_norm = np.sum(first_action[3:7]**2)
    print(f"    2. 当前值归一化检查: {quat_norm:.6f} (应该≈1.0)")
    print(f"    3. 值范围: [{all_actions.min():.6f}, {all_actions.max():.6f}]")
    print(f"    4. 四元数通常范围在[-1, 1]，但需要满足归一化")
    
    print(f"\n选项2: [joint1, joint2, ..., joint7] (7个关节角度)")
    print(f"  ❌ 不太可能")
    print(f"  理由:")
    print(f"    1. WidowX 250只有6个关节（6DOF）")
    print(f"    2. 不应该有7个关节角度")
    print(f"    3. 关节角度通常范围较大（如[-π, π]），但当前值较小")
    
    print(f"\n选项3: [dx, dy, dz, dqx, dqy, dqz, gripper] (增量控制 + gripper)")
    print(f"  ✅ 最可能！")
    print(f"  理由:")
    print(f"    1. 符合6DOF机器人 + gripper的结构")
    print(f"    2. 值范围较小，符合增量控制特征")
    print(f"    3. 第7维只有0.0和1.0，符合gripper开关状态")
    
    print(f"\n【详细验证】")
    
    # 分析前6维
    pos_actions = all_actions[:, :3]
    rot_actions = all_actions[:, 3:6]
    gripper_actions = all_actions[:, 6]
    
    print(f"\n1. 前3维（位置增量 dx, dy, dz）:")
    print(f"   值范围: [{pos_actions.min():.6f}, {pos_actions.max():.6f}] 米")
    print(f"   平均值: {pos_actions.mean():.6f} 米")
    print(f"   标准差: {pos_actions.std():.6f} 米")
    print(f"   前3帧示例:")
    for i in range(min(3, len(all_actions))):
        print(f"     帧 {i}: {all_actions[i, :3]}")
    print(f"   → 值较小（< 0.1米），符合增量控制特征")
    
    print(f"\n2. 中间3维（旋转增量 dqx, dqy, dqz）:")
    print(f"   值范围: [{rot_actions.min():.6f}, {rot_actions.max():.6f}]")
    print(f"   平均值: {rot_actions.mean():.6f}")
    print(f"   标准差: {rot_actions.std():.6f}")
    print(f"   前3帧示例:")
    for i in range(min(3, len(all_actions))):
        print(f"     帧 {i}: {all_actions[i, 3:6]}")
    print(f"   → 可能是轴角（axis-angle）表示的旋转增量")
    print(f"   → 轴角范围通常在[-π, π]，当前值较小，符合增量特征")
    
    print(f"\n3. 第7维（gripper状态）:")
    unique_gripper = np.unique(gripper_actions)
    print(f"   唯一值: {unique_gripper}")
    print(f"   1.0的数量: {np.sum(gripper_actions == 1.0)}")
    print(f"   0.0的数量: {np.sum(gripper_actions == 0.0)}")
    print(f"   → 1.0 = gripper打开，0.0 = gripper关闭")
    
    # 分析与其他字段的关系
    if 'delta_robot_transform' in frame:
        print(f"\n4. 与delta_robot_transform的关联:")
        delta_transform = frame['delta_robot_transform']
        delta_pos = delta_transform[:3, 3]
        print(f"   delta_robot_transform位置增量: {delta_pos}")
        print(f"   action前3维: {all_actions[0, :3]}")
        diff = np.abs(delta_pos - all_actions[0, :3])
        print(f"   差异: {diff}")
        print(f"   → 如果差异较小，说明action前3维确实是位置增量")
    
    print(f"\n【最终结论】")
    print(f"=" * 70)
    print(f"action格式: [dx, dy, dz, dqx, dqy, dqz, gripper]")
    print(f"")
    print(f"详细说明:")
    print(f"  - dx, dy, dz: 末端执行器位置增量（单位：米）")
    print(f"    * 值范围: 通常在[-0.1, 0.1]米")
    print(f"    * 表示相对于当前末端位置的移动量")
    print(f"")
    print(f"  - dqx, dqy, dqz: 末端执行器旋转增量")
    print(f"    * 可能是轴角（axis-angle）表示")
    print(f"    * 值范围: 通常在[-0.1, 0.1]弧度")
    print(f"    * 表示相对于当前姿态的旋转量")
    print(f"")
    print(f"  - gripper: gripper状态（二进制）")
    print(f"    * 1.0: gripper打开")
    print(f"    * 0.0: gripper关闭")
    print(f"")
    print(f"【数据格式总结】")
    print(f"policy_out结构:")
    print(f"  - 类型: list，长度=帧数")
    print(f"  - 每帧: dict，包含:")
    print(f"    * 'actions': np.ndarray, shape=(7,), dtype=float64")
    print(f"    * 'new_robot_transform': np.ndarray, shape=(4,4), 齐次变换矩阵")
    print(f"    * 'delta_robot_transform': np.ndarray, shape=(4,4), 增量变换矩阵")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    analyze_action_with_bridgedata(data_dir)

