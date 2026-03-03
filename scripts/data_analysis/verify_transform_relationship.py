#!/usr/bin/env python3
"""
验证transform之间的关系
"""
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys

def verify_relationships(data_dir):
    """验证transform之间的关系"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    print("=" * 70)
    print("验证：从上一帧new_robot_transform和delta_robot_transform计算当前帧")
    print("=" * 70)
    
    print(f"\n【理论关系1】")
    print(f"new_robot_transform[t] = new_robot_transform[t-1] @ delta_robot_transform[t]")
    
    errors1 = []
    for i in range(1, len(policy_out)):
        prev_new = policy_out[i-1]['new_robot_transform']
        curr_delta = policy_out[i]['delta_robot_transform']
        curr_new_actual = policy_out[i]['new_robot_transform']
        
        predicted_new = prev_new @ curr_delta
        error = np.abs(predicted_new - curr_new_actual).max()
        errors1.append(error)
    
    print(f"\n验证结果:")
    print(f"  验证帧数: {len(errors1)}")
    print(f"  平均误差: {np.mean(errors1):.6f}")
    print(f"  最大误差: {np.max(errors1):.6f}")
    
    if np.max(errors1) < 1e-5:
        print(f"  ✅ 关系1成立！")
    else:
        print(f"  ❌ 关系1不成立")
    
    print(f"\n【理论关系2】")
    print(f"从actions构建delta_robot_transform")
    
    errors2 = []
    for i in range(len(policy_out)):
        action = policy_out[i]['actions']
        actual_delta = policy_out[i]['delta_robot_transform']
        
        dx, dy, dz = action[:3]
        dqx, dqy, dqz = action[3:6]
        
        rot_vec = np.array([dqx, dqy, dqz])
        rot = Rotation.from_rotvec(rot_vec)
        
        predicted_delta = np.eye(4)
        predicted_delta[:3, :3] = rot.as_matrix()
        predicted_delta[:3, 3] = [dx, dy, dz]
        
        error = np.abs(predicted_delta - actual_delta).max()
        errors2.append(error)
    
    print(f"\n验证结果:")
    print(f"  总帧数: {len(errors2)}")
    print(f"  平均误差: {np.mean(errors2):.6f}")
    print(f"  最大误差: {np.max(errors2):.6f}")
    
    if np.max(errors2) < 1e-3:
        print(f"  ✅ 可以从actions构建delta_robot_transform！")
    else:
        print(f"  ❌ 无法从actions构建delta_robot_transform")
    
    print(f"\n【完整验证】")
    print(f"从actions和上一帧new_robot_transform计算当前帧new_robot_transform")
    
    errors3 = []
    for i in range(1, len(policy_out)):
        prev_new = policy_out[i-1]['new_robot_transform']
        action = policy_out[i]['actions']
        curr_new_actual = policy_out[i]['new_robot_transform']
        
        # 从actions构建delta
        dx, dy, dz = action[:3]
        dqx, dqy, dqz = action[3:6]
        rot_vec = np.array([dqx, dqy, dqz])
        rot = Rotation.from_rotvec(rot_vec)
        
        delta_from_action = np.eye(4)
        delta_from_action[:3, :3] = rot.as_matrix()
        delta_from_action[:3, 3] = [dx, dy, dz]
        
        predicted_new = prev_new @ delta_from_action
        error = np.abs(predicted_new - curr_new_actual).max()
        errors3.append(error)
        
        if i <= 3:
            print(f"\n帧 {i}:")
            print(f"  上一帧位置: {prev_new[:3, 3]}")
            print(f"  action增量: pos={action[:3]}, rot={action[3:6]}")
            print(f"  预测位置: {predicted_new[:3, 3]}")
            print(f"  实际位置: {curr_new_actual[:3, 3]}")
            print(f"  位置误差: {np.abs(predicted_new[:3, 3] - curr_new_actual[:3, 3]).max():.6f}")
    
    print(f"\n验证结果:")
    print(f"  验证帧数: {len(errors3)}")
    print(f"  平均误差: {np.mean(errors3):.6f}")
    print(f"  最大误差: {np.max(errors3):.6f}")
    
    if np.max(errors3) < 1e-3:
        print(f"\n✅ 完整验证成功！")
        print(f"   可以从actions和上一帧new_robot_transform计算当前帧的new_robot_transform")
    else:
        print(f"\n❓ 存在误差，需要进一步分析")
    
    print(f"\n【总结】")
    print(f"关系1: new_robot_transform[t] = new_robot_transform[t-1] @ delta_robot_transform[t]")
    print(f"  {'✅ 成立' if np.max(errors1) < 1e-5 else '❌ 不成立'}")
    print(f"")
    print(f"关系2: delta_robot_transform = build_transform_from_action(actions)")
    print(f"  {'✅ 成立' if np.max(errors2) < 1e-3 else '❌ 不成立'}")
    print(f"")
    print(f"关系3: new_robot_transform[t] = new_robot_transform[t-1] @ build_transform_from_action(actions[t])")
    print(f"  {'✅ 成立' if np.max(errors3) < 1e-3 else '❌ 不成立'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    verify_relationships(data_dir)

