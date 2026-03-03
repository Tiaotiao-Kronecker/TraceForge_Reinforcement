#!/usr/bin/env python3
"""
深入分析transform之间的关系
"""
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys

def analyze_transform_relationship(data_dir):
    """分析transform之间的关系"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    print("=" * 70)
    print("深入分析：delta_robot_transform和new_robot_transform的关系")
    print("=" * 70)
    
    print(f"\n【检查不同的组合方式】")
    
    errors_method1 = []  # new[t] = new[t-1] @ delta[t]
    errors_method2 = []  # new[t] = delta[t] @ new[t-1]
    errors_method3 = []  # new[t] = new[t-1] @ inv(delta[t])
    
    for i in range(1, len(policy_out)):
        prev_new = policy_out[i-1]['new_robot_transform']
        curr_delta = policy_out[i]['delta_robot_transform']
        curr_new = policy_out[i]['new_robot_transform']
        
        # 方法1
        pred1 = prev_new @ curr_delta
        err1 = np.abs(pred1 - curr_new).max()
        errors_method1.append(err1)
        
        # 方法2
        pred2 = curr_delta @ prev_new
        err2 = np.abs(pred2 - curr_new).max()
        errors_method2.append(err2)
        
        # 方法3
        pred3 = prev_new @ np.linalg.inv(curr_delta)
        err3 = np.abs(pred3 - curr_new).max()
        errors_method3.append(err3)
    
    print(f"\n方法1: new[t] = new[t-1] @ delta[t]")
    print(f"  平均误差: {np.mean(errors_method1):.6f}")
    print(f"  最大误差: {np.max(errors_method1):.6f}")
    
    print(f"\n方法2: new[t] = delta[t] @ new[t-1]")
    print(f"  平均误差: {np.mean(errors_method2):.6f}")
    print(f"  最大误差: {np.max(errors_method2):.6f}")
    
    print(f"\n方法3: new[t] = new[t-1] @ inv(delta[t])")
    print(f"  平均误差: {np.mean(errors_method3):.6f}")
    print(f"  最大误差: {np.max(errors_method3):.6f}")
    
    print(f"\n【检查delta和实际变化的关系】")
    print(f"直接对比delta和new的变化量:")
    
    for i in range(1, min(5, len(policy_out))):
        prev_new = policy_out[i-1]['new_robot_transform']
        curr_new = policy_out[i]['new_robot_transform']
        curr_delta = policy_out[i]['delta_robot_transform']
        
        # 计算实际的位置变化（在世界坐标系中）
        actual_pos_change = curr_new[:3, 3] - prev_new[:3, 3]
        
        # delta中的位置
        delta_pos = curr_delta[:3, 3]
        
        print(f"\n帧 {i}:")
        print(f"  实际位置变化（世界坐标系）: {actual_pos_change}")
        print(f"  delta中的位置: {delta_pos}")
        print(f"  差异: {np.abs(actual_pos_change - delta_pos)}")
        
        # 检查旋转
        prev_rot = prev_new[:3, :3]
        curr_rot = curr_new[:3, :3]
        actual_rot_change = curr_rot @ prev_rot.T  # 相对旋转
        
        delta_rot = curr_delta[:3, :3]
        rot_diff = np.abs(actual_rot_change - delta_rot).max()
        print(f"  旋转差异: {rot_diff:.6f}")
        
        if rot_diff < 0.01:
            print(f"  → 旋转部分匹配！")
    
    print(f"\n【结论】")
    print(f"基于分析，delta_robot_transform可能:")
    print(f"1. 不是简单的增量变换")
    print(f"2. 可能在不同的坐标系中定义")
    print(f"3. 可能与actions有更复杂的关系")
    print(f"")
    print(f"建议:")
    print(f"- 查看BridgeData V2的代码实现")
    print(f"- 查看相关文档或论文补充材料")
    print(f"- 直接使用delta_robot_transform，而不是从actions推导")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    analyze_transform_relationship(data_dir)

