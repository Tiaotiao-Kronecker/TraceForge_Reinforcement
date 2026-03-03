#!/usr/bin/env python3
"""
检查policy_out.pkl中action的具体数据格式
"""
import pickle
import numpy as np
import os
import sys

def analyze_action_format(data_dir):
    """分析action的数据格式"""
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    
    if not os.path.exists(policy_path):
        print(f"文件不存在: {policy_path}")
        return
    
    print("=" * 70)
    print("policy_out.pkl 中 action 的详细数据格式")
    print("=" * 70)
    
    with open(policy_path, 'rb') as f:
        policy_out = pickle.load(f)
    
    print(f"\n1. 整体结构:")
    print(f"   类型: {type(policy_out)}")
    print(f"   长度: {len(policy_out)} (帧数)")
    
    if len(policy_out) == 0:
        print("   警告: policy_out为空")
        return
    
    # 分析第一帧
    frame = policy_out[0]
    print(f"\n2. 单帧数据结构:")
    print(f"   类型: {type(frame)}")
    print(f"   键: {list(frame.keys())}")
    
    if 'actions' not in frame:
        print("   警告: 未找到'actions'键")
        return
    
    actions = frame['actions']
    print(f"\n3. actions 字段详细格式:")
    print(f"   类型: {type(actions)}")
    
    if isinstance(actions, np.ndarray):
        print(f"   数据类型: numpy.ndarray")
        print(f"   shape: {actions.shape}")
        print(f"   dtype: {actions.dtype}")
        print(f"   维度数: {len(actions.shape)}D")
        
        if len(actions.shape) == 1:
            print(f"\n   格式说明:")
            print(f"     - 这是一个1D数组，长度为 {actions.shape[0]}")
            print(f"     - 每个元素代表一个控制维度")
            print(f"     - 通常用于机器人关节控制或末端执行器控制")
            
            print(f"\n   第一帧的action值:")
            print(f"     {actions}")
            print(f"\n   值统计:")
            print(f"     最小值: {actions.min():.6f}")
            print(f"     最大值: {actions.max():.6f}")
            print(f"     平均值: {actions.mean():.6f}")
            print(f"     标准差: {actions.std():.6f}")
            
            # 分析所有帧
            all_actions = [f['actions'] for f in policy_out if 'actions' in f]
            if len(all_actions) > 0:
                actions_array = np.array(all_actions)
                print(f"\n   所有帧的统计:")
                print(f"     完整数组shape: {actions_array.shape}  # (帧数={len(all_actions)}, action维度={actions.shape[0]})")
                print(f"     整体值范围: [{actions_array.min():.6f}, {actions_array.max():.6f}]")
                print(f"     每帧平均值范围: [{actions_array.mean(axis=-1).min():.6f}, {actions_array.mean(axis=-1).max():.6f}]")
                
                # 显示前3帧和后3帧
                print(f"\n   前3帧的action值:")
                for i in range(min(3, len(all_actions))):
                    print(f"     帧 {i}: {all_actions[i]}")
                
                if len(all_actions) > 3:
                    print(f"\n   后3帧的action值:")
                    for i in range(max(0, len(all_actions)-3), len(all_actions)):
                        print(f"     帧 {i}: {all_actions[i]}")
            
            # 推测action含义
            action_dim = actions.shape[0]
            print(f"\n   可能的含义（根据维度 {action_dim} 推测）:")
            if action_dim == 7:
                print(f"     → 7DOF机械臂控制")
                print(f"       可能是: [x, y, z, qx, qy, qz, qw] (位置+四元数)")
                print(f"       或: [joint1, joint2, ..., joint7] (7个关节角度)")
            elif action_dim == 6:
                print(f"     → 6DOF控制")
                print(f"       可能是: [x, y, z, roll, pitch, yaw] (位置+欧拉角)")
            elif action_dim == 14:
                print(f"     → 双臂机器人（每臂7DOF）")
                print(f"       可能是: [arm1_joint1, ..., arm1_joint7, arm2_joint1, ..., arm2_joint7]")
            elif action_dim == 8:
                print(f"     → 可能是7DOF + gripper控制")
            else:
                print(f"     → {action_dim}维控制空间")
                print(f"       需要根据具体机器人配置确定含义")
    
    # 其他字段
    print(f"\n4. 其他字段信息:")
    for key in frame.keys():
        if key != 'actions':
            val = frame[key]
            print(f"   {key}:")
            print(f"     类型: {type(val)}")
            if isinstance(val, np.ndarray):
                print(f"     shape: {val.shape}, dtype: {val.dtype}")
            elif isinstance(val, (list, tuple)):
                print(f"     长度: {len(val)}")
            else:
                print(f"     值: {str(val)[:50]}")
    
    print(f"\n5. 读取代码示例:")
    print(f"```python")
    print(f"import pickle")
    print(f"import numpy as np")
    print(f"")
    print(f"# 读取policy_out.pkl")
    print(f"with open('{policy_path}', 'rb') as f:")
    print(f"    policy_out = pickle.load(f)")
    print(f"")
    print(f"# 提取所有actions")
    print(f"actions = [frame['actions'] for frame in policy_out if 'actions' in frame]")
    print(f"actions_array = np.array(actions)  # shape: ({len(policy_out)}, {actions.shape[0] if isinstance(actions, np.ndarray) else '?'})")
    print(f"")
    print(f"# 访问第t帧的action")
    print(f"action_t = actions_array[t]  # shape: ({actions.shape[0] if isinstance(actions, np.ndarray) else '?'},)")
    print(f"```")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    analyze_action_format(data_dir)

