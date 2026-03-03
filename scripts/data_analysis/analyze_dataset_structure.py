#!/usr/bin/env python3
"""
分析BridgeData V2数据集中的所有信息
"""
import pickle
import numpy as np
import os
import sys

def analyze_dataset(data_dir):
    """分析数据集结构"""
    print("=" * 70)
    print("BridgeData V2 数据集完整结构分析")
    print("=" * 70)
    
    # 1. lang.txt
    lang_path = os.path.join(data_dir, "lang.txt")
    if os.path.exists(lang_path):
        with open(lang_path, 'r') as f:
            lang_text = f.read().strip()
        print(f"\n1. lang.txt (任务描述)")
        print(f"   内容: \"{lang_text}\"")
        print(f"   含义: 自然语言任务指令，描述机器人要执行的任务")
        print(f"   示例: \"pick up pot\", \"put the carrot on the plate\"")
    
    # 2. policy_out.pkl
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    if os.path.exists(policy_path):
        with open(policy_path, 'rb') as f:
            policy_out = pickle.load(f)
        print(f"\n2. policy_out.pkl (策略输出)")
        print(f"   帧数: {len(policy_out)}")
        if len(policy_out) > 0:
            print(f"   字段: {list(policy_out[0].keys())}")
            print(f"   含义:")
            print(f"     - actions: 7维action向量 [dx,dy,dz,dqx,dqy,dqz,gripper]")
            print(f"     - new_robot_transform: 4x4齐次变换矩阵（当前帧机器人位姿）")
            print(f"     - delta_robot_transform: 4x4齐次变换矩阵（增量变换）")
    
    # 3. obs_dict.pkl
    obs_path = os.path.join(data_dir, "obs_dict.pkl")
    if os.path.exists(obs_path):
        try:
            with open(obs_path, 'rb') as f:
                obs_dict = pickle.load(f)
            
            print(f"\n3. obs_dict.pkl (观察数据字典)")
            print(f"   类型: {type(obs_dict)}")
            
            if isinstance(obs_dict, dict):
                print(f"   键: {list(obs_dict.keys())}")
                for key in obs_dict.keys():
                    val = obs_dict[key]
                    print(f"\n   字段 '{key}':")
                    print(f"     类型: {type(val)}")
                    
                    if isinstance(val, list):
                        print(f"     长度: {len(val)}")
                        if len(val) > 0:
                            if isinstance(val[0], np.ndarray):
                                print(f"     第一帧shape: {val[0].shape}")
                                print(f"     第一帧dtype: {val[0].dtype}")
                            elif isinstance(val[0], dict):
                                print(f"     第一帧键: {list(val[0].keys())}")
                    
                    elif isinstance(val, np.ndarray):
                        print(f"     shape: {val.shape}")
                        print(f"     dtype: {val.dtype}")
                        if val.size > 0:
                            print(f"     值范围: [{val.min():.6f}, {val.max():.6f}]")
            
            elif isinstance(obs_dict, list):
                print(f"   长度: {len(obs_dict)}")
                if len(obs_dict) > 0:
                    frame = obs_dict[0]
                    if isinstance(frame, dict):
                        print(f"   第一帧键: {list(frame.keys())}")
                        for key in frame.keys():
                            val = frame[key]
                            print(f"     {key}: {type(val)}")
                            if isinstance(val, np.ndarray):
                                print(f"       shape: {val.shape}")
                                print(f"       dtype: {val.dtype}")
        
        except Exception as e:
            print(f"\n3. obs_dict.pkl")
            print(f"   读取错误: {e}")
    
    # 4. agent_data.pkl
    agent_path = os.path.join(data_dir, "agent_data.pkl")
    if os.path.exists(agent_path):
        try:
            with open(agent_path, 'rb') as f:
                agent_data = pickle.load(f)
            
            print(f"\n4. agent_data.pkl (Agent数据)")
            print(f"   类型: {type(agent_data)}")
            
            if isinstance(agent_data, dict):
                print(f"   键: {list(agent_data.keys())}")
                for key in list(agent_data.keys())[:10]:  # 只显示前10个
                    val = agent_data[key]
                    print(f"\n   字段 '{key}':")
                    print(f"     类型: {type(val)}")
                    
                    if isinstance(val, list):
                        print(f"     长度: {len(val)}")
                        if len(val) > 0:
                            if isinstance(val[0], np.ndarray):
                                print(f"     第一帧shape: {val[0].shape}")
                            elif isinstance(val[0], dict):
                                print(f"     第一帧键: {list(val[0].keys())[:5]}")
                    
                    elif isinstance(val, np.ndarray):
                        print(f"     shape: {val.shape}")
                        print(f"     dtype: {val.dtype}")
                    
                    elif isinstance(val, dict):
                        print(f"     子键数量: {len(val)}")
                        print(f"     子键示例: {list(val.keys())[:5]}")
        
        except Exception as e:
            print(f"\n4. agent_data.pkl")
            print(f"   读取错误: {e}")
            print(f"   说明: 可能需要ROS依赖才能读取")
    
    # 5. 图像目录
    print(f"\n5. 图像数据")
    
    # RGB图像
    for img_dir in ['images0', 'images1', 'images2']:
        img_path = os.path.join(data_dir, img_dir)
        if os.path.exists(img_path):
            img_files = sorted([f for f in os.listdir(img_path) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
            if len(img_files) > 0:
                print(f"\n   {img_dir}/ (RGB图像)")
                print(f"     图像数量: {len(img_files)}")
                print(f"     文件命名: {img_files[0]} ... {img_files[-1]}")
                print(f"     含义: RGB图像，视角{img_dir[-1]}")
                print(f"     说明: images0=主视角(over-the-shoulder), images1/2=随机视角")
    
    # 深度图像
    for depth_dir in ['depth_images0', 'depth_images1', 'depth_images2']:
        depth_path = os.path.join(data_dir, depth_dir)
        if os.path.exists(depth_path):
            depth_files = sorted([f for f in os.listdir(depth_path) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            if len(depth_files) > 0:
                print(f"\n   {depth_dir}/ (深度图像)")
                print(f"     图像数量: {len(depth_files)}")
                print(f"     文件命名: {depth_files[0]} ... {depth_files[-1]}")
                print(f"     含义: 深度图像，与对应RGB图像对齐")
                print(f"     说明: 16位PNG格式，单位可能是毫米或厘米")
    
    print(f"\n" + "=" * 70)
    print(f"总结")
    print(f"=" * 70)
    print(f"数据集包含以下信息:")
    print(f"1. 任务描述 (lang.txt)")
    print(f"2. 策略输出 (policy_out.pkl) - actions和机器人位姿")
    print(f"3. 观察数据 (obs_dict.pkl) - 可能包含传感器数据")
    print(f"4. Agent数据 (agent_data.pkl) - 可能需要ROS依赖")
    print(f"5. RGB图像 (images0/1/2) - 多视角RGB图像")
    print(f"6. 深度图像 (depth_images0/1/2) - 多视角深度图像")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    analyze_dataset(data_dir)

