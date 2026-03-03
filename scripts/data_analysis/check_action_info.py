#!/usr/bin/env python3
"""
检查bridge_depth数据集中的action信息
"""
import pickle
import os
import sys

def check_action_info(data_dir):
    """检查数据目录中的action信息"""
    print(f"=== 检查目录: {data_dir} ===\n")
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 1. 检查 lang.txt (任务描述)
    lang_path = os.path.join(data_dir, "lang.txt")
    if os.path.exists(lang_path):
        with open(lang_path, 'r') as f:
            lang_text = f.read().strip()
        print(f"📄 lang.txt (任务描述/指令):")
        print(f"   {lang_text}\n")
    
    # 2. 检查 agent_data.pkl
    agent_path = os.path.join(data_dir, "agent_data.pkl")
    if os.path.exists(agent_path):
        print(f"📄 agent_data.pkl:")
        try:
            with open(agent_path, 'rb') as f:
                data = pickle.load(f)
                print(f"   类型: {type(data)}")
                
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"   键 ({len(keys)}个): {keys}")
                    
                    # 检查action相关键
                    action_keys = [k for k in keys if 'action' in k.lower() or 'act' in k.lower()]
                    if action_keys:
                        print(f"   ✓ 找到action相关键: {action_keys}")
                        for key in action_keys:
                            val = data[key]
                            print(f"     - {key}: {type(val)}", end="")
                            try:
                                import numpy as np
                                if isinstance(val, np.ndarray):
                                    print(f", shape={val.shape}, dtype={val.dtype}")
                                elif isinstance(val, (list, tuple)):
                                    print(f", length={len(val)}")
                                else:
                                    print(f", value={str(val)[:50]}")
                            except:
                                print()
                elif isinstance(data, (list, tuple)):
                    print(f"   长度: {len(data)}")
                    if len(data) > 0:
                        print(f"   第一个元素类型: {type(data[0])}")
                        if isinstance(data[0], dict):
                            keys = list(data[0].keys())
                            print(f"   第一个元素的键: {keys}")
                            action_keys = [k for k in keys if 'action' in k.lower() or 'act' in k.lower()]
                            if action_keys:
                                print(f"   ✓ 找到action相关键: {action_keys}")
        except Exception as e:
            print(f"   ❌ 读取错误: {e}")
            print(f"   提示: 可能需要安装特定依赖（如numpy, sensor_msgs等）")
        print()
    
    # 3. 检查 policy_out.pkl
    policy_path = os.path.join(data_dir, "policy_out.pkl")
    if os.path.exists(policy_path):
        print(f"📄 policy_out.pkl:")
        try:
            with open(policy_path, 'rb') as f:
                data = pickle.load(f)
                print(f"   类型: {type(data)}")
                
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"   键 ({len(keys)}个): {keys}")
                    
                    action_keys = [k for k in keys if 'action' in k.lower() or 'act' in k.lower()]
                    if action_keys:
                        print(f"   ✓ 找到action相关键: {action_keys}")
                elif isinstance(data, (list, tuple)):
                    print(f"   长度: {len(data)}")
                    if len(data) > 0:
                        print(f"   第一个元素类型: {type(data[0])}")
                        try:
                            import numpy as np
                            if isinstance(data[0], np.ndarray):
                                print(f"   第一个元素: numpy.ndarray, shape={data[0].shape}, dtype={data[0].dtype}")
                                print(f"   ✓ 这可能是action序列（每个元素是一帧的action）")
                            elif isinstance(data[0], dict):
                                keys = list(data[0].keys())
                                print(f"   第一个元素的键: {keys}")
                                action_keys = [k for k in keys if 'action' in k.lower() or 'act' in k.lower()]
                                if action_keys:
                                    print(f"   ✓ 找到action相关键: {action_keys}")
                            else:
                                print(f"   第一个元素示例: {str(data[0])[:100]}")
                        except ImportError:
                            print(f"   第一个元素: {type(data[0])}")
        except Exception as e:
            print(f"   ❌ 读取错误: {e}")
        print()
    
    # 4. 检查 obs_dict.pkl
    obs_path = os.path.join(data_dir, "obs_dict.pkl")
    if os.path.exists(obs_path):
        print(f"📄 obs_dict.pkl:")
        try:
            with open(obs_path, 'rb') as f:
                data = pickle.load(f)
                print(f"   类型: {type(data)}")
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"   键 ({len(keys)}个): {keys[:10]}...")
        except Exception as e:
            print(f"   ❌ 读取错误: {e}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000"
    
    check_action_info(data_dir)

