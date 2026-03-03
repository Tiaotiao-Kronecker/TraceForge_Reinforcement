#!/usr/bin/env python3
"""
检查 agent_data.pkl 中是否包含URDF文件或robot_description参数
"""

import pickle
import os
import sys
import re
from collections import defaultdict

def check_for_urdf_content(data, path="", depth=0, max_depth=10):
    """
    递归检查数据中是否包含URDF相关内容
    """
    urdf_indicators = []
    
    if depth > max_depth:
        return urdf_indicators
    
    # 检查字符串内容
    if isinstance(data, (str, bytes)):
        content = data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
        
        # URDF关键词
        urdf_keywords = [
            'robot_description',
            'robot_description_semantic',
            'robot_description_kinematics',
            '<robot',
            '</robot>',
            '<link',
            '<joint',
            'urdf',
            'xacro',
        ]
        
        # 检查是否包含URDF XML标签
        if '<robot' in content and '</robot>' in content:
            # 提取robot标签内容
            robot_match = re.search(r'<robot[^>]*>.*?</robot>', content, re.DOTALL)
            if robot_match:
                urdf_indicators.append({
                    'type': 'URDF_XML',
                    'path': path,
                    'length': len(robot_match.group()),
                    'preview': robot_match.group()[:500]  # 前500字符
                })
        
        # 检查关键词
        for keyword in urdf_keywords:
            if keyword.lower() in content.lower():
                urdf_indicators.append({
                    'type': 'KEYWORD',
                    'keyword': keyword,
                    'path': path,
                    'context': content[max(0, content.lower().find(keyword.lower())-50):
                                      min(len(content), content.lower().find(keyword.lower())+50)]
                })
    
    # 检查字典
    elif isinstance(data, dict):
        # 检查键名
        for key, value in data.items():
            key_str = str(key).lower()
            
            # 检查键名是否包含URDF相关
            if any(kw in key_str for kw in ['urdf', 'robot_description', 'description']):
                urdf_indicators.append({
                    'type': 'KEY_NAME',
                    'key': key,
                    'path': f"{path}.{key}" if path else str(key),
                    'value_type': type(value).__name__,
                    'value_preview': str(value)[:200] if not isinstance(value, (dict, list)) else f"<{type(value).__name__}>"
                })
            
            # 递归检查值
            new_path = f"{path}.{key}" if path else str(key)
            urdf_indicators.extend(check_for_urdf_content(value, new_path, depth+1, max_depth))
    
    # 检查列表
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            urdf_indicators.extend(check_for_urdf_content(item, new_path, depth+1, max_depth))
    
    # 检查对象属性（ROS消息对象）
    elif hasattr(data, '__dict__'):
        # 检查属性名
        for attr_name in dir(data):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(data, attr_name)
                attr_str = str(attr_name).lower()
                
                if any(kw in attr_str for kw in ['urdf', 'robot_description', 'description']):
                    urdf_indicators.append({
                        'type': 'ATTRIBUTE',
                        'attribute': attr_name,
                        'path': f"{path}.{attr_name}" if path else attr_name,
                        'value_type': type(attr_value).__name__,
                        'value_preview': str(attr_value)[:200] if not isinstance(attr_value, (dict, list)) else f"<{type(attr_value).__name__}>"
                    })
                
                # 递归检查属性值
                new_path = f"{path}.{attr_name}" if path else attr_name
                urdf_indicators.extend(check_for_urdf_content(attr_value, new_path, depth+1, max_depth))
            except:
                pass
    
    return urdf_indicators

def analyze_agent_data(agent_data_path):
    """
    分析agent_data.pkl文件
    """
    print("=" * 80)
    print("检查 agent_data.pkl 中是否包含URDF文件")
    print("=" * 80)
    
    if not os.path.exists(agent_data_path):
        print(f"❌ 文件不存在: {agent_data_path}")
        return
    
    print(f"\n📁 文件路径: {agent_data_path}")
    print(f"📊 文件大小: {os.path.getsize(agent_data_path) / 1024:.2f} KB")
    
    # 方法1: 直接搜索二进制文件中的URDF字符串
    print("\n" + "=" * 80)
    print("方法1: 在二进制文件中搜索URDF字符串")
    print("=" * 80)
    
    with open(agent_data_path, 'rb') as f:
        binary_content = f.read()
    
    # 搜索URDF相关字符串
    urdf_strings = [
        b'robot_description',
        b'robot_description_semantic',
        b'robot_description_kinematics',
        b'<robot',
        b'</robot>',
        b'<link',
        b'<joint',
        b'.urdf',
        b'.xacro',
    ]
    
    found_strings = []
    for search_str in urdf_strings:
        if search_str in binary_content:
            # 找到所有出现位置
            positions = []
            start = 0
            while True:
                pos = binary_content.find(search_str, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            found_strings.append({
                'string': search_str.decode('utf-8', errors='ignore'),
                'count': len(positions),
                'positions': positions[:5]  # 只保存前5个位置
            })
    
    if found_strings:
        print(f"\n✅ 在二进制文件中找到 {len(found_strings)} 种URDF相关字符串:\n")
        for item in found_strings:
            print(f"  '{item['string']}': 出现 {item['count']} 次")
            if item['positions']:
                print(f"    位置: {item['positions']}")
    else:
        print("\n❌ 在二进制文件中未找到URDF相关字符串")
    
    # 方法2: 尝试加载pickle（可能需要ROS环境）
    print("\n" + "=" * 80)
    print("方法2: 尝试加载pickle文件（可能需要ROS环境）")
    print("=" * 80)
    
    try:
        print("\n正在加载 agent_data.pkl...")
        with open(agent_data_path, 'rb') as f:
            agent_data = pickle.load(f)
        
        print(f"✅ 加载成功")
        print(f"📦 数据类型: {type(agent_data).__name__}")
        
        # 分析数据结构
        print("\n" + "=" * 80)
        print("数据结构分析")
        print("=" * 80)
        
        if isinstance(agent_data, dict):
            print(f"\n字典包含 {len(agent_data)} 个键:")
            for i, key in enumerate(list(agent_data.keys())[:20], 1):
                value = agent_data[key]
                print(f"  {i}. '{key}': {type(value).__name__}")
                if isinstance(value, (str, bytes)):
                    print(f"     长度: {len(value)} 字符")
                elif isinstance(value, (dict, list)):
                    print(f"     元素数: {len(value)}")
            if len(agent_data) > 20:
                print(f"  ... 还有 {len(agent_data) - 20} 个键")
        
        elif isinstance(agent_data, list):
            print(f"\n列表包含 {len(agent_data)} 个元素")
            if len(agent_data) > 0:
                print(f"第一个元素类型: {type(agent_data[0]).__name__}")
        
        # 检查URDF内容
        print("\n" + "=" * 80)
        print("URDF内容检查")
        print("=" * 80)
        
        print("\n正在搜索URDF相关内容...")
        urdf_indicators = check_for_urdf_content(agent_data)
        
        if urdf_indicators:
            print(f"\n✅ 找到 {len(urdf_indicators)} 个URDF相关指示器\n")
            
            # 按类型分组
            by_type = defaultdict(list)
            for indicator in urdf_indicators:
                by_type[indicator['type']].append(indicator)
            
            # 显示结果
            for indicator_type, items in by_type.items():
                print(f"\n【{indicator_type}】 ({len(items)} 个)")
                for item in items[:10]:  # 只显示前10个
                    print(f"  路径: {item.get('path', 'N/A')}")
                    if 'keyword' in item:
                        print(f"    关键词: {item['keyword']}")
                    if 'key' in item:
                        print(f"    键名: {item['key']}")
                    if 'attribute' in item:
                        print(f"    属性: {item['attribute']}")
                    if 'preview' in item:
                        preview = item['preview'].replace('\n', ' ')
                        print(f"    预览: {preview[:100]}...")
                    if 'context' in item:
                        context = item['context'].replace('\n', ' ')
                        print(f"    上下文: {context[:100]}...")
                    print()
                
                if len(items) > 10:
                    print(f"  ... 还有 {len(items) - 10} 个")
            
            # 检查是否有完整的URDF XML
            urdf_xml = [i for i in urdf_indicators if i['type'] == 'URDF_XML']
            if urdf_xml:
                print("\n" + "=" * 80)
                print("🎉 发现完整的URDF XML内容！")
                print("=" * 80)
                for item in urdf_xml:
                    print(f"\n路径: {item['path']}")
                    print(f"长度: {item['length']} 字符")
                    print(f"\n前500字符预览:")
                    print(item['preview'])
                    print("\n" + "-" * 80)
        else:
            print("\n❌ 未找到URDF相关内容")
            print("\n可能的原因:")
            print("1. URDF文件不在agent_data.pkl中")
            print("2. URDF以其他格式存储（如二进制、压缩等）")
            print("3. URDF存储在ROS参数服务器中，未包含在数据中")
        
        # 额外检查：查找可能的参数消息
        print("\n" + "=" * 80)
        print("额外检查：ROS参数相关")
        print("=" * 80)
        
        if isinstance(agent_data, dict):
            param_keys = [k for k in agent_data.keys() if 'param' in str(k).lower() or 'config' in str(k).lower()]
            if param_keys:
                print(f"\n找到 {len(param_keys)} 个可能的参数相关键:")
                for key in param_keys[:10]:
                    print(f"  - {key}")
        
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: 可能需要ROS环境才能正确加载ROS消息对象")

def main():
    # 默认路径
    default_path = "/usr/data/dataset/opt/dataset_temp/bridge_depth/00000/agent_data.pkl"
    
    if len(sys.argv) > 1:
        agent_data_path = sys.argv[1]
    else:
        agent_data_path = default_path
    
    analyze_agent_data(agent_data_path)

if __name__ == "__main__":
    main()

