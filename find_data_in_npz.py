import numpy as np

# 加载npz数据
data = np.load('/home/peter/GameFormer-Planner/nuplan/processed_data/us-ma-boston_0a0aaeab5a25507e.npz')

# 列出文件中的所有键名
keys = ['map_name', 'token', 'ego_agent_past', 'ego_agent_future', 'neighbor_agents_past',
        'neighbor_agents_future', 'lanes', 'crosswalks', 'route_lanes']

# 对于每个键，打印出第一帧的数据（如果是数组的话）
for key in keys:
    print(f"Data under key '{key}':")
    try:
        # 取第一帧数据，假设数据是按帧存储的数组
        if data[key].shape[0] > 0:  # 检查是否有数据
            print(data[key][0])
        else:
            print("No data available for this key.")
    except IndexError:
        # 如果数据不是数组形式，直接打印
        print(data[key])
    except KeyError:
        # 如果键不存在
        print(f"No data available for key '{key}'.")
    print("\n")  # 打印空行以分隔不同的数据键
