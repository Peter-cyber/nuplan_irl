import numpy as np
import csv
import matplotlib.pyplot as plt
from data_process import create_map_raster, create_ego_raster, create_agents_raster, draw_trajectory

# 加载npz数据
data = np.load('/home/peter/GameFormer-Planner/nuplan/processed_data/us-nv-las-vegas-strip_449c72fc98805e7a.npz')

# 提取所需数据
ego_past = data['ego_agent_past']
ego_future = data['ego_agent_future']
print('ego past shape is', ego_past.shape, 'ego future shape is', ego_future.shape)
neighbors_past = data['neighbor_agents_past']
neighbors_future = data['neighbor_agents_future']
print('neighbors past shape is', neighbors_past.shape, 'neighbors future shape is', neighbors_future.shape)
lanes = data['lanes']
route_lanes = data['route_lanes']
crosswalks = data['crosswalks']
print('lanes shape is', lanes.shape, 'route lanes shape is', route_lanes.shape, 'crosswalks shape is', crosswalks.shape)

def extract_features(ego_past, ego_future, neighbors_past, neighbors_future, lanes, route_lanes, crosswalks,
                     ego_length=100, neighbors_length=4000, lanes_length=14000, route_lanes_length=1500, crosswalks_length=450):
    # 提取ego车辆的过去轨迹特征
    ego_past_features = ego_past.reshape(-1)[:ego_length]
    ego_past_features = np.pad(ego_past_features, (0, ego_length - ego_past_features.shape[0]), 'constant')

    # 提取ego车辆的未来轨迹特征
    ego_future_features = ego_future.reshape(-1)[:ego_length]
    ego_future_features = np.pad(ego_future_features, (0, ego_length - ego_future_features.shape[0]), 'constant')

    # 提取邻居车辆的过去轨迹特征
    neighbors_past_features = neighbors_past.reshape(-1)[:neighbors_length]
    neighbors_past_features = np.pad(neighbors_past_features, (0, neighbors_length - neighbors_past_features.shape[0]), 'constant')

    # 提取邻居车辆的未来轨迹特征
    neighbors_future_features = neighbors_future.reshape(-1)[:neighbors_length]
    neighbors_future_features = np.pad(neighbors_future_features, (0, neighbors_length - neighbors_future_features.shape[0]), 'constant')

    # 提取车道线特征
    lanes_features = lanes.reshape(-1)[:lanes_length]
    lanes_features = np.pad(lanes_features, (0, lanes_length - lanes_features.shape[0]), 'constant')

    # 提取路径车道线特征
    route_lanes_features = route_lanes.reshape(-1)[:route_lanes_length]
    route_lanes_features = np.pad(route_lanes_features, (0, route_lanes_length - route_lanes_features.shape[0]), 'constant')

    # 提取人行横道特征
    crosswalks_features = crosswalks.reshape(-1)[:crosswalks_length]
    crosswalks_features = np.pad(crosswalks_features, (0, crosswalks_length - crosswalks_features.shape[0]), 'constant')

    # 将所有特征拼接在一起
    traj_features = np.concatenate((ego_past_features, ego_future_features, neighbors_past_features, neighbors_future_features,
                                    lanes_features, route_lanes_features, crosswalks_features))


    return traj_features

def normalize_and_compute_likeness(traj_features, ego_future, ego_past):
    # 归一化特征
    max_feat = np.max(traj_features, axis=0, keepdims=True)
    max_feat[max_feat == 0] = 1  # 避免除以0
    traj_features /= max_feat

    # 计算human_likeness
    human_likeness = np.exp(-np.linalg.norm(ego_future[:2] - ego_past[-2:]))

    return traj_features, human_likeness, max_feat

def polynomial_trajectory_sampler(current_state, target_state, obstacle_states, T=1):
    # 从起点状态到目标状态采样一条多项式轨迹
    # current_state: 起点状态 [x, y, v, heading]
    # target_state: 目标状态 [x, y, v, heading]
    # obstacle_states: 障碍物状态列表,每个元素为 [x, y, v, heading]
    # T: 轨迹持续时间

    # 提取起点和终点状态
    start_x, start_y, start_v, start_heading = current_state
    end_x, end_y, end_v, end_heading = target_state

    # 生成随机的中间速度和航向角
    mid_v = np.random.uniform(min(start_v, end_v), max(start_v, end_v))
    mid_heading = np.random.uniform(min(start_heading, end_heading), max(start_heading, end_heading))

    # 纵向轨迹的边界条件
    x_conditions = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 2, 0, 0],
        [T**4, T**3, T**2, T, 1],
        [4*T**3, 3*T**2, 2*T, 1, 0]
    ])
    x_boundary = np.array([start_x, start_v * np.cos(start_heading), mid_v * np.sin(mid_heading),
                           end_x, end_v * np.cos(end_heading)])

    # 横向轨迹的边界条件
    y_conditions = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [T**5, T**4, T**3, T**2, T, 1],
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]
    ])
    y_boundary = np.array([start_y, start_v * np.sin(start_heading), mid_v * np.cos(mid_heading),
                           end_y, end_v * np.sin(end_heading), 0])

    # 解多项式系数
    x_coeffs = np.linalg.solve(x_conditions, x_boundary)
    y_coeffs = np.linalg.solve(y_conditions, y_boundary)

    # 采样轨迹点
    times = np.linspace(0, T, num=T*10)
    traj_x = np.polyval(x_coeffs, times)
    traj_y = np.polyval(y_coeffs, times)
    traj_v = np.sqrt(np.gradient(traj_x, times)**2 + np.gradient(traj_y, times)**2)
    traj_heading = np.arctan2(np.gradient(traj_y, times), np.gradient(traj_x, times))

    # 将轨迹点组装成轨迹
    sampled_traj = np.column_stack((traj_x, traj_y, traj_v, traj_heading))
    return sampled_traj

def plot_scenario(ego_past, ego_future, neighbors_past, neighbors_future, lanes, route_lanes, crosswalks, sampled_trajs):
    # Create map layers
    create_map_raster(lanes, crosswalks, route_lanes)

    # Create agent layers
    create_ego_raster(ego_past[-1])
    create_agents_raster(neighbors_past[:, -1])

    # Draw past and future trajectories
    draw_trajectory(ego_past, neighbors_past)
    draw_trajectory(ego_future, neighbors_future)

    # Draw sampled trajectories
    for traj in sampled_trajs:
        plt.plot(traj[:, 0], traj[:, 1], 'y.', linewidth=3) # 在这里修改绘图
        plt.plot(traj[:, 0], traj[:, 1], 'y-', linewidth=1)

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def weighted_trajectory_sampler(current_state, target_state, obstacle_states, theta, num_samples=10):
    sampled_trajs = []
    for _ in range(num_samples):
        sampled_traj = polynomial_trajectory_sampler(current_state, target_state, obstacle_states)
        traj_features = extract_features(sampled_traj[:, :4], ego_future[-1], neighbors_past[:, -1], neighbors_future[:, -1], lanes[-1], route_lanes[-1], crosswalks[-1])
        traj_features, _, _ = normalize_and_compute_likeness(traj_features, ego_future[-1], sampled_traj[-1, :4])
        reward = np.dot(traj_features, theta)
        sampled_trajs.append((sampled_traj, reward))

    # 根据reward的大小对采样轨迹进行排序
    sampled_trajs.sort(key=lambda x: x[1], reverse=True)

    # 返回reward最大的轨迹
    return sampled_trajs[0][0]

# 在 weighted_trajectory_sampler 函数中,我们首先使用 polynomial_trajectory_sampler 函数生成多条采样轨迹。
# 然后,对于每条采样轨迹,我们提取其特征,并使用训练得到的权重 theta 计算 reward。
# 最后,我们根据 reward 的大小对采样轨迹进行排序,并返回 reward 最大的轨迹作为最终的采样结果。

# 定义超参数
n_iters = 100
lr = 0.01
lam = 0.1

# 获取特征数量
sample_features = extract_features(ego_past[0], ego_future[0], neighbors_past[:, 0], neighbors_future[:, 0], lanes[0], route_lanes[0], crosswalks[0])
sample_features, _, max_feat = normalize_and_compute_likeness(sample_features, ego_future[0], ego_past[0])
feature_num = sample_features.shape[-1]

# 初始化权重
theta_personal = np.random.normal(0, 0.05, size=feature_num)

# 创建训练日志
with open('maxent_irl_training_log_personal.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['iteration', 'human feature', 'trajectory feature', 'feature norm', 'weights', 'human likeness', 'likelihood'])

# 确定最小的时间步数
min_steps = min(ego_past.shape[0], neighbors_past.shape[1], lanes.shape[0], route_lanes.shape[0], crosswalks.shape[0])
# print(min_steps)

# 个性化建模
for iteration in range(n_iters):
    # print('iteration: {}/{}'.format(iteration + 1, n_iters)) # 训练进度可视化

    for i in range(min_steps):
        traj_features = extract_features(ego_past[i], ego_future[i], neighbors_past[:, i], neighbors_future[:, i], lanes[i], route_lanes[i], crosswalks[i])
        traj_features, human_likeness, _ = normalize_and_compute_likeness(traj_features, ego_future[i], ego_past[i])

        # 采样多条轨迹
        sampled_trajs = []
        current_state = np.concatenate((ego_past[-1, :2],
                                        [np.linalg.norm(ego_past[-1, 2:4])],
                                        [ego_past[-1, -1]]))

        target_state = np.concatenate((ego_future[-1, :2],
                                    [np.linalg.norm(ego_future[-1, :2])],
                                    [ego_future[-1, -1]]))
        obstacle_states = [np.concatenate((neighbors_past[j, -1, :2], [np.linalg.norm(neighbors_past[j, -1, 2:4])], [neighbors_past[j, -1, -1]])) for j in range(neighbors_past.shape[0])]

        for _ in range(10):  # 采样10条轨迹
            sampled_traj = polynomial_trajectory_sampler(current_state, target_state, obstacle_states)
            sampled_trajs.append(sampled_traj)

        trajs_features = []
        for traj in sampled_trajs:
            # 在环境模型中向前模拟轨迹,得到轨迹特征
            traj_features_tmp = extract_features(traj[:, :4], ego_future[i], neighbors_past[:, i], neighbors_future[:, i], lanes[i], route_lanes[i], crosswalks[i])
            traj_features_tmp, _, _ = normalize_and_compute_likeness(traj_features_tmp, ego_future[i], traj[-1, :4])
            trajs_features.append(traj_features_tmp)

        # 计算interaction awareness特征
        ego_influenced_speeds = []
        for j in range(neighbors_past.shape[0]):
            if neighbors_past[j, -1, 0] < ego_past[-1, 0] and np.abs(neighbors_past[j, -1, 1] - ego_past[-1, 1]) < 2:  # 位于自车后方且与自车在同一车道
                influenced_speed = np.min(neighbors_future[j, :, 2])  # 取后续时间步速度的最小值
                ego_influenced_speeds.append(influenced_speed)
        interaction_feature = np.sum(ego_influenced_speeds)  # interaction awareness特征

        # 计算reward和概率
        rewards = []
        for traj_feat in trajs_features:
            reward = np.dot(traj_feat, theta_personal)
            rewards.append(reward)
        probs = np.exp(rewards) / np.sum(np.exp(rewards))

        # 累加特征期望
        feature_exp = np.zeros(feature_num)
        for j in range(len(probs)):
            feature_exp += probs[j] * trajs_features[j]
        feature_exp /= len(probs)

        # 计算human trajectory特征
        human_feature_exp = traj_features

        # 计算log likelihood
        log_like = np.dot(traj_features, theta_personal) - np.log(np.sum(np.exp(rewards)))

        # 计算梯度并更新权重
        grad = human_feature_exp - feature_exp - 2 * lam * theta_personal
        theta_personal += lr * grad

        # 添加个性化建模结果到训练日志
        with open('maxent_irl_training_log_personal.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([iteration + 1, human_feature_exp, feature_exp,
                                np.linalg.norm(human_feature_exp - feature_exp),
                                theta_personal, human_likeness, log_like])


# 使用训练得到的权重对轨迹进行采样
sampled_trajs = []
current_state = np.concatenate((ego_past[-1, :2],
                                [np.linalg.norm(ego_past[-1, 2:4])],
                                [ego_past[-1, -1]]))

target_state = np.concatenate((ego_future[-1, :2],
                               [np.linalg.norm(ego_future[-1, :2])],
                               [ego_future[-1, -1]]))
obstacle_states = [np.concatenate((neighbors_past[j, -1, :2], [np.linalg.norm(neighbors_past[j, -1, 2:4])], [neighbors_past[j, -1, -1]])) for j in range(neighbors_past.shape[0])]

for _ in range(10):  # 采样10条轨迹
    sampled_traj = weighted_trajectory_sampler(current_state, target_state, obstacle_states, theta_personal)
    sampled_trajs.append(sampled_traj)

# 绘制最终的轨迹
plot_scenario(ego_past, ego_future, neighbors_past, neighbors_future, lanes, route_lanes, crosswalks, sampled_trajs)