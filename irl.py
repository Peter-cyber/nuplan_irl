import numpy as np
import csv

# 加载npz数据
data = np.load('/home/peter/GameFormer-Planner/nuplan/processed_data/us-ma-boston_0a0aaeab5a25507e.npz')

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
    # 提取ego车辆的特征
    ego_features = np.concatenate((ego_past, ego_future))
    ego_features = ego_features[:ego_length]
    ego_features = np.pad(ego_features, (0, ego_length - ego_features.shape[0]), 'constant')

    # 提取邻居车辆的过去轨迹特征
    neighbors_past_features = neighbors_past.reshape(-1)
    neighbors_past_features = neighbors_past_features[:neighbors_length]
    neighbors_past_features = np.pad(neighbors_past_features, (0, neighbors_length - neighbors_past_features.shape[0]), 'constant')

    # 提取邻居车辆的未来轨迹特征
    neighbors_future_features = neighbors_future.reshape(-1)
    neighbors_future_features = neighbors_future_features[:neighbors_length]
    neighbors_future_features = np.pad(neighbors_future_features, (0, neighbors_length - neighbors_future_features.shape[0]), 'constant')

    # 提取车道线特征
    lanes_features = lanes.reshape(-1)
    lanes_features = lanes_features[:lanes_length]
    lanes_features = np.pad(lanes_features, (0, lanes_length - lanes_features.shape[0]), 'constant')

    # 提取路径车道线特征
    route_lanes_features = route_lanes.reshape(-1)
    route_lanes_features = route_lanes_features[:route_lanes_length]
    route_lanes_features = np.pad(route_lanes_features, (0, route_lanes_length - route_lanes_features.shape[0]), 'constant')

    # 提取人行横道特征
    crosswalks_features = crosswalks.reshape(-1)
    crosswalks_features = crosswalks_features[:crosswalks_length]
    crosswalks_features = np.pad(crosswalks_features, (0, crosswalks_length - crosswalks_features.shape[0]), 'constant')

    # 将所有特征拼接在一起
    traj_features = np.concatenate((ego_features, neighbors_past_features, neighbors_future_features,
                                    lanes_features, route_lanes_features, crosswalks_features))

    human_likeness = 0 # 这里需要根据具体逻辑计算human_likeness

    return traj_features, human_likeness

# 定义超参数
n_iters = 100
lr = 0.01
lam = 0.1

# 获取特征数量
sample_features, _ = extract_features(ego_past[0], ego_future[0], neighbors_past[0], neighbors_future[0], lanes[0], route_lanes[0], crosswalks[0])
feature_num = sample_features.shape[0]

# 初始化权重
theta = np.random.normal(0, 0.05, size=feature_num)

# 创建训练日志
with open('maxent_irl_training_log.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['iteration', 'human feature', 'trajectory feature', 'feature norm', 'weights', 'human likeness', 'mean human likeness', 'likelihood'])

# 打印特征向量的长度
print('Feature vector length:', sample_features.shape[0])

# 确定最小的时间步数
min_steps = min(ego_past.shape[0], neighbors_past.shape[1], lanes.shape[0], route_lanes.shape[0], crosswalks.shape[0])

# 预先提取所有时间步的特征
traj_features_all = []
human_likeness_all = []
for i in range(min_steps):
    traj_features, human_likeness = extract_features(ego_past[i], ego_future[i], neighbors_past[:, i], neighbors_future[:, i], lanes[i], route_lanes[i], crosswalks[i])
    traj_features_all.append(traj_features)
    human_likeness_all.append(human_likeness)
traj_features_all = np.array(traj_features_all)
human_likeness_all = np.array(human_likeness_all)

# 训练迭代
for iteration in range(n_iters):
    print('iteration: {}/{}'.format(iteration+1, n_iters))

    feature_exp = np.zeros([feature_num])
    human_feature_exp = np.zeros([feature_num])
    log_like_list = []
    num_samples = traj_features_all.shape[0]

    for i in range(num_samples):
        traj_features = traj_features_all[i]
        human_likeness = human_likeness_all[i]

        # 计算reward和概率
        reward = np.dot(traj_features, theta)
        prob = np.exp(reward) / np.sum(np.exp(reward))

        # 累加特征期望
        feature_exp += prob * traj_features

        # 计算human trajectory feature
        human_feature_exp += traj_features

        # 计算log likelihood
        log_like = reward - np.log(np.sum(np.exp(reward)))
        log_like_list.append(log_like)

    # 计算梯度并更新权重
    grad = human_feature_exp - feature_exp - 2*lam*theta
    theta += lr * grad

    # 添加到训练日志
    with open('maxent_irl_training_log.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([iteration+1, np.array(human_feature_exp/num_samples), np.array(feature_exp/num_samples),
                            np.linalg.norm(human_feature_exp/num_samples - feature_exp/num_samples),
                            theta, human_likeness_all, np.mean(human_likeness_all), np.mean(log_like_list)])
