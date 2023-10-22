from . import global_Value as G

import random
import copy

def Initialize_X(N):
    X = {}
    for i in range(1, N + 1):
        # 随机生成4位二进制数
        state = random.randint(0, 15)
        X[i] = state
    return X

def Reassign_states(X):
    new_X = {}
    for key, state in X.items():
        # 将state转为4位二进制字符串
        binary_str = format(state, '04b')
        new_values = []
        for bit in binary_str:
            value = random.uniform(0, 0.1)
            if bit == '0':
                value = -value
            new_values.append(value)
        new_X[key] = new_values
    return new_X

def Initialize_road_network_random(N):
    # 计算矩阵的大小
    side_length = int(N ** 0.5)
    if side_length * side_length != N:
        raise ValueError("N should be a perfect square for a square grid")

    # 初始化路网字典
    road_network = {}

    for i in range(1, N + 1):
        # 计算当前节点的坐标
        x = (i - 1) % side_length
        y = (i - 1) // side_length

        # 初始化四个方向为-1
        east, south, west, north = -1, -1, -1, -1

        # 计算四个方向的节点
        if x + 1 < side_length:
            east = y * side_length + (x + 1) + 1
        if y + 1 < side_length:
            south = (y + 1) * side_length + x + 1
        if x - 1 >= 0:
            west = y * side_length + (x - 1) + 1
        if y - 1 >= 0:
            north = (y - 1) * side_length + x + 1

        # 为当前节点更新路网信息
        road_network[i] = {
            "east": east,
            "south": south,
            "west": west,
            "north": north
        }

    # 确保双向连通性
    for i, neighbors in road_network.items():
        for direction, j in neighbors.items():
            if j != -1:
                opposite_direction = {
                    "east": "west",
                    "west": "east",
                    "north": "south",
                    "south": "north"
                }
                if road_network[j][opposite_direction[direction]] != i:
                    road_network[j][opposite_direction[direction]] = i

    return road_network


def Generate_road_attributes(road_network):
    attributes = {}

    for node, directions in road_network.items():
        for direction, next_node in directions.items():
            if next_node != -1:  # 检查是否有连接的节点
                key = (node, next_node)
                reverse_key = (next_node, node)

                # 如果已经为此链接生成过属性，就跳过
                if key in attributes or reverse_key in attributes:
                    continue

                # 生成随机的左转和右转概率
                a = random.uniform(0, 1)
                b = random.uniform(0, 1 - a)  # 确保 a + b <= 1

                # 生成随机的车流量
                q = random.randint(0, 100)

                attributes[key] = {
                    "left_turn_probability": a,
                    "right_turn_probability": b,
                    "traffic_flow": q,
                    "direction": direction  # 记录当前道路的朝向
                }

                # 为反方向的道路设置相同的属性
                opposite_direction = {
                    "east": "west",
                    "west": "east",
                    "north": "south",
                    "south": "north"
                }
                attributes[reverse_key] = {
                    "left_turn_probability": a,
                    "right_turn_probability": b,
                    "traffic_flow": q,
                    "direction": opposite_direction[direction]
                }

    return attributes



road_net = Initialize_road_network_random(G.N)

X = Initialize_X(G.N)
X_last = Initialize_X(G.N)

road_attributes = Generate_road_attributes(road_net)
road_attributes_last = Generate_road_attributes(road_net)