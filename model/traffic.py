from . import global_Value as G

import random
import numpy as np
import copy

def Initialize_X(N):
    X = {}
    for i in range(N):
        # 随机生成4位二进制数
        state = random.randint(0, 15)
        X[i] = state
    return X


def ReassignStates(X):
    num_nodes = len(X)
    matrix = np.zeros((4, num_nodes))
    
    for col, state in enumerate(X.values()):
        binary_str = format(state, '04b')
        
        for row, bit in enumerate(binary_str):
            # value = random.uniform(0, 0.1)
            value = 1
            if bit == '0':
                value = -value
            matrix[3 - row, col] = value
            
    return matrix


def Initialize_road_network_random(N):
    # 计算矩阵的大小
    side_length = int(N ** 0.5)
    if side_length * side_length != N:
        raise ValueError("N should be a perfect square for a square grid")

    # 初始化路网字典
    road_network = {}

    for i in range(N):
        # 计算当前节点的坐标
        x = i % side_length
        y = i // side_length

        # 初始化四个方向为-1
        east, south, west, north = -1, -1, -1, -1

        # 计算四个方向的节点
        if x + 1 < side_length:
            east = y * side_length + (x + 1)
        if y + 1 < side_length:
            south = (y + 1) * side_length + x
        if x - 1 >= 0:
            west = y * side_length + (x - 1)
        if y - 1 >= 0:
            north = (y - 1) * side_length + x

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


def Generate_road_attributes(road_network, N):
    # 初始化矩阵
    left_turn_probability = [[0 for _ in range(4)] for _ in range(N)]
    right_turn_probability = [[0 for _ in range(4)] for _ in range(N)]
    traffic_flow = [[0 for _ in range(4)] for _ in range(N)]

    # 初始化方向字典
    attributes = {}

    # 方向映射
    direction_map = {
        "east": 0,
        "north": 1,
        "west": 2,
        "south": 3
    }

    # 反方向映射
    opposite_direction_map = {0: 2, 1: 3, 2: 0, 3: 1}
    opposite_direction = {
        "east": "west",
        "west": "east",
        "north": "south",
        "south": "north"
    }

    # 遍历网络
    for node, directions in road_network.items():
        for direction, next_node in directions.items():
            if next_node != -1:  # 检查是否有连接的节点
                key = (node, next_node)
                reverse_key = (next_node, node)

                if key in attributes or reverse_key in attributes:
                    continue

                # 随机生成左转和右转概率
                a = random.uniform(0, 1)
                b = random.uniform(0, 1 - a)

                # 随机生成车流量
                q = random.randint(0, 100)

                # 更新矩阵
                dir_index = direction_map[direction]
                left_turn_probability[node][dir_index] = a
                right_turn_probability[node][dir_index] = b
                traffic_flow[node][dir_index] = q

                # 更新反方向
                opposite_dir_index = opposite_direction_map[dir_index]
                left_turn_probability[next_node][opposite_dir_index] = a
                right_turn_probability[next_node][opposite_dir_index] = b
                traffic_flow[next_node][opposite_dir_index] = q

                # 更新方向字典
                attributes[key] = {
                    "direction": direction
                }
                attributes[reverse_key] = {
                    "direction": opposite_direction[direction]
                }

    nd = {"a": np.array(left_turn_probability),
          "b": np.array(right_turn_probability),
          "q": np.array(traffic_flow)}

    return nd, attributes

def Get_connected_nodes(road_attributes):
    connections = {}

    for (i, j), attributes in road_attributes.items():
        # 如果i不在字典中，添加空列表
        if i not in connections:
            connections[i] = []
        connections[i].append(j)

        # 如果j不在字典中，添加空列表
        if j not in connections:
            connections[j] = []
        connections[j].append(i)

    # 删除重复的连接
    for key, value in connections.items():
        connections[key] = list(set(value))

    return connections

def InitAll():
    road_net = Initialize_road_network_random(G.N)

    X = Initialize_X(G.N)
    X_last = Initialize_X(G.N)

    road_attributes, road_toward = Generate_road_attributes(road_net,G.N)
    road_attributes_last, road_toward = Generate_road_attributes(road_net,G.N)

    connected_nodes = Get_connected_nodes(road_toward)
    return X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward

# road_net = Initialize_road_network_random(G.N)

# X = Initialize_X(G.N)
# X_last = Initialize_X(G.N)

# road_attributes = Generate_road_attributes(road_net)
# road_attributes_last = Generate_road_attributes(road_net)

# connected_nodes = Get_connected_nodes(road_attributes)