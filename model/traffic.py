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
    # for i, neighbors in road_network.items():
    #     for direction, j in neighbors.items():
    #         if j != -1:
    #             opposite_direction = {
    #                 "east": "west",
    #                 "west": "east",
    #                 "north": "south",
    #                 "south": "north"
    #             }
    #             if road_network[j][opposite_direction[direction]] != i:
    #                 road_network[j][opposite_direction[direction]] = i

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

    road_attributes = Generate_road_attributes(road_net)
    road_attributes_last = Generate_road_attributes(road_net)

    connected_nodes = Get_connected_nodes(road_attributes)
    return X, X_last, road_attributes, road_attributes_last, connected_nodes

# road_net = Initialize_road_network_random(G.N)

# X = Initialize_X(G.N)
# X_last = Initialize_X(G.N)

# road_attributes = Generate_road_attributes(road_net)
# road_attributes_last = Generate_road_attributes(road_net)

# connected_nodes = Get_connected_nodes(road_attributes)