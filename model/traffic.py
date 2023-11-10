from . import global_Value as G

import random
import numpy as np
import copy

import geopandas as gpd
from shapely.geometry import box
from scipy.sparse import lil_matrix, csr_matrix

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

####################################################################
####################################################################
# 读取数据文件，绘制路网函数
####################################################################
####################################################################

def direction_to_index(direction):
    mapping = {'east': 0, 'north': 1, 'west': 2, 'south': 3}
    return mapping.get(direction, -1)  # 如果方向是unknown，返回-1


def determine_direction(lat1, lon1, lat2, lon2):
    # 计算两点之间的角度（以北方为0度，顺时针）
    angle = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360
    if (angle <= 45) or (angle > 315):
        return 'north'
    elif 45 < angle <= 135:
        return 'east'
    elif 135 < angle <= 225:
        return 'south'
    else:  # 225 < angle <= 315
        return 'west'


def update_road_attributes(road_attributes, road_attributes_last, connected_nodes, road_toward):
    # 遍历所有的路口
    for node_id, neighbors in connected_nodes.items():
        for neighbor in neighbors:
            # 获取两个节点之间的方向
            direction_info = road_toward.get((node_id, neighbor))
            
            # 如果没有方向信息或者方向是unknown，则跳过
            if not direction_info or direction_info['direction'] == 'unknown':
                continue

            # 根据方向获取对应的列索引
            col_index = direction_to_index(direction_info['direction'])

            # 生成随机的a和b的值，确保它们的和小于等于1
            a_value = np.random.uniform(0, 1)
            b_value = np.random.uniform(0, 1 - a_value)
            
            # 生成随机的q的值，范围在1到100之间
            q_value = np.random.randint(1, 101)
            
            # 更新road_attributes的值
            road_attributes['a'][node_id, col_index] = a_value
            road_attributes['b'][node_id, col_index] = b_value
            road_attributes['q'][node_id, col_index] = q_value
            
            # 更新road_attributes_last的值，这里我们假设它们与road_attributes相同
            # 如果需要不同的逻辑，您可以在这里进行调整
            road_attributes_last['a'][node_id, col_index] = a_value
            road_attributes_last['b'][node_id, col_index] = b_value
            road_attributes_last['q'][node_id, col_index] = q_value

    return road_attributes, road_attributes_last


def extract_intersections_and_roads(shapefile_path, x_min, y_min, x_max, y_max):
    """
    Extracts intersections and roads within a bounding box from a given shapefile.

    Parameters:
    shapefile_path (str): The file path to the shapefile.
    x_min (float): The minimum x coordinate of the bounding box.
    y_min (float): The minimum y coordinate of the bounding box.
    x_max (float): The maximum x coordinate of the bounding box.
    y_max (float): The maximum y coordinate of the bounding box.

    Returns:
    intersections (GeoDataFrame): The intersections within the bounding box.
    roads_within_bbox (GeoDataFrame): The roads within the bounding box.
    """

    # 读取Shapefile文件
    gdf = gpd.read_file(shapefile_path)

    # 定义区域边界框
    bbox = box(x_min, y_min, x_max, y_max)

    # 筛选出与边界框相交的路网数据
    roads_within_bbox = gdf[gdf.intersects(bbox)]

    # 创建一个空的GeoDataFrame来存储路口
    intersections = gpd.GeoDataFrame(columns=['geometry'], crs=gdf.crs)

    # 找出所有道路的相交点
    for road1 in roads_within_bbox.geometry:
        for road2 in roads_within_bbox.geometry:
            if road1.equals(road2):  # 避免自己与自己相交
                continue
            intersection = road1.intersection(road2)
            if intersection.is_empty:  # 如果没有交点则跳过
                continue
            if "Point" == intersection.geom_type:
                intersections = intersections.append({'geometry': intersection}, ignore_index=True)
            elif "MultiPoint" == intersection.geom_type:
                for point in intersection.geoms:
                    intersections = intersections.append({'geometry': point}, ignore_index=True)
            # 不考虑线与线相交作为路口

    # 移除重复的路口位置
    intersections = intersections.drop_duplicates(subset=['geometry'])

    return intersections, roads_within_bbox


# Helper function to find the index of the closest intersection to a given point
def find_closest_index(point, intersection_dict, threshold=300):
    closest_idx = None
    min_dist = float('inf')
    for idx, p in intersection_dict.items():
        dist = np.linalg.norm(np.array(point) - np.array(p))
        if dist < min_dist:
            closest_idx = idx
            min_dist = dist
    if min_dist > threshold:
        print(f"No close intersection found for point {point}, closest distance: {min_dist}")
        return None
    # print(f"Closest intersection for point {point} is {closest_idx} at distance {min_dist}")
    return closest_idx


def create_road_network(intersections_gdf, roads_gdf):
    # Create a dictionary of intersection positions
    intersection_dict = {}
    for idx, intersection in enumerate(intersections_gdf.geometry):
        coords = intersection.coords[:]  # 获取坐标数组
        if len(coords) >= 1:
            # 将坐标数组的第一个坐标作为路口位置
            intersection_dict[idx] = coords[0]
    
    # Initialize the adjacency matrix
    num_intersections = len(intersection_dict)
    adjacency_matrix = lil_matrix((num_intersections, num_intersections), dtype=int)

    for road in roads_gdf.itertuples():
        start_point = road.geometry.coords[0]
        end_point = road.geometry.coords[-1]
        start_idx = find_closest_index(start_point, intersection_dict)
        end_idx = find_closest_index(end_point, intersection_dict)
        if start_idx is not None and end_idx is not None:
            adjacency_matrix[start_idx, end_idx] = 1
            adjacency_matrix[end_idx, start_idx] = 1
        else:
            print(f"Failed to find intersections for road between {start_point} and {end_point}")


    # Converting the adjacency matrix to CSR format for more efficient matrix operations
    adjacency_matrix = adjacency_matrix.tocsr()
    
    return intersection_dict, adjacency_matrix


def extract_parameters(adjacency_matrix, location_dict):
    num_nodes = adjacency_matrix.shape[0]
    
    # 确保 adjacency_matrix 是稀疏矩阵形式
    if not isinstance(adjacency_matrix, csr_matrix):
        adjacency_matrix = csr_matrix(adjacency_matrix)
    
    # 初始化 X 和 X_last
    X = Initialize_X(num_nodes)
    X_last = Initialize_X(num_nodes)
    
    # 初始化 road_attributes 和 road_attributes_last
    road_attributes = {'a': np.zeros((num_nodes, 4)), 'b': np.zeros((num_nodes, 4)), 'q': np.zeros((num_nodes, 4))}
    road_attributes_last = {'a': np.zeros((num_nodes, 4)), 'b': np.zeros((num_nodes, 4)), 'q': np.zeros((num_nodes, 4))}

    
    # 初始化 connected_nodes 和 road_toward 字典
    connected_nodes = {}
    road_toward = {}

    # 构建 connected_nodes 字典
    for i in range(num_nodes):
        # 对于每个节点，找到所有相邻的节点（即邻接矩阵中的非零元素索引）
        neighbors = adjacency_matrix[i].nonzero()[1]
        connected_nodes[i] = neighbors.tolist()
    
    # 构建 road_toward 字典
    for i in range(num_nodes):
        for j in connected_nodes[i]:
            if (i, j) not in road_toward:
                direction = 'unknown'
                if i in location_dict and j in location_dict:
                    lat1, lon1 = location_dict[i]
                    lat2, lon2 = location_dict[j]
                    # 计算方向
                    direction = determine_direction(lat1, lon1, lat2, lon2)
                
                road_toward[(i, j)] = {'direction': direction}
                # 如果道路是双向的，也为相反方向添加条目
                if adjacency_matrix[j, i] > 0 and (j, i) not in road_toward:
                    # 需要根据(j, i)重新计算方向
                    opposite_direction = determine_direction(lat2, lon2, lat1, lon1)
                    road_toward[(j, i)] = {'direction': opposite_direction}
    
    road_attributes, road_attributes_last = update_road_attributes(road_attributes, road_attributes_last, connected_nodes, road_toward)

    return X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward


def simplify_network(adjacency_matrix, intersection_dict):
    if not isinstance(adjacency_matrix, lil_matrix):
        adjacency_matrix = adjacency_matrix.tolil()

    nodes_to_remove = set()
    for i in range(adjacency_matrix.shape[0]):
        # Get the indices of the non-zero elements in the row and column
        non_zero_in_row = adjacency_matrix[i].nonzero()[1]
        non_zero_in_col = adjacency_matrix[:, i].nonzero()[0]
        
        # Create a set of unique indices excluding the self-loop
        unique_indices = set(non_zero_in_row).union(set(non_zero_in_col)) - {i}

        if len(unique_indices) == 2:  # Node i has only two neighbors
            # Mark the node for removal
            nodes_to_remove.add(i)

            # Connect the two neighbors directly
            j, k = unique_indices
            adjacency_matrix[j, k] = 1
            adjacency_matrix[k, j] = 1

    # Remove the nodes from the adjacency matrix
    adjacency_matrix = adjacency_matrix.tolil()
    for node in nodes_to_remove:
        adjacency_matrix[node, :] = 0  # Remove all connections from node
        adjacency_matrix[:, node] = 0  # Remove all connections to node

    # Remove the nodes from the intersection dictionary
    for node in sorted(nodes_to_remove, reverse=True):
        del intersection_dict[node]

    # Create a new mapping for the remaining nodes
    remaining_nodes = list(set(range(adjacency_matrix.shape[0])) - nodes_to_remove)
    simplified_intersection_dict = {new_idx: intersection_dict[old_idx] for new_idx, old_idx in enumerate(remaining_nodes)}

    # Construct the new adjacency matrix with remaining nodes
    new_adjacency_matrix = adjacency_matrix[remaining_nodes, :][:, remaining_nodes]

    return simplified_intersection_dict, new_adjacency_matrix.tocsr()


def visualize_simplified_network(intersection_dict, adjacency_matrix):
    # This function will plot the network using the intersection_dict and adjacency_matrix
    fig, ax = plt.subplots()

    # Plot the edges using adjacency_matrix
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                point1 = intersection_dict[i]
                point2 = intersection_dict[j]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-', lw=0.5)

    # Plot the intersections using intersection_dict
    for idx, point in intersection_dict.items():
        ax.plot(point[0], point[1], 'o', markersize=5, color='blue')

    plt.show()


def InitializeRoadNetwork(shapefile_path='./Data/Network/Network_Final_Y2022.shp', x_min=490000, y_min=302000, x_max=495000, y_max=305000):
    # 提取交叉点和边界框内的路网
    intersections, roads_within_bbox = extract_intersections_and_roads(shapefile_path, x_min, y_min, x_max, y_max)

    # 创建路网网络
    intersection_dict, adjacency_matrix = create_road_network(intersections, roads_within_bbox)

    # 提取参数
    X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = extract_parameters(adjacency_matrix, intersection_dict)

    return X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward

# 现在您可以调用此函数并获取所需的变量
# X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = InitializeRoadNetwork()

# 使用示例
# simplified_intersection_dict, simplified_adjacency_matrix = simplify_network(adjacency_matrix, intersection_dict)

# shapefile_path = './Data/Network/Network_Final_Y2022.shp'
# x_min, y_min, x_max, y_max = [490000, 302000, 495000, 305000]
# intersections, roads_within_bbox = extract_intersections_and_roads(shapefile_path, x_min, y_min, x_max, y_max)

# # print(f"Number of roads within bbox: {len(roads_within_bbox)}")
# # print(f"Sample road geometries: {roads_within_bbox.geometry.head()}")

# intersection_dict, adjacency_matrix = create_road_network(intersections, roads_within_bbox)

# # Correct usage of the function
# visualize_simplified_network(intersection_dict, adjacency_matrix)

# X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = extract_parameters(adjacency_matrix, intersection_dict)

# print("X:", X)
# print("X_last:", X_last)
# print("road_attributes:", road_attributes)
# print("connected_nodes:", connected_nodes)
# print("road_toward:", road_toward)