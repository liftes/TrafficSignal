from . import traffic as gv
from . import global_Value as G

import math
import random
import copy

# ------------------------------------------------------------
# =================== 基本哈密顿量计算函数 ====================
# ------------------------------------------------------------

# 提取X[i]中的特定二进制位
def Get_bit(x, position):
    return 2 * ((x >> (position - 1)) & 1) - 1

def Calculate_q_ew(pair, X = gv.X):
    i, j = pair

    # 从X中获取二进制位
    x_i1 = (Get_bit(X[i], 1) + 1) / 2
    x_i3 = (Get_bit(X[i], 3) + 1) / 2
    x_j3 = (Get_bit(X[j], 3) + 1) / 2
    x_j4 = (Get_bit(X[j], 4) + 1) / 2

    # 从全局字典中取得相关值
    alpha_i1 = gv.road_attributes[(i, j)]["left_turn_probability"]
    alpha_j1 = gv.road_attributes[(j, i)]["left_turn_probability"]
    alpha_j4 = gv.road_attributes[(j, i)]["left_turn_probability"]
    beta_j2 = gv.road_attributes[(j, i)]["right_turn_probability"]
    q_i1 = gv.road_attributes[(i, j)]["traffic_flow"]
    q_j1 = gv.road_attributes[(j, i)]["traffic_flow"]
    q_j2 = gv.road_attributes[(j, i)]["traffic_flow"]
    q_j4 = gv.road_attributes[(j, i)]["traffic_flow"]

    # 使用上面的公式计算q_ew
    term1 = (1 - x_j3) * x_j4 * q_j4 * alpha_j4
    term2 = x_j3 * x_j4 * q_j2 * beta_j2
    term3 = x_j3 * (1 - x_j4) * q_j1 * (1 - alpha_j1 - alpha_j4)
    term4 = (1 - x_i1) * (1 - x_j3) * q_i1 * alpha_i1
    term5 = x_i3 * (1 - x_j4) * q_i1 * (1 - alpha_i1)

    q_ew = term1 + term2 + term3 - term4 - term5
    return q_ew

def Calculate_q_sn(pair, X = gv.X):
    i, j = pair

    # 从X中获取二进制位
    x_j3 = (Get_bit(X[j], 3) + 1) / 2
    x_j4 = (Get_bit(X[j], 4) + 1) / 2
    x_i3 = (Get_bit(X[i], 3) + 1) / 2
    x_i4 = (Get_bit(X[i], 4) + 1) / 2

    # 从全局字典中取得相关值
    alpha_j3 = gv.road_attributes[(j, i)]["left_turn_probability"]
    beta_j1 = gv.road_attributes[(j, i)]["right_turn_probability"]
    alpha_i4 = gv.road_attributes[(i, j)]["left_turn_probability"]
    beta_i4 = gv.road_attributes[(i, j)]["right_turn_probability"]
    q_j3 = gv.road_attributes[(j, i)]["traffic_flow"]
    q_j1 = gv.road_attributes[(j, i)]["traffic_flow"]
    q_j4 = gv.road_attributes[(j, i)]["traffic_flow"]
    q_i4 = gv.road_attributes[(i, j)]["traffic_flow"]

    # 使用上面的公式计算q_sn
    term1 = (1 - x_j3) * (1 - x_j4) * q_j3 * alpha_j3
    term2 = x_j3 * (1 - x_j4) * q_j1 * beta_j1
    term3 = x_i4 * x_i3 * q_i4 * (1 - alpha_i4 - beta_i4)
    term4 = (1 - x_i3) * x_i4 * q_i4 * alpha_i4
    term5 = x_i3 * x_i4 * q_i4 * (1 - alpha_i4)

    q_sn = term1 + term2 + term3 - term4 - term5
    return q_sn

def Calculate_H_q(road_attributes = gv.road_attributes):
    N = G.N
    H_q = 0
    
    for i in range(1, N+1):
        # 计算平均流量
        adjacent_list = [j for j in range(1, N+1) if (j, i) in road_attributes]
        total_flow_i = sum([road_attributes[(j, i)]["traffic_flow"] for j in adjacent_list])
        avg_q_i = total_flow_i / len(adjacent_list)

        # 计算流量偏差代价
        for j in adjacent_list:
            q_ji = road_attributes[(j, i)]["traffic_flow"]
            H_q += (q_ji - avg_q_i)**2 / len(adjacent_list)
            
    return H_q

def D_i_c(i, c, X_i_c):
    """
    计算给定道路节点的D_i_c值

    参数:
        i: 道路节点
        c: 相位状态

    返回:
        D_i_c的值
    """
    epsilon = G.epsilon_in_Di_c
    X_previous = Get_bit(gv.X_last[i], c)
    difference = X_i_c - X_previous
    return 1 - math.exp(-difference**2 / (2 * epsilon**2))

def D_i(i, X_now = gv.X):
    """
    计算给定道路节点的D_i值

    参数:
        i: 道路节点

    返回:
        D_i的值
    """
    return sum([D_i_c(i, c, Get_bit(X_now[i], c)) for c in range(1, 5)])

def Calculate_H_d(eta = G.eta, N = G.N, X_now = gv.X):
    """
    根据给定的N计算H_d

    参数:
        eta: 参数η
        N: 路网节点数

    返回:
        H_d的值
    """
    return eta * sum([D_i(i, X_now) for i in range(1, N+1)])

def f(x, m):
    """
    计算函数f(x,m)的值。

    参数:
        x: 一个列表，表示四组二进制数中的状态（用1和-1来表示）
        m: 一个列表，表示目标状态

    返回:
        f(x,m)的值
    """
    # print(Get_bit(x,1), m[0])
    return sum([abs(Get_bit(x,i) - m[i-1]) for i in range(1,5)])

def Calculate_H_w_i(x):
    """
    计算Hw_i的值。

    参数:
        x: 一个列表，表示四组二进制数中的状态（用1和-1来表示）
        a: 一个大的数值

    返回:
        Hw的值
    """
    
    # 定义允许的四组二进制数状态
    allowed_states = G.allowed_states
    
    # 计算Hw
    product = 1
    for state in allowed_states:
        product *= f(x, state)
    
    return G.H_w_a * product

def Calculate_H_w(X_now = gv.X):
    return sum([Calculate_H_w_i(X_now[i]) for i in range(1,G.N+1)])

def H_total(road_attributes = gv.road_attributes, X = gv.X):
    # print(Calculate_H_q(road_attributes), Calculate_H_d(X_now = X), Calculate_H_w(X))
    tmpHq = Calculate_H_q(road_attributes)
    tmpHd = Calculate_H_d(X_now = X)
    tmpHw = Calculate_H_w(X)
    return tmpHd+tmpHq+tmpHw, tmpHq, tmpHd, tmpHw

# ------------------------------------------------------------
# ===================   道路状态生成函数   ====================
# ------------------------------------------------------------

def Random_change_X(X_now = gv.X):
    i = random.randint(1, G.N)
    X_now[i] = random.randint(0, 15)
    return X_now

def Update_traffic_flow(attributes, tmpX):
    new_attributes = {}

    for (node, next_node), attribute in attributes.items():
        q = copy.deepcopy(attribute["traffic_flow"])

        # 根据道路方向添加q_ew或q_sn
        if attribute["direction"] in ["east", "west"]:
            q += Calculate_q_ew((node, next_node), X = tmpX)
        elif attribute["direction"] in ["north", "south"]:
            q += Calculate_q_sn((node, next_node), X = tmpX)

        # 更新车流量
        new_attributes[(node, next_node)] = {
            "left_turn_probability": attribute["left_turn_probability"],
            "right_turn_probability": attribute["right_turn_probability"],
            "traffic_flow": q,
            "direction": attribute["direction"]
        }

    return new_attributes

def Normalize_traffic_flow(attributes):
    # 获取所有q的值
    all_q_values = [attr["traffic_flow"] for attr in attributes.values()]

    # 找到q的最大和最小值
    min_q = min(all_q_values)
    max_q = max(all_q_values)

    # 根据q的最大和最小值对所有的q进行缩放
    for key, attr in attributes.items():
        normalized_q = ((attr["traffic_flow"] - min_q) / (max_q - min_q)) * 100
        attributes[key]["traffic_flow"] = normalized_q

    return attributes


# 示例
# H_q_value = Calculate_H_q()
# print(H_q_value)

# pair = (1, 2)
# q_sn_value = Calculate_q_sn(pair)
# print(q_sn_value)
# q_ew_value = Calculate_q_ew(pair)
# print(q_ew_value)

# pair = (1, 3)
# q_sn_value = Calculate_q_sn(pair)
# print(q_sn_value)
# q_ew_value = Calculate_q_ew(pair)
# print(q_ew_value)

# H_w_value = Calculate_H_w(gv.X)
# print(H_w_value)

# Random_change_X(3)
# new_attributes = Update_traffic_flow(gv.road_net, gv.road_attributes)
# print(new_attributes, gv.road_attributes)
