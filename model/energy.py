from . import traffic as gv
from . import global_Value as G

import math
import random
import copy
import numpy as np

# ------------------------------------------------------------
# =================== 基本哈密顿量计算函数 ====================
# ------------------------------------------------------------

# 提取X[i]中的特定二进制位


def Get_bit(x, position):
    return 2 * ((x >> (position - 1)) & 1) - 1

def Get_X(x,i,c):
    if isinstance(x, dict):
        # 执行字典相关的操作 (A)
        return Get_bit(x[i], c+1)
    elif isinstance(x, np.ndarray):
        # 执行NumPy数组相关的操作 (B)
        return x[i,c]

def q_sn_j_i(x, nd, j, i):
    q, alpha, beta = nd["q"], nd["a"], nd["b"]
    part1 = (1 - Get_X(x, j, 0)) * (1 - Get_X(x, j, 1)) * Get_X(q, j, 2) * Get_X(alpha, j, 2)
    part2 = (1 - Get_X(x, j, 0)) * Get_X(x, j, 1) * Get_X(q, j, 0) * Get_X(beta, j, 0)
    part3 = Get_X(x, j, 2) * Get_X(x, j, 3) * Get_X(q, j, 3) * (1 - Get_X(alpha, j, 3) - Get_X(beta, j, 3))
    part4 = -Get_X(x, i, 0) * (1 - Get_X(x, i, 1)) * Get_X(q, i, 3) * Get_X(alpha, i, 3)
    part5 = -Get_X(x, i, 2) * Get_X(x, i, 3) * Get_X(q, i, 3) * (1 - Get_X(alpha, i, 3))
    return part1 + part2 + part3 + part4 + part5

def q_sn_i_j(x, nd, i, j):
    q, alpha, beta = nd["q"], nd["a"], nd["b"]
    part1 = (1 - Get_X(x, i, 2)) * (1 - Get_X(x, i, 3)) * Get_X(q, i, 0) * Get_X(alpha, i, 0)
    part2 = (1 - Get_X(x, i, 2)) * Get_X(x, i, 3) * Get_X(q, i, 2) * Get_X(beta, i, 2)
    part3 = Get_X(x, i, 0) * Get_X(x, i, 1) * Get_X(q, i, 1) * (1 - Get_X(alpha, i, 1) - Get_X(beta, i, 1))
    part4 = -Get_X(x, j, 2) * (1 - Get_X(x, j, 3)) * Get_X(q, j, 1) * Get_X(alpha, j, 1)
    part5 = -Get_X(x, j, 0) * Get_X(x, j, 1) * Get_X(q, j, 1) * (1 - Get_X(alpha, j, 1))
    return part1 + part2 + part3 + part4 + part5

def q_ew_j_i(x, nd, j, i):
    q, alpha, beta = nd["q"], nd["a"], nd["b"]
    part1 = Get_X(x, j, 0) * (1 - Get_X(x, j, 1)) * Get_X(q, j, 3) * Get_X(alpha, j, 3)
    part2 = Get_X(x, j, 0) * Get_X(x, j, 1) * Get_X(q, j, 1) * Get_X(beta, j, 1)
    part3 = (1 - Get_X(x, j, 0)) * Get_X(x, j, 1) * Get_X(q, j, 0) * (1 - Get_X(alpha, j, 0) - Get_X(beta, j, 0))
    part4 = -(1 - Get_X(x, i, 2)) * (1 - Get_X(x, i, 3)) * Get_X(q, i, 0) * Get_X(alpha, i, 0)
    part5 = -Get_X(x, i, 1) * (1 - Get_X(x, i, 0)) * Get_X(q, i, 0) * (1 - Get_X(alpha, i, 0))
    return part1 + part2 + part3 + part4 + part5


def q_ew_i_j(x, nd, i, j):
    q, alpha, beta = nd["q"], nd["a"], nd["b"]
    part1 = Get_X(x, i, 2) * (1 - Get_X(x, i, 3)) * Get_X(q, i, 1) * Get_X(alpha, i, 1)
    part2 = Get_X(x, i, 2) * Get_X(x, i, 3) * Get_X(q, i, 3) * Get_X(beta, i, 3)
    part3 = (1 - Get_X(x, i, 2)) * Get_X(x, i, 3) * Get_X(q, i, 2) * (1 - Get_X(alpha, i, 2) - Get_X(beta, i, 2))
    part4 = -(1 - Get_X(x, j, 0)) * (1 - Get_X(x, j, 1)) * Get_X(q, j, 2) * Get_X(alpha, j, 2)
    part5 = -Get_X(x, j, 3) * (1 - Get_X(x, j, 2)) * Get_X(q, j, 2) * (1 - Get_X(alpha, j, 2))
    return part1 + part2 + part3 + part4 + part5


def Calculate_H_q(road_attributes):
    N = G.N

    qlist = road_attributes["q"]
    variances = [np.var(row[row != 0]) if np.any(row != 0) else 0 for row in qlist]

    return np.sum(variances)


def D_i_c(i, c, X_i_c, X_last):
    """
    计算给定道路节点的D_i_c值

    参数:
        i: 道路节点
        c: 相位状态

    返回:
        D_i_c的值
    """
    epsilon = G.epsilon_in_Di_c
    X_previous = Get_bit(X_last[i], c)
    difference = X_i_c - X_previous
    return 1 - math.exp(-difference**2 / (2 * epsilon**2))


def D_i(i, X_now, X_last):
    """
    计算给定道路节点的D_i值

    参数:
        i: 道路节点

    返回:
        D_i的值
    """
    return sum([D_i_c(i, c, Get_bit(X_now[i], c), X_last) for c in range(1, 5)])


def Calculate_H_d(X_now, X_last):
    """
    根据给定的N计算H_d

    参数:
        eta: 参数η
        N: 路网节点数

    返回:
        H_d的值
    """
    eta=G.eta
    N=G.N
    return eta * sum([D_i(i, X_now, X_last) for i in range(N)])


def Calculate_H_d_forbsb(X_now, X_last, epsilon=G.epsilon_in_Di_c, eta=G.eta):
    """
    使用矩阵运算计算H_d

    参数:
        eta: 参数η
        X_now: 当前的X矩阵
        X_last: 上一个时间步的X矩阵
        epsilon: 用于D_i_c_forbsb的参数

    返回:
        H_d的值
    """

    # 计算差异矩阵
    difference_matrix = X_now - X_last

    # 使用公式D_i_c_forbsb计算新的矩阵
    D_matrix = 1 - np.exp(-np.power(difference_matrix, 2) / (2 * epsilon**2))

    # 求得矩阵的每列的和
    D_sum = np.sum(D_matrix, axis=0)

    # 计算H_d
    H_d = eta * np.sum(D_sum)

    return H_d


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
    return sum([abs(Get_bit(x, i) - m[i-1]) for i in range(1, 5)])


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


def f_forbsb(x, m):
    """
    计算函数f(x,m)的值。

    参数:
        x: 一个列表，表示四组二进制数中的状态（用1和-1来表示）
        m: 一个列表，表示目标状态

    返回:
        f(x,m)的值
    """
    # print(Get_bit(x,1), m[0])
    return sum([abs(x[i] - m[i]) for i in range(0, 4)])


def Calculate_H_w_i_forbsb(x):
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
        product *= f_forbsb(x, state)

    return G.H_w_a * product


def Calculate_H_w(X_now):
    return sum([Calculate_H_w_i(X_now[i]) for i in range(G.N)])


def Calculate_H_w_forbsb(X_now):
    return sum([Calculate_H_w_i_forbsb(X_now[:, i]) for i in range(G.N)])


def H_total(road_attributes, X, X_last):
    # print(Calculate_H_q(road_attributes), Calculate_H_d(X_now = X), Calculate_H_w(X))
    tmpHq = Calculate_H_q(road_attributes)/G.N
    tmpHd = Calculate_H_d(X_now=X, X_last=X_last)/G.N
    tmpHw = Calculate_H_w(X)/G.N
    return tmpHd+tmpHq+tmpHw, tmpHq, tmpHd, tmpHw


def H_total_forbsb(road_attributes, X, X_last):
    # print(Calculate_H_q(road_attributes), Calculate_H_d(X_now = X), Calculate_H_w(X))
    tmpHq = Calculate_H_q(road_attributes)/G.N
    tmpHd = Calculate_H_d_forbsb(X, X_last)/G.N
    tmpHw = Calculate_H_w_forbsb(X)/G.N
    return tmpHd+tmpHq+tmpHw, tmpHq, tmpHd, tmpHw

# ------------------------------------------------------------
# ===================   道路状态生成函数   ====================
# ------------------------------------------------------------


def Random_change_X(X_now):
    i = random.randint(1, G.N)
    X_now[i] = random.randint(0, 15)
    return X_now


def Update_traffic_flow(attributes, tmpX, road_toward):

    new_q = copy.deepcopy(attributes["q"])

    for (node, next_node), _ in road_toward.items():
        direction = road_toward.get((node, next_node), {}).get("direction", None)

        if direction == "east":
            new_q[node,0] += q_ew_i_j(tmpX, attributes, node, next_node)
        elif direction == "west":
            new_q[node,2] += q_ew_j_i(tmpX, attributes, node, next_node)
        elif direction == "north":
            new_q[node,1] += q_sn_j_i(tmpX, attributes, node, next_node)
        elif direction == "south":
            new_q[node,3] += q_sn_i_j(tmpX, attributes, node, next_node)

    attributes["q"] = new_q

    return attributes


def Normalize_traffic_matrix(q_matrix):
    # q_matrix 是一个二维矩阵，其中存储着交通流量的q值

    # 找到矩阵中的最大值和最小值
    min_q = np.min(q_matrix)
    max_q = np.max(q_matrix)

    # 避免除以零的情况，如果最大值和最小值相同，则返回原矩阵
    if min_q == max_q:
        return q_matrix

    # 对矩阵中的每个元素进行归一化
    normalized_matrix = ((q_matrix - min_q) / (max_q - min_q)) * 100

    return normalized_matrix



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
