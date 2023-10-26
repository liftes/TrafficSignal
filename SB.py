import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

import math
import copy
import random
import numpy as np

direction_map = {
    "east": 0,
    "north": 1,
    "west": 2,
    "south": 3
}

def State_to_str(state):
    return ''.join([str(int(val)) for val in state])


def Str_to_state(s):
    return [1 if val == '1' else -1 for val in s if val != '-']


def Map_column(column):
    # 定义映射
    mapping = {
        "-1-1-11": "-1-1-1-1",
        "1-1-11": "-1-11-1",
        "-111-1": "-1111",
        "1-111": "1-11-1",
        "-11-1-1": "-11-11",
        "1-1-11": "11-1-1",
        "11-1-1": "11-11",
        "111-1": "1111"
    }

    col_str = State_to_str(column)
    return Str_to_state(mapping.get(col_str, col_str))


def ReassignStates(X):
    num_cols = X.shape[1]
    mapped_matrix = np.zeros_like(X)

    for col in range(num_cols):
        mapped_matrix[:, col] = Map_column(X[:, col])

    return mapped_matrix


def AllOnesOrMinusOnes(column):
    """检查列的所有值是否都是1或-1"""
    return all(val == 1 or val == -1 for val in column)


def ReplaceColumns(X, X_last):
    """替换X中不符合要求的列为X_last的对应列"""
    num_cols = X.shape[1]

    for col in range(num_cols):
        if not AllOnesOrMinusOnes(X[:, col]):
            X[:, col] = X_last[:, col]

    return X


def InitializeY(shape, mean=0, std_dev=0.1):
    """
    Initialize matrix Y with values sampled from a Gaussian (normal) distribution.

    Parameters:
    - shape: tuple, shape of the matrix Y (e.g., (4, n) for 4 rows and n columns)
    - mean: float, mean of the Gaussian distribution (default is 0)
    - std_dev: float, standard deviation of the Gaussian distribution (default is 0.1)

    Returns:
    - Y: matrix initialized with values sampled from the Gaussian distribution
    """
    Y = np.random.normal(mean, std_dev, shape)
    return Y


def RescaleMatrix(matrix, scale=0.3):
    """
    Rescale the elements of the matrix such that elements >=1 become 'scale' 
    and elements <=-1 become '-scale'.
    """
    # matrix[matrix >= 1] = scale * matrix
    # matrix[matrix <= -1] = scale * matrix
    return scale * matrix


def a(t):
    # You can implement the function a(t) here
    return G.k*t


def q(qdict, road_toward, j, i):
    global direction_map
    direction = road_toward[(j,i)]["direction"]
    return qdict["q"][j,direction_map[direction]]

def H_w_dot(X, m):
    tmp1 = energy.Calculate_H_w_i_forbsb(X)
    tmp2 = energy.f_forbsb(X, m)
    if tmp2 == 0:
        return 0
    else:
        return G.H_w_a*tmp1/tmp2


def X_dot(Y, i, c):
    a_0 = G.a_0  # 从全局字典中获取a_0的值
    y_i_c = Y[c][i]  # 获取对应的y值
    return a_0 * y_i_c


def Y_dot(X, X_last, i, c, t, qdict, connected_nodes, road_toward):
    """
    Calculate the value of y_dot for given parameters.

    Parameters:
    - X: The matrix X.
    - i, c: The indices for which y_dot is being calculated.
    - a0: Constant value for a_0.
    - t: Time variable for function a(t).
    - q_deriv: Function to compute the derivative of q with respect to x.
    - mu: Matrix of mean values.
    - sigma: Standard deviation.
    - H_w: Constant value for H_w.
    - M: Set M for summing.
    - Ni: Set Ni for node i.

    Returns:
    - y_dot: Value of y_dot for the given parameters.
    """
    # Init Global variables
    a0 = G.a_0
    Ni = connected_nodes[i]
    Ni_len = len(Ni)
    mu = X_last
    sigma = G.epsilon_in_Di_c
    M = G.allowed_states
    # Calculating individual parts of the equation
    part1 = -(a0 - a(t)) * X[c, i]

    q_i_bar = sum([q(qdict, road_toward, j, i) for j in Ni])/Ni_len
    q_i_bar_deriv = sum([energy.q_diff(X, qdict, j, i, c, road_toward, "ji") for j in Ni])/Ni_len

    # 与中心表征节点i_0相关的导数，即汇入i_0道路的Hq项导数
    part2 = -sum([(2/Ni_len) * (q(qdict,road_toward,j,i)-q_i_bar) * (energy.q_diff(X, qdict, j, i, c, road_toward, "ji") - q_i_bar_deriv) for j in Ni])

    # 流出中心节点i_0的导数
    part3 = 0
    for j0 in Ni:
        Nj = connected_nodes[j0]
        Nj_len = len(Nj)
        q_j_bar = sum([q(qdict, road_toward, j, j0) for j in Nj])/Nj_len
        part3 -= (2*(Nj_len-1)/(Nj_len**2)) * (q(qdict,road_toward,i,j0)-q_j_bar) * energy.q_diff(X, qdict, i, j0, c, road_toward, "ij")

    part5 = - G.eta * (X[c, i] - mu[c, i])/sigma**2 * \
        np.exp(-((X[c, i] - mu[c, i])**2)/(2*sigma**2))
    part6 = - sum([H_w_dot(X[:, i], m) for m in M])

    y_dot_value = (part1 + part2 + part3 + part5 + part6)/G.N
    return y_dot_value


def SimulatedBifurcation(X, X_last, Y, road_attributes, road_attributes_last, connected_nodes, road_toward):
    """
    Implement ballistic simulated bifurcation.

    Parameters:
    - X: Initial state matrix
    - Y: Target state matrix

    Returns:
    - Optimized X matrix
    - Record of parameters during the iteration
    """

    # Rescale the X matrix
    X = RescaleMatrix(X, 0.01)

    # Placeholder for record of parameters during the iteration
    parameters_record = []

    # Define parameters for the simulation
    dt = 0.005  # Time step
    iterations = 2000  # Number of iterations

    # Iteratively update X and Y using Euler's method (alternating implicit scheme)
    for iteration in range(iterations):
        # Update each element in X and Y matrix
        for i in range(G.N):
            for c in range(4):
                # Check if X meets the condition
                if abs(X[c][i]) >= 1:
                    # Retain the sign of X but set its absolute value to 1
                    X[c][i] = 1 if X[c][i] > 0 else -1
                    Y[c][i] = 0  # Clear the corresponding value in Y
                else:
                    # If X does not meet the condition, compute the derivatives
                    # Update X based on the Xdot function
                    X[c][i] += dt * X_dot(Y, i, c)
                    Y[c][i] += dt * Y_dot(X, X_last, i, c, iteration*dt,
                                          road_attributes, connected_nodes, road_toward)

        # Calculate the current energy and related parameters
        road_attributes = energy.Update_traffic_flow(
            road_attributes_last, X, road_toward)
        road_attributes["q"] = energy.Normalize_traffic_matrix(road_attributes["q"])
        current_energy, hq, hd, hw = energy.H_total_forbsb(
            road_attributes, X, X_last)
        current_h_list = {"Hq": hq, "Hd": hd, "Hw": hw}

        print(current_energy, current_h_list, iteration)
        # Append the parameters to the record
        parameters_record.append({
            "iteration": iteration,
            "current_energy": current_energy,
            "current_h_list": current_h_list
        })

    # # # 重映射X的值
    # X = ReplaceColumns(X, X_last)
    # road_attributes = energy.Update_traffic_flow_forbsb(
    #     road_attributes_last, X)
    # road_attributes = energy.Normalize_traffic_flow(road_attributes)
    # current_energy, hq, hd, hw = energy.H_total_forbsb(
    #     road_attributes, connected_nodes, X, X_last)
    # current_h_list = {"Hq": hq, "Hd": hd, "Hw": hw}

    # # print(current_energy, current_h_list)
    # # Append the parameters to the record
    # parameters_record.append({
    #     "iteration": iterations,
    #     "current_energy": current_energy,
    #     "current_h_list": current_h_list
    # })

    return X, parameters_record


if __name__ == "__main__":
    X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = gv.InitAll()
    X = gv.ReassignStates(X)
    X_last = gv.ReassignStates(X_last)
    Y = InitializeY(X.shape, 0, 0.01)

    best_solution, temp_data_list = SimulatedBifurcation(
        X_last, X_last, Y, road_attributes, road_attributes_last, connected_nodes, road_toward)
    ft.Save_data(temp_data_list, "Result/SB/test_result.pkl")
    ft.Save_data([best_solution, X_last], "Result/SB/test_result_X.pkl")

    print(best_solution)
    print(X_last)
