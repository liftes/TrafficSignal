import SA
import SB
import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

from tqdm import trange

import time
import copy
import numpy as np

def Random_reflesh_road_attributes(road_attributes):
    new_road_attributes = {}
    new_road_attributes["q"] = np.random.uniform(0, 100, road_attributes["q"].shape)
    new_road_attributes["a"] = np.random.uniform(0, 0.5, road_attributes["a"].shape)
    new_road_attributes["b"] = np.random.uniform(0, 0.5, road_attributes["b"].shape)
    return new_road_attributes

def Time_sequential_optimization(num_time_steps):
    resultH = np.zeros((2,num_time_steps))
    # 初始状态设置
    # G.N = 100
    # X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = gv.InitAll() # 使用实际数据时，这里只需要给出connected_nodes, road_toward，然后随机化X和X_last
    X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = gv.InitializeRoadNetwork()
    G.N = len(X)
    X_copy = copy.deepcopy(X)

    for time_step in range(num_time_steps):
        print(f"Optimizing for time step: {time_step}")

        # 如果不是第一步，根据实际数据更新路况数据
        road_attributes = Random_reflesh_road_attributes(road_attributes)
        road_attributes_last = copy.deepcopy(road_attributes)

        # 如果不是第一步，使用上一个时刻的最优解作为当前初始状态
        if time_step != 0:
            X = copy.deepcopy(best_solution_SA)
            X_last = copy.deepcopy(best_solution_SA)

        # 模拟退火优化
        _, tmpHq, tmpHd, _ = energy.H_total(road_attributes_last, X, X_last)
        kbT = tmpHd + tmpHq
        best_solution_SA, temp_data_list = SA.Simulated_annealing(
            initial_temperature=10, 
            min_temperature=0.0001, 
            cooling_rate=0.20, 
            kbT = kbT,
            X = X,
            X_last = X_last,
            road_attributes = road_attributes,
            road_attributes_last = road_attributes_last,
            road_toward = road_toward,
            connected_nodes = connected_nodes)

        resultH[0,time_step] = temp_data_list[-1]["current_energy"]

        # 可以选择存储或进一步处理模拟退火的最优解
        ft.Save_data(best_solution_SA, f"Result/TestTimeList/SA_time_step_{time_step}.pkl")

        # 如果不是第一步，使用上一个时刻的最优解作为当前初始状态
        if time_step != 0:
            X = copy.deepcopy(best_solution_SB)
            X_last = copy.deepcopy(best_solution_SB)
        else:
            X = gv.ReassignStates(X_copy)
            X_last = gv.ReassignStates(X_last)

        # 模拟分叉（SB）优化
        Y = SB.InitializeY(X.shape, 0, 0.01)
        best_solution_SB, temp_data_list = SB.SimulatedBifurcation(
            X, X_last, Y, road_attributes, road_attributes_last, connected_nodes, road_toward, 1000)

        resultH[1,time_step] = temp_data_list[-1]["current_energy"]

        # 存储模拟分叉（SB）的最优解
        ft.Save_data(best_solution_SB, f"Result/TestTimeList/SB_time_step_{time_step}.pkl")

    return resultH

result = Time_sequential_optimization(3)
print(result)