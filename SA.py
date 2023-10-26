import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

import math
import copy
import random

def Simulated_annealing(initial_temperature, min_temperature, cooling_rate, kbT, X, X_last, road_attributes, road_attributes_last, road_toward, connected_nodes):
    current_energy, hq, hd, hw = energy.H_total(road_attributes_last, X_last, X_last)
    current_h_list = {"Hq":hq, "Hd":hd, "Hw":hw}

    best_state = copy.deepcopy(X)
    tmpX = copy.deepcopy(X)
    best_energy = current_energy
    temperature = initial_temperature

    # 用于存储不同温度下的数据
    temperature_data_list = []

    while temperature > min_temperature:
        for _ in range(15*G.N):
            X = energy.Random_change_X(X)
            road_attributes = energy.Update_traffic_flow(road_attributes_last, X, road_toward)
            road_attributes["q"] = energy.Normalize_traffic_matrix(road_attributes["q"])
            new_energy, hq, hd, hw = energy.H_total(road_attributes, X = X, X_last = X_last)
            new_h_list = {"Hq":hq, "Hd":hd, "Hw":hw}

            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.uniform(0, 1) < math.exp(-delta_energy / temperature / kbT):
                tmpX = copy.deepcopy(X)
                current_energy = new_energy
                current_h_list = new_h_list

                if current_energy < best_energy:
                    best_state = copy.deepcopy(X)
                    best_energy = current_energy
            else:
                X = copy.deepcopy(tmpX)

        # 在列表中添加当前温度下的数据
        temperature_data_list.append({
            "current_energy": current_energy,
            "current_h_list": current_h_list,
            "temperature": temperature,
            "X": copy.deepcopy(X)
        })

        print(current_energy, current_h_list, temperature)

        # 更新温度
        temperature = temperature * (1 - cooling_rate)

    return best_state, temperature_data_list

if __name__ == "__main__":
    # 使用模拟退火函数的示例：
    X, X_last, road_attributes, road_attributes_last, connected_nodes, road_toward = gv.InitAll()
    _, tmpHq, tmpHd, _ = energy.H_total(road_attributes_last, X, X_last)
    kbT = tmpHd + tmpHq
    best_solution, temp_data_list = Simulated_annealing(
        initial_temperature=10, 
        min_temperature=0.001, 
        cooling_rate=0.20, 
        kbT = kbT,
        X = X,
        X_last = X_last,
        road_attributes = road_attributes,
        road_attributes_last = road_attributes_last,
        road_toward = road_toward,
        connected_nodes = connected_nodes)

    ft.Save_data(temp_data_list, "Result/SA/test_result.pkl")
    ft.Save_data([best_solution,X_last], "Result/SA/test_result_X.pkl")

    print(best_solution)
    print(X_last)