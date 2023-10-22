import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

import math
import copy
import random

def Simulated_annealing(initial_temperature, min_temperature, cooling_rate, kbT):

    current_energy, hq, hd, hw = energy.H_total(gv.road_attributes_last, gv.X_last)
    current_h_list = {"Hq":hq, "Hd":hd, "Hw":hw}

    best_state = copy.deepcopy(gv.X)
    best_energy = current_energy
    temperature = initial_temperature

    # 用于存储不同温度下的数据
    temperature_data_list = []

    while temperature > min_temperature:
        for _ in range(15*G.N):
            gv.X = energy.Random_change_X(gv.X)
            gv.road_attributes = energy.Update_traffic_flow(gv.road_attributes_last, gv.X)
            gv.road_attributes = energy.Normalize_traffic_flow(gv.road_attributes)
            new_energy, hq, hd, hw = energy.H_total(road_attributes=gv.road_attributes, X = gv.X)
            new_h_list = {"Hq":hq, "Hd":hd, "Hw":hw}

            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.uniform(0, 1) < math.exp(-delta_energy / temperature / kbT):
                gv.X_last = copy.deepcopy(gv.X)
                current_energy = new_energy
                current_h_list = new_h_list

                if current_energy < best_energy:
                    best_state = copy.deepcopy(gv.X)
                    best_energy = current_energy
            else:
                gv.X = copy.deepcopy(gv.X_last)

        # 在列表中添加当前温度下的数据
        temperature_data_list.append({
            "current_energy": current_energy,
            "current_h_list": current_h_list,
            "temperature": temperature,
            "X": copy.deepcopy(gv.X)
        })

        print(current_energy, current_h_list, temperature)

        # 更新温度
        temperature = temperature * (1 - cooling_rate)

    return best_state, temperature_data_list

# 使用模拟退火函数的示例：
_, tmpHq, tmpHd, _ = energy.H_total(gv.road_attributes_last, gv.X_last)
kbT = tmpHd + tmpHq
best_solution, temp_data_list = Simulated_annealing(
    initial_temperature=10, min_temperature=0.001, cooling_rate=0.05, kbT = kbT)

ft.Save_data(temp_data_list, "Result/SA/test_result.pkl")
