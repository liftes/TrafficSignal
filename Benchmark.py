import SA
import SB
import model.global_Value as G
import model.traffic as gv
import model.energy as energy
import tools.file as ft

from tqdm import trange

import time
import numpy as np

def Time_test_SB():
    X, X_last, road_attributes, road_attributes_last, connected_nodes = gv.InitAll()
    time1 = time.time()
    X = gv.ReassignStates(X)
    X_last = gv.ReassignStates(X_last)
    Y = SB.InitializeY(X.shape, 0, 0.01)

    best_solution, temp_data_list = SB.SimulatedBifurcation(X_last, X_last, Y, road_attributes, road_attributes_last, connected_nodes)
    time2 = time.time()
    return time2 - time1

def Time_test_SA():
    X, X_last, road_attributes, road_attributes_last, connected_nodes = gv.InitAll()
    time1 = time.time()
    _, tmpHq, tmpHd, _ = energy.H_total(road_attributes_last, connected_nodes, X, X_last)
    kbT = tmpHd + tmpHq
    best_solution, temp_data_list = SA.Simulated_annealing(
        initial_temperature=1000, 
        min_temperature=0.001, 
        cooling_rate=0.20, 
        kbT = kbT,
        X = X,
        X_last = X_last,
        road_attributes = road_attributes,
        road_attributes_last = road_attributes_last,
        connected_nodes = connected_nodes)
    time2 = time.time()
    return time2 - time1

N = 5
repeats = 20
timeSAlist = np.zeros((repeats,N))
timeSBlist = np.zeros((repeats,N))
for i in range(2,N+2):
    for j in trange(repeats):
        G.N = i**2
        timeSAlist[j,i-2] = Time_test_SA()
        timeSBlist[j,i-2] = Time_test_SB()

ft.Save_data([timeSAlist, timeSBlist], "Result/Benchmark/test_time.pkl")