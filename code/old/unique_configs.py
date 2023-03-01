import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
import atomic_units as unit
from data import *
from utils import maxreal, minreal

A_2P_48Js_fab = (
    np.array([36.6, 36.3, 35.7, 35.2, 34.9, 34.8, 34.5, 33.0, 35.8, 33.9]) * unit.MHz
)
pos_1P = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])

pos_2P_1 = np.array([[10, 11, 12, 10, 11, 12], [5, 5, 5, 6, 6, 6]])

pos_2P_2 = np.array([[10, 11, 12, 10, 11, 12], [-5, -5, -5, -4, -4, -4]])


unique_configs = {}


unique_hyperfines = {}
A_dict = {}
i_A = 0


def distance(r1, r2):
    return np.sqrt(np.dot(r1 - r2, r1 - r2))


for i1 in range(6):
    for i2 in range(6):
        for i3 in range(6):
            r1 = pos_1P[:, i1]
            r2 = pos_2P_1[:, i2]
            r3 = pos_2P_2[:, i3]
            d1 = distance(r1, r2)
            d2 = distance(r1, r3)
            d3 = distance(r2, r3)

            if d1 > d2:
                dtemp = d1
                d1 = d2
                d2 = dtemp

            d_trip = f"{d1:.2f},{d2:.2f},{d3:.2f}"

            if d3 not in A_dict.keys():
                A_dict[d3] = A_2P_48Js_fab[i_A]
                i_A += 1
            unique_configs[d_trip] = unique_configs.get(d_trip, []) + [[d1, d2, d3]]
            unique_hyperfines[f"{d3:.2f}"] = unique_hyperfines.get(f"{d3:.2f}", []) + [
                [i2, i3]
            ]


A_1P_2P = [[] for i in range(69)]
keys = list(unique_configs.keys())

for i in range(69):
    A_key = unique_configs[keys[i]][0][2]
    A_1P_2P[i] = [A_dict[A_key], -A_P]


J_100_110_18nm = pt.cat((J_100_18nm, J_110_18nm))
J_rand = pt.rand(39) * (maxreal(J_100_110_18nm) - minreal(J_100_110_18nm)) + minreal(
    J_100_110_18nm
)

J_69 = pt.cat((J_100_110_18nm, J_rand))

pt.save(pt.tensor(A_1P_2P) / unit.MHz, f"{exch_data_folder}A_2P_69")
pt.save(pt.tensor(J_69) / unit.MHz, f"{exch_data_folder}J_69")


print(f"{len(unique_configs.keys())} unique configurations")

for key in unique_configs.keys():
    print(key)
    print(unique_configs[key])


print(f"{len(unique_hyperfines.keys())} unique hyperfines")
for key in unique_hyperfines.keys():
    print(key)
    print(unique_hyperfines[key])


