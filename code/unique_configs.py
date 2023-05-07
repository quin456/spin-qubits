import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
import atomic_units as unit
from data import *
from utils import maxreal, minreal

folder = "exchange_data_fab/"


def get_unique_configs(fp):
    pos_1P_og = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])
    pos_1P = np.array([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    pos_2P_1_offset_1 = np.array([5, 100])
    pos_2P_2_offset_1 = np.array([-5, 100])
    pos_2P_1_offset_2 = np.array([40, 0])
    pos_2P_2_offset_2 = np.array([50, 0])

    # 3 and 4 are dumb
    pos_2P_1_offset_3 = np.array([100, 5])
    pos_2P_2_offset_3 = np.array([100, -5])
    pos_2P_1_offset_4 = np.array([0, 60])
    pos_2P_2_offset_4 = np.array([0, 50])
    pos_2P_1 = pos_1P + np.einsum("i,j->ji", np.ones(6), pos_2P_1_offset_2)
    pos_2P_2 = pos_1P + np.einsum("i,j->ji", np.ones(6), pos_2P_2_offset_2)

    # pos_2P_1 = np.array([[10, 11, 12, 10, 11, 12], [5, 5, 5, 6, 6, 6]])
    # pos_2P_2 = np.array([[10, 11, 12, 10, 11, 12], [-5, -5, -5, -4, -4, -4]])

    unique_configs = {}

    unique_hyperfines = {}
    A_dict = {}
    i_A = 0

    # J_1P_2P = pt.cat((J_100_110_18nm, J_rand))
    A_1P_2P = np.array(
        [36.6, 36.3, 35.7, 35.2, 35.1, 34.9, 34.8, 34.5, 33.0, 35.8, 33.9]
    )

    # A_1P_2P = pt.load(f"{folder}{fp}_A_gen")
    J_1P_2P = pt.load(f"{folder}{fp}_J_gen")

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
                    A_dict[d3] = A_1P_2P[i_A]
                    i_A += 1
                unique_configs[d_trip] = unique_configs.get(d_trip, []) + [[d1, d2, d3]]
                unique_hyperfines[f"{d3:.2f}"] = unique_hyperfines.get(
                    f"{d3:.2f}", []
                ) + [[i2, i3]]

    n_sys = len(unique_configs.keys())

    A_1P_2P = [[] for i in range(n_sys)]
    keys = list(unique_configs.keys())

    for i in range(n_sys):
        A_key = unique_configs[keys[i]][0][2]
        A_1P_2P[i] = [A_P / unit.MHz, A_dict[A_key]]

    J_100_110_18nm = pt.cat((J_100_18nm, J_110_18nm))
    J_rand = pt.rand(39) * (
        maxreal(J_100_110_18nm) - minreal(J_100_110_18nm)
    ) + minreal(J_100_110_18nm)

    print(f"{len(unique_configs.keys())} unique configurations")

    # for key in unique_configs.keys():
    #     print(key)
    #     print(unique_configs[key])
    return A_1P_2P, J_1P_2P


def save_unique_configs(fp="U_1P_2P"):
    A, J = get_unique_configs(fp)
    pt.save(pt.tensor(A), f"{exch_data_folder}{fp}_A")
    pt.save(pt.tensor(J), f"{exch_data_folder}{fp}_J")


def generate_A_and_J_from_uniform(
    nA=100, nJ=710, Amin=32, Amax=40, Jmin=10, Jmax=100, fp=None
):

    A_gen = np.random.uniform(Amin, Amax, nA)
    J_gen = np.random.uniform(Jmin, Jmax, nJ)

    pt.save(A_gen, f"{folder}{fp}_A_gen")
    pt.save(J_gen, f"{folder}{fp}_J_gen")


def inspect_A_J(fp="U_1P_2P"):
    A = pt.load(f"{exch_data_folder}{fp}_A")
    J = pt.load(f"{exch_data_folder}{fp}_J")
    print(A)
    print(J)
    print(f"nA = {len(A)}")
    print(f"nJ = {len(J)}")


def add_one_J():
    pass


if __name__ == "__main__":
    fp = "U_1P_2P"
    fp_big = "U_1P_2P_big"
    # generate_A_and_J_from_uniform(fp=fp_big)
    get_unique_configs(fp_big)
    # save_unique_configs(fp)
    # inspect_A_J(fp)

