import torch as pt
import numpy as np

from utils import *
from visualisation import *
from GRAPE import load_grape
from run_n_entangle import get_2P_EE_swap_kwargs, run_2P_1P_N_entangle
from electrons import get_electron_X
from pulse_maker import frame_transform_pulse


def get_1P_2P_NE_hamiltonian(Bz, A, J):
    pass


def get_2e_flip_grape_pulse(tN = 500*unit.ns, B0=2*unit.T):
    tN = lock_to_frequency(gamma_e*B0, tN)
    run_2P_1P_N_entangle(tN=tN, N=1000, nS=1, lam=0, max_time=120)

def run_2e_swap():

    B0 = 0.00002 * unit.T

    grape_fp = "fields/c1273_2S_2q_500ns_1000step"
    kwargs = get_2P_EE_swap_kwargs()
    grape = load_grape(grape_fp, **kwargs)
    grape.print_result()

    Bx, By = grape.sum_XY_fields()
    Bx *= unit.T
    By *= unit.T
    T = linspace(0, grape.tN, grape.N)
    Bx, By = frame_transform_pulse(Bx, By, T, -0.5*gamma_e * B0)
    X = get_electron_X(grape.tN, grape.N, B0, grape.A[1], grape.J[0], Bx, By)
    X_reduced = pt.stack((X[-1, :, 0], X[-1, :, 3])).T
    print_rank2_tensor(X[-1])
    print(f"fidelity = {fidelity(grape.target[1], X_reduced)}")

    print(f"max field = {maxreal(pt.sqrt(Bx**2 + By**2))/unit.mT} mT")
    plt.plot(T / unit.ns, Bx / unit.mT)
    plt.plot(T / unit.ns, By / unit.mT)
    plt.xlabel("time (ns)")
    plt.ylabel("B-field (mT)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #get_2e_flip_grape_pulse()
    run_2e_swap()

