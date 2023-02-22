import torch as pt

from GRAPE import CouplerGrape
from data import get_A, get_J
import gates as gate
import atomic_units as unit
from visualisation import *
from pulse_maker import load_XY
from electrons import get_electron_X
from utils import batch_fidelity
from run_grape import run_CNOTs


def test_coupler_pulse(fp="fields/c1212_1S_3q_100ns_500step"):

    Bx, By, T = load_XY(fp)

    X = get_electron_X(T[-1], len(T), 0, get_A(1, 3), get_J(1, 3), Bx, By)

    X2 = matmul3(dagger(gate.coupler_Id), X, gate.coupler_Id)

    fid = batch_fidelity(X2, gate.CX)

    plot_fidelity(plt.subplot(), fid, T)

    # psi0 = gate.spin_110
    # psi = X@psi0
    # plot_psi_with_phase(psi, T)


def run_coupler_grape(tN, N, nS, max_time=10, kappa=1, lam=0, prev_grape_fp=None):

    J = get_J(nS, 3, J1=J_100_18nm[:9], J2=J_100_18nm[:9] / 2.3)
    A = get_A(nS, 3)

    grape = CouplerGrape(J, A, tN, N, max_time=max_time, lam=lam, kappa=kappa)
    grape.run()
    grape.print_result()
    grape.plot_result()
    grape.save()


if __name__ == "__main__":
    run_coupler_grape(
        tN=3000 * unit.ns,
        N=5000,
        nS=81,
        max_time=23.5 * 3600,
        kappa=1,
        lam=1e8
        # prev_grape_fp="fields/g284_69S_2q_3000ns_5000step",
    )
    # test_coupler_pulse()
    if not pt.cuda.is_available():
        plt.show()

