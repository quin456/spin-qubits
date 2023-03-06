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


def run_coupler_grape(
    tN,
    N,
    nS,
    kappa=1,
    lam=0,
    max_time=10,
    prev_grape_fp=None,
    J_modulated=False,
    save_data=True,
    run_optimisation=True,
):
    J = get_J(nS, 3, J1=J_100_18nm[:9], J2=J_100_18nm[:9] / 2.3)
    A = get_A(nS, 3)

    run_CNOTs(
        tN=tN,
        N=N,
        nq=3,
        nS=nS,
        max_time=max_time,
        J=J,
        A=A,
        save_data=save_data,
        prev_grape_fp=prev_grape_fp,
        alpha=0,
        lam=lam,
        kappa=kappa,
        run_optimisation=run_optimisation,
        Grape=CouplerGrape,
        J_modulated=J_modulated,
    )


if __name__ == "__main__":
    run_coupler_grape(
        tN=4000 * unit.ns,
        N=8000,
        nS=81,
        max_time=1 * 3600,
        kappa=1,
        lam=1e9,
        J_modulated=False,
        run_optimisation=True,
    )
    # run_coupler_grape(
    #     tN=700 * unit.ns,
    #     N=1000,
    #     nS=2,
    #     max_time=120,
    #     kappa=1,
    #     lam=1e8,
    #     run_optimisation=False,
    #     J_modulated=True,
    #     save_data=True,
    #     prev_grape_fp='fields/g316_2S_3q_700ns_1000step'
    # )
    # test_coupler_pulse()
    if not pt.cuda.is_available():
        plt.show()

