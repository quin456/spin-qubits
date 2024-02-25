import os

import torch as pt

from GRAPE import CouplerGrape, load_grape
from data import get_A, get_J
import gates as gate
import atomic_units as unit
from visualisation import *
from pulse_maker import load_XY
from electrons import get_electron_X
from utils import batch_fidelity
from run_grape import run_CNOTs, get_fids_and_field_from_fp


def test_coupler_pulse(fp="fields/c1212_1S_3q_100ns_500step"):
    Bx, By, T = load_XY(fp)

    X = get_electron_X(T[-1], len(T), 0, get_A(1, 3), get_J(1, 3), Bx, By)

    X2 = matmul3(dagger(gate.coupler_Id), X, gate.coupler_Id)

    fid = batch_fidelity(X2, gate.CX)

    plot_fidelity(plt.subplot(), fid, T)

    # psi0 = gate.spin_110
    # psi = X@psi0
    # plot_psi_with_phase(psi, T)


def coupler_fidelity_bars(
    fp="fields/g329_81S_3q_4000ns_8000step", ax=None, simulate_spectators=True
):
    fids = get_fids_from_fp
    print(fids)
    if ax is None:
        ax = plt.subplot()
    fidelity_bar_plot(
        fids,
        ax,
        f=[0.99999, 0.9999, 0.999],
        labels=["99.999%", "99.99%", "99.9% "],
        colours=["darkgreen", "green", "orange"],
    )
    ax.set_ylim(0.999, 1.0002)
    ax.set_yticks([0.999, 0.9992, 0.9994, 0.9996, 0.9998, 1])


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
    simulate_spectators=True,
    matrix_exp_batches=1,
):
    J = get_J(nS, 3, J1=J_low[:9], J2=J_low[:9])
    A = get_A(nS, 3, NucSpin=[0, 0, 1])

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
        simulate_spectators=simulate_spectators,
        matrix_exp_batches=matrix_exp_batches,
    )


if __name__ == "__main__":
    # coupler_fidelity_bars(fp = 'fields/g337_100S_3q_4000ns_8000step')
    run_coupler_grape(
        tN=4000 * unit.ns,
        N=8000,
        nS=81,
        max_time=23.5 * 3600,
        kappa=1,
        lam=1e9,
        J_modulated=False,
        run_optimisation=True,
        matrix_exp_batches=1,
    )
    # run_coupler_grape(
    #     tN=700 * unit.ns,
    #     N=1000,
    #     nS=2,
    #     max_time=160,
    #     kappa=1,
    #     lam=1e8,
    #     run_optimisation=True,
    #     save_data=True,
    #     simulate_spectators=True
    #     #prev_grape_fp='fields/g316_2S_3q_700ns_1000step'
    # )
    # test_coupler_pulse()
    if not pt.cuda.is_available():
        plt.show()
