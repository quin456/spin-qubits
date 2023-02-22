from email.policy import default
import torch as pt
import numpy as np
import matplotlib
import pickle


if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


import gates as gate
import atomic_units as unit
from data import *
from utils import *
from eigentools import *
from data import (
    get_A,
    get_J,
    J_100_18nm,
    J_100_14nm,
    cplx_dtype,
    default_device,
    gamma_e,
    gamma_n,
    A_P,
    A_2P,
)
from visualisation import visualise_Hw, plot_psi, show_fidelity, fidelity_bar_plot
from hamiltonians import get_U0, get_H0, get_X_from_H
from GRAPE import (
    GrapeESR,
    CNOT_targets,
    CXr_targets,
    GrapeESR_AJ_Modulation,
    load_grape,
)
from electrons import get_electron_X

from pulse_maker import get_smooth_E


def grape_48S_fid_bars():
    grape = load_grape(
        "fields-gadi/fields/g228_48S_2q_5000ns_10000step", Grape=GrapeESR_AJ_Modulation
    )
    fids = grape.fidelity()[0]
    fidelity_bar_plot(fids)


def run_2P_1P_CNOTs(
    tN,
    N,
    nS=15,
    Bz=0,
    A=None,
    J=None,
    max_time=24 * 3600,
    lam=0,
    prev_grape_fn=None,
    verbosity=2,
    reverse_CX=False,
    kappa=1,
    simulate_spectators=True,
):

    nq = 2
    if A is None:
        A = get_A_1P_2P(nS)
    if J is None:
        J = get_J_1P_2P(nS)
    target = CNOT_targets(nS, nq)
    if reverse_CX:
        target = CXr_targets(nS)
    if prev_grape_fn is None:
        grape = GrapeESR(
            J,
            A,
            tN,
            N,
            Bz=Bz,
            target=target,
            max_time=max_time,
            lam=lam,
            verbosity=verbosity,
            simulate_spectators=simulate_spectators,
            kappa=kappa,
        )
    else:
        grape = load_grape(
            prev_grape_fn,
            max_time=max_time,
            Grape=GrapeESR,
            verbosity=verbosity,
            kappa=kappa,
            simulate_spectators=simulate_spectators,
            lam=lam,
        )

    grape.run()
    grape.print_result()
    if not pt.cuda.is_available():
        grape.plot_result()
        plt.show()
    grape.save()


def run_1P_2P_uniform_J_CNOTs(
    tN,
    N,
    nS,
    A=None,
    J=np.float64(100) * unit.MHz,
    max_time=60,
    kappa=1,
    lam=0,
    prev_grape_fp=None,
    simulate_spectators=True,
):

    A = get_A_1P_2P_uniform_J(nS)
    J = get_J_1P_2P_uniform_J(nS, J=J)
    run_2P_1P_CNOTs(
        tN,
        N,
        nS,
        A=A,
        J=J,
        max_time=max_time,
        kappa=kappa,
        lam=lam,
        simulate_spectators=simulate_spectators,
        prev_grape_fn=prev_grape_fp,
    )


if __name__ == "__main__":

    # run_2P_1P_CNOTs(5000*unit.ns, 10000, nS=48, max_time = 23.5*3600, lam=1e8, prev_grape_fn='fields/g254_48S_2q_5000ns_10000step', reverse_CX=False, kappa=1e2)
    run_2P_1P_CNOTs(
        300 * unit.ns,
        2000,
        nS=2,
        max_time=10,
        lam=1e4,
        kappa=1,
        simulate_spectators=True,
        # prev_grape_fn="fields/g284_69S_2q_3000ns_5000step",
    )
    # run_1P_2P_uniform_J_CNOTs(
    #     1000 * unit.ns,
    #     400,
    #     nS=9,
    #     J=10 * unit.MHz,
    #     lam=1e8,
    #     max_time=360,
    #     kappa=1e4,
    #     simulate_spectators=True,
    #     prev_grape_fp="fields/c1183_9S_2q_1000ns_400step",
    # )
    # grape_48S_fid_bars(); plt.show()

