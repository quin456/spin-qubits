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
from run_grape import run_CNOTs


def run_2P_1P_N_entangle(
    tN=3000 * unit.ns,
    N=6000,
    nS=69,
    Bz=0,
    max_time=24 * 3600,
    lam=1e9,
    prev_grape_fn=None,
    kappa=1,
    simulate_spectators=False,
    Grape=GrapeESR,
    A_spec=None,
    prev_grape_fp=None,
    save_data=True,
    run_optimisation=True,
):

    nq = 2
    A = get_A_1P_2P(nS, NucSpin=[1, -1])
    J = get_J_1P_2P(nS)

    X0 = pt.tensor(
        [[1, 0], [0, 0], [0, 0], [0, 1]], dtype=cplx_dtype, device=default_device
    )

    target = pt.tensor(
        [[0, 1], [0, 0], [0, 0], [1, 0]], dtype=cplx_dtype, device=default_device
    )

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
        Grape=GrapeESR,
        simulate_spectators=simulate_spectators,
        A_spec=A_spec,
        target=target,
        X0=X0,
    )


if __name__ == "__main__":

    run_2P_1P_N_entangle(tN=200 * unit.ns, N=500, nS=1)

