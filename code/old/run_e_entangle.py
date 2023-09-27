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
from run_n_entangle import get_2P_EE_swap_kwargs


def run_2P_1P_ee_entangle(tN, N, nS, lam=1, prev_grape_fp=None, step=1, max_time=60):

    combine_A = pt.cat if nS > 1 else pt.stack
    A = combine_A((get_A_1P_2P(nS, NucSpin=[1, -1]), get_A_1P_2P(nS, NucSpin=[1, 1])))
    if nS == 1:
        J = pt.tensor(2 * [get_J_1P_2P(nS)], dtype=cplx_dtype, device=default_device)
    else:
        J = pt.cat((get_J_1P_2P(nS), get_J_1P_2P(nS)))

    if step == 1:
        X0 = pt.tensor(
            [[1, 0], [0, 0], [0, 1], [0, 0]], dtype=cplx_dtype, device=default_device
        )

        target_e_flip = pt.tensor(
            [[1, 0], [0, 0], [0, 0], [0, 1]], dtype=cplx_dtype, device=default_device
        )
    else:
        X0 = pt.tensor(
            [[1, 0], [0, 0], [0, 0], [0, 1]], dtype=cplx_dtype, device=default_device
        )

        target_e_flip = pt.tensor(
            [[1, 0], [0, 0], [0, 1], [0, 0]], dtype=cplx_dtype, device=default_device
        )
    ones = pt.ones(nS, dtype=cplx_dtype, device=default_device)
    target = pt.cat(
        (pt.einsum("s,ab->sab", ones, X0), pt.einsum("s,ab->sab", ones, target_e_flip))
    )

    grape = run_CNOTs(
        tN=tN,
        N=N,
        nq=3,
        nS=nS,
        max_time=max_time,
        J=J,
        A=A,
        save_data=True,
        prev_grape_fp=prev_grape_fp,
        alpha=0,
        lam=lam,
        kappa=1,
        run_optimisation=True,
        Grape=GrapeESR,
        target=target,
        **get_2P_EE_swap_kwargs()
    )


if __name__ == "__main__":

    run_2P_1P_ee_entangle(
        tN=3000 * unit.ns, N=6000, nS=69, lam=1e9, max_time=23.5 * 3600
    )

    # run_2P_1P_N_entangle(
    #     prev_grape_fp="fields/g366_138S_2q_6000ns_8000step",
    #     save_data=False,
    #     run_optimisation=False,
    # )

