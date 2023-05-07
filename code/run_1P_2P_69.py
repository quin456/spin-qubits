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


def grape_48S_fid_bars():
    grape = load_grape(
        "fields-gadi/fields/g228_48S_2q_5000ns_10000step", Grape=GrapeESR_AJ_Modulation
    )
    fids = grape.fidelity()[0]
    fidelity_bar_plot(fids)


# def get_2P_EE_CX_kwargs():

#     target_spec = pt.stack((gate.Id, gate.Id))
#     A_spec = get_A(1, 2, [1, 0])
#     X0 = pt.tensor(
#         [[1, 0], [0, 0], [0, 0], [0, 1]], dtype=cplx_dtype, device=default_device
#     )

#     kwargs = {}
#     kwargs["target_spec"] = target_spec
#     kwargs["A_spec"] = A_spec
#     kwargs["X0"] = X0
#     kwargs["simulate_spectators"] = True

#     return kwargs


def run_2P_1P_CNOTs(
    tN,
    N,
    nS=15,
    Bz=0,
    max_time=24 * 3600,
    lam=0,
    prev_grape_fn=None,
    verbosity=2,
    reverse_CX=False,
    kappa=1,
    Grape=GrapeESR,
    A_spec=get_A_spec_single(),
    prev_grape_fp=None,
    J_modulated=False,
    save_data=True,
    run_optimisation=True,
):

    nq = 2
    A = get_A_1P_2P(nS, NucSpin=[-1, 1], donor_composition=[2, 1])
    J = get_J_1P_2P(nS)
    target = CNOT_targets(nS, nq)
    if reverse_CX:
        target = CXr_targets(nS)
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
        J_modulated=J_modulated,
        A_spec=A_spec,
        target=target,
        verbosity=verbosity,
    )


if __name__ == "__main__":

    # run_2P_1P_CNOTs(
    #     3000 * unit.ns,
    #     8000,
    #     nS=70,
    #     max_time=23.5 * 3600,
    #     lam=1e9,
    #     kappa=1,
    #     Grape=GrapeESR,
    #     A_spec=pt.tensor([get_A(1, 1)], device=default_device),
    # )

    run_2P_1P_CNOTs(500 * unit.ns, 1000, nS=1, max_time=60, lam=0, reverse_CX=False)

