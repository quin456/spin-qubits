import os
import torch as pt
import numpy as np
import matplotlib
import pickle
from scipy import fftpack


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
from visualisation import *
from hamiltonians import get_U0, get_H0, get_X_from_H
from GRAPE import (
    GrapeESR,
    CNOT_targets,
    GrapeESR_AJ_Modulation,
    load_grape,
    CouplerGrape,
)
from electrons import get_electron_X
from single_spin import test_grape_pulse_on_non_res_spin, get_single_spin_X
from pulse_maker import get_smooth_E


def get_fids_from_fp(fp, **kwargs):
    fp_fids = fp + "_fids"
    if os.path.isfile(fp_fids):
        fids = pt.load(fp_fids)
    else:
        grape = load_grape(fp=fp, **kwargs)
        fids = grape.fidelity()[0]
        pt.save(fids, fp_fids)
    return fids


def analyse_grape_pulse(fp="fields/c1095_1S_2q_300ns_2000step", Grape=CouplerGrape):
    grape = load_grape(fp, Grape=Grape)
    Bx, By = grape.sum_XY_fields()
    Bx *= unit.T
    By *= unit.T
    rf = get_multi_system_resonant_frequencies(grape.get_H0())
    T = linspace(0, grape.tN, grape.N)
    fig, ax = plt.subplots(1, 1)
    # plot_fields(Bx, By, T=T, ax=ax[0])
    dt = T[-1] / len(T)
    freq = np.fft.fftfreq(T.shape[-1]) / dt
    fBx = np.fft.fft(Bx)
    fBy = np.fft.fft(Bx)
    ax.plot(freq / unit.MHz, np.imag(fBx), color="blue")
    # ax[1].plot(freq / unit.MHz, np.real(fBy), color='orange')

    for w in rf:
        ax.axvline(w / (2 * np.pi) / unit.MHz, color="red", linestyle="--", linewidth=1)


def run_grape_pulse_on_system(
    A, J, fp="fields/g329_81S_3q_4000ns_8000step", psi0=gate.spin_10, **kwargs
):
    grape = load_grape(fp, **kwargs)
    Bx, By = grape.sum_XY_fields()
    Bx *= unit.T
    By *= unit.T
    X = get_electron_X(grape.tN, grape.N, Bz=np.float64(0), A=A, J=J, Bx=Bx, By=By)
    print_rank2_tensor(X[-1])

    plot_psi(X @ psi0, tN=grape.tN)


def inspect_system():
    J = get_J(3, 3)[2:3]
    J[0, 0] /= 5
    tN = 200.0 * unit.ns
    N = 1500
    nq = 3
    nS = 1
    max_time = 15
    kappa = 1
    rf = None
    save_data = True
    init_u_fn = None
    mergeprop = False
    A = get_A(1, 3, NucSpin=[-1, -1, -1]) * 0

    grape = GrapeESR(
        J, A, tN, N, Bz=0.02 * unit.T, max_time=max_time, save_data=save_data
    )
    Hw = grape.get_Hw()

    eigs = pt.linalg.eig(grape.H0)
    E = eigs.eigenvalues[0]
    S = eigs.eigenvectors[0]
    D = pt.diag(E)
    UD = get_U0(D, tN, N)
    Hwd = dagger(UD) @ S.T @ Hw[0] @ UD
    visualise_Hw(Hw[0], tN)

    plt.show()


def sum_grapes(grapes):

    grape = grapes[0].copy()
    nG = len(grapes)

    m = sum([grape.m for grape in grapes])
    N = grape.N
    u_sum = zeros_like_reshape(grapes[0].u, (m, N))
    rf = zeros_like_reshape(grapes[0].rf, (m // 2,))

    k0 = 0
    for grape in grapes:
        kmid = grape.m // 2
        k1 = k0 + kmid
        u_sum[k0:k1] = grape.u_mat()[:kmid]
        u_sum[m // 2 + k0 : m // 2 + k1] = grape.u_mat()[kmid:]
        rf[k0:k1] = grape.rf
        k0 = k1
    u_sum = uToVector(u_sum)

    grape.u = u_sum
    grape.rf = rf
    grape.m = m
    grape.N = N
    grape.initialise_control_fields()

    return grape


def run_CNOTs(
    tN=1000 * unit.ns,
    N=2500,
    nq=2,
    nS=15,
    Bz=np.float64(0),
    max_time=24 * 3600,
    J=None,
    A=None,
    target=None,
    save_data=True,
    rf=None,
    prev_grape_fp=None,
    kappa=1,
    lam=0,
    alpha=0,
    noise_model=None,
    ensemble_size=1,
    run_optimisation=True,
    cost_momentum=0,
    simulate_spectators=False,
    verbosity=2,
    Grape=GrapeESR,
    simulation_steps=False,
    J_modulated=False,
    matrix_exp_batches=1,
    A_spec=None,
    X0=None,
):

    J1_low = J_100_18nm / 50
    J2_low = get_J_low(nS, nq)
    J_low = J2_low
    if A is None:
        A = get_A(nS, nq)
    if J is None:
        J = get_J(nS, nq)

    H0 = get_H0(A, J)
    H0_phys = get_H0(A, J, B0)
    S, D = get_ordered_eigensystem(H0, H0_phys)

    # rf,u0 = get_low_J_rf_u0(S, D, tN, N)
    rf = None
    u0 = None

    if prev_grape_fp is None:
        grape = Grape(
            tN,
            N,
            J,
            A=A,
            Bz=Bz,
            target=target,
            rf=rf,
            u0=u0,
            lam=lam,
            alpha=alpha,
            noise_model=noise_model,
            kappa=kappa,
            ensemble_size=ensemble_size,
            cost_momentum=cost_momentum,
            simulate_spectators=simulate_spectators,
            verbosity=verbosity,
            simulation_steps=simulation_steps,
            J_modulated=J_modulated,
            matrix_exp_batches=matrix_exp_batches,
            A_spec=A_spec,
            X0=X0,
        )
    else:
        grape = load_grape(
            prev_grape_fp,
            kappa=kappa,
            lam=lam,
            simulate_spectators=simulate_spectators,
            verbosity=verbosity,
            Grape=Grape,
            simulation_steps=simulation_steps,
            J_modulated=J_modulated,
            A_spec=A_spec,
            X0=X0,
        )

    if run_optimisation:
        grape.run(max_time=max_time)
    else:
        grape.propagate()
    grape.print_result()
    grape.plot_result()
    if save_data:
        grape.save()

    if not pt.cuda.is_available():
        plt.show()


if __name__ == "__main__":

    run_grape_pulse_on_system(
        get_A_1P_2P(1, [0, 0]),
        get_J_1P_2P(1),
        fp="fields/g287_69S_2q_3000ns_5000step",
        simulate_spectators=True,
        Grape=GrapeESR,
    )

    # run_CNOTs(
    #     prev_grape_fp="fields/g339_81S_3q_4000ns_8000step",
    #     run_optimisation=False,
    #     Grape=CouplerGrape,
    #     save_data=False,
    #     simulate_spectators=True,
    # )

    # run_CNOTs(
    #     300 * unit.ns,
    #     N=500,
    #     nS=2,
    #     nq=2,
    #     max_time=120,
    #     lam=1e8,
    #     kappa=1,
    #     simulate_spectators=False,
    #     prev_grape_fp="fields/g315_2S_2q_300ns_500step",
    #     run_optimisation=False,
    #     verbosity=0,
    #     save_data=True,
    #     Grape=GrapeESR,
    #     simulation_steps=False,
    # )
    # run_CNOTs(
    #     tN=2000.0 * unit.ns,
    #     N=2500,
    #     nq=2,
    #     nS=15,
    #     J=get_J(15, 2),
    #     max_time=60,
    #     save_data=True,
    #     lam=0,
    #     kappa=1,
    #     noise_model=None,
    #     ensemble_size=1,
    # prev_grape_fn="fields/c1009_15S_2q_2000ns_2500step"
    # run_optimisation=False
    # )
    # run_CNOTs(prev_grape_fn='fields/c965_15S_2q_2000ns_2500step', max_time=1, run_optimisation=False)
    # run_CNOTs(
    #     tN=200.0 * unit.ns,
    #     N=500,
    #     nq=2,
    #     nS=1,
    #     max_time=10,
    #     save_data=False,
    #     lam=0,
    #     alpha=0,
    #     prev_grape_fn=None,
    #     kappa=1,
    #     noise_model=None,
    #     ensemble_size=1,
    #     cost_momentum=0,
    # )

    if not pt.cuda.is_available():
        plt.show()

