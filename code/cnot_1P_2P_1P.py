import torch as pt
import numpy as np

from utils import *
from eigentools import *
from visualisation import *
from GRAPE import load_grape, GrapeESR, Grape_ee_Flip
from electrons import get_electron_X
from pulse_maker import (
    frame_transform_pulse,
    block_interpolate_pulse,
    pi_pulse_square,
    pi_pulse_duration,
)
from single_NE import NE_transition_pulse, get_NE_X
from visualisation import *
from run_1P_2P import *
from run_grape import (
    get_fids_and_field_from_fp,
    get_fids_and_field_saved,
    get_separate_fids_and_fields_save,
)
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, get_X_from_H

from matplotlib.colors import LinearSegmentedColormap


def get_2P_1P_CX_pulse():
    tN = 300 * unit.ns
    N = 2000
    grape = run_2P_1P_CNOTs(
        tN,
        N,
        1,
        max_time=360,
        lam=1e7,
        simulate_spectators=True,
        A_spec=-get_A_spec_single(),
    )


def show_2e_flip(grape_fp="fields/c1326_1S_3q_479ns_1000step", fp=None):
    fp_2 = "fields/c1296_2S_2q_200ns_1000step"  # one n-flip total

    grape = load_grape(grape_fp, Grape=Grape_ee_Flip, step=1, verbosity=2)
    grape.print_result(verbosity=2)
    print(f"max field = {get_max_field(*grape.sum_XY_fields())*1e3} mT")
    Bx, By = grape.get_Bx_By()
    X = grape.X
    X_spec, _P_spec = grape.get_spec_propagators()
    T = linspace(0, grape.tN, grape.N)

    psi0 = gate.spin_0
    psi0_spec = gate.spin_0

    psi1 = X[0, :, :4, 0]
    psi2 = X[0, :, 4:, 1]

    psi1_spec = X_spec[0, :, :2, :2] @ psi0_spec
    psi2_spec = X_spec[0, :, 2:, 2:] @ psi0_spec
    fig, ax = plt.subplot_mosaic(
        [
            ["upper", "upper"],
            ["middle left", "middle right"],
            ["lower left", "lower right"],
        ],
        figsize=(7.5, 3.5),
        layout="constrained",
    )  # , gridspec_kw={})

    # fig, ax = plt.subplots(2, 2)
    label_getter = lambda x: electron_arrow_spin_state_label_getter(x, 2)
    label_getter1 = lambda x: electron_arrow_spin_state_label_getter(x, 1)
    legend_loc = "center left"
    plot_fields(Bx, By, T=T, ax=ax["upper"])
    plot_psi(
        psi1,
        T=T,
        ax=ax["middle left"],
        legend_loc=legend_loc,
        label_getter=label_getter,
    )
    plot_psi(
        psi2,
        T=T,
        ax=ax["middle right"],
        legend_loc=legend_loc,
        label_getter=label_getter,
    )
    plot_psi(
        psi1_spec,
        T=T,
        ax=ax["lower left"],
        legend_loc=legend_loc,
        label_getter=label_getter1,
    )
    plot_psi(
        psi2_spec,
        T=T,
        ax=ax["lower right"],
        legend_loc=legend_loc,
        label_getter=label_getter1,
    )

    print_rank2_tensor(X_spec[0, -1])

    spin_states = {
        "middle left": Downarrow,
        "middle right": Uparrow,
        "lower left": Downarrow,
        "lower right": Uparrow,
    }
    nucleus = {
        "middle left": "n_1",
        "middle right": "n_1",
        "lower left": "n_3",
        "lower right": "n_3",
    }
    for key in ["middle left", "middle right", "lower right"]:
        ax[key].set_yticks([0, 1])
        label_axis(
            ax[key],
            f"$|{nucleus[key]}{rangle} = |{spin_states[key]}{rangle}$",
            x_offset=0.35,
            y_offset=0.8,
        )

    label_axis(
        ax["lower left"],
        f"$|n_3{rangle} = |{Downarrow}{rangle}$",
        x_offset=0.35,
        y_offset=0.5,
    )
    ax["lower left"].set_yticks([0, 1])

    fig.set_size_inches(8, 6.4)
    if fp != None:
        fig.savefig(fp)


def show_2P_1P_CX_pulse(grape_fp="fields/c1303_1S_2q_300ns_2000step", fp=None):
    grape = load_grape(grape_fp, A_spec=-get_A_spec_single())
    T = grape.get_T()

    Bx, By = grape.get_Bx_By()

    fig, ax = plt.subplots(2, 1)
    plot_fields(Bx, By, T=T, ax=ax[0])

    X = grape.X[0]
    X_IP = dagger(get_U0(grape.get_H0()[0], T=T)) @ X
    psi0 = gate.spin_10
    psi = pt.abs(X @ psi0) ** 2
    psi_IP = pt.abs(X_IP @ psi0) ** 2

    ax[1].plot(T / unit.ns, psi[:, 0], color=color_cycle[0], label="00")
    ax[1].plot(
        T / unit.ns,
        psi[:, 1],
        color=color_cycle[1],
        linestyle="--",
        linewidth=0.5,
        label="01",
    )
    ax[1].plot(
        T / unit.ns,
        psi[:, 2],
        color=color_cycle[2],
        linestyle="--",
        linewidth=0.5,
        label="10",
    )
    ax[1].plot(T / unit.ns, psi[:, 3], color=color_cycle[3], label="11")
    ax[1].plot(T / unit.ns, psi_IP[:, 1], color=color_cycle[1], label="01 (IP)")
    ax[1].plot(T / unit.ns, psi_IP[:, 2], color=color_cycle[2], label="10 (IP)")
    ax[1].legend(loc="upper center")
    ax[1].set_xlabel("Time (ns)")

    fig.set_size_inches(4.5, 4)
    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def nuclear_spin_flip():
    Bz = 2 * unit.T
    tN = 20000 * unit.ns
    N = 40000
    A = get_A(1, 1)

    H0 = get_NE_H0(A, Bz)

    tN = pi_pulse_duration(gamma_n, 1.5 * unit.mT)
    T = linspace(0, tN, N)

    # Bx, By, T = NE_transition_pulse(1,3, tN, N, A, Bz)

    Bx, By = pi_pulse_square(-gamma_n * Bz + 2 * A, gamma_n, T=T)

    X = get_NE_X(N, Bz, A, Bx, By, T=T, interaction_picture=True)

    print_rank2_tensor(X[-1])
    # print_rank2_tensor(pt.abs(X[-1]))
    print(f"abs fidelity = {fidelity(pt.abs(X[-1]), real(gate.CX))}")


def cnot_2P_1P():
    tN = 200 * unit.ns
    N = 500
    grape = run_2P_1P_CNOTs(tN, N, 1, max_time=10)


def multi_2P_1P_CX(fp=None, grape_fp="fields/g370_69S_2q_3000ns_8000step"):
    fids, fields = get_fids_and_field_from_fp(
        grape_fp, get_from_grape=False, A_spec=-get_A_spec_single()
    )
    grape = load_grape(fp=grape_fp, A_spec=get_A_spec_single())
    Bx, By, T = fields
    # grape = load_grape(grape_fp, A_spec=-get_A_spec_single())
    # grape.print_result(verbosity=2)
    # Bx, By = grape.get_Bx_By()
    # T = grape.get_T()
    # fids = grape.fidelity()[0]
    print(f"min fidelity = {minreal(fids)}")
    fig, ax = plt.subplots(2, 1)

    axt = ax[0].twinx()
    xcol = color_cycle[0]
    ycol = color_cycle[1]
    ax[0].plot(T / unit.ns, Bx / unit.mT, color=xcol)
    axt.plot(T / unit.ns, By / unit.mT, color=ycol)
    ax[0].set_ylim([-2.5, 1])
    axt.set_ylim([-1, 2.5])
    ax[0].set_yticks([-1, 1], [1, -1], color=xcol)
    axt.set_yticks([-1, 1], [1, -1], color=ycol)
    ax[0].set_ylabel("Bx (mT)", color=xcol)
    axt.set_ylabel("By (mT)", color=ycol)
    ax[0].set_xlabel("time (ns)")

    div = 3
    n_2P_1P = 69
    systems_ax = np.concatenate(
        (np.linspace(1, n_2P_1P, n_2P_1P), np.array([1]) + n_2P_1P + div)
    )
    fidelity_bar_plot(
        fids,
        systems_ax=systems_ax,
        ax=ax[1],
        f=[0.99999, 0.9999, 0.999],
        colours=["green", "orange"],
    )
    ax[1].set_ylim([0.9998, 1.0001])
    ax[1].set_yticks([0.9998, 1], ["99.98", "100.0"])
    ax[1].set_ylabel("Fidelity (%)")
    ax[1].set_xticks([1, n_2P_1P], [1, n_2P_1P])

    y_offset = 0.67
    label_axis(
        ax[1],
        f"$T_{{CX}}$",
        x_offset=0.2,
        y_offset=y_offset,
    )
    label_axis(
        ax[1],
        f"$T_{{Spec}}$",
        x_offset=0.9,
        y_offset=y_offset,
    )

    fig.tight_layout()
    if fp != None:
        fig.savefig(fp)


def check_e_flip_phases(grape_fp="fields/c1314_3S_3q_500ns_1000step"):
    grape = load_grape(fp=grape_fp, Grape=Grape_ee_Flip, simulate_spectators=False)

    fids = pt.zeros(69)
    for k in range(69):
        fids[k] = fidelity(grape.X[k, -1] @ gate.X, grape.X[69 + k, -1])

    print(fids)


def test_ee_flip_grape_pulse(grape_fp="fields/c1315_3S_3q_500ns_1000step"):
    grape_fp = "fields/c1331_1S_3q_479ns_1000step"
    grape = load_grape(
        fp=grape_fp, Grape=Grape_ee_Flip, simulate_spectators=True, step=1
    )

    A = get_A_1P_2P(1, [-1, 1], fp=None)
    J = get_J_1P_2P(1, fp=None)
    Bx, By = grape.get_Bx_By()

    H0 = Grape_ee_Flip.get_H0_1n2e(J, A)[0]
    Sx = Grape_ee_Flip.Sx1 + Grape_ee_Flip.Sx2
    Sy = Grape_ee_Flip.Sy1 + Grape_ee_Flip.Sy2
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, 2 * Sx, 2 * Sy)
    U0 = get_U0(H0, tN=grape.tN, N=grape.N)
    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN=grape.tN, N=grape.N)

    Xe0 = get_electron_X(grape.tN, grape.N, 0, A=A, J=J, Bx=Bx, By=By)
    A1 = get_A_1P_2P(1, [-1, -1], fp=None)
    Xe1 = get_electron_X(grape.tN, grape.N, 0, A=A1, J=J, Bx=Bx, By=By)
    # print_rank2_tensor(X[-1])

    Uf = remove_leading_phase(X[-1])
    print_rank2_tensor(Uf)
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_unitary(Uf, ax=ax)
    # print_rank2_tensor(grape.X[0, -1])


def get_unitary_X_from_grape(grape, Xn, Yn):
    H0 = grape.get_H0()[0]
    Bx, By = grape.get_Bx_By()

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=Xn, Y=Yn)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, grape.tN, grape.N)
    return X


def test_stupid_ham():
    nS = 1
    A = get_A_1P_2P(nS, NucSpin=[-1, 1])
    J = get_J_1P_2P(nS)
    tN = 500 * unit.ns
    tN = lock_to_frequency(A[1], tN)
    N = 1000

    grape = Grape_ee_Flip(tN, N, J, A, simulate_spectators=True)

    H0 = grape.spectator_H0()[0]
    Bx, By = pi_pulse_square(-2 * A[1], gamma_e, tN, N)
    B_mag = pi_pulse_field_strength(gamma_e, tN)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, gate.IX, gate.IY)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    print_rank2_tensor(X[-1])
    u_mat = pt.zeros_like(grape.u_mat())
    u_mat[-1] = pt.ones_like(u_mat[-1]) * B_mag / unit.T
    grape.u = uToVector(u_mat)
    grape.print_result()

    pass


def all_multi_system_pulses_old(
    grape_fp1="fields/g378_69S_3q_6974ns_8000step",
    grape_fp2="fields/g370_69S_2q_3000ns_8000step",
    grape_fp3="fields/g379_69S_3q_5983ns_8000step",
    fp=None,
):
    fids1, fields1 = get_fids_and_field_from_fp(grape_fp1, Grape=Grape_ee_Flip, step=1)
    fids2, fields2 = get_fids_and_field_from_fp(grape_fp2, Grape=GrapeESR)
    fids3, fields3 = get_fids_and_field_from_fp(grape_fp3, Grape=Grape_ee_Flip, step=2)

    Bx1, By1, T1 = fields1
    Bx2, By2, T2 = fields2
    Bx3, By3, T3 = fields3

    T_RF = pt.linspace(0, 20 * unit.us, 2000)
    Bx_RF = 1.5 * unit.mT * pt.cos(get_A(1, 1) * (T_RF))
    By_RF = 1.5 * unit.mT * pt.sin(get_A(1, 1) * (T_RF))

    fig, ax = plt.subplot_mosaic(
        [
            ["upper left", "upper middle", "upper right"],
            ["middle", "middle", "middle"],
            ["lower", "lower", "lower"],
        ],
        figsize=(7.5, 3.5),
        layout="constrained",
    )

    # Bx = pt.cat((Bx1, Bx_RF, Bx2, Bx_RF, Bx3))
    # By = pt.cat((By2, By_RF, By2, By_RF, By3))
    # T = pt.cat((T1, T1[-1]+T_RF, T1[-1]+T_RF[-1]+T2, T1[-1]+T_RF[-1]+T2[-1]+T_RF, T1[-1]+2*T_RF[-1]+T2[-1]+T3))

    near_lim = 1.1
    far_lim = 2.5
    plot_fields_twinx(
        Bx1,
        By1,
        T1,
        ax=ax["upper left"],
        near_lim=near_lim,
        far_lim=far_lim,
        tick_lim=1,
        ylabels=False,
    )
    plot_fields_twinx(
        Bx2,
        By2,
        T2,
        ax=ax["upper middle"],
        near_lim=near_lim,
        far_lim=far_lim,
        tick_lim=1,
        ylabels=False,
    )
    axt2 = plot_fields_twinx(
        Bx3,
        By3,
        T3,
        ax=ax["upper right"],
        near_lim=near_lim,
        far_lim=far_lim,
        tick_lim=1,
        ylabels=False,
    )
    ax["upper left"].set_ylabel("Bx (mT)", color=color_cycle[0])
    axt2.set_ylabel("By (mT)", color=color_cycle[1])
    # ax["upper left"].twinx()
    # ax[0].plot(T/unit.ns, Bx/unit.mT)

    ee_flip_ax_key = "middle"
    systems_ax = np.linspace(0, len(fids1) - 1, len(fids1))
    fidelity_bar_plot(
        1 - fids1, ax=ax[ee_flip_ax_key], width=0.8, systems_ax=systems_ax
    )
    fidelity_bar_plot(
        1000000 * (1 - fids2),
        ax=ax[ee_flip_ax_key],
        width=0.8,
        systems_ax=systems_ax,
        bottom=0.01,
    )
    fidelity_bar_plot(
        -(1 - fids3), ax=ax[ee_flip_ax_key], width=0.8, systems_ax=systems_ax
    )

    ax[ee_flip_ax_key].set_ylim([-0.01, 0.02])

    if fp is not None:
        fig.savefig(fp)


def multi_system_n_entangled_CX(
    grape_fp1="fields/g420_70S_3q_3966ns_8000step",
    grape_fp2="fields/g421_70S_3q_3966ns_8000step",
    fp=None,
):
    fids1, fields1 = get_fids_and_field_from_fp(grape_fp1, Grape=Grape_ee_Flip, step=10)
    fids3, fields3 = get_fids_and_field_from_fp(grape_fp2, Grape=Grape_ee_Flip, step=11)

    Bx1, By1, T1 = map(real, fields1)
    Bx3, By3, T3 = map(real, fields3)

    print(
        f"Max fields: {get_max_field(Bx1, By1)/unit.mT:.1f} mT, {get_max_field(Bx3, By3)/unit.mT:.1f} mT"
    )

    Bx_col = color_cycle[0]
    By_col = color_cycle[1]

    fig, ax = plt.subplots(2, 2, gridspec_kw={"width_ratios": [2, 1.5]})
    plot_fields_twinx(
        Bx1[100:],
        By1[100:],
        T1[100:],
        ax=ax[0, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        far_lim=2.4,
        near_lim=0.3,
        tick_lim=1,
    )
    plot_fields_twinx(
        Bx3[100:],
        By3[100:],
        T3[100:],
        ax=ax[1, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        far_lim=2.4,
        near_lim=0.3,
        tick_lim=1,
    )

    n_sys = 70
    n_spec = 1
    systems_ax = np.concatenate(
        (np.linspace(1, n_sys, n_sys), np.array([n_sys + n_spec + 3]))
    )

    fidelity_bar_plot(
        fids1, systems_ax=systems_ax, ax=ax[0, 0], f=[0.9999], colours=["green"]
    )
    fidelity_bar_plot(
        fids3,
        systems_ax=systems_ax,
        f=[0.9999, 0.999],
        colours=["green", "orange"],
        ax=ax[1, 0],
    )

    x_offset, y_offset = [-0.12, -0.35]
    fontsize = 11
    label_axis(ax[0, 0], "1. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 0], "2. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[0, 1], "1. (b)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 1], "2. (b)", x_offset, y_offset, fontsize=fontsize)

    fig.set_size_inches(9.2, 3.7)
    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def all_multi_system_pulses(
    grape_fp1="fields/g378_69S_3q_6974ns_8000step",
    grape_fp2="fields/g370_69S_2q_3000ns_8000step",
    grape_fp3="fields/g379_69S_3q_5983ns_8000step",
    fp=None,
):
    fids1, fields1 = get_fids_and_field_from_fp(
        grape_fp1, Grape=Grape_ee_Flip, step=1, get_from_grape=False
    )
    fids2, fields2 = get_fids_and_field_from_fp(
        grape_fp2, Grape=GrapeESR, A_spec=get_A_spec_single()
    )
    fids3, fields3 = get_fids_and_field_from_fp(grape_fp3, Grape=Grape_ee_Flip, step=3)

    Bx1, By1, T1 = map(real, fields1)
    Bx2, By2, T2 = map(real, fields2)
    Bx3, By3, T3 = map(real, fields3)

    Bx_col = color_cycle[0]
    By_col = color_cycle[1]

    fig, ax = plt.subplots(3, 2, gridspec_kw={"width_ratios": [2, 1.5]})
    plot_fields_twinx(
        Bx1[100:],
        By1[100:],
        T1[100:],
        ax=ax[0, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        far_lim=1,
        near_lim=0.3,
        tick_lim=0.5,
    )
    plot_fields_twinx(
        Bx2, By2, T2, ax=ax[1, 1], prop_zoom_start=0.3, prop_zoom_end=0.31
    )
    plot_fields_twinx(
        Bx3[100:],
        By3[100:],
        T3[100:],
        ax=ax[2, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        far_lim=1,
        near_lim=0.3,
        tick_lim=0.5,
    )

    n_sys = 70
    n_spec = 1
    systems_ax = np.concatenate(
        (np.linspace(1, n_sys, n_sys), np.array([n_sys + n_spec + 3]))
    )

    fidelity_bar_plot(
        fids1, systems_ax=systems_ax, ax=ax[0, 0], f=[0.9999], colours=["green"]
    )
    fidelity_bar_plot(
        fids2,
        systems_ax=systems_ax,
        ax=ax[1, 0],
        f=[0.9999],
        colours=["green"],
    )
    fidelity_bar_plot(
        fids3,
        systems_ax=systems_ax,
        f=[0.9999, 0.999],
        colours=["green", "orange"],
        ax=ax[2, 0],
    )

    x_offset, y_offset = [-0.12, -0.35]
    fontsize = 11
    label_axis(ax[0, 0], "1. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 0], "2. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[2, 0], "3. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[0, 1], "1. (b)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 1], "2. (b)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[2, 1], "3. (b)", x_offset, y_offset, fontsize=fontsize)

    fig.set_size_inches(9.2, 5.2)
    # fig.set_size_inches(18 / 2.52, 18 / 2.54 * 5 / 9)
    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def exchange_vs_detuning(tc=1):
    n = 1000
    eps = linspace(0, 100, n)

    J1 = 0.5 * eps + np.sqrt(tc**2 + (eps / 2) ** 2)
    J2 = -0.5 * eps + np.sqrt(tc**2 + (eps / 2) ** 2)
    k = np.argmax(real(J1) > 100)
    print(J1[k], J2[k])
    ax = plt.subplot()
    ax.plot(eps, J1, label="1P-2P")
    ax.plot(eps, J2, label="2P-1P")
    ax.plot(eps, J1 / J2, label="1P-2P")


def single_system_pulses_and_unitaries(
    fp1="fields/c1326_1S_3q_479ns_1000step",
    fp2="fields/c1345_1S_2q_300ns_1000step",
    fp3="fields/c1350_1S_3q_479ns_2500step",
):
    fp1 = "fields/c1326_1S_3q_479ns_1000step"
    grape1 = load_grape(fp1, Grape_ee_Flip, step=1)
    grape2 = load_grape(fp2, GrapeESR, A_spec=get_A_spec_single())
    grape3 = load_grape(fp3, Grape_ee_Flip, step=2)

    Xn = gate.IXI + gate.IIX
    Yn = gate.IYI + gate.IIY
    XN1 = get_unitary_X_from_grape(grape1, Xn, Yn)[-1]
    XN2 = grape2.X[0, -1]
    XN3 = get_unitary_X_from_grape(grape3, Xn, Yn)[-1]

    def reduce(X):
        return pt.tensor(
            [
                [X[0, 0], X[0, 3], X[0, 4], X[0, 7]],
                [X[3, 0], X[3, 3], X[3, 4], X[3, 7]],
                [X[4, 0], X[4, 3], X[4, 4], X[4, 7]],
                [X[7, 0], X[7, 3], X[7, 4], X[7, 7]],
            ],
            dtype=X.dtype,
            device=X.device,
        )

    def label_getter_1n2e(j):
        if j == 0:
            return f"{Downarrow}{downarrow}{downarrow}"
        elif j == 1:
            return f"{Downarrow}{uparrow}{uparrow}"
        elif j == 2:
            return f"{Uparrow}{downarrow}{downarrow}"
        elif j == 3:
            return f"{Uparrow}{uparrow}{uparrow}"

    def label_getter_2e(j):
        if j == 0:
            return f"{downarrow}{downarrow}"
        elif j == 1:
            return f"{downarrow}{uparrow}"
        elif j == 2:
            return f"{uparrow}{downarrow}"
        elif j == 3:
            return f"{uparrow}{uparrow}"

    # Define the mosaic layout using a list of lists
    # mosaic_layout = [["A", "A", "A"], ["B", "B", "B"], ["C", "C", "C"], ["D", "E", "F"]]

    # Create a figure and add subplots according to the mosaic layout
    # fig, ax = plt.subplot_mosaic(
    #     mosaic_layout, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 1, 1, 2]}
    # )
    # ax["D"].remove()
    # ax["D"] = fig.add_subplot(337, projection="3d")
    # ax["E"].remove()
    # ax["E"] = fig.add_subplot(338, projection="3d")
    # ax["F"].remove()
    # ax["F"] = fig.add_subplot(339, projection="3d")

    # fig0, ax0 = plt.subplots(3, 1)
    # fig0.set_size_inches(8, 4)
    # ax0 = [ax["A"], ax["B"], ax["C"]]

    grid = GridSpec(4, 3, height_ratios=[1, 1, 1, 2.4])
    subgrid0 = GridSpec(3, 3)
    subgrid1 = GridSpec(1, 3)
    fig = plt.figure(figsize=(9, 6))

    subfig0_spec = grid.new_subplotspec((0, 0), rowspan=3, colspan=3)
    subfig0 = fig.add_subfigure(subplotspec=subfig0_spec)

    subfig1_spec = grid.new_subplotspec((3, 0), rowspan=1, colspan=3)
    subfig1 = fig.add_subfigure(subplotspec=subfig1_spec)

    ax0 = np.array([None] * 3)
    twinx0 = np.array([None] * 3)
    ax1 = np.array([None] * 3)
    for i in range(3):
        ax0[i] = subfig0.add_subplot(subgrid0[i, :])
        ax1[i] = subfig1.add_subplot(subgrid1[i], projection="3d")
        twinx0[i] = ax0[i].twinx()

    psi0 = (gate.spin_0 + gate.spin_1) / np.sqrt(2)
    psi = grape1.X[0] @ psi0
    psi = psi[:, [0, 4, 7]]
    plot_psi(
        psi,
        grape1.tN,
        label_getter=label_getter_1n2e_sub,
        ax=twinx0[0],
        legend_loc="right",
    )

    # fig1 = plt.figure()
    # ax1 = [
    #     fig1.add_subplot(131, projection="3d"),
    #     fig1.add_subplot(132, projection="3d"),
    #     fig1.add_subplot(133, projection="3d"),
    # ]
    # ax1 = [ax["D"], ax["E"], ax["F"]]

    hsv = plt.cm.get_cmap("hsv", 256)

    # Remove the last color to avoid repeating the starting color
    start = -90
    arr = np.linspace(0, 1, 256)
    arr = np.concatenate((arr[start:], arr[:start]))
    colors = hsv(arr)[:-1]

    # Create a custom cyclic colormap
    custom_cyclic = LinearSegmentedColormap.from_list("custom_cyclic", colors, N=256)

    cmap = custom_cyclic
    plot_unitary(
        remove_leading_phase(reduce(XN1)),
        label_getter=label_getter_1n2e,
        ax=ax1[0],
        cmap=cmap,
    )
    plot_unitary(
        remove_leading_phase(XN2), label_getter=label_getter_2e, ax=ax1[1], cmap=cmap
    )
    plot_unitary(
        remove_leading_phase(reduce(XN3)),
        label_getter=label_getter_1n2e,
        ax=ax1[2],
        cmap=cmap,
    )
    cbar_ax = subfig1.add_axes([0.01, 0.2, 0.02, 0.6])
    add_colorbar(cmap, cbar_ax)

    plot_fields(*grape1.get_Bx_By(), T=grape1.get_T(), ax=ax0[0])
    plot_fields(*grape2.get_Bx_By(), T=grape2.get_T(), ax=ax0[1])
    plot_fields(*grape3.get_Bx_By(), T=grape3.get_T(), ax=ax0[2])

    ax0[0].set_xticks([])
    ax0[1].set_xticks([])
    ax0[2].set_xticks(
        [0, grape1.tN / unit.ns, grape2.tN / unit.ns, grape3.tN / unit.ns]
    )
    ax0[0].set_xlabel(None)
    ax0[1].set_xlabel(None)

    for k in range(3):
        ax0[k].set_ylabel("B-field (mT)")
        ax0[k].set_xlim([0, ax0[2].get_xlim()[1] * 1.06])

    x_offset = 0.63
    label_axis(ax0[0], "MW1", x_offset, y_offset=0.1)
    label_axis(ax0[1], "MW2", x_offset, y_offset=0.3)
    label_axis(ax0[2], "MW3", x_offset, y_offset=0.0)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    fig.subplots_adjust(top=0.975, bottom=0.150, left=0.080, right=0.980)
    # fig.subplots_adjust(top=0.9)
    # # fig0.tight_layout()
    # # fig1.tight_layout()
    # subfig.tight_layout()


def label_getter_1n2e_sub(j, subs=[1, 1, 2]):
    if j == 0:
        return f"${Downarrow}_{subs[0]}{downarrow}_{subs[1]}{downarrow}_{subs[2]}$"
    elif j == 1:
        return f"${Uparrow}_{subs[0]}{downarrow}_{subs[1]}{downarrow}_{subs[2]}$"
    elif j == 2:
        return f"${Uparrow}_{subs[0]}{uparrow}_{subs[1]}{uparrow}_{subs[2]}$"


def label_getter_1n1e_sub(j, subs=[3, 3]):
    if j == 0:
        return f"${Downarrow}_{subs[0]}{downarrow}_{subs[1]}$"
    elif j == 1:
        return f"${Uparrow}_{subs[0]}{downarrow}_{subs[1]}$"
    elif j == 2:
        return f"${Uparrow}_{subs[0]}{uparrow}_{subs[1]}$"


def label_getter_1e(j, sub=1):
    if j == 0:
        return f"${downarrow}_{sub}$"
    elif j == 1:
        return f"${uparrow}_{sub}$"


def label_getter_2e_reduced(j, subs=[2, 3]):
    if j == 0:
        return f"${uparrow}_{subs[0]}{downarrow}_{subs[1]}$"
    elif j == 1:
        return f"${uparrow}_{subs[0]}{uparrow}_{subs[1]}$"


def small_MW_1_3(grape_fp="fields/c1326_1S_3q_479ns_1000step", fp=None):
    grape = load_grape(grape_fp, Grape_ee_Flip, step=1)

    fig, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1.5, 1.5]})

    plot_fields(
        *grape.get_Bx_By(), T=grape.get_T(), ax=ax[0], linewidth=1, legend_loc="right"
    )
    ax[0].set_ylabel("B-field (mT)")

    colors = ["black", "red", "red"]
    linestyles = ["-", "-", "--"]
    linewidth = 1
    psi0 = (gate.spin_0 + gate.spin_1) / np.sqrt(2)
    psi = grape.X[0] @ psi0
    psi = psi[:, [0, 4, 7]]
    plot_psi(
        psi,
        grape.tN,
        label_getter=label_getter_1n2e_sub,
        ax=ax[1],
        legend_loc="right",
        colors=colors,
        linestyles=linestyles,
        linewidth=linewidth,
        ylabel="$|\psi_{{12}}|^2$",
    )

    psi0_spec = (gate.spin_00 + gate.spin_10) / np.sqrt(2)
    psi_spec = grape.X_spec[0] @ psi0_spec
    psi_spec = psi_spec[:, [0, 2, 3]]
    plot_psi(
        psi_spec,
        grape.tN,
        label_getter=label_getter_1n1e_sub,
        ax=ax[2],
        legend_loc="right",
        colors=colors,
        linestyles=linestyles,
        linewidth=linewidth,
        ylabel="$|\psi_3|^2$",
    )

    for axis in ax:
        axis.set_xticks([0, 100, 200, 300, 400, grape.tN / unit.ns])

    x_offset = -0.07
    label_axis(ax[0], "(a)", x_offset, -0.55)
    label_axis(ax[1], "(b)", x_offset, -0.45)
    label_axis(ax[2], "(c)", x_offset, -0.45)

    fig.set_size_inches(5, 4.5)
    fig.tight_layout()
    if fp is not None:
        fig.savefig(fp)


def set_spine_color(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


def single_systems(
    fp1="fields/c1326_1S_3q_479ns_1000step",
    fp2="fields/c1345_1S_2q_300ns_1000step",
    fp3="fields/c1350_1S_3q_479ns_2500step",
    fp=None,
):
    grape1 = load_grape(fp1, Grape_ee_Flip, step=1)
    grape2 = load_grape(fp2, GrapeESR, A_spec=get_A_spec_single())
    grape3 = load_grape(fp3, Grape_ee_Flip, step=2)
    grape = [grape1, grape2, grape3]

    # grid = GridSpec(5, 3, height_ratios=[1, 1, 1, 2, 2])
    # subgrid1 = GridSpec(3, 3)
    # subgrid2 = GridSpec(2, 3)
    # fig = plt.figure(figsize=(8, 6))

    # subfig1_spec = grid.new_subplotspec((0, 0), rowspan=3, colspan=3)
    # subfig1 = fig.add_subfigure(subplotspec=subfig1_spec)

    # subfig2_spec = grid.new_subplotspec((3, 0), rowspan=2, colspan=3)
    # subfig2 = fig.add_subfigure(subplotspec=subfig2_spec)
    # ax1 = np.array([None] * 3)
    # ax2 = np.array([[None] * 3 for _ in range(2)])

    mosaic = [
        ["A", "A", "A"],
        ["B", "B", "B"],
        ["C", "C", "C"],
        ["D", "E", "F"],
        ["G", "H", "I"],
    ]
    fig, ax = plt.subplot_mosaic(
        mosaic,
        gridspec_kw={"height_ratios": [1, 1, 1, 2, 2]},
        figsize=(20 / 2.54, 20 / 2.54),
    )
    ax1 = [ax["A"], ax["B"], ax["C"]]
    ax2 = np.array([[ax["D"], ax["E"], ax["F"]], [ax["G"], ax["H"], ax["I"]]])

    colors1 = ["black", "black", "red"]
    linestyles1 = ["--", "-", "-"]
    linewidth = 0.8
    psi0_13 = (gate.spin_0 + gate.spin_1) / np.sqrt(2)
    psi0_13_spec = (gate.spin_00 + gate.spin_10) / np.sqrt(2)

    psi0_2 = gate.spin_10
    psi0_2_spec = gate.spin_0

    psi0 = [psi0_13, psi0_2, psi0_13]
    psi0_spec = [psi0_13_spec, psi0_2_spec, psi0_13_spec]

    pulse_labels = ["MW1", "MW2", "MW3"]
    pulse_labels = ["1. (a)", "2. (a)", "3. (a)"]
    facecolors = ["red", "blue", "green"]
    x_offset = -0.3
    y_offset = -0.2
    for i in range(3):
        # ax1[i] = subfig1.add_subplot(subgrid1[i, :])
        plot_fields(
            *grape[i].get_Bx_By(), T=grape[i].get_T(), ax=ax1[i], legend_loc="right"
        )
        ax1[i].set_ylabel("B-field (mT)")
        set_spine_color(ax1[i], facecolors[i])
        set_spine_color(ax2[0, i], facecolors[i])
        set_spine_color(ax2[1, i], facecolors[i])
        # for j in range(2):
        # ax2[j, i] = subfig2.add_subplot(subgrid2[j, i])
        if i in [0, 2]:
            psi = grape[i].X[0] @ psi0[i]
            psi_spec = grape[i].X_spec[0] @ psi0_spec[i]
            psi = psi[:, [0, 4, 7]]
            psi_spec = psi_spec[:, [0, 2, 3]]
            label_getter1 = label_getter_1n2e_sub
            label_getter2 = label_getter_1n1e_sub
            colors = colors1
            linestyles = linestyles1
            ylabel1 = "$|\psi_{{12}}|^2$"
            ylabel2 = "$|\psi_{{3}}|^2$"
        else:
            X = grape[i].X[0]
            X_IP = dagger(get_U0(grape[i].get_H0()[0], T=grape[i].get_T())) @ X
            psi_spec = X_IP @ psi0[i]
            psi = grape[i].X_spec[0] @ psi0_spec[i]
            psi_spec = psi_spec[:, [2, 3]]
            label_getter1 = label_getter_1e
            label_getter2 = label_getter_2e_reduced
            colors = colors1[1:]
            linestyles = linestyles1[1:]
            ylabel1 = "$|\psi_{{1}}|^2$"
            ylabel2 = "$|\psi_{{23}}|^2$"
        plot_psi(
            psi,
            grape[i].tN,
            label_getter=label_getter1,
            ax=ax2[0, i],
            legend_loc="right",
            colors=colors,
            linestyles=linestyles,
            linewidth=linewidth,
            ylabel=ylabel1,
        )
        plot_psi(
            psi_spec,
            grape[i].tN,
            label_getter=label_getter2,
            ax=ax2[1, i],
            legend_loc="right",
            colors=colors,
            linestyles=linestyles,
            linewidth=linewidth,
            ylabel=ylabel2,
        )

        xf = round(grape[i].tN / unit.ns)
        ax1[i].set_yticks([-0.25, 0, 0.25])
        ax1[i].set_xticks([0, 100, 200, 300, 400, xf])
        ax1[i].set_xlim([-20, 570])
        ax1[i].set_ylim([-0.28, 0.28])
        ax2[0, i].set_xticks([0, 250, xf])
        ax2[1, i].set_xticks([0, 250, xf])

        fontsize = 12

    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def multi_n_entangled_3_pulse(
    grape_fp1="fields/g378_69S_3q_6974ns_8000step",
    grape_fp2="fields/g421_70S_3q_3966ns_8000step",
    grape_fp3="fields/g379_69S_3q_5983ns_8000step",
    fp=None,
):
    fids1, fields1 = get_fids_and_field_saved(grape_fp1)
    fids2, fields2 = get_fids_and_field_saved(grape_fp2)
    fids3, fields3 = get_fids_and_field_saved(grape_fp3)

    Bx1, By1, T1 = map(real, fields1)
    Bx2, By2, T2 = map(real, fields2)
    Bx3, By3, T3 = map(real, fields3)

    Bx_col = color_cycle[0]
    By_col = color_cycle[1]

    fig, ax = plt.subplots(3, 2, gridspec_kw={"width_ratios": [2, 1.5]})
    plot_fields_twinx(
        Bx1[100:],
        By1[100:],
        T1[100:],
        ax=ax[0, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        far_lim=1,
        near_lim=0.3,
        tick_lim=0.5,
    )
    plot_fields_twinx(
        Bx2, By2, T2, ax=ax[1, 1], prop_zoom_start=0.3, prop_zoom_end=0.31
    )
    plot_fields_twinx(
        Bx3[100:],
        By3[100:],
        T3[100:],
        ax=ax[2, 1],
        prop_zoom_start=0.301,
        prop_zoom_end=0.3053,
        # far_lim=1,
        # near_lim=0.3,
        # tick_lim=0.5,
    )

    n_sys = 70
    n_spec = 1
    systems_ax = np.concatenate(
        (np.linspace(1, n_sys, n_sys), np.array([n_sys + n_spec + 3]))
    )

    fidelity_bar_plot(
        fids1, systems_ax=systems_ax, ax=ax[0, 0], f=[0.9999], colours=["green"]
    )
    fidelity_bar_plot(
        fids2,
        systems_ax=systems_ax,
        ax=ax[1, 0],
        f=[0.9999],
        colours=["green"],
    )
    fidelity_bar_plot(
        fids3,
        systems_ax=systems_ax,
        f=[0.9999, 0.999],
        colours=["green", "orange"],
        ax=ax[2, 0],
    )

    x_offset, y_offset = [-0.12, -0.35]
    fontsize = 11
    label_axis(ax[0, 0], "1. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 0], "2. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[2, 0], "3. (a)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[0, 1], "1. (b)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[1, 1], "2. (b)", x_offset, y_offset, fontsize=fontsize)
    label_axis(ax[2, 1], "3. (b)", x_offset, y_offset, fontsize=fontsize)

    fig.set_size_inches(9.2, 5.2)
    # fig.set_size_inches(18 / 2.52, 18 / 2.54 * 5 / 9)
    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def single_sys_n_entangled_CX(fp=None):
    tN1 = 300 * unit.ns  # lock_to_frequency(get_A(1, 1), 800 * unit.ns)
    tN2 = 300 * unit.ns
    N = 1500
    kappa = 1
    lam = 0

    J = pt.tensor(50 * unit.MHz, dtype=real_dtype)
    A = get_A_1P_2P(1)

    grape1 = run_CNOTs(
        tN=tN1,
        N=N,
        J=J,
        A=A,
        lam=lam,
        kappa=kappa,
        Grape=Grape_ee_Flip,
        verbosity=0,
        step=4,
        stop_fid_avg=0.999,
        stop_fid_min=0.999,
        save_data=False,
        prev_grape_fp="fields/c1438_1S_3q_300ns_1500step",
        run_optimisation=False,
    )
    grape2 = run_CNOTs(
        tN=tN2,
        N=N,
        J=J,
        A=A,
        lam=lam,
        kappa=kappa,
        Grape=Grape_ee_Flip,
        verbosity=0,
        step=3,
        stop_fid_avg=0.999,
        stop_fid_min=0.999,
        save_data=False,
        prev_grape_fp="fields/c1439_1S_3q_300ns_1500step",
        run_optimisation=False,
    )

    Bx1, By1 = grape1.get_Bx_By()
    Bx2, By2 = grape2.get_Bx_By()

    mosaic = [["1a", "1a"], ["2a", "2a"], ["1b", "1c"], ["2b", "2c"]]
    fig, ax = plt.subplot_mosaic(mosaic)

    plot_fields(*grape1.get_Bx_By(), T=grape1.get_T(), ax=ax["1a"], legend_loc="right")
    plot_fields(*grape2.get_Bx_By(), T=grape2.get_T(), ax=ax["2a"], legend_loc="right")

    ax["1a"].set_yticks([-1, 0, 1])
    ax["2a"].set_yticks([-0.5, 0, 0.5])

    psi0_1 = gate.spin_1
    psi0_2 = normalise(gate.spin_10)
    psi0_spec_1 = normalise(gate.spin_00 + gate.spin_10 / np.sqrt(2))
    psi0_spec_2 = normalise(gate.spin_00 + gate.spin_11 / np.sqrt(2))

    psi1 = (dagger(get_U0(grape1.get_H0()[0], T=grape1.get_T())) @ grape1.X[0]) @ psi0_1
    psi2 = (dagger(get_U0(grape2.get_H0()[0], T=grape2.get_T())) @ grape2.X[0]) @ psi0_2
    psi1_spec = grape1.X_spec[0] @ psi0_spec_1
    psi2_spec = grape2.X_spec[0] @ psi0_spec_2

    label_getter_1 = lambda i: ["⇑↑↓", "⇑↑↑"][i]
    label_getter_2 = lambda i: ["⇓↓↑", "⇓↑↑"][i]
    label_getter_spec_1 = lambda i: ["⇓↓", "⇑↓"][i]
    label_getter_spec_2 = lambda i: ["⇓↓", "⇑↑"][i]

    legend_loc = "center"
    colors = ["black", "red"]
    plot_psi(
        psi1[:, [6, 7]],
        T=grape1.get_T(),
        ax=ax["1b"],
        label_getter=label_getter_1,
        legend_loc="center",
        colors=colors,
    )
    plot_psi(
        psi2[:, [1, 3]],
        T=grape2.get_T(),
        ax=ax["2b"],
        label_getter=label_getter_2,
        legend_loc="center",
        colors=colors,
    )
    plot_psi(
        psi1_spec[:, [0, 2]],
        T=grape1.get_T(),
        ax=ax["1c"],
        label_getter=label_getter_spec_1,
        legend_loc="center",
        colors=colors,
    )
    plot_psi(
        psi2_spec[:, [0, 3]],
        T=grape2.get_T(),
        ax=ax["2c"],
        label_getter=label_getter_spec_2,
        legend_loc="center",
        colors=colors,
    )
    ax["1a"].set_ylabel("B-field (mT)")
    ax["2a"].set_ylabel("B-field (mT)")
    ax["1b"].set_ylabel("$|\psi_{{12}}|^2$")
    ax["1c"].set_ylabel("$|\psi_{{3}}|^2$")
    ax["2b"].set_ylabel("$|\psi_{{23}}|^2$")
    ax["2c"].set_ylabel("$|\psi_{{1}}|^2$")
    for ax in [ax["1b"], ax["1c"], ax["2b"], ax["2c"]]:
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])

    fig.set_size_inches(14 / 2.54, 14 / 2.54)
    fig.tight_layout()
    if fp is not None:
        fig.savefig(fp)


def latest_multi_sys(fp=None):
    # grape_fp1 = "fields/g448_70S_3q_3966ns_8000step"
    grape_fp1 = "fields/g513_70S_3q_1983ns_8000step"
    grape_fp2 = "fields/g491_70S_3q_2974ns_8000step"
    grape_fp3 = "fields/g490_70S_3q_2974ns_8000step"

    multi_n_entangled_3_pulse(
        grape_fp1=grape_fp1, grape_fp2=grape_fp2, grape_fp3=grape_fp3, fp=fp
    )


def three_pulses_one_ax(grape1, grape2, grape3, ax):
    """
    Puts three pulses one on top of other similar to with three axes, but
    on a single axis, with ticks set appropriately.
    """
    plot_fields()


def latest_single_sys(
    fp1="fields/c1457_1S_3q_274ns_800step",
    fp2="fields/c1469_1S_3q_342ns_500step",
    fp3="fields/c1463_1S_3q_274ns_500step",
    fp=None,
):
    grape1 = load_grape(fp1, Grape_ee_Flip, step=1)
    grape2 = load_grape(fp2, Grape_ee_Flip, step=2)
    grape3 = load_grape(fp3, Grape_ee_Flip, step=3)
    grape = [grape1, grape2, grape3]

    ax1 = np.array([None] * 3)
    ax2 = np.array([[None] * 3 for _ in range(2)])

    mosaic = [
        ["A", "A", "A"],
        ["D", "D", "D"],
    ]
    fig, ax = plt.subplot_mosaic(
        mosaic,
        gridspec_kw={"height_ratios": [1.5, 1]},
        figsize=(20 / 2.54, 16 / 2.54),
    )

    fontsize = 12
    colors1 = ["black", "black", "red"]
    linestyles1 = ["--", "-", "-"]
    linewidth = 0.8
    psi0_13 = (gate.spin_0 + gate.spin_1) / np.sqrt(2)
    psi0_13_spec = (gate.spin_00 + gate.spin_10) / np.sqrt(2)

    psi0_2 = gate.spin_10
    psi0_2_spec = gate.spin_00

    psi0 = [psi0_13, psi0_2, psi0_13]
    psi0_spec = [psi0_13_spec, psi0_2_spec, psi0_13_spec]

    pulse_labels = ["(a)", "(b)", "(c)"]
    facecolors = ["red", "blue", "green"]
    x_offset = -0.3
    y_offset = -0.2

    def label_getter_5spin(j):
        # if j == 0:
        #     return "⇓↓↓↓⇓"
        if j == 16:
            return "⇑↓↓↓⇓"
        elif j == 18:
            return "⇑↓↓↑⇓"
        elif j == 28:
            return "⇑↑↑↓⇓"
        elif j == 30:
            return "⇑↑↑↑⇓"

    def label_getter_reduced_5spin(j):
        return ["⇑↓↓↓⇓", "⇑↓↓↑⇓", "⇑↑↑↓⇓", "⇑↑↑↑⇓"][j]

    colors_5spin = ["gray"] * 32
    # colors_5spin[0] = "black"
    colors_5spin[16] = "red"
    colors_5spin[18] = "orange"
    colors_5spin[28] = "purple"
    colors_5spin[30] = "green"

    label_getter_1 = lambda i: ["⇓↓↓↓⇓", "⇑↓↓↓⇓", "⇑↑↑↓⇓"][i]
    label_getter_2 = lambda i: ["⇓↓↓↓⇓", "⇑↑↑↓⇓", "⇑↑↑↑⇓"][i]
    label_getters = [label_getter_1, label_getter_2, label_getter_1]

    T1 = grape[0].get_T()
    T2 = T1[-1] + grape[1].get_T()
    T3 = T2[-1] + grape[2].get_T()

    T = pt.cat((T1, T2, T3))

    field_sep = 0.9

    ytick_locs = np.array([-0.25, 0, 0.25])
    yticks = [-0.25, 0, 0.25]
    ax["A"].set_yticks(
        np.concatenate(
            (ytick_locs, ytick_locs + field_sep, ytick_locs + 2 * field_sep)
        ),
        3 * yticks,
    )
    ax["A"].set_ylim([-0.3, field_sep * 2 + 0.3])
    ax["A"].set_ylabel("B-field (mT)")
    ax["A"].set_xticks([0] + [g.tN / unit.ns for g in grape])
    ax["A"].set_xlim([-20, 350])
    border_width = 0.7
    ax["A"].axhline(
        field_sep / 2, linestyle="--", linewidth=border_width, color="black"
    )
    ax["A"].axhline(
        3 * field_sep / 2, linestyle="--", linewidth=border_width, color="black"
    )
    legend_locs = ["upper right", False, False]

    for i in range(3):
        # ax1[i] = subfig1.add_subplot(subgrid1[i, :])
        ax["A"].text(-18, i * field_sep - 0.28, f"MW{3-i}", fontsize=12)
        Bx, By = grape[i].get_Bx_By()
        By += field_sep * unit.mT * (2 - i)
        Bx += field_sep * unit.mT * (2 - i)
        plot_fields(
            Bx,
            By,
            T=grape[i].get_T(),
            ax=ax["A"],
            legend_loc=legend_locs[i],
        )
        # for j in range(2):
        # ax2[j, i] = subfig2.add_subplot(subgrid2[j, i])
        psi = grape[i].X[0] @ psi0[i]
        psi_spec = grape[i].X_spec[0] @ psi0_spec[i]
        psi = psi[:, [0, 4, 7]]
        psi_spec = psi_spec[:, [0, 2, 3]]

        xf = round(grape[i].tN / unit.ns)

    # WAVE FUNCTION PLOT

    # interested in psi[i] for i in [16, 18, 28, 30]

    psi1 = grape[0].X[0] @ gate.spin_1
    psi1_spec = (grape[0].X_spec[0] @ gate.spin_00)[:, [0, 2, 1, 3]]
    psi1_5s = batch_kron(psi1, psi1_spec)
    psi2 = (grape[1].X[0] @ gate.spin_10)[:, [0, 4, 2, 6, 1, 5, 3, 7]]
    psi2_spec = grape[1].X_spec[0] @ gate.spin_11
    psi2_5s = batch_kron(psi2_spec, psi2)
    psi3 = grape[2].X[0] @ gate.spin_1
    psi3_spec = (grape[2].X_spec[0] @ gate.spin_01)[:, [0, 2, 1, 3]]
    psi3_5s = batch_kron(psi3, psi3_spec)

    psi = pt.cat((psi1_5s, psi2_5s, psi3_5s))
    psi = psi[:, [16, 18, 28, 30]]
    plot_psi(
        psi,
        T=T,
        label_getter=label_getter_reduced_5spin,
        colors=["red", "green", "gray", "black"],
        ax=ax["D"],
        linestyles=["-", "-", "--", ":"],
    )
    ax["D"].set_xlim(0, real(T[-1] / unit.ns) + 140)
    ax["D"].set_ylim([0, 1.2])
    t12 = real(T1[-1] / unit.ns)
    t23 = real(T2[-1] / unit.ns)
    tf = real(T3[-1] / unit.ns)
    ax["D"].set_xticks([0, pt.round(t12), pt.round(t23), pt.round(tf)])
    linewidth = 0.6
    color = "gray"
    linestyle = "-"
    ax["D"].axvline(t12, color="black", linestyle=linestyle, linewidth=linewidth)
    ax["D"].axvline(t23, color="black", linestyle=linestyle, linewidth=linewidth)
    ax["D"].axvline(tf, color="black", linestyle=linestyle, linewidth=linewidth)
    ax["D"].set_yticks([0, 1])
    ax["D"].set_ylabel("$|\psi_k|$")
    y_offset = 0.8
    fontsize = 13
    label_axis(ax["D"], "Steps 2-3", 0.05, y_offset, fontsize)
    label_axis(ax["D"], "Step 4", 0.35, y_offset, fontsize)
    label_axis(ax["D"], "Steps 5-6", 0.63, y_offset, fontsize)

    label_axis(ax["A"], "(a)", x_offset=-0.07, y_offset=-0.1, fontsize=fontsize)
    label_axis(ax["D"], "(b)", x_offset=-0.07, y_offset=-0.25, fontsize=fontsize)

    if fp is not None:
        fig.savefig(fp)


def multi_sys_pie_with_cost(grape_fp1="fields/g508_70S_3q_3966ns_8500step"):
    fids1, fields1 = get_fids_and_field_saved(grape_fp1)

    cost_hist = pt.load(grape_fp1 + "_cost_hist")
    # Bx1, By1, T1 = map(real, fields1)

    fig, ax = plt.subplots(1, 1)

    # fidelity_pie_chart(fids1, [0.9999, 0.999, 0.99])
    plot_cost_hist(cost_hist, ax)

    # plot_fields_twinx(
    #     Bx1[100:],
    #     By1[100:],
    #     T1[100:],
    #     ax=ax[0],
    #     prop_zoom_start=0.301,
    #     prop_zoom_end=0.3053,
    #     far_lim=1,
    #     near_lim=0.3,
    #     tick_lim=0.5,
    # )


def test_multi_sys_grape(grape_fp, step):
    grape = load_grape(grape_fp, Grape=Grape_ee_Flip, step=step, verbosity=-1)
    # print(grape.J/unit.MHz)
    # print(grape.A/unit.MHz)
    # fids, fields = get_fids_and_field_from_fp(grape_fp, get_from_grape=True, Grape=Grape_ee_Flip, step=step)
    # fidelity_bar_plot(fids, ylim = [0,1])


def multisys_upgraded(
    grape_fp1="fields/g513_70S_3q_1983ns_8000step",
    grape_fp2="fields/g491_70S_3q_2974ns_8000step",
    grape_fp3="fields/g521_70S_3q_2974ns_8000step",
    figure_fp=None,
):
    (fids1, fields1), (fids2, fields2), (fids3, fields3) = map(
        get_fids_and_field_saved, [grape_fp1, grape_fp2, grape_fp3]
    )

    Bx1, By1, T1 = map(real, fields1)
    Bx2, By2, T2 = map(real, fields2)
    Bx3, By3, T3 = map(real, fields3)
    Bx = [Bx1, Bx2, Bx3]
    By = [By1, By2, By3]
    T = [T1, T2, T3]

    cost1, cost2, cost3 = map(
        pt.load,
        [grape_fp1 + "_cost_hist", grape_fp2 + "_cost_hist", grape_fp3 + "_cost_hist"],
    )

    Bx_col = color_cycle[0]
    By_col = color_cycle[1]
    fig, ax = plt.subplot_mosaic(
        [["A", "B"], ["C", "C"]],
        gridspec_kw={"height_ratios": [1, 1.5], "width_ratios": [1.5, 1]},
    )
    fig.set_size_inches(16 / 2.54, 18 / 2.54)

    #### FIDELITY BAR PLOT
    avgfid1, avgfid2, avgfid3 = map(pt.mean, [fids1[:70], fids2[:70], fids3[:70]])
    minfid1, minfid2, minfid3 = map(pt.min, [fids1[:70], fids2[:70], fids3[:70]])
    (specfid1, specfid2, specfid3) = (fids1[-1], fids2[-1], fids3[-1])

    fids = np.array(
        [
            avgfid1,
            minfid1,
            specfid1,
            avgfid2,
            minfid2,
            specfid2,
            avgfid3,
            minfid3,
            specfid3,
        ]
    )
    x = np.array([0, 1, 2])
    x = np.concatenate((x, x + 4, x + 8))
    colors = 3 * ["orange", "blue", "green"]
    colors = 3 * ["red"] + 3 * ["blue"] + 3 * ["green"]

    ax["A"].set_ylim(0.99, 1.001)
    ax["A"].set_yticks([0.99, 1.00])
    ax["A"].set_ylabel("Fidelity")
    fewer_fid_bar(
        fids,
        ax["A"],
        x=x,
        labels=["Average", "Minimum", "Spectator"] * 3,
        colors=colors,
        label_fontsize=10,
    )
    ax["A"].set_xticks([1, 5, 9], ["MW1", "MW2", "MW3"])

    #### COST HIST PLOT
    plot_cost_hist(cost1, ax["B"], legend_label="cost 1", color="red")
    plot_cost_hist(cost2, ax["B"], legend_label="cost 2", color="purple")
    plot_cost_hist(cost3, ax["B"], legend_label="cost 3", color="green")
    ax["B"].legend(loc="lower left")

    #### FIELD PLOT
    axt = ax["C"].twinx()
    field_sep = 5
    for i in range(3):
        plot_fields_twinx(
            Bx[i][100:] + (2 - i) * field_sep * unit.mT,
            By[i][100:] + (2 - i) * field_sep * unit.mT,
            T[i][100:],
            ax=ax["C"],
            axt=axt,
            prop_zoom_start=0.301,
            prop_zoom_end=0.32,
            far_lim=2,
            near_lim=0.3,
            tick_lim=1,
        )
        ax["C"].text(
            -0.025,
            i * field_sep - 0.5,
            f"MW{i+1}",
            ha="center",
            va="center",
            fontsize=12,
            rotation=90,
        )
    for axis in [ax["C"], axt]:
        ylim = axis.get_ylim()
        axis.set_ylim(ylim[0], 2 * field_sep + ylim[1])
        yticks = axis.get_yticks()
        yticks_new = np.concatenate(
            (yticks, yticks + field_sep, yticks + 2 * field_sep)
        )
        ytick_labels = np.concatenate((yticks, yticks, yticks))
        axis.set_yticks(yticks_new, ytick_labels)

    label_axis(ax["C"], "Zoomed in", x_offset=0.27, y_offset=0.65, fontsize=12)
    # label_axis(ax["A"], "(a)", y_offset=-0.2)
    # label_axis(ax["B"], "(b)", x_offset=-0.01, y_offset=-0.0)
    # label_axis(ax["C"], "(c)", x_offset=-0.06, y_offset=-0.1)
    if figure_fp is not None:
        fig.savefig(figure_fp)


def three_fidelity_bar_plots(
    grape_fp1="fields/g510_70S_3q_2974ns_8500step",
    grape_fp2="fields/g491_70S_3q_2974ns_8000step",
    grape_fp3="fields/g490_70S_3q_2974ns_8000step",
    figure_fp=None,
):
    fids1, fields1 = get_fids_and_field_saved(grape_fp1)
    fids2, fields2 = get_fids_and_field_saved(grape_fp2)
    fids3, fields3 = get_fids_and_field_saved(grape_fp3)

    fig, ax = plt.subplots(3, 1)

    n_sys = 70
    n_spec = 1
    systems_ax = np.concatenate(
        (np.linspace(1, n_sys, n_sys), np.array([n_sys + n_spec + 3]))
    )

    fidelity_bar_plot(
        fids1, systems_ax=systems_ax, ax=ax[0], f=[0.9999], colours=["green"]
    )
    fidelity_bar_plot(
        fids2,
        systems_ax=systems_ax,
        ax=ax[1],
        f=[0.9999],
        colours=["green"],
    )
    fidelity_bar_plot(
        fids3,
        systems_ax=systems_ax,
        f=[0.9999, 0.999],
        colours=["green", "orange"],
        ax=ax[2],
    )

    for axis in ax:
        axis.set_xticks([1, 30, 60, 74], [1, 30, 60, "Spec"])

    fig.set_size_inches(16 / 2.54, 13 / 2.54)
    fig.tight_layout()
    if figure_fp is not None:
        fig.savefig(figure_fp)


def test_check_spike(grape_fp):
    grape = load_grape(grape_fp, Grape=Grape_ee_Flip, step=1)
    pass


if __name__ == "__main__":
    # get_2e_flip_grape_pulse()
    # run_2e_swap()
    # nuclear_spin_flip()
    # show_2e_flip()
    # get_2P_1P_CX_pulse()
    # show_2P_1P_CX_pulse()
    # multi_2P_1P_CX()
    # check_e_flip_phases()
    # assess_field()
    # test_ee_flip_grape_pulse(grape_fp="fields/c1326_1S_3q_479ns_1000step")
    # test_stupid_ham()
    # all_multi_system_pulses()
    # exchange_vs_detuning()
    # single_system_pulses_and_unitaries()
    # small_MW_1_3("fields/c1350_1S_3q_479ns_2500step")
    # single_systems()
    # multi_system_n_entangled_CX()
    # single_sys_n_entangled_CX()
    # multi_n_entangled_3_pulse()
    # latest_multi_sys()
    # latest_single_sys()
    # test_multi_sys_grape("fields/g506_70S_3q_3966ns_8000step", step=1)
    # multi_sys_pie_with_cost()


    # grape_fp1 = "fields/g517_70S_3q_2769ns_8000step"
    grape_fp1 = "fields/g510_70S_3q_2974ns_8500step"
    grape_fp2 = "fields/g491_70S_3q_2974ns_8000step"
    # grape_fp3 = "fields/g490_70S_3q_2974ns_8000step"
    grape_fp3 = "fields/g521_70S_3q_2974ns_8000step"


    # three_fidelity_bar_plots(grape_fp1, grape_fp2, grape_fp3)
    multisys_upgraded(grape_fp1, grape_fp2, grape_fp3)
    # test_check_spike(grape_fp=grape_fp1)
    plt.show()
