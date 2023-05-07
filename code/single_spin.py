import torch as pt
import matplotlib
import numpy as np

if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


import gates as gate
from pulse_maker import pi_pulse_square, frame_transform_pulse
import atomic_units as unit
from gates import spin_up, spin_down
from visualisation import plot_fields, plot_psi, show_fidelity, plot_phases
from data import gamma_e, dir, cplx_dtype
from utils import forward_prop, NoiseModels
from eigentools import lock_to_frequency
from hamiltonians import (
    get_pulse_hamiltonian,
    sum_H0_Hw,
    get_U0,
    get_X_from_H,
    multi_sys_single_electron_H0,
)
from data import get_A
from GRAPE import Grape, load_total_field
from hamiltonians import single_electron_H0, single_J_coupled_electron_H0
from visualisation import *


from pdb import set_trace


def label_getter(j):
    if j == 0:
        return "$|<0|\psi>|$"
    return "$|<1|\psi>|$"


def show_single_spin_evolution(
    Bz=0 * unit.T,
    A=get_A(1, 1),
    tN=500 * unit.ns,
    N=100000,
    target=gate.X,
    psi0=spin_up,
    fp=None,
):

    tN = lock_to_frequency(A, tN)

    w_res = gamma_e * Bz + 2 * A
    # fig,ax = plt.subplots(2,2)
    fig, ax = plt.subplots(3, 1)
    Bx, By = pi_pulse_square(w_res, gamma_e / 2, tN, N)
    plot_fields(Bx, By, tN, ax[0])

    H0 = single_electron_H0(Bz, A)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    U0 = get_U0(H0, tN=tN, N=N)

    U = pt.matrix_exp(-1j * H * tN / N)
    X = forward_prop(U)
    show_fidelity(X, tN=tN, target=gate.X, ax=ax[1])
    psi = X @ psi0

    plot_psi(psi, tN=tN, ax=ax[2], label_getter=label_getter)
    # plot_phases(psi, tN, ax[1,1])
    plt.tight_layout()

    if fp is not None:
        plt.savefig(fp)


def get_single_spin_X(
    Bz=0 * unit.T,
    A=get_A(1, 1),
    gamma=gamma_e,
    T=None,
    Bx=None,
    By=None,
    tN=None,
    N=None,
):
    evaluate_timestep_inputs(T, tN, N)
    if Bx is None:
        Bx = pt.zeros(N, dtype=cplx_dtype)
    if By is None:
        By = pt.zeros(N, dtype=cplx_dtype)
    H0 = single_electron_H0(Bz, A)
    Hw = get_pulse_hamiltonian(Bx, By, gamma)
    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, T=T)
    return X


def test_grape_pulse_with_varied_J(
    fp="fields/c1095_1S_2q_300ns_2000step_XY", Bx=None, By=None, T=None
):

    n = 100
    J = pt.linspace(0 * unit.kHz, 4 * unit.kHz, n)
    fids = pt.zeros(n)
    if Bx is None:
        Bx, By, T = load_total_field(fp)
        print(f"Max field = {get_max_field(Bx, By)/unit.mT} mT")
    for i in range(n):
        X = get_single_spin_X(Bx=Bx, By=By, T=T, A=get_A(1, 1) + 2 * J[i])
        fids[i] = fidelity(X[-1], gate.Id)
    ax = plt.subplot()
    ax.plot(J / unit.kHz, fids)
    ax.set_xlabel("J (kHz)")
    ax.set_ylabel("Fidelity")


def test_grape_pulse_on_non_res_spin(
    fp="fields/g338_69S_2q_4000ns_8000step", Bx=None, By=None, T=None, ax=None
):
    A = get_A(1, 1)
    if Bx is None:
        Bx, By, T = load_total_field(fp)
    X = get_single_spin_X(Bx=Bx, By=By, T=T, A=A)
    fids = fidelity_progress(X, gate.Id)

    # plot_fidelity(fids, T=T, ax=ax)

    psi0 = (gate.spin_0 + gate.spin_1) / np.sqrt(2)
    plot_psi_with_phase(X @ psi0, T=T)

    print_rank2_tensor(X[-1])
    print(f"tested fidelity = {fidelity(X[-1], gate.Id)}")
    # plot_fields(Bx, By, T=T)


def test_multi_grape_pulse_on_non_res_spin():
    fps = ["fields/c1095_1S_2q_300ns_2000step_XY"]
    ax = plt.subplot()
    for fp in fps:
        test_grape_pulse_on_non_res_spin(fp=fp, ax=ax)


class SingleElectronGRAPE(Grape):
    def __init__(self, tN, N, Bz=0, A=get_A(1, 1), J=None, **kwargs):
        if J is not None:
            print("J != None: assume this is for SWAP X-gates.")
            self.target = pt.cat(
                (
                    pt.einsum("k,ab->kab", pt.ones(len(J)), gate.X),
                    pt.einsum("k,ab->kab", pt.ones(len(J)), gate.Id),
                )
            )
        self.nS = self.get_nS(A, J)
        self.nq = 1
        self.Bz = Bz
        self.A = A
        self.J = J
        super().__init__(tN, N, **kwargs)
        self.Hw = self.get_Hw()
        self.fun = self.cost

    def get_default_targets(self):
        return gate.H

    def get_nS(self, A, J):
        try:
            return len(A)
        except:
            try:
                return 2 * len(J)
            except:
                return 1  # Ew

    def print_setup_info(self):
        super().print_setup_info()
        print(f"Bz = {self.Bz/unit.T}")
        print(f"A = {self.A/unit.MHz}")

    def get_H0(self, Bz=0):
        if self.J is None:
            if self.nS == 1:
                H0 = single_electron_H0(Bz, self.A)
                return H0.reshape(1, *H0.shape)
            else:
                H0 = multi_sys_single_electron_H0(Bz, self.A)
                return H0
        else:
            return pt.cat(
                (
                    single_J_coupled_electron_H0(Bz, -self.A, self.J),
                    single_J_coupled_electron_H0(Bz, self.A, self.J),
                )
            )

    def get_Hw(self):
        return pt.einsum("kj,ab->kjab", self.x_cf, gate.X) + pt.einsum(
            "kj,ab->kjab", self.y_cf, gate.Y
        )

    def get_all_resonant_frequencies(self):
        return pt.tensor([gamma_e * self.Bz + 2 * self.A])

    def print_result(self):
        super().print_result()
        fidelities = self.fidelity()[0]
        if self.verbosity >= 2:
            print(
                f"All fidelities: " + ", ".join([str(fid.item()) for fid in fidelities])
            )
        for system, fidelity in enumerate(fidelities):
            print(f"system {system}: fidelity = {fidelity}")

    def plot_result(self, psi0=spin_up):
        fig, ax = plt.subplots(2, 2)

        psi = self.X[0] @ psi0

        self.plot_u(ax[0, 0], legend_loc=False)
        # self.plot_control_fields(ax[0,1])
        self.plot_cost_hist(ax[0, 1])

        # plot_psi(psi, tN=self.tN, ax=ax[1,1], label_getter=label_getter)
        Bx, By = self.sum_XY_fields()
        self.plot_XY_fields(ax[1, 0], Bx, By)
        # plot_phases(psi, self.tN, ax[0])
        show_fidelity(self.X[0], tN=self.tN, target=self.target, ax=ax[1, 1])
        fig.set_size_inches(1.1 * fig_width_double, 1.1 * 0.8 * fig_height_double_long)
        fig.tight_layout()

        x_offset = -0.11
        y_offset = -0.25
        label_axis(ax[0, 0], "(a)", x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[0, 1], "(b)", x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[1, 0], "(c)", x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[1, 1], "(d)", x_offset=x_offset, y_offset=y_offset)

        return ax


def run_single_electron_grape(
    A=get_A(1, 1), fp=None, tN=100 * unit.ns, N=100, kappa=1, lam=0
):
    target = gate.Id
    Bz = 0

    u0 = None

    grape = SingleElectronGRAPE(
        tN,
        N,
        Bz=Bz,
        A=A,
        target=target,
        u0=u0,
        kappa=kappa,
        lam=lam,
        noise_model=None,
        ensemble_size=1,
        cost_momentum=0,
    )

    grape.run(max_time=2)
    grape.print_result()
    grape.save_field("fields/single_electron")
    # ax = grape.plot_result()

    if fp is not None:
        ax[0, 0].get_figure().savefig(fp)


def identity_hyperfines(N, tN, nS, kappa=1, max_time=60):
    """
    Perform identity element on all hyperfines simultaneously.
    Eventually would like pulse which does this while performing all 1P-2P CNOTs...
    """
    A = pt.unique(pt.flatten(real(get_A_1P_2P_uniform_J(nS))))
    target = gate.Id
    grape = SingleElectronGRAPE(
        tN,
        N,
        target,
        Bz=np.float64(0),
        A=A,
        u0=None,
        noise_model=None,
        ensemble_size=1,
        cost_momentum=0,
        max_time=max_time,
        kappa=kappa,
    )
    grape.run()
    grape.print_result()
    grape.plot_result()


def J_coupled_X(
    nS,
    N=2000,
    tN=100 * unit.ns,
    max_time=60,
    kappa=1,
    lam=0,
    prev_grape_fp=None,
    run_optimisation=True,
):
    """
    Needed to perform nuclear-electron swap with active exchange.
    """
    target = gate.X
    Bz = 0
    u0 = None

    if prev_grape_fp is None:
        grape = SingleElectronGRAPE(
            tN,
            N,
            target,
            Bz=Bz,
            A=get_A(1, 1),
            J=get_J_low(nS, 2),
            u0=u0,
            noise_model=None,
            ensemble_size=1,
            cost_momentum=0,
            max_time=max_time,
            kappa=kappa,
            lam=lam,
        )
    else:
        grape = load_grape(prev_grape_fp, Grape=SingleElectronGRAPE, max_time=max_time)

    if run_optimisation:
        grape.run()
    grape.print_result()
    grape.save_field("fields/electron_flip")
    ax = grape.plot_result()


def test_pulse_frame_change():
    tN = 200 * unit.ns
    N = 4000
    Bz = 2 * unit.T
    A = get_A(1, 1)
    w0 = gamma_e * Bz + 2 * A
    tN = lock_to_frequency(A, tN)
    T = linspace(0, tN, N)
    Hz = (0.5 * gamma_e * Bz) * gate.Z 
    HA = A * gate.Z

    Uz = get_U0(Hz, T=T)

    Bx, By = pi_pulse_square(2*A, gamma_e, tN, N)
    Bx, By = frame_transform_pulse(Bx, By, T, gamma_e*Bz)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    Hw = pt.einsum('jab,jbc,jcd->jad', dagger(Uz), Hw, Uz)
    H = sum_H0_Hw(HA, Hw)

    visualise_Hw(H, tN)

    X = get_X_from_H(H, T=T)

    #plot_fields(Bx, By, T=T)

    print_rank2_tensor(X[-1])
    print(f"fidelity = {fidelity(gate.X,X[-1])}")


if __name__ == "__main__":

    # show_single_spin_evolution(N=500, tN=100*unit.ns); plt.show()
    # run_single_electron_grape(A=get_A(1, 1), kappa=1e7, tN=400 * unit.ns, N=500)
    # J_coupled_X(
    #     nS=15, N=1000, tN=200 * unit.ns, max_time=None, kappa=1e5, lam=0
    # )  # This one gives high fidelity for all 30 systems
    # J_coupled_X(nS=15, N=1000, tN=500 * unit.ns, max_time=None, kappa=1e6, lam=1e9)
    # identity_hyperfines(2000, 1000 * unit.ns, nS=1, kappa=1e4, max_time=10)
    # test_grape_pulse_on_non_res_spin(fp="fields/g337_100S_3q_4000ns_8000step_XY")
    test_pulse_frame_change()
    plt.show()
    # test_grape_pulse_with_varied_J(fp="fields/c1224_2S_2q_500ns_1000step_XY"); plt.show()

    # plt.show()

