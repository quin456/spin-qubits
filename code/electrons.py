import matplotlib
import numpy as np
import torch as pt

if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt

import atomic_units as unit
import gates as gate
from data import *
from eigentools import *
from hamiltonians import (
    get_H0,
    get_pulse_hamiltonian,
    get_U0,
    get_X_from_H,
    sum_H0_Hw,
    get_1S_HA,
    get_IP_X,
)
from pulse_maker import pi_pulse_square
from utils import *
from visualisation import (
    eigenstate_label_getter,
    plot_psi,
    show_fidelity,
    visualise_Hw,
    plot_alpha_beta,
    plot_psi_with_phase,
)

spin_up = pt.tensor([1, 0], dtype=cplx_dtype)
spin_down = pt.tensor([0, 1], dtype=cplx_dtype)


def get_K(A, J):
    dA = A[0] - A[1]
    return (2 * (4 * J ** 2 + dA ** 2 + dA * np.sqrt(4 * J ** 2 + dA ** 2))) ** (-1 / 2)


def get_alpha(A, J):
    dA = A[0] - A[1]
    return get_K(A, J) * (np.sqrt(4 * J ** 2 + dA ** 2) + dA)


def get_beta(A, J):
    return 2 * get_K(A, J) * J


def plot_energy_spectrum(E, ax=None):
    dim = E.shape[-1]
    if ax is None:
        ax = plt.subplot()
    for sys in E:
        for i in range(len(sys)):
            ax.axhline(
                pt.real(sys[i] / unit.MHz), label=f"E{dim-1-i}", color=color_cycle[i]
            )
    ax.legend()


def get_lowest_energy_indices(D):

    # singlet is at 0, T- is at 3 (for 2q system)
    return 3, 0


def put_diagonal(E):
    dim = E.shape[-1]
    D = pt.zeros(dim, dim, dtype=cplx_dtype)
    for j in range(4):
        D[j, j] = E[0, j]
    return D


def print_E_system_info(A, J, tN, N):
    nS, nq = get_nS_nq_from_A(A)
    print(f"Simulating {nq} electron system.")
    print(f"A = {A/unit.MHz} MHz")
    print(f"J = {J/unit.MHz} MHz")
    print(f"tN = {tN/unit.ns} ns")
    print(f"N = {N}")


def get_free_electron_evolution(tN, N, A=get_A(1, 3), J=get_J(1, 3), psi0=None):
    print_E_system_info(A, J, tN, N)
    nS, nq = get_nS_nq_from_A(A)

    print("Getting electron free evolution.")
    print(f"J = {J/unit.MHz} MHz")
    print(f"A = {A/unit.MHz} MHz")
    print(f"tN = {tN/unit.ns} ns")

    if psi0 is None:
        if nq == 2:
            psi0 = pt.tensor([0, 0, 1, 0], dtype=cplx_dtype)
        elif nq == 3:
            psi0 = pt.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=cplx_dtype)

    H0 = get_H0(A, J)
    U0 = get_U0(H0, tN, N)
    psi = U0 @ psi0
    return psi


def plot_free_electron_evolution(tN, N, A, J, psi0=None, ax=None, label_getter=None):
    """
    Simulates and plots the free evolution of a system of 2 or 3 electrons coupled via exchange.
    """
    psi = get_free_electron_evolution(A, J, tN, N, psi0=psi0)
    plot_psi(psi, tN, ax, label_getter=label_getter)


def project_psi(psi, n, S):
    return pt.abs(pt.einsum("ja,a->j", psi, S[0, :, n]))


def plot_2q_eigenstates(psi, S, tN, ax=None):
    if ax is None:
        ax = plt.subplot()
    N = len(psi)
    T = linspace(0, tN / unit.ns, N)
    ax.plot(T, project_psi(psi, 0, S), label="T+")
    ax.plot(T, project_psi(psi, 1, S), label="T0")
    ax.plot(T, project_psi(psi, 2, S), label="S")
    ax.plot(T, project_psi(psi, 3, S), label="T-")


def plot_eigenstates(psi, S, tN, ax=None):
    dim = S.shape[-1]
    if dim == 4:
        plot_2q_eigenstates(psi, S, tN, ax)
        return

    if ax is None:
        ax = plt.subplot()

    N = len(psi)
    T = linspace(0, tN / unit.ns, N)
    for i in range(dim):
        ax.plot(T, project_psi(psi, i, S), label=f"E{dim-1-i}")


def visualise_hamiltonian(H, tN):
    """
    Takes (N,d,d) tensor H which is Hamiltonian at each of N timesteps. 
    Plots real and imaginary parts of each cell.
    """
    N, d, d = H.shape
    T = linspace(0, tN / unit.ns, N)
    fig, ax = plt.subplots(d, d)
    maxval = 0
    for a in range(d):
        for b in range(d):
            localmax = max(pt.real(H[:, a, b]))
            if localmax > maxval:
                maxval = localmax
            ax[a, b].plot(T, pt.real(H[:, a, b]))
            ax[a, b].plot(T, pt.imag(H[:, a, b]))

    for a in range(d):
        for b in range(d):
            ax[a, b].set_ylim(-1.2 * maxval, 1.2 * maxval)


def get_transition_frequency(A, J, i, j, include_HZ=False):
    H0 = get_H0(A, J, include_HZ=include_HZ)
    E, S = get_ordered_eigensystem(A, J, include_HZ=include_HZ)
    return E[:, i] - E[:, j]


def ground_state(nq):
    if nq == 2:
        return pt.tensor([0, 0, 0, 1], dtype=cplx_dtype)
    elif nq == 3:
        return pt.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=cplx_dtype)


def get_Bw_field(tN, N, J, A, multisys=True, include_HZ=False):
    """
    Accepts pulse time tN (nanoseconds) and A,J couplngs (Mhz), and returns control pulse (unit.T) to achieve
    desired eigenstate transition.
    """
    A *= unit.MHz
    J *= unit.MHz
    tN *= unit.ns
    if not multisys:
        J = J.reshape(1, *J.shape)
        A = A.reshape(1, *A.shape)

    C = get_couplings_over_gamma_e(A, J)
    omega = get_transition_frequency(A, J, -2, -1, include_HZ=include_HZ)
    phase = pt.zeros_like(omega)

    T = linspace(0, tN, N)
    wt = pt.einsum("k,j->kj", omega, T)

    # get unitless field direction vectors
    x_cf, y_cf = unit.get_unit_CFs(omega, phase, tN, N)

    # get magnetic field strength (teslas)
    Bw = np.pi / (C[0][-2, -1] * g_e * mu_B * unit.T * tN)

    x_field = Bw * x_cf
    y_field = Bw * y_cf

    if not multisys:
        return x_field[0], y_field[0]
    return x_field, y_field


def forward_prop(U, device=default_device):
    """
    Forward propagates U suboperators. U has shape (N,d,d) or (nS,N,d,d)
    """
    if len(U.shape) == 3:
        U = U.reshape(1, *U.shape)
        sys_axis = False
    else:
        sys_axis = True

    nS, N, dim, dim = U.shape
    nq = get_nq_from_dim(dim)
    X = pt.zeros((nS, N, dim, dim), dtype=cplx_dtype, device=device)
    X[:, 0, :, :] = U[:, 0]  # forward propagated time evolution operator

    for j in range(1, N):
        X[:, j, :, :] = pt.matmul(U[:, j, :, :], X[:, j - 1, :, :])

    if sys_axis:
        return X
    else:
        return X[0]


def get_electron_X(
    J, A, Bz=np.float64(0), tN=None, N=None, Bx=None, By=None, T=None, IP=False
):
    if Bx is None:
        Bx = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    if By is None:
        By = pt.zeros(N, dtype=cplx_dtype, device=default_device)

    _nS, nq = get_nS_nq_from_A(A)
    if T == None:
        T = linspace(0, tN, N)

    H0 = get_H0(A, J, Bz)

    # print("Printing RF's:")
    # rf = get_resonant_frequencies(H0)
    # for w in rf:
    #     print(f"{w/Mhz} MHz")

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=gate.get_Xn(nq), Y=gate.get_Yn(nq))
    if IP:
        U0 = get_U0(H0, T=T)
        Hw = pt.einsum("jab,jbc,jcd->jad", dagger(U0), Hw, U0)
        H = Hw
    else:
        H = sum_H0_Hw(H0, Hw)

    X = get_X_from_H(H, tN, N, T=T)
    return X


def electron_wf_evolution(tN, N, Bz, A, J, Bx=None, By=None, psi0=gate.spin_111):
    """
    Simulates electron evolution. Bx and By = None for free evolution.
    """
    X = get_electron_X(tN, N, Bz, A, J, Bx, By)
    psi = X @ psi0
    return psi


def simulate_electron_wf(psi0, tN, N, Bz, A, J, Bx, By, ax=None):

    psi = electron_wf_evolution(tN, N, Bz, A, J, Bx, By, psi0)

    label_getter = None

    H0 = get_H0(A, J, Bz)
    S, D = get_ordered_eigensystem(H0)
    if S is not None:
        psi = pt.einsum("ab,jb->ja", S.T, psi)
        label_getter = lambda j: f"E{j}"
    plot_psi(psi, tN, label_getter=label_getter, ax=ax)


def get_n_e_label(j):
    """ Returns state label corresponding to integer j\in[0,dim] """
    uparrow = "\u2191"
    downarrow = "\u2193"
    b = np.binary_repr(j, 4)
    if b[2] == "0":
        L2 = uparrow
    else:
        L2 = downarrow
    if b[3] == "0":
        L3 = uparrow
    else:
        L3 = downarrow

    return b[0] + b[1] + L2 + L3


def get_nuclear_spins(A):
    nq = len(A)
    spins = [0] * nq
    for i in range(nq):
        if pt.real(A[i]) > 0:
            spins[i] = spin_up
        else:
            spins[i] = spin_down
    return spins


def analyse_3E_system(Bz=2 * unit.T, A=get_A(1, 3), J=get_J(1, 3)):

    H0 = get_H0(A=A, J=J, Bz=Bz)
    S, D = get_ordered_eigensystem(H0)

    dim = len(S)

    print("Electron eigenstates:")
    for j in range(dim):
        print(f"|e{j}> = {psi_to_string(S[:,j], pmin=0.001)}")


def drive_electron_transition(
    S,
    D,
    transition,
    tN=100 * unit.ns,
    N=10000,
    ax=None,
    label_getter=eigenstate_label_getter,
):
    H0 = S @ D @ S.T

    nq = get_nq_from_dim(H0.shape[-1])
    E = pt.diag(D)

    w_res = E[transition[0]] - E[transition[1]]
    couplings = get_couplings(S)
    coupling = couplings[transition]
    print(
        f"Driving electron transition: |E{transition[0]}> <--> |E{transition[1]}> with frequency {pt.real(w_res)/unit.MHz} MHz, coupling {pt.real(coupling)*unit.T/unit.MHz} MHz/unit.T."
    )

    Bx, By = pi_pulse_square(w_res, coupling, tN, N)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(nq), gate.get_Yn(nq))

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    psi0 = S[:, transition[0]]
    psi_target = S[:, transition[1]]

    psi = X @ psi0

    psi_eig = pt.einsum("ab,jb->ja", S.T, psi)
    plot_psi(psi_eig, tN, label_getter=label_getter, ax=ax)
    label_axis(ax, transition, x_offset=0, y_offset=0)


def investigate_3E_resfreqs(tN=1000 * unit.ns, N=50000, fp=None):
    nq = 3
    H0 = get_H0(get_A(1, nq), get_J(1, nq, J1=J_100_18nm, J2=J_100_14nm))
    S, D = get_ordered_eigensystem(H0)

    fig, ax = plt.subplots(5, 3)

    allowed_transitions = get_allowed_transitions(H0, S=S, D=D)

    for j, transition in enumerate(allowed_transitions):
        # if j+1 in [5,6,7,8,9,10]:
        #    drive_electron_transition(S, D, (transition[1], transition[0]), tN=tN)
        label_getter = lambda i: eigenstate_label_getter(i, states_to_label=transition)
        drive_electron_transition(
            S,
            D,
            transition,
            tN=tN,
            N=N,
            ax=ax[j // 3, j % 3],
            label_getter=label_getter,
        )
    fig.set_size_inches(16, 12)

    if fp is not None:
        fig.savefig(fp)


def visualise_3E_Hw(A=get_A(1, 3), J=get_J(1, 3), Bz=0, tN=10 * unit.ns, N=1000):

    nq = 3
    H0 = get_H0(A=A, J=J, Bz=Bz)
    S, D = get_ordered_eigensystem(H0)
    E = pt.diag(D)
    trans_idx = 8
    transitions = get_allowed_transitions(H0, S=S, E=E)
    transition = transitions[trans_idx]
    omega = E[transition[0]] - E[transition[1]]

    Bx, By = pi_pulse_square(omega, gamma_e, tN, N)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=gate.get_Xn(nq), Y=gate.get_Yn(nq))

    dim = H0.shape[-1]

    Hw_eig = pt.einsum("ab,jbd->jad", S.T, pt.einsum("jbc,cd->jbd", Hw, S))

    print(
        f"w_res = {pt.real(omega)/unit.MHz:.2f} MHz for transition |E{transition[0]}> <--> |E{transition[1]}>"
    )

    for a in range(dim):
        for b in range(dim):
            if (
                pt.max(pt.real(Hw_eig[:, a, b])) + pt.max(pt.imag(Hw_eig[:, a, b]))
            ) / unit.MHz < 1e-9:
                Hw_eig[:, a, b] = pt.zeros_like(Hw_eig[:, a, b])

    exp_iDt = get_U0(D, tN, N)
    Hw_eig_IP = pt.einsum(
        "jab,jbd->jad", dagger(exp_iDt), pt.einsum("jbc,jcd->jbd", Hw_eig, exp_iDt)
    )

    visualise_Hw(Hw_eig_IP, tN)


def examine_electron_eigensystem(
    Bz=B0, NucSpin=[0, 1], A_mags=[A_P, A_P], n=5000, dim=4, pmin=0.997
):

    A = get_A(n, 2, NucSpin, A_mags)
    print(A[0] / unit.MHz)
    J = linspace(0.1, 30, n) * unit.MHz
    S = pt.zeros(n, dim, dim, dtype=cplx_dtype, device=default_device)
    D = pt.zeros_like(S)

    H0 = get_H0(A, J, 1 * unit.mT)
    H0_phys = get_H0(A, J, B0)

    S, D = get_multi_ordered_eigensystems(H0, H0_phys)

    rf = pt.zeros(4, n)
    for k in range(n):
        rf[:, k] = get_resonant_frequencies(H0[k])

    alpha = pt.real(S[:, 1, 1])
    beta = pt.real(S[:, 1, 2])

    fig, ax = plt.subplots(1, 3)
    for i in range(4):
        ax[2].plot(real(J) / unit.MHz, real(rf[i]) / unit.MHz, label=f"$\omega_{i}$")
    ax[2].plot(real(J) / unit.MHz, 4 * real(J) / unit.MHz, label="4J")
    ax[2].legend()

    alpha2 = pt.tensor([get_alpha(A[j], J[j]) for j in range(500)], dtype=real_dtype)
    beta2 = pt.tensor([get_beta(A[j], J[j]) for j in range(500)], dtype=real_dtype)

    ax[1].plot(real(J) / unit.MHz, alpha ** 2, label="$α^2$")
    ax[1].plot(real(J) / unit.MHz, beta ** 2, label="$ß^2$")
    ax[1].set_xlabel("J (MHz)")

    ax[0].plot(real(J) / unit.MHz, real(D[:, 0, 0]) / unit.MHz, label="$|T^+>$")
    ax[0].plot(real(J) / unit.MHz, real(D[:, 1, 1]) / unit.MHz, label="$|S_0>$")
    ax[0].plot(real(J) / unit.MHz, real(D[:, 2, 2]) / unit.MHz, label="$|T_0>$")
    ax[0].plot(real(J) / unit.MHz, real(D[:, 3, 3]) / unit.MHz, label="$|T_->$")
    ax[0].legend()
    i = 0

    i = np.argmax(2 * beta ** 2 > 1 - pmin)
    Jmax = J[i]
    print(f"Jmax = {Jmax/unit.MHz} MHz")
    print(
        f"alpha[{i}]^2 = {alpha[i]**2}, beta[{i}]^2 = {beta[i]**2} at J[{i}] = {J[i]/unit.MHz} MHz"
    )
    ax[1].axvline(
        real(J[i]) / unit.MHz,
        linestyle="--",
        color="red",
        label="Low J cutoff",
        linewidth=1,
    )
    ax[1].legend()


def get_2E_low_J_CNOT_pulse(
    tN=20 * unit.ns, N=500, J=1 * unit.MHz, A=get_A(1, 2), Bz=0
):

    H0 = get_H0(A, J, Bz)
    H0_phys = get_H0(A, J, B0)
    S, D = get_ordered_eigensystem(H0, H0_phys, ascending=False)
    E = pt.diag(D)
    omega = E[2] - E[3]
    couplings = get_couplings(S)
    return pi_pulse_square(omega, couplings[2, 3], tN, N)


def J_dynamical_decoupling_2E(tN=100 * unit.ns, N=None):
    """
    Attempts to perform dynamical dycoupling of exchange between two 1P electrons by applying π-pulses to one of them.
    """
    Bz = np.float64(0.1 * unit.T)
    A = get_A(1, 2)
    J = get_J(1, 2) * 0
    H0 = get_H0(A, J, Bz)
    # tN = lock_to_frequency(A[0], tN)
    rf = get_resonant_frequencies(H0)
    if N is None:
        N = 2 * get_rec_min_N(rf, tN, N_period=200, verbosity=100)
    T = linspace(0, tN, N, dtype=real_dtype)
    Bx = pt.zeros(N, dtype=real_dtype)
    By = pt.zeros(N, dtype=real_dtype)
    ########################################################
    # put fields in here
    # S,D = get_ordered_eigensystem(H0)
    Bx, By = pi_pulse_square(2 * (A[0] + J), gamma_e / 2, tN, N)

    ########################################################
    if Bx is not None:
        print(f"Max field: {maxreal(pt.sqrt(Bx**2+By**2))/unit.mT} mT")
    HA = get_1S_HA(A)
    X = get_electron_X(tN, N, Bz, A, J, Bx=Bx, By=By)
    # X = get_IP_X(X, HA, T=T)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=gate.get_Xn(2), Y=gate.get_Yn(2))
    HA_Hw = sum_H0_Hw(HA, Hw)
    X0 = get_X_from_H(HA_Hw, T=T)
    # X = dagger(X0) @ X
    psi0 = (
        gate.spin_00
    )  # (gate.spin_00 + gate.spin_01 + gate.spin_10 + gate.spin_11) / 2
    plot_psi_with_phase(X @ psi0, T)


if __name__ == "__main__":
    examine_electron_eigensystem(
        NucSpin=[1, 0], A_mags=[120 * unit.MHz, 29.25 * unit.MHz], pmin=0.999
    )
    # J_dynamical_decoupling_2E(tN=1000 * unit.ns)

    # print(get_alpha([5*unit.MHz, 0], .2*unit.MHz)**2)

    plt.show()
