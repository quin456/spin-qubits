from email.policy import default
import torch as pt
import numpy as np


import atomic_units as unit
import gates as gate
from utils import *
from data import *
from hamiltonians import get_H0
from pulse_maker import pi_pulse_field_strength


def get_allowed_transitions(
    H0=None, Hw_shape=None, S=None, D=None, device=default_device
):

    if S is None:
        if H0 is None:
            raise Exception("No Hamiltonian information provided.")
        eig = pt.linalg.eig(H0)
        E = eig.eigenvalues
        S = eig.eigenvectors
        S = S.to(device)
    else:
        E = pt.diag(D)

    S_T = pt.transpose(S, -2, -1)
    d = len(E)

    if Hw_shape is None:
        nq = get_nq_from_dim(S.shape[-1])
        Hw_shape = (gate.get_Xn(nq) + gate.get_Yn(nq)) / np.sqrt(2)

    # transform shape of control Hamiltonian to basis of energy eigenstates
    Hw_trans = matmul3(S_T, Hw_shape, S)
    Hw_nz = (pt.abs(Hw_trans) > 1e-8).to(int)
    Hw_angle = pt.angle(Hw_trans ** 2)

    allowed_transitions = []
    for i in range(d):
        for j in range(d):
            if Hw_nz[i, j] and Hw_angle[i, j] < 0:
                allowed_transitions.append((i, j))

    return allowed_transitions


def get_resonant_frequencies(
    H0, Hw_shape=None, S=None, D=None, device=default_device, return_transitions=False
):
    """
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0. 
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. 
    """
    if D is None:
        eig = pt.linalg.eig(H0)
        E = eig.eigenvalues
        D = pt.diag(E)
    allowed_transitions = get_allowed_transitions(
        H0, S=S, D=D, Hw_shape=Hw_shape, device=device
    )
    transition_freqs = get_transition_freqs(allowed_transitions, E=E, device=device)
    if return_transitions:
        return transition_freqs, allowed_transitions
    return transition_freqs


def get_rf_matrix(S, D, device=default_device, Hw_shape=None):
    if len(S.shape) > 2:
        return get_multisys_rf_tensor(S, D, device=device, Hw_shape=Hw_shape)
    E = pt.diag(D)
    allowed_transitions = get_allowed_transitions(
        S=S, D=D, Hw_shape=Hw_shape, device=device
    )
    rf_mat = pt.zeros_like(S)
    for transition in allowed_transitions:
        rf_mat[transition] = pt.real(E[transition[0]] - E[transition[1]])
        rf_mat[transition[::-1]] = pt.real(E[transition[0]] - E[transition[1]])
    return rf_mat


def get_multisys_rf_tensor(S, D, device=default_device, Hw_shape=None):
    nS = len(S)
    rf_tens = pt.zeros_like(S)
    for q in range(nS):
        rf_tens[q] = get_rf_matrix(S[q], D[q], device=device, Hw_shape=Hw_shape)
    return rf_tens


def get_rf_from_J_A_Bz(J, A, Bz=0):
    H0 = get_H0(A, J, Bz=Bz)
    return get_resonant_frequencies(H0)


def get_transition_freqs(allowed_transitions, E, device=default_device):
    freqs = []
    for transition in allowed_transitions:
        freqs.append((pt.real(E[transition[0]] - E[transition[1]])))
    freqs = pt.tensor(remove_duplicates(freqs), dtype=real_dtype, device=device)
    return freqs


def get_low_J_rf_u0(S, D, tN, N):
    """
    Gets resonant frequencies and u0 in low J limit, where transition is driven mostly by a single frequency.
    u0 is chosen to give appropriate pi-pulse.
    """

    E = pt.diag(D)
    H0 = S @ D @ S.T
    allowed_transitions = get_allowed_transitions(H0, S=S, E=E)
    target_transition = (2, 3)
    if target_transition in allowed_transitions:
        idx = allowed_transitions.index(target_transition)
    elif target_transition[::-1] in allowed_transitions:
        idx = allowed_transitions.index(target_transition[::-1])
    else:
        raise Exception(
            "Failed to find low J resonant frequencies and u0: target transition not in allowed transitions."
        )

    target_transition = allowed_transitions[idx]
    allowed_transitions[idx] = allowed_transitions[0]
    allowed_transitions[0] = target_transition
    rf = get_transition_freqs(allowed_transitions, E)

    couplings = get_couplings(S)
    print_rank2_tensor(couplings)
    m = len(rf)
    u0 = pt.zeros(m, N, dtype=cplx_dtype, device=default_device)
    u0[0] = pi_pulse_field_strength(couplings[target_transition], tN) / unit.T

    u0 = pt.cat((u0, pt.zeros_like(u0)))
    u0 = uToVector(u0)

    return rf, u0


def get_all_low_J_rf_u0(S, D, tN, N, device=default_device):
    rf = pt.tensor([], dtype=real_dtype, device=device)
    u0 = pt.tensor([], dtype=real_dtype, device=device)
    nS = len(S)
    nq = get_nq_from_dim(S.shape[-1])
    Hw_shape = (gate.get_Xn(nq) + gate.get_Yn(nq)) / np.sqrt(2)
    for q in range(nS):
        rf_q, u0_q = get_low_J_rf_u0(S[q], D[q], tN, N)
        rf = pt.cat((rf, rf_q))
        u0 = pt.cat((u0, u0_q))
    return rf


def get_multi_system_resonant_frequencies(H0s, device=default_device):
    nS = len(H0s)
    nq = get_nq_from_dim(H0s.shape[-1])
    Hw_shape = (gate.get_Xn(nq) + gate.get_Yn(nq)) / np.sqrt(2)
    if len(H0s.shape) == 2:
        return get_resonant_frequencies(H0s, Hw_shape)
    rf = pt.tensor([], dtype=real_dtype, device=device)
    for q in range(nS):
        rf_q = get_resonant_frequencies(H0s[q], Hw_shape)
        rf = pt.cat((rf, rf_q))
    return rf


def get_couplings_over_gamma_e(S, D):
    """
    Takes input S which has eigenstates of free Hamiltonian as columns.

    Determines the resulting coupling strengths between these eigenstates which arise 
    when a transverse control field is applied.

    The coupling strengths refer to the magnitudes of the terms in the eigenbasis ctrl Hamiltonian.

    Couplings are unitless.
    """

    nq = get_nq_from_dim(len(S[0]))
    couplings = pt.zeros_like(S)
    Xn = gate.get_Xn(nq)
    couplings = S.T @ Xn @ S
    return couplings


def get_multi_system_couplings(S, Hw_mag=None):
    nS, dim, dim = S.shape
    couplings = pt.zeros_like(S)
    for q in range(nS):
        couplings[q] = get_couplings(S[q], Hw_mag=Hw_mag)
    return couplings


def get_couplings(S, Hw_mag=None):
    if len(S.shape) == 3:
        return get_multi_system_couplings(S, Hw_mag)
    nq = get_nq_from_dim(S.shape[-1])
    if Hw_mag is None:
        if VERBOSE:
            print("No Hw_mag specified, providing couplings for electron spin system.")
        Hw_mag = gamma_e * gate.get_Xn(nq)
    return gamma_e * S.T @ Hw_mag @ S


def get_pi_pulse_tN_from_field_strength(B_mag, coupling, coupling_lock=None):
    tN = np.pi / (coupling * B_mag)
    print(
        f"Pi-pulse duration for field strength {B_mag/unit.mT} mT with coupling {pt.real(coupling)*unit.T/unit.MHz} MHz/unit.T is {pt.real(tN)/unit.ns} ns."
    )
    if coupling_lock is not None:
        return lock_to_frequency(coupling_lock, tN)
    return tN


def get_ordered_eigensystem(H0, H0_phys=None, ascending=False):
    """
    Gets eigenvectors and eigenvalues of Hamiltonian H0 corresponding to hyperfine A, exchange J.
    Orders from lowest energy to highest. Zeeman splitting is accounted for in ordering, but not 
    included in eigenvalues, so eigenvalues will likely not appear to be in order.
    """
    if H0_phys is None:
        H0_phys = H0

    if len(H0.shape) == 3:
        return get_multi_ordered_eigensystems(H0, H0_phys, ascending=ascending)

    # ordering is always based of physical energy levels (so include_HZ always True)
    E_phys = pt.real(pt.linalg.eig(H0_phys).eigenvalues)

    E, S = order_eigensystem(H0, E_phys, ascending=ascending)
    D = pt.diag(E)
    return S, D


def get_multi_ordered_eigensystems(H0, H0_phys=None, ascending=False):
    nS, dim, dim = H0.shape
    if H0_phys is None:
        H0_phys = H0
    S = pt.zeros(nS, dim, dim, dtype=cplx_dtype, device=default_device)
    D = pt.zeros_like(S)
    for q in range(nS):
        S[q], D[q] = get_ordered_eigensystem(H0[q], H0_phys[q])
    return S, D


def order_eigensystem(H0, E_order, ascending=True):

    idx_order = pt.topk(E_order, len(E_order), largest=not ascending).indices

    # get unsorted eigensystem
    eig = pt.linalg.eig(H0)
    E_us = eig.eigenvalues
    S_us = eig.eigenvectors

    E = pt.zeros_like(E_us)
    S = pt.zeros_like(S_us)
    for i, j in enumerate(idx_order):
        E[i] = E_us[j]
        S[:, i] = S_us[:, j]
    return E, S


def get_max_allowed_coupling(H0, p=0.9999):

    rf = get_resonant_frequencies(H0)

    # first find smallest difference in rf's
    min_delta_rf = 1e30
    for i in range(len(rf)):
        for j in range(i + 1, len(rf)):
            if pt.abs(rf[i] - rf[j]) < min_delta_rf:
                min_delta_rf = pt.abs(rf[i] - rf[j])
    return min_delta_rf / np.pi * np.arccos(np.sqrt(p))


def lock_to_frequency(c, tN):
    t_HF = 2 * np.pi / c
    tN_locked = int(tN / t_HF) * t_HF
    if tN_locked == 0:
        tN_locked = t_HF
        # print(f"tN={pt.real(tN)/unit.ns:.2f}ns too small to lock to coupling period {t_HF/unit.ns:.2f}ns.")
        # return tN_locked
    print(
        f"Locking tN={real(tN/unit.ns):.2f} ns to coupling period {t_HF/unit.ns:.2f} ns. New tN={tN_locked/unit.ns:.2f} ns."
    )
    return tN_locked


def get_2E_rf_analytic(J, A, Bz=0):
    """
    Gets resonant frequencies for single 2E system with exchange J and hyperfine A.
    """
    dA = A[0] - A[1]
    Ab = A[0] + A[1]
    w0 = gamma_e * Bz

    w12 = w0 + Ab + 2 * J - pt.sqrt(4 * J ** 2 + dA ** 2)
    w13 = w0 + Ab + 2 * J + pt.sqrt(4 * J ** 2 + dA ** 2)
    w24 = w0 + Ab - 2 * J + pt.sqrt(4 * J ** 2 + dA ** 2)
    w34 = w0 + Ab - 2 * J - pt.sqrt(4 * J ** 2 + dA ** 2)

    return pt.tensor([w12, w13, w24, w34])


def get_2E_multi_system_rf_analytic(J, A, Bz=0):
    """
    Gets resonant frequencies for array of 2E systems with exchanges J and hyperfines A
    using analytic formulae rather than by generating H0 array.
    """
    nS = len(A)
    rf = pt.zeros(4 * nS, dtype=real_dtype, device=default_device)
    for q in range(nS):
        rf[q : q + 4] = get_2E_rf_analytic(J[q], A[q], Bz)
    return rf


def map_time_dep_operator(A, U):
    return pt.einsum("jab,jbc,jcd->jad", U, A, dagger(U))


if __name__ == "__main__":
    pass
