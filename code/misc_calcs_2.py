from data import *
import atomic_units as unit
from hamiltonians import get_H0
from eigentools import (
    get_multi_system_resonant_frequencies,
    get_multi_ordered_eigensystems,
    get_rf_matrix,
)
from utils import real

import matplotlib as mpl
mpl.use("Qt5Agg")
from matplotlib import pyplot as plt

def extrapolate_exchange():
    x = np.array([10**2, 10**0, 10**(-2)]) * 1e9
    y = np.array([9, 15, 21])
    X = np.stack((np.ones(len(x)), np.log2(x))).T

    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    def get_x(J):
        return theta[0] + theta[1] * np.log2((J))
    print(f"theta = {theta}")
    x0 = get_x(1e3)

    J = 10**(np.linspace(3,9,1000))
    x = get_x(J)
    for k in range(len(J)):
        print(f"{x[k]:.1f} nm: {J[k]:.2e} Hz")

    print(f"Separation for 1 kHz exchange is {x0} nm")

    ax = plt.subplot()
    ax.plot(x,J)
    ax.set_xlabel('Separation (nm)')
    ax.set_ylabel("Exchange (Hz)")
    ax.set_yscale('log')
    plt.show()


def hyperfine_modulation_E_field_strength():

    """
    Determine electric field strength required to address single qubits by modulating hyperfine.
    """

    P = 0.999
    t_1q_gate = 0.3 * unit.us

    A = get_A(1, 1)

    B_ac = 2 * np.pi / (gamma_e * t_1q_gate)
    E = np.sqrt(gamma_e * B_ac / (-2 * eta2 * A)) * (1 / P - 1) ** (1 / 4)

    # print(f"B_ac = {B_ac/unit.mT} mT")
    print(f"Electric field requires: {E/unit.MV * unit.m} MV/m")


def rabi_prob(dw, B_ac=1 * unit.mT, c=gamma_e):
    Pr = (c * B_ac / 2) ** 2 / ((c * B_ac) ** 2 + dw ** 2)
    return Pr


def coupler_unique_configs():
    """
    Determines number of unique coupler configs. A config is characterised
    by a tuple of distances (d12, d23).
    """

    def get_all_displacements(p1, p2):
        return np.stack((p2[i] - p1[j] for i in range(len(p2)) for j in range(len(p1))))

    patch = np.array([[a, b] for a in range(0, 2) for b in range(0, 3)])

    q_sep = 30
    q1 = np.array([0, 0])
    q2 = np.array([q_sep, 0])
    q3 = np.array([2 * q_sep, 0])

    p1 = patch + q1
    p2 = patch + q2
    p3 = patch + q3

    d12 = np.linalg.norm(get_all_displacements(p1, p2), axis=1)
    d23 = np.linalg.norm(get_all_displacements(p2, p3), axis=1)

    d_tup = [(d12[i], d23[j]) for i in range(len(d12)) for j in range(len(d23))]

    print(len(d12))
    print(len(set(d12)))

    print(len(d_tup))
    print(len(set(d_tup)))

    print(set(d12))
    pass


def get_rabi_prob(w_ac, w_res, B_ac, c):
    """
    Inputs
        w_ac: frequency to be applied to system
        w_res: resonant frequency of transition
        B_ac: strength of applied field
        c: coupling of B field to transition

    Determines disruption to system caused by w_ac using Rabi solution.
    """
    W = c * B_ac / 2
    dw = w_ac - w_res
    Omega = np.sqrt(dw ** 2 + W ** 2)
    Pr = np.abs(W ** 2 / Omega ** 2)
    return Pr


def frequencies_are_similar(w_ac, w_res, B_ac, c, maxp=0.01, print_info=True):
    p = get_rabi_prob(w_ac, w_res, B_ac, c).item()
    if print_info:
        print(
            f"Field B_ac = {B_ac/unit.mT} mT, w_ac = {real(w_ac/unit.MHz):.1f} MHz has probability p={p:.4f} of exciting transition with w_res = {real(w_res/unit.MHz):.1f} MHz, coupling = {real(c*unit.T/unit.MHz):.1f} MHz/T"
        )

    if p > maxp:
        return True
    return False


def partition_frequencies(
    A=get_A(2, 2, NucSpin=[0, 1, 0]), J=get_J(2, 2, J1=J_100_18nm, J2=J_100_14nm)
):

    H0 = get_H0(A=A, J=J)
    S, D = get_multi_ordered_eigensystems(H0, H0_phys=get_H0(Bz=2 * unit.T, A=A, J=J))
    rf_mat = get_rf_matrix(S, D)
    dim = S.shape[-1]
    nS = len(S)

    rf, transitions = get_multi_system_resonant_frequencies(
        S=S, D=D, return_transitions=True
    )
    rf_sort = get_multi_system_resonant_frequencies(H0).sort()
    rf = rf_sort.values
    idxs = rf_sort.indices

    n_freqs = len(rf)

    # Iterate over sorted frequency list. Partitions will already be grouped
    # together, just need to find the big enough gaps.

    # define array of indices of partition starts
    partition_idxs = [0]

    for j in range(1, n_freqs):
        if not frequencies_are_similar(rf[j - 1], rf[j], 10 * unit.uT, gamma_e, 0.0001):
            partition_idxs.append(j)

    M = pt.zeros(len(partition_idxs), n_freqs, dim, dim, dtype=real_dtype)
    for i in range(1, len(partition_idxs)):
        for k in range(partition_idxs[i - 1], partition_idxs[i]):
            M[(i, *transitions[k])] = 1

    print(f"{n_freqs} frequencies, {len(partition_idxs)} partitions.")
    print(partition_idxs)

    # print frequency span of each partition
    d_freqs = []
    for k in range(1, len(partition_idxs)):
        min_freq = rf[partition_idxs[k - 1]]
        max_freq = rf[partition_idxs[k] - 1]
        d_freq = max_freq - min_freq
        d_freqs.append(d_freq)

    print(f"Frequency spans:")
    for d_freq in d_freqs:
        print(f"{d_freq/unit.MHz} MHz")
    print(f"Maximum frequency span is {max(d_freqs)/unit.MHz} MHz")


if __name__ == "__main__":
    # hyperfine_modulation_E_field_strength()
    # print(f"Pr = {rabi_prob(2*unit.MHz, 1*unit.mT)}")
    extrapolate_exchange()
    # coupler_unique_configs()
    #partition_frequencies()

