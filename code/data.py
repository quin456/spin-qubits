"""
Data file, responsible for reading in and defining values of data used in 
spin simulations and optimisations, particularly exchange and hyperfine data.
Includes functions which allow data to be retrieved in formats appropriate for
usage in GRAPE optimisation.
"""


import pickle
import torch as pt
import atomic_units as unit
from atomic_units import hbar, qE, mE
import numpy as np


cplx_dtype = pt.complex128
real_dtype = pt.float64

VERBOSE = False

default_device = "cuda:0" if pt.cuda.is_available() else "cpu"

dir = "./"
exch_filename = f"exchange_data_updated.p"
exch_data_folder = "exchange_data_fab/"
exch_data = pickle.load(open(exch_filename, "rb"))

# max J for low J regime, ensuring 99.9% spin-state eigenstate similarity
J_low_max = 1.9 * unit.MHz

J_100_18nm = (
    pt.tensor(exch_data["100_18"], dtype=cplx_dtype, device=default_device) * unit.MHz
)

J_low = J_100_18nm * J_low_max / pt.max(pt.real(J_100_18nm))


J_100_14nm = (
    pt.tensor(exch_data["100_14"], dtype=cplx_dtype, device=default_device) * unit.MHz
)
J_110_18nm = (
    pt.tensor(exch_data["110_18"], dtype=cplx_dtype, device=default_device) * unit.MHz
)
J_110_14nm = (
    pt.tensor(exch_data["110_14"], dtype=cplx_dtype, device=default_device) * unit.MHz
)
J_2P_1P_fab = (
    pt.load(f"{exch_data_folder}J_big").to(cplx_dtype).to(device=default_device)
    * unit.MHz
)
A_2P_1P_fab = (
    pt.load(f"{exch_data_folder}A_2P_1P").to(cplx_dtype).to(default_device) * unit.MHz
)
A_2P_48Js_fab = (
    pt.load(f"{exch_data_folder}A_2P").to(cplx_dtype).to(device=default_device)
    * unit.MHz
)
A_2P_fab = pt.tensor(list(set(pt.real(A_2P_48Js_fab).tolist())), dtype=real_dtype)
A_2P_69 = (
    pt.load(f"{exch_data_folder}A_2P_69").to(cplx_dtype).to(device=default_device)
    * unit.MHz
)
J_69 = (
    pt.load(f"{exch_data_folder}J_69").to(cplx_dtype).to(device=default_device)
    * unit.MHz
)
J_extended = pt.cat(
    (
        J_100_18nm,
        J_100_18nm * 1.1,
        J_100_18nm * 1.2,
        J_100_18nm * 1.3,
        J_100_18nm * 1.4,
        J_100_18nm * 1.5,
        J_100_18nm * 1.6,
        J_100_18nm * 1.7,
        J_100_18nm * 1.8,
        J_100_18nm * 1.9,
    )
)


# gyromagnetic ratios (Mrad/T)
gamma_n = 17.235 * unit.MHz / unit.T
gamma_e = 28025 * unit.MHz / unit.T


B0 = 2 * unit.T  # static background field
g_e = 2.0023  # electron g-factor


mu_B = qE * hbar / (2 * mE)  # Bohr magneton
omega0 = g_e * mu_B * B0 / hbar  # Larmor frequency


# exchange values
A_P = np.float64(58.5 / 2) * unit.MHz
delta_A_kane = np.float(2) * A_P
A_kane = pt.tensor([A_P, -A_P])
A_kane3 = pt.tensor([A_P, -A_P, A_P])
A_2P = np.float64(150 / 4) * unit.MHz

A_As = 198.35 / 4 * unit.MHz
A_Sb = 186.8 / 4

eta2 = -3e-3 * (unit.um / unit.V) ** 2

J_single = pt.tensor(60 * unit.MHz, dtype=real_dtype)
A_1P_2P_single = pt.tensor([A_P, 70 * unit.MHz], dtype=real_dtype)
A_2P_1P_single = pt.tensor([70 * unit.MHz, A_P], dtype=real_dtype)


def get_A_from_num_P_donors(num_P_donors):
    if num_P_donors == 1:
        return A_P
    elif num_P_donors == 2:
        return A_2P


def get_A_spec_single():
    return pt.tensor([get_A(1, 1)], device=default_device)


def get_A(nS, nq, NucSpin=None, A_mags=None, device=default_device):
    try:
        if 0 in NucSpin:
            raise Exception("Use +1, -1 for NucSpin, not computational state.")
    except:
        pass
    if nq == 1:
        return pt.tensor(A_P, device=default_device)
    if A_mags is None:
        A_mags = nq * [A_P]
    if nq == 2:
        if NucSpin is None:
            NucSpin = [1, -1]
        A = pt.tensor(
            nS * [[NucSpin[0] * A_mags[0], NucSpin[1] * A_mags[1]]],
            device=device,
            dtype=cplx_dtype,
        )
    elif nq == 3:
        if NucSpin is None:
            NucSpin = [1, -1, 1]
        A = pt.tensor(
            nS
            * [
                [NucSpin[0] * A_mags[0], NucSpin[1] * A_mags[1], NucSpin[2] * A_mags[2]]
            ],
            device=device,
            dtype=cplx_dtype,
        )

    if nS == 1:
        return A[0]
    return A


def all_J_pairs(J1, J2, device=default_device):
    nJ = min((len(J1), len(J2)))
    J = pt.zeros(nJ**2, 2, device=device, dtype=cplx_dtype)
    for i in range(nJ):
        for j in range(nJ):
            J[i * nJ + j, 0] = J1[i]
            J[i * nJ + j, 1] = J2[j]
    return J


def get_J(
    nS,
    nq,
    J1=J_100_18nm,
    J2=J_110_18nm / 3.217,
    N=1,
    device=default_device,
    E_rise_time=1 * unit.ns,
):
    # default to small exchange for testing single triple donor
    # if nS==1 and nq==3:
    #     return pt.tensor([0.37*unit.MHz, 0.21*unit.MHz], dtype=cplx_dtype)

    if nq == 2:
        J = J1[:nS].to(device)

    elif nq == 3:
        J = all_J_pairs(J1, J2)[:nS]

    if nS == 1:
        return J[0]
    return J


def get_J_low(nS, nq):
    Jmax = 1.87 * unit.MHz
    J_low = J_100_18nm * Jmax / pt.max(pt.real(J_100_18nm))
    return get_J(nS, nq, J1=J_low)


def get_A_1P_2P(nS, NucSpin=[1, -1], donor_composition=[1, 2], fp="A_70"):
    if fp is None:
        A_data = A_2P_69
        NucSpin[1] *= -1  # because 1P A is down by default
    else:
        A_data = (
            pt.load(f"exchange_data_fab/{fp}").to(
                dtype=cplx_dtype, device=default_device
            )
            * unit.MHz
        )
    try:
        if 0 in NucSpin:
            raise Exception("Use +1, -1 for NucSpin, not computational state.")
    except:
        pass
    A = A_data[:nS]
    A[:, 1] *= 2
    if donor_composition == [2, 1] and fp is not None:
        A = pt.flip(A, (1,))
    elif donor_composition != [1, 2]:
        raise Exception("Invalid donor composition: must be [1,2] or [2,1].")
    A = pt.einsum("sq,q->sq", A, pt.tensor(NucSpin, device=default_device))

    return A if nS > 1 else A[0]


def get_J_1P_2P(nS, fp="J_70"):
    if fp is None:
        J_data = J_69
    else:
        J_data = (
            pt.load(f"exchange_data_fab/{fp}").to(
                dtype=cplx_dtype, device=default_device
            )
            * unit.MHz
        )
    return get_J(nS, 2, J1=J_data)


def J_HF(R, a=1.8 * unit.nm):
    return (R / a) ** (5 / 2) * np.exp(-2 * R / a)


def get_A_1P_2P_uniform_J(nS):
    if nS > 9:
        raise Exception("Only 9 2P hyperfines to be optimised in parallel.")
    A = (pt.stack((A_2P_fab[:nS], -pt.tensor([A_P] * nS))).T).to(cplx_dtype)
    if len(A) == 1:
        return A[0]
    return A


def get_J_1P_2P_uniform_J(nS, J=100 * unit.MHz):
    if nS > 10:
        raise Exception("Only 10 2P hyperfines to be optimised in parallel.")
    J = pt.tensor([J] * nS, dtype=cplx_dtype)
    if len(J) == 1:
        return J[0]
    return J


J_1s3q = J = get_J(1, 3, J1=J_100_18nm, J2=J_100_14nm[8:])


if __name__ == "__main__":
    print("============================================")
    print("Printing data used in spin-qubit simulations")
    print("============================================\n")
    print(
        f"gamma_e = {gamma_e*unit.T/unit.MHz:.1f} MHz, gamma_n = {gamma_n*unit.T/unit.MHz:.1f} MHz"
    )
    print(f"\nHyperfine coupling: A_1P = {get_A(1,1)/unit.MHz:.2f} MHz")
    print("\nExchange valiues for (100) 18nm separation:")
    for i in range(15):
        print(f"J_18nm_100_{i} = {pt.real(J_100_18nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (100) 18nm separation:")
    for i in range(15):
        print(f"J_18nm_110_{i} = {pt.real(J_110_18nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (100) 14nm separation:")
    for i in range(15):
        print(f"J_14nm_{i} = {pt.real(J_100_14nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (110) 14nm separation:")
    for i in range(15):
        print(f"J_14nm_{i} = {pt.real(J_110_14nm[i]).item()/unit.MHz:.1f} MHz")

    print(f"\nHyperfine coupling 2P-1P (fabricated):")
    for i in range(len(A_2P_1P_fab)):
        print(f"A_2P_{i} = {A_2P_48Js_fab[i].item()/unit.MHz:.2f} MHz")
    print(f"Exchange 2P-1P (fabricated)")
    for i in range(len(A_2P_1P_fab)):
        print(f"J_2P_1P_{i} = {J_2P_1P_fab[i].item()/unit.MHz:.2f} MHz")

    breakpoint()
