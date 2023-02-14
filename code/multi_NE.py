import torch as pt

import gates as gate
import atomic_units as unit
from data import *
from hamiltonians import multi_NE_H0, multi_NE_Hw, sum_H0_Hw, get_X_from_H


def multi_NE_evol(
    Bx,
    By,
    Bz=2 * unit.T,
    A=get_A(1, 1),
    J=get_A(1, 3),
    tN=1000 * unit.ns,
    psi0=pt.kron(gate.spin_100, gate.spin_111),
):
    N = len(Bx)
    Hw = multi_NE_Hw(Bx, By, 3)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    psi = X @ psi0
    return psi


def get_multi_NE_X(
    N,
    Bz=2 * unit.T,
    A=get_A(1, 1),
    J=None,
    nq=2,
    Bx=None,
    By=None,
    T=None,
    tN=None,
    deactivate_exchange=False,
):
    if J is None:
        J = get_J(1, nq)
    if tN is None and T is None:
        print("No time information provided")

    # Bx and By set to None is free evolution
    if Bx is None:
        Bx = pt.zeros(N)
    if By is None:
        By = pt.zeros(N)

    Hw = multi_NE_Hw(Bx, By, nq)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J, deactivate_exchange=deactivate_exchange)
    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, T=T)

    return X
