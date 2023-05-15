import numpy as np
import torch as pt

from visualisation import *
from GRAPE import load_grape, Grape_ee_Flip
from electrons import *
from hamiltonians import H_hyperfine


def get_max_J(dA, pmin=0.001):
    """
    Returns maximum allowed J for a given dA, which is the J for which the 
    Rabi probability for 01 <-> 10 transitions is pmin.
    """

    Jmin = 1
    Jmax = 100
    n = 10000
    J = linspace(Jmin, Jmax, n)

    p = (
        4
        * J
        / np.sqrt(8 * J ** 2 + 2 * dA ** 2 + 2 * dA * np.sqrt(4 * J ** 2 + dA ** 2))
    ) ** 2
    Jmax = J[np.argmax(real(p) > pmin)]

    print(Jmax)

    return 0.5 * dA * np.sqrt(pmin / (1 - pmin))


def measure_J_phase_effect(
    phase=pt.tensor([0.1,0], dtype=cplx_dtype) * unit.MHz,
    fp_grape="fields/c1366_1S_2q_500ns_1000step",
):
    """
    Tests the effect of exchange ZZ-coupling to neighbouring fixed spin on gate fidelity 
    """
    grape = load_grape(fp_grape, A_spec=get_A_spec_single())
    Bx, By = grape.get_Bx_By()
    T = grape.get_T()
    J = grape.J
    A = grape.A

    # add phase term to one or both qubits
    A += phase
    H_phase = pt.tensor(
        [
            [phase[0] + phase[1], 0, 0, 0],
            [0, phase[0] - phase[1], 0, 0],
            [0, 0, phase[1] - phase[0], 0],
            [0, 0, 0, -phase[0] - phase[1]],
        ],
        dtype=cplx_dtype,
    )

    U_phase = get_U0(H_phase, T=T)

    X = get_electron_X(J, A, T=T, Bx=Bx, By=By)

    CX = U_phase[-1] @ gate.CX
    CX = gate.CX

    print_rank2_tensor(remove_leading_phase(X[-1]))
    print_rank2_tensor(CX)
    print(f"fidelity = {fidelity(CX, X[-1])}")
    print(f"abs fidelity = {fidelity(pt.real(CX), pt.abs(X[-1]))}")


if __name__ == "__main__":
    # print(get_max_J(10e3))
    measure_J_phase_effect()

