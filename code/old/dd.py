import torch as pt
import numpy as np
import atomic_units as unit
from data import *
from utils import *
from eigentools import *
from pulse_maker import *
from hamiltonians import *
from visualisation import *


# dynamical decoupling maybe


def decouple_weak_exchange_from_1P(
    n_flips=5,
    A=get_A_1P_2P(1, donor_composition=[2, 1]),
    J=pt.tensor(0.1 * unit.MHz),
    tN=1000 * unit.ns,
    N=20000,
):
    """
    2P fixed in 0-state. J << dA => 2P applies phase to 1P. 
    Target 1P hyperfine to rotate 1P and hopefully decouple exchange.
    """
    H0 = get_H0(A, J)

    # lock tN to 1P hyperfine
    tN = lock_to_frequency(A[1], tN)

    # Apply pulse which flips 1P n_flips times over the duration of tN
    Bx, By = pi_pulse_square(2 * A[1], gamma_e / (2 * n_flips), tN, N)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(2), gate.get_Yn(2))

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    U0 = get_U0(H0, tN=tN, N=N)


    psi0 = gate.spin_00

    psi = X @ psi0

    plot_psi(psi, T=linspace(0, tN, N))
    print_rank2_tensor(remove_leading_phase(X[-1]))
    print_rank2_tensor(remove_leading_phase(U0[-1]))

    return


if __name__ == "__main__":
    decouple_weak_exchange_from_1P()
    plt.show()
