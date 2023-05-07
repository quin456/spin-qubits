

import torch as pt 
import numpy as np
import atomic_units as unit
from data import * 
from utils import *
from eigentools import *
from pulse_maker import *
from hamiltonians import *






# dynamical decoupling maybe




def decouple_weak_exchange_from_1P(n_flips = 1, A = get_A_1P_2P(1), J=get_J_1P_2P(1), tN = 1000*unit.ns, N=10000):
    """
    2P fixed in 0-state. J << dA => 2P applies phase to 1P. 
    Target 1P hyperfine to rotate 1P and hopefully decouple exchange.
    """

    H0 = get_H0(A, J)

    # lock tN to 1P hyperfine
    tN = lock_to_frequency(A[0], tN)

    # Apply pulse which flips 1P n_flips times over the duration of tN
    Bx, By = pi_pulse_square(A[0], gamma_e, tN/n_flips, N)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(2), gate.get_Yn(2))

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)


    return

