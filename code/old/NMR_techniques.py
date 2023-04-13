



import numpy as np 
import torch as pt

import atomic_units as unit
from gates import *
from data import *
from visualisation import *
from hamiltonians import single_electron_H0, get_U0, get_X_from_H, sum_H0_Hw, get_pulse_hamiltonian
from pulse_maker import pi_pulse_duration, pi_pulse_field_strength, pi_pulse_square
from electrons import get_electron_X



def simulate_exchange_with_refocussing():
    Bz = 0 * unit.T
    A = get_A(1,2)
    J = pt.tensor(0 * unit.MHz, dtype=cplx_dtype)

    print(f'A = {A/unit.MHz} MHz')

    tN = 500 * unit.ns 
    N = 500
    T = linspace(0,tN, N, dtype=real_dtype)

    H0 = get_H0(A, J, Bz)
    H0_phys = get_H0(A, J, 2*unit.T)
    S,D = get_ordered_eigensystem(H0, H0_phys)
    rf = get_rf_matrix(S, D)
    C = get_couplings(S)


    Bx, By = pi_pulse_square(rf[2,3], C[2,3], tN, 500)
    pad = pt.zeros(2250)
    #Bx = pt.cat((pad, Bx, pad))
    #By = pt.cat((pad, By, pad))

    U0 = pt.matrix_exp(-1j*pt.einsum('ab,j->jab',H0,T))
    X = get_electron_X(tN, N, Bz, A, J, Bx, By)

    psi0 = spin_01
    psi = X@psi0 

    phi_0 = pt.angle(X[:,0,0])
    phi_1 = pt.angle(X[:,1,1])
    phi_2 = pt.angle(X[:,2,2])
    phi_3 = pt.angle(X[:,3,3])

    phi_01 = phi_1 - phi_0
    phi_12 = phi_2 - phi_1
    phi_23 = phi_3 - phi_2
    phi_02 = phi_2 - phi_0


    fig,ax = plt.subplots(1,2) 

    ax[0].plot(T/unit.ns, pt.abs(psi[:,0]), label = '|psi_0|')
    ax[0].plot(T/unit.ns, pt.abs(psi[:,1]), label = '|psi_1|')
    ax[0].plot(T/unit.ns, pt.abs(psi[:,2]), label = '|psi_2|')
    ax[0].plot(T/unit.ns, pt.abs(psi[:,3]), label = '|psi_3|')

    ax[1].plot(T/unit.ns, phi_0, label='phi_0')
    ax[1].plot(T/unit.ns, phi_1, label='phi_1')
    ax[1].plot(T/unit.ns, phi_2, label='phi_2')
    ax[1].plot(T/unit.ns, phi_3, label='phi_3')

    # ax.plot(T/unit.ns, phi_01, label = 'phi_01')
    # ax.plot(T/unit.ns, phi_12, label = 'phi_12')
    # ax.plot(T/unit.ns, phi_23, label = 'phi_23')
    # ax.plot(T/unit.ns, phi_02, label = 'phi_02')
    ax[0].legend()
    ax[1].legend()

    



if __name__ == '__main__':
    simulate_exchange_with_refocussing()
    plt.show()











