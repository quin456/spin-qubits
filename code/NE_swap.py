

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt 
import torch as pt
from scipy.optimize import minimize


import gates as gate 
from atomic_units import *
from visualisation import plot_spin_states, plot_psi_and_fields, visualise_Hw, plot_fidelity_progress, plot_fields, plot_phases, plot_energy_spectrum
from utils import forward_prop, get_pulse_hamiltonian, sum_H0_Hw, fidelity, fidelity_progress, get_U0, dagger, get_IP_X, get_IP_eigen_X, show_fidelity
from pulse_maker import pi_rot_square_pulse
from data import get_A, gamma_e, gamma_n, cplx_dtype

from pdb import set_trace

Sz = 0.5*gate.IZ; Sy = 0.5*gate.IY; Sx = 0.5*gate.IX 
Iz = 0.5*gate.ZI; Iy = 0.5*gate.YI; Ix = 0.5*gate.XI

from gates import spin_up, spin_down
spin_down_down = pt.kron(spin_down, spin_down)


def E_CX(A, Bz, tN, N, psi0=spin_up):
    '''
    CNOT gate with electron spin as target, nuclear spin as control.
    '''
    Bx,By = NE_CX_pulse(tN,N,A,Bz)
    H0 = (A+gamma_e*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx,By,gamma_e)
    T = pt.linspace(0,tN,N)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)


    plot_psi_and_fields(psi,Bx,By,tN)


def N_CX(A, Bz, tN, N, psi0=spin_up):
    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    H0 = (-A-gamma_n*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx, By, gamma_n)
    T = pt.linspace(0,tN,N)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)
    plot_psi_and_fields(psi,Bx,By,tN)



def H_zeeman(Bz):
    return gamma_e*Bz*Sz - gamma_n*Bz*Iz

def H_hyperfine(A):
    return A * gate.sigDotSig

def NE_H0(A,Bz):
    return H_zeeman(Bz) + H_hyperfine(A)


def NE_eigensystem(H0):
    eig = pt.linalg.eig(H0)
    E = eig.eigenvalues
    S = eig.eigenvectors

    def reorder(A):
        A_new = pt.zeros_like(A)
        A_new[:,0] = A[:,2]
        A_new[:,1] = A[:,1]
        A_new[:,2] = A[:,0]
        A_new[:,3] = A[:,3]
        return A_new
    E = reorder(E.reshape((1,len(S)))).flatten()
    D = pt.diag(E)

    return reorder(S), D

def NE_couplings(H0):
    S,D = NE_eigensystem(H0)
    Hw_mag = 0.5*gamma_e*gate.IX - 0.5*gamma_n*gate.XI
    couplings = S.T @ Hw_mag @ S
    return couplings

def get_coupling(A,Bz):
    Gbar = (gamma_e + gamma_n) * Bz / 2 # remove Bz from Gamma later on 
    K = ( 2 * (4*A**2 + Gbar**2 - Gbar*np.sqrt(4*A**2+Gbar**2)) )**(-1/2)
    alpha = -Gbar + np.sqrt(4*A**2+Gbar**2)
    beta = 2*K*A
    Ge = gamma_e/2
    Gn = gamma_n/2
    return alpha*Gn + beta*Ge


def NE_CX_pulse(tN,N,A,Bz, ax=None):
    w_res = -2*A + gamma_e * Bz
    phase = 0


    H0 = NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c = pt.abs(couplings[2,3])
    w_eigenres = D[2,2]-D[3,3]
    Bx,By = pi_rot_square_pulse(w_eigenres, c, tN, N, phase)

    #Bx,By = pi_rot_pulse(w_res, gamma_e/2, tN, N, phase)
    if ax is not None:
        plot_fields(Bx,By,tN,ax)

    return Bx,By

def show_NE_CX(A,Bz,tN,N, psi0=spin_down_down): 
    fig,ax = plt.subplots(1,4)
    Bx,By = NE_CX_pulse(tN,N,A,Bz, ax[0])
    H0 = NE_H0(A,Bz)
    X = get_NE_X(Bx, By, H0, tN, N)
    show_fidelity(X,tN,N, gate.CX, ax=ax[1])

    psi = pt.matmul(X,psi0)
    plot_spin_states(psi,tN,ax[2])
    plot_phases(psi,tN,ax[3])

    Hw = get_NE_Hw(Bx,By)

    H0 = NE_H0(A,Bz)


def EN_CX_pulse(tN,N,A,Bz, ax=None):

    w_res = -(2*A+gamma_n*Bz)
    phase = 0


    # get coupling of transition to transverse field
    # Gbar = (gamma_e + gamma_P) * Bz / 2 # remove Bz from Gamma later on 
    # alpha = -Gbar - np.sqrt(4*A**2+Gbar**2)
    # beta = 2*A
    # K = 1/np.sqrt(alpha**2+beta**2)
    # alpha*=K; beta*=K 

    # Ge = gamma_e/2
    # Gn = -gamma_P/2
    # c = alpha*Gn + beta*Ge
    # E = pt.linalg.eig(H0).eigenvalues
    # w_eigenres = E[1]-E[3]



    # get actual resonant freq
    H0 = NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c = pt.abs(couplings[1,3])
    w_eigenres = D[1,1]-D[3,3]
    Bx,By = pi_rot_square_pulse(w_eigenres, c, tN, N, phase)

    if ax is not None: 
        plot_fields(Bx,By,tN,ax)
    return Bx,By








def get_NE_Hw(Bx,By):
    return -get_pulse_hamiltonian(Bx, By, gamma_n, 2*Ix, 2*Iy) + get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy)

def get_NE_X(Bx, By, H0, tN, N):

    #Bx,By = NE_CX_pulse(tN,N,A,Bz)
    Hw = get_NE_Hw(Bx,By)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)

    X = forward_prop(U)
    S,D = NE_eigensystem(H0)
    #X = get_IP_X(X,H0,tN,N)
    #X = dagger(get_U0(H0,tN,N))@X

    #undo only zeeman evolution
    UZ = get_U0(H_zeeman(2*tesla),tN,N)
    X = dagger(UZ) @ X

    return X


def show_EN_CX(A,Bz,tN,N, psi0=spin_down_down):
    fig,ax = plt.subplots(1,4)
    Bx,By = EN_CX_pulse(tN,N,A,Bz, ax[0])
    H0 = NE_H0(A,Bz)
    X = get_NE_X(Bx, By, H0, tN, N)
    show_fidelity(X,tN,N, gate.CXr, ax=ax[1])

    psi = pt.matmul(X,psi0)
    plot_spin_states(psi,tN,ax[2])
    plot_phases(psi,tN,ax[3])

    Hw = get_NE_Hw(Bx,By)

    H0 = NE_H0(A,Bz)


def get_swap_pulse_times(tN, A):

    tN_EN = 0.9*tN 
    tN_NE = (tN-tN_EN)/2 

    tN_EN = lock_to_coupling(A, tN_EN)
    tN_NE = lock_to_coupling(A, tN_NE)

    return tN_NE, tN_EN


def NE_swap_pulse(tN,N,A,Bz, ax=None):
    
    N_NE = N//10
    N_EN = N-2*N_NE
    tN_NE = N_NE/N * tN 
    tN_EN = N_EN/N * tN

    #tN_NE, tN_EN = get_swap_pulse_times(tN, A)

    Bx_NE,By_NE = NE_CX_pulse(tN_NE, N_NE, A, Bz)
    Bx_EN,By_EN = EN_CX_pulse(tN_EN, N_EN, A, Bz)

    Bx = pt.cat((Bx_NE,Bx_EN,Bx_NE))
    By = pt.cat((By_NE,By_EN,By_NE))

    if ax is not None:
        plot_fields(Bx,By,tN,ax)
    return Bx,By

def NE_swap(A,Bz,tN,N):

    H0 = NE_H0(A,Bz)

    Bx,By = NE_swap_pulse(tN,N,A,Bz)
    #Bx,By = NE_CX_pulse(tN,N,A,Bz)
    Hw = - get_pulse_hamiltonian(Bx, By, gamma_n, 2*Ix, 2*Iy) + get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy) 
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)

    U0 = get_U0
    X = forward_prop(U)
    U0 = get_U0(H0, tN, N)
    X = pt.matmul(dagger(U0),X)

    return X

def NE_swap_fidelity(A,Bz,tN,N):
    Bx,By = NE_swap_pulse(tN,N,A,Bz)
    X = NE_swap(A,Bz,tN,N)

    fig,ax = plt.subplots(1,2)
    fids = fidelity_progress(X,gate.swap)
    plot_fidelity_progress(ax[0],fids,tN)
    plot_fields(Bx,By,tN,ax[1])
    
    print(f"Unitary achieved ] \n{X[-1]}")

def show_NE_swap(A,Bz,tN,N, psi0=spin_down_down):
    fig,ax = plt.subplots(1,3)
    Bx,By = NE_swap_pulse(tN,N,A,Bz, ax[0])

    H0 = NE_H0(A, Bz)
    X = get_NE_X(Bx, By, H0, tN, N)
    X = get_IP_X(X,H0,tN,N)

    show_fidelity(X,tN,gate.swap,ax[1])


    psi = X @ psi0 
    plot_spin_states(psi,tN,ax[2])



def get_NE_label(j):
    ''' Returns state label corresponding to integer j\in[0,dim] '''
    uparrow = u'\u2191'
    downarrow = u'\u2193'
    b= np.binary_repr(j,4)
    if b[2]=='0':
        L2 = uparrow 
    else:
        L2=downarrow
    if b[3]=='0':
        L3 = uparrow
    else:
        L3 = downarrow
    
    return b[0]+b[1]+L2+L3
    
def NE_energy_levels(A=get_A(1,1)*Mhz, Bz=2*tesla):
    H0 = NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)
    E = pt.diagonal(D)/Mhz
    #set_trace()
    #plot_energy_spectrum(E)

    ax = plt.subplot()
    ax.axhline(1, label=get_NE_label(0))
    ax.axhline(10, label=f'$\psi$')
    ax.legend()



def get_subops(H,dt):
    ''' Gets suboperators for time-independent Hamiltonian H '''
    return pt.matrix_exp(-1j*H*dt)


def lock_to_coupling(c, tN):
    t_HF = 2*np.pi/c
    tN_locked = int(tN / (t_HF) ) * t_HF
    return tN_locked


if __name__ == '__main__':

    psi0=pt.kron(spin_up,spin_down)

    tN = lock_to_coupling(get_A(1,1),50*nanosecond)
    #show_NE_CX(get_A(1,1)*Mhz, 2*tesla, tN, 100000, psi0=pt.kron(spin_down, spin_up)); plt.show()

    tN_locked = lock_to_coupling(get_A(1,1),499*nanosecond)
    #show_EN_CX(get_A(1,1)*Mhz, 2*tesla, tN_locked, 5000); plt.show()


    tN_locked = lock_to_coupling(get_A(1,1),500*nanosecond)
    show_NE_swap(get_A(1,1), 2*tesla, tN_locked, 10000); plt.show()


#NE_energy_levels(); plt.show()


