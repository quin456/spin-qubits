

import numpy as np
import torch as pt
import matplotlib

if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt
from scipy.optimize import minimize

from GRAPE import Grape
import gates as gate 
from atomic_units import *
from visualisation import plot_psi, plot_psi_and_fields, visualise_Hw, plot_fidelity, plot_fields, plot_phases, plot_energy_spectrum, show_fidelity
from utils import *
import utils
from pulse_maker import pi_rot_square_pulse
from data import get_A, gamma_e, gamma_n, cplx_dtype
from visualisation import double_long_width, double_long_height, single_long_height, multi_NE_label_getter
from hamiltonians import get_IP_X, get_U0, get_pulse_hamiltonian, sum_H0_Hw, get_NE_H0, H_zeeman, H_hyperfine

from pdb import set_trace

Sz = 0.5*gate.IZ; Sy = 0.5*gate.IY; Sx = 0.5*gate.IX 
Iz = 0.5*gate.ZI; Iy = 0.5*gate.YI; Ix = 0.5*gate.XI

B_mag = 1e-3 * tesla

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
    Hw_mag = gamma_e*gate.IX - gamma_n*gate.XI
    couplings = S.T @ Hw_mag @ S

    print("Couplings (MHz/T):")
    utils.print_rank2_tensor(pt.real(couplings)*tesla/Mhz)
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


    H0 = get_NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c = pt.abs(couplings[2,3])
    w_eigenres = D[2,2]-D[3,3]
    Bx,By = pi_rot_square_pulse(w_eigenres, c, tN, N, phase)

    #Bx,By = pi_rot_pulse(w_res, gamma_e/2, tN, N, phase)
    if ax is not None:
        plot_fields(Bx,By,tN,ax)

    return Bx,By

def print_specs(A, Bz, tN, N, gamma):
    print("System specifications:")
    print(f"Hyperfine A = {A/Mhz} MHz")
    print(f"Static field Bz = {Bz/tesla} T")
    if tN is None:
        print(f"Control field strength = {B_mag/nanosecond} ns")
    else:
        print(f"Pulse duration tN = {tN/nanosecond} ns")
    print(f"Timesteps N={N}")
    H0 = get_NE_H0(A,Bz)
    max_coupling = get_max_allowed_coupling(H0)
    print(f"Max allowed coupling = {max_coupling/Mhz} MHz, corresponding to field strength B={max_coupling/(0.5*gamma) / mT} mT")

def show_NE_CX(A,Bz,tN,N, psi0=spin_down_down): 
    print("Performing NE_CX, which flips electron spin conditionally on nuclear spin being down.")
    print_specs(A, Bz, tN, N, gamma=gamma_e)
    fig,ax = plt.subplots(1,4)
    Bx,By = NE_CX_pulse(tN,N,A,Bz, ax[0])
    X = get_NE_X(tN, N, Bz, A, Bx, By)
    fids = show_fidelity(X,tN, gate.CX, ax=ax[1])
    psi = pt.matmul(X,psi0)
    plot_psi(psi,tN,ax[2])
    plot_phases(psi,tN,ax[3])



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
    H0 = get_NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c = pt.abs(couplings[1,3])
    w_eigenres = D[1,1]-D[3,3]

    if tN is None:
        tN = get_pi_pulse_tN_from_field_strength(B_mag, c)

    Bx,By = pi_rot_square_pulse(w_eigenres, c, tN, N, phase)

    if ax is not None: 
        plot_fields(Bx,By,tN,ax)


    return Bx, By, tN








def get_NE_Hw(Bx,By):
    return -get_pulse_hamiltonian(Bx, By, gamma_n, 2*Ix, 2*Iy) + get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy)

def get_NE_X(tN, N, Bz, A, Bx=None, By=None):
    # Bx and By both None results in free evolution X
    if Bx==None:
        Bx = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    if By==None:
        By = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    #Bx,By = NE_CX_pulse(tN,N,A,Bz)
    Hw = get_NE_Hw(Bx,By)
    H0 = get_NE_H0(A, Bz)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)

    X = forward_prop(U)
    S,D = NE_eigensystem(H0)
    #X = get_IP_X(X,H0,tN,N)
    #X = dagger(get_U0(H0,tN,N))@X



    #undo only zeeman evolution
    Hz=H_zeeman(Bz)
    UZ = get_U0(Hz,tN,N)
    U0 = get_U0(H0, tN, N)


    X = dagger(UZ) @ X

    #visualise_Hw(dagger(S)@dagger(U0)@Hw@U0@S,tN); plt.show()

    return X

def get_phase_correction(Bz, A, Xf, N_search, tN_search):
    
    X_search = Xf@get_NE_X(tN_search, N_search, Bz, A)
    fids = fidelity_progress(X_search, gate.CXr)

    N_wait = pt.argmax(fids)
    tN_wait = N_wait*tN_search/N_search

    print(f"Adding {tN_wait/nanosecond} ns free evolution to correct phase, achieving fidelity {fids[N_wait]}")

    X_wait = X_search[:N_wait]
    T_wait = pt.linspace(0, tN_wait, N_wait)
    return X_wait, T_wait 


def show_EN_CX(A,Bz,N, tN=None, psi0=spin_down_down):
    print("Performing EN_CX, which flips nuclear spin conditionally on electron spin being down.")
    print_specs(A, Bz, tN, N, gamma_n)
    fig,ax = plt.subplots(1,3)
    Bx,By,tN = EN_CX_pulse(tN,N,A,Bz)
    X_CX = get_NE_X(tN, N, Bz, A, Bx, By)
    T_CX = pt.linspace(0,tN,N)

    X_wait,T_wait = get_phase_correction(Bz, A, X_CX[-1], N, 500*nanosecond)
    T = pt.cat((T_CX, T_CX[-1]+T_wait))
    X = pt.cat((X_CX, X_wait))


    fids=show_fidelity(X,T=T, ax=ax[2])


    psi = pt.matmul(X,psi0)
    plot_psi(psi,T=T, ax=ax[0])
    plot_phases(psi,T=T, ax=ax[1])



def get_swap_pulse_times(tN, A):

    tN_EN = 0.9*tN 
    tN_NE = (tN-tN_EN)/2 

    tN_EN = lock_to_frequency(A, tN_EN)
    tN_NE = lock_to_frequency(A, tN_NE)

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

    H0 = get_NE_H0(A,Bz)

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
    plot_fidelity(ax[0],fids,tN)
    plot_fields(Bx,By,tN,ax[1])
    
    print(f"Unitary achieved ] \n{X[-1]}")

def show_NE_swap(A,Bz,tN,N, psi0=spin_down_down):
    fig,ax = plt.subplots(3,1)
    Bx,By = NE_swap_pulse(tN,N,A,Bz, ax[0])

    H0 = get_NE_H0(A, Bz)
    X = get_NE_X(tN, N, Bz, A, Bx, By)
    X = get_IP_X(X,H0,tN,N)

    show_fidelity(X,tN,gate.swap,ax[1])


    psi = X @ psi0 
    plot_psi(psi,tN,ax[2])

    fig.set_size_inches(double_long_width, double_long_height)



def multi_NE_label_getter(j):
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
    H0 = get_NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)
    E = pt.diagonal(D)/Mhz
    #plot_energy_spectrum(E)

    ax = plt.subplot()
    ax.axhline(1, label=multi_NE_label_getter(0))
    ax.axhline(10, label=f'$\psi$')
    ax.legend()

def triple_NE_H0():
    return gamma_e*Bz*gate.get_Sz_sum(3) - gamma_n*Bz*gate.get_Iz_sum(3) # unfinished

def get_subops(H,dt):
    ''' Gets suboperators for time-independent Hamiltonian H '''
    return pt.matrix_exp(-1j*H*dt)


class NuclearElectronGrape(Grape):

    def __init__(self, tN, N, Bz=2*tesla, A=get_A(1,1), nq=2, target=gate.swap, rf=None, u0=None,  save_data=False, max_time=99999):

        # save system data before super().__init__
        self.A = A
        self.Bz = Bz
        self.nq=nq
        super().__init__(tN, N, target, rf, u0=u0, save_data=save_data, max_time=max_time)

        # perform calculations of system parameters after super().__init__
        self.Hw = self.get_Hw()
        self.S, self.D = NE_eigensystem(self.H0[0])

    def get_H0(self, interaction_picture=False):
        if self.nq==6:
            H0 = triple_NE_H0()
        if interaction_picture:
            H0 = H_hyperfine(self.A)
        else:
            H0 = get_NE_H0(self.A, self.Bz)
        return H0.reshape(1,*H0.shape)

    def get_Hw(self, interaction_picture=False):
        return get_NE_Hw(self.x_cf, self.y_cf)




def run_NE_grape():
    grape = NuclearElectronGrape(Bz=0.02*tesla, tN=100*nanosecond, N=2500, target = gate.CXr, max_time = 60)
    
    grape.run()
    grape.plot_result()



def test():
    Bz = 2*tesla 
    A = get_A(1,1)
    tN = lock_to_frequency(A, 50*nanosecond)
    N=10000

    H0 = get_NE_H0(A,Bz)
    U0 = get_U0(H0, tN, N)
    S,D = NE_eigensystem(H0)
    U0_d = get_U0(D,tN,N)

    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    Hw = get_NE_Hw(Bx,By)

    visualise_Hw(dagger(S)@dagger(U0)@Hw@U0@S,tN)




if __name__ == '__main__':

    psi0=pt.kron(spin_up,spin_down)

    Bz=2*tesla
    steps_per_nanosecond = 20 * gamma_e*Bz/Ghz

    def min_steps(tN):
        return int(steps_per_nanosecond * tN/nanosecond)


    # tN = 1000*nanosecond#lock_to_coupling(get_A(1,1),100*nanosecond)
    # show_NE_CX(get_A(1,1), Bz, tN, min_steps(tN), psi0=pt.kron(spin_down, spin_up)); plt.show()

    # tN = 1001*nanosecond
    # tN = lock_to_frequency(get_A(1,1),tN)
    show_EN_CX(get_A(1,1), Bz=Bz, N=40000); plt.show()
    #test(); plt.show()
    #tN_locked = lock_to_coupling(get_A(1,1),500*nanosecond)
    #show_NE_swap(A=get_A(1,1), Bz=0.02*tesla, tN=10*nanosecond, N=10000); plt.show()

    # tN = 1*nanosecond
    # psi0 = 0.5 * (gate.spin_00 + gate.spin_01 + gate.spin_10 + gate.spin_11)
    # X_free = get_NE_X(tN, 1000, 2*tesla, get_A(1,1))
    # psi = X_free @ psi0 
    # fig,ax = plt.subplots(2,1)
    # plot_psi(psi, tN, ax=ax[0])
    # plot_phases(psi, tN, ax=ax[1])


    # plt.show()

    # set_trace()


