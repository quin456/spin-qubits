

from turtle import forward
import numpy as np
import torch as pt
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt


import gates as gate 
import atomic_units as unit
from utils import *
from eigentools import *
from visualisation import plot_psi, plot_psi_and_fields, visualise_Hw, plot_fidelity, plot_fields, plot_phases, plot_energy_spectrum, show_fidelity
from pulse_maker import pi_pulse_square
from data import *
from visualisation import *
from hamiltonians import get_IP_X, get_U0, get_pulse_hamiltonian, sum_H0_Hw, get_NE_H0, H_zeeman, H_hyperfine, get_NE_Hw, get_X_from_H
from GRAPE import Grape


from pdb import set_trace

Sz = 0.5*gate.IZ; Sy = 0.5*gate.IY; Sx = 0.5*gate.IX 
Iz = 0.5*gate.ZI; Iy = 0.5*gate.YI; Ix = 0.5*gate.XI

B_mag = 1e-3 * unit.T

from gates import spin_up, spin_down
spin_down_down = pt.kron(spin_down, spin_down)


def E_CX(A, Bz, tN, N, psi0=spin_up):
    '''
    CNOT gate with electron spin as target, nuclear spin as control.
    '''
    Bx,By = NE_CX_pulse(tN,N,A,Bz)
    H0 = (A+gamma_e*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx,By,gamma_e)
    T = linspace(0,tN,N, dtype=cplx_dtype, device=default_device)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)


    plot_psi_and_fields(psi,Bx,By,tN)


def N_CX(A, Bz, tN, N, psi0=spin_up):
    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    H0 = (-A-gamma_n*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx, By, gamma_n)
    T = linspace(0,tN,N, dtype=cplx_dtype, device=default_device)
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
    print_rank2_tensor(pt.real(couplings)*unit.T/unit.MHz)
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
    c = couplings[2,3]
    w_res = D[2,2]-D[3,3]
    if tN is None:
        Omega = np.sqrt(16*A**2 + 0.25*gamma_e**2*0.5227**2*unit.mT**2) / 4
        tN = lock_to_frequency(A,get_pi_pulse_tN_from_field_strength(B_mag, c))
    Bx,By = pi_pulse_square(w_res, c, tN, N, phase)

    #Bx,By = pi_rot_pulse(w_res, gamma_e/2, tN, N, phase)
    if ax is not None:
        plot_fields(Bx,By,tN,ax)

    T = linspace(0, tN, N)

    Bx,By,T = correct_phase(Bx, By, T, N, Bz, A, gate.CX)
    return Bx,By,T

def EN_CX_pulse(tN,N,A,Bz, ax=None):

    H0 = get_NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c = pt.abs(couplings[1,3])
    w_eigenres = D[1,1]-D[3,3]

    if tN is None:
        tN = get_pi_pulse_tN_from_field_strength(B_mag, c)

    Bx_CX, By_CX = pi_pulse_square(w_eigenres, c, tN, N, 0)

    if ax is not None: 
        plot_fields(Bx,By,tN,ax)

    T_CX = linspace(0,tN,N, dtype=cplx_dtype, device=default_device)

    Bx, By, T = correct_phase(Bx_CX, By_CX, T_CX, N, Bz, A, gate.CXr)

    #Bx = Bx_CX; By=By_CX; T=T_CX

    return Bx, By, T

def get_swap_pulse_times(tN, A):

    tN_EN = 0.9*tN 
    tN_NE = (tN-tN_EN)/2 

    tN_EN = lock_to_frequency(A, tN_EN)
    tN_NE = lock_to_frequency(A, tN_NE)

    return tN_NE, tN_EN


def NE_swap_pulse(N_e, N_n, A, Bz, ax=None):

    Bx_NE,By_NE,T_e = NE_CX_pulse(None,N_e, A, Bz)
    Bx_EN,By_EN, T_n = EN_CX_pulse(None, N_n, A, Bz)

    Bx_swap = pt.cat((Bx_NE,Bx_EN,Bx_NE))
    By_swap = pt.cat((By_NE,By_EN,By_NE))
    T_swap = pt.cat((T_e, T_e[-1]+T_n, T_e[-1]+T_n[-1]+T_e))


    N_swap = len(Bx_swap)
    X_swap = get_NE_X(N_swap, Bz, A, Bx_swap, By_swap, T=T_swap)

    return Bx_swap, By_swap, T_swap

    #Bx=Bx_swap; By=By_swap; T=T_swap

    if ax is not None:
        plot_fields(Bx,By,T=T,ax=ax)
    return Bx,By,T
def print_specs(A, Bz, tN, N, gamma):
    print("System specifications:")
    print(f"Hyperfine A = {A/unit.MHz} MHz")
    print(f"Static field Bz = {Bz/unit.T} T")
    if tN is None:
        print(f"Control field strength = {B_mag/unit.mT} mT")
    else:
        print(f"Pulse duration tN = {tN/unit.ns} ns")
    print(f"Timesteps N={N}")
    H0 = get_NE_H0(A,Bz)
    max_coupling = get_max_allowed_coupling(H0)
    print(f"Max allowed coupling = {max_coupling/unit.MHz} MHz, corresponding to field strength B={max_coupling/(0.5*gamma) / unit.mT} mT")

def show_NE_CX(A,Bz,N, tN=None, psi0=(gate.spin_00+gate.spin_10)/np.sqrt(2), fp=None, ax=None, fig=None, legend_loc='best'): 
    print("Performing NE_CX, which flips electron spin conditionally on nuclear spin being down.")
    print_specs(A, Bz, tN, N, gamma=gamma_e)
    if ax is None:
        fig,ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [1.5, 0.5,  1]})
        #fig,ax = plt.subplots(1,2)
    Bx,By,T = NE_CX_pulse(tN,N,A,Bz)
    X = get_NE_X(N, Bz, A, Bx, By, T=T)
    show_fidelity(X, T=T, target=gate.CX, ax=ax[0])
    t_zoom = 0.2*unit.ns
    N_zoom = 1
    while pt.real(T[-1]-T[-N_zoom])<t_zoom: 
        N_zoom+=1
    T_zoom = T[-N_zoom:]
    X_zoom = X[-N_zoom:]
    show_fidelity(X_zoom, T=T_zoom, target=gate.CX, ax=ax[1])
    box_ax(ax[0], xlim = [(T[-1]-t_zoom)/unit.ns, T[-1]/unit.ns], padding=0)
    box_ax(ax[1], xlim=[(T[-1]-t_zoom)/unit.ns, T[-1]/unit.ns], padding=0.1)
    psi = pt.matmul(X,psi0)
    plot_psi(psi,tN=tN, T=T, ax=ax[2], label_getter = NE_label_getter, legend_loc=legend_loc)
    if fig is not None:
        fig.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig.tight_layout()
    if fp is not None:
        fig.savefig(fp)




#def get_EN_X_with_wait()





def get_NE_X(N, Bz, A, Bx=None, By=None, T=None, tN=None):
    # Bx and By both None results in free evolution X
    if Bx==None:
        Bx = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    if By==None:
        By = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    if T is None:
        if tN is None:
            raise Exception("No time specified for get_NE_X.")
        else:
            T = linspace(0,tN,N, dtype=cplx_dtype, device=default_device)
    Hw = get_NE_Hw(Bx,By)
    H0 = get_NE_H0(A, Bz)
    H = sum_H0_Hw(H0,Hw)

    X = get_X_from_H(H, T=T)

    S,D = NE_eigensystem(H0)
    #X = get_IP_X(X,H0,tN,N)
    #X = dagger(get_U0(H0,tN,N))@X


    return X
    #undo only zeeman evolution
    Hz=H_zeeman(Bz)
    UZ = get_U0(H0, N, T=T)
    X = dagger(UZ) @ X
    return X


def get_phase_correction(Bz, A, Xf, N_search, tN_search, target=gate.CXr):

    T_search = linspace(0,tN_search,N_search)
    X_free = get_NE_X(N_search, Bz, A, T=T_search)
    X_search = X_free@Xf
    fids = fidelity_progress(X_search, target)

    N_wait = pt.argmax(fids)+1
    tN_wait = N_wait*tN_search/N_search

    print(f"Adding {tN_wait/unit.ns} ns free evolution to correct phase, achieving fidelity {fids[N_wait]}, up from {fids[0]}")

    Bx_wait = pt.zeros(N_wait, dtype=cplx_dtype, device=default_device)
    By_wait = pt.zeros(N_wait, dtype=cplx_dtype, device=default_device)
    T_wait = T_search[:N_wait]

    return Bx_wait, By_wait, T_wait

def correct_phase(Bx_CX, By_CX, T_CX, N, Bz, A, target):
    X_CX = get_NE_X(N, Bz, A, Bx_CX, By_CX, T=T_CX)
    Bx_wait, By_wait, T_wait = get_phase_correction(Bz, A, X_CX[-1], N, 500*unit.ns, target=target)
    T = pt.cat((T_CX, T_CX[-1]+T_wait))
    Bx =pt.cat((Bx_CX, Bx_wait))
    By =pt.cat((By_CX, By_wait))
    print(f"Phase correction finished. Pulse now runs from {T[0]/unit.ns} ns to {T[-1]/unit.ns} ns")
    return Bx, By, T

def show_EN_CX(A,Bz,N, tN=None, psi0=(gate.spin_00+gate.spin_01)/np.sqrt(2), target = gate.CXr, fp=None, ax=None, fig=None, legend_loc = 'best'):
    print("Performing EN_CX, which flips nuclear spin conditionally on electron spin being down.")
    print_specs(A, Bz, tN, N, gamma_n)

    if ax is None:
        fig,ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [1.5, 0.5,  1]})

    Bx,By,T = EN_CX_pulse(tN,N,A,Bz)

    X = get_NE_X(N, Bz, A, Bx, By, T=T)

    fids=show_fidelity(X, T=T, target=target, ax=ax[0])
    t_zoom = 0.1*unit.ns
    N_zoom = 1
    while pt.real(T[-1]-T[-N_zoom])<t_zoom: 
        N_zoom+=1
    T_zoom = pt.round(pt.real(T[-N_zoom:]),decimals=2)
    X_zoom = X[-N_zoom:]
    show_fidelity(X_zoom, T=T_zoom, target=gate.CXr, ax=ax[1])
    ax[1].set_xlabel('')
    ax[1].set_xticks(np.round(np.array([T[-1]-t_zoom,T[-1]]),decimals=1)/unit.ns)
    box_ax(ax[0], xlim = [(T[-1]-t_zoom)/unit.ns, T[-1]/unit.ns], padding=0)
    box_ax(ax[1], xlim=[(T[-1]-t_zoom)/unit.ns, T[-1]/unit.ns], padding=0)

    psi = pt.matmul(X,psi0)
    plot_psi(psi,T=T, ax=ax[2], label_getter=NE_label_getter, legend_loc = legend_loc)
    #plot_phases(psi,T=T, ax=ax[1])

    if fig is not None:
        fig.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig.tight_layout()
    if fp is not None: fig.savefig(fp)





def save_NE_swap_pulse():
    A = get_A(1,1)
    Bz = 2*unit.T
    N_e = 100000
    N_n = 40000

    fig,ax = plt.subplots(1,2)
    Bx,By,T = NE_swap_pulse(N_e, N_n, A, Bz)
    Bx_By_T = pt.stack((Bx, By, T)).T
    pt.save(Bx_By_T, "fields/swap_fields")


def show_NE_swap(A, Bz, N_e, N_n, psi0=None, fp=None):

    if psi0 is None: psi0 = (gate.spin_00 + gate.spin_01)/np.sqrt(2)

    fig,ax = plt.subplots(1,2)
    Bx,By,T = NE_swap_pulse(N_e, N_n, A, Bz)

    
    X = get_NE_X(0, Bz, A, Bx=Bx, By=By, T=T)

    show_fidelity(X, T=T, target=gate.swap, ax=ax[0])


    psi = X @ psi0 
    plot_psi(psi, T=T, ax=ax[1], label_getter = NE_label_getter)


    fig.set_size_inches(fig_width_double_long, fig_height_double_long)
    fig.tight_layout()
    if fp is not None: fig.savefig(fp)



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
    
def NE_energy_levels(A=get_A(1,1)*unit.MHz, Bz=2*unit.T):
    H0 = get_NE_H0(A,Bz)
    S,D = NE_eigensystem(H0)
    E = pt.diagonal(D)/unit.MHz
    #plot_energy_spectrum(E)

    ax = plt.subplot()
    ax.axhline(1, label=multi_NE_label_getter(0))
    ax.axhline(10, label=f'$\psi$')
    ax.legend()

def triple_NE_H0():
    return gamma_e*B0*gate.get_Sz_sum(3) - gamma_n*B0*gate.get_Iz_sum(3) # unfinished

def get_subops(H,dt):
    ''' Gets suboperators for time-independent Hamiltonian H '''
    return pt.matrix_exp(-1j*H*dt)


class NuclearElectronGrape(Grape):

    def __init__(self, tN, N, Bz=2*unit.T, A=get_A(1,1), nq=2, target=gate.swap, rf=None, u0=None,  save_data=False, max_time=99999):

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
    grape = NuclearElectronGrape(Bz=0.02*unit.T, tN=100*unit.ns, N=2500, target = gate.CXr, max_time = 60)
    
    grape.run()
    grape.plot_result()



def test():
    Bz = 2*unit.T 
    A = get_A(1,1)
    tN = lock_to_frequency(A, 50*unit.ns)
    N=10000

    H0 = get_NE_H0(A,Bz)
    U0 = get_U0(H0, tN, N)
    S,D = NE_eigensystem(H0)
    U0_d = get_U0(D,tN,N)

    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    Hw = get_NE_Hw(Bx,By)

    visualise_Hw(dagger(S)@dagger(U0)@Hw@U0@S,tN)


def kms(N=5, Bz = 2*unit.T, A=get_A(1,1)):

    tN = 100*unit.ns 
    T = linspace(0,tN,N, dtype=cplx_dtype)

    X1 = get_NE_X(N, Bz, A, tN=tN)
    X2 = get_NE_X(N, Bz, A, T=T)


    H0 = get_NE_H0(A, Bz)

    U01 = get_U0(H0, N, tN=tN)
    U02 = get_U0(H0, N, T=T)




    print(U01==U02)



def IP_NE_SWAP_things(A = get_A(1,1), tN_e=100*unit.ns, tN_n=12000*unit.ns, N = 20000):
    
    H0 = get_NE_H0(A, B0)
   

    S,D = NE_eigensystem(H0)

    couplings = NE_couplings(H0)
    c_e = couplings[2,3]
    w_res_e = D[2,2]-D[3,3]
    Bx_e,By_e = pi_pulse_square(w_res_e, c_e, tN_e, N, 0)
    T_e = linspace(0,tN_e,N)

    c_n = couplings[1,3]
    w_res_n = D[1,1]-D[3,3]
    Bx_n,By_n = pi_pulse_square(w_res_n, c_n, tN_n, N, 0)
    T_n = linspace(0,tN_n,N)

    set_trace()

    Bx = Bx_n; By=By_n; T=T_n
    #Bx = pt.zeros(N); By=pt.zeros_like(Bx)
    H_ac = get_NE_Hw(Bx, By)
    #U0 = get_U0(H0, N, T=T)
    U0 = get_U0(D, N, T=T)
    #H_ac_IP = pt.einsum('jac,jcd->jad',pt.einsum('jab,jbc->jac',dagger(U0),H_ac),U0)
    H_ac_IP = pt.einsum('jac,cd->jad',pt.einsum('ab,jbc->jac',dagger(S),H_ac),S)
    H_ac_IP = pt.einsum('jac,jcd->jad',pt.einsum('jab,jbc->jac',dagger(U0),H_ac_IP),U0)
    H_ac_IP[:,0,1] = pt.zeros_like(H_ac_IP[:,0,1])
    H_ac_IP[:,0,2] = pt.zeros_like(H_ac_IP[:,0,1])
    H_ac_IP[:,1,0] = pt.zeros_like(H_ac_IP[:,0,1])
    H_ac_IP[:,2,0] = pt.zeros_like(H_ac_IP[:,0,1])
    H_ac_IP[:,2,3] = pt.zeros_like(H_ac_IP[:,0,1])
    H_ac_IP[:,3,2] = pt.zeros_like(H_ac_IP[:,0,1])
    visualise_Hw(H_ac_IP,tN_n)
    
    dT = get_dT(T)
    Ht = pt.einsum('j,jab->jab', dT, H_ac_IP)
    U = pt.matrix_exp(-1j*Ht)

    X = forward_prop(U)

    psi0 = gate.spin_01
    fig,ax = plt.subplots(1,2)
    plot_psi(X@psi0, T=T, ax=ax[1])
    show_fidelity(X, T=T, target=gate.CXr_native, ax=ax[0])
    






if __name__ == '__main__':


    #tN = lock_to_frequency(get_A(1,1),100*unit.ns)

    show_NE_CX(get_A(1,1), 2*unit.T,  300000, psi0 = (gate.spin_10)); plt.show()

    #show_EN_CX(get_A(1,1), Bz=2*unit.T, N=400); plt.show()

    #show_NE_swap(get_A(1,1), 2*unit.T, 100000, 40000)
    #save_NE_swap_pulse()
    #IP_NE_SWAP_things()


    plt.show()

    #kms()
    


