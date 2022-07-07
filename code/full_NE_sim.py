
import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from GRAPE import *
import gates as gate
from atomic_units import *
from data import get_A, get_J
from visualisation import plot_spin_states
from pulse_maker import square_pulse
from utils import get_pulse_hamiltonian, lock_to_coupling
from NE_swap import NE_swap_pulse, NE_CX_pulse


from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']




spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)




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


def plot_nuclear_electron_wf(psi,tN, ax=None):
    N,dim = psi.shape
    nq = get_nq(dim)
    T = pt.linspace(0,tN/nanosecond,N)
    
    if ax is None: ax=plt.subplot()
    for j in range(dim):
        ax.plot(T,pt.abs(psi[:,j]), label = get_NE_label(j))
    ax.legend()



def get_nuclear_spins(A):
    nq=len(A)
    spins = [0]*nq
    for i in range(nq):
        if pt.real(A[i])>0:
            spins[i]=spin_up 
        else:
            spins[i]=spin_down 
    return spins


def multi_NE_H0(Bz, A, J, nq):
    """
    Returns free evolution Hamiltonian of nq==2 or nq==3 electron-nucleus pairs. Each electron interacts with its
    nucleus via hyperfine term 4AS.I, each neighboring electron interacts via exchange 4JS.S, and nulear and electrons
    experience Zeeman splitting due to background static field, Bz.
    """

    Iz = gate.get_Iz_sum(nq)
    Sz = gate.get_Sz_sum(nq)
    if nq==2:
        o_n1e1 = gate.o4_13 
        o_n2e2 = gate.o4_24 
        o_e1e2 = gate.o4_34
        H0 = gamma_e*Bz*Sz - gamma_n*Bz*Iz + A*o_n1e1 + A*o_n2e2 + J*o_e1e2/10
    elif nq==3:
        o_n1e1 = gate.o6_14
        o_n2e2 = gate.o6_25
        o_n3e3 = gate.o6_36
        o_e1e2 = gate.o6_45 
        o_e2e3 = gate.o6_56 
        H0 = gamma_e*Bz*Sz - gamma_n*Bz*Iz + A*(o_n1e1+o_n2e2+o_n3e3) + J[0]*o_e1e2 + J[1]*o_e2e3
    return H0

def get_NE_Hw(Bx, By, nq):
    """
    Returns Hamiltonian resulting from transverse magnetic field (Bx, By, 0) applied to system of
    nq==2 or nq==3 nuclear-electron pairs.
    """
    Ix = gate.get_Ix_sum(nq)
    Iy = gate.get_Iy_sum(nq)
    Sx = gate.get_Sx_sum(nq)
    Sy = gate.get_Sy_sum(nq)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy) - get_pulse_hamiltonian(Bx, By, gamma_n, 2*Ix, 2*Iy)
    return Hw



def nuclear_electron_sim(Bx,By,tN,nq,A=get_A(1,1)*Mhz, J=None, psi0=None):
    '''
    Simulation of nuclear and electron spins for single CNOT system.

    Order Hilbert space as |n1,...,e1,...>

    Inputs:
        (Bx,By): Tensors describing x and y components of control field at each timestep (in teslas).
        A: hyperfines
        J: Exchange coupling (0-dim or 2 element 1-dim)
    '''
    N = len(Bx)
    Bz = 2*tesla

    if psi0 is None:
        if nq==2:
            psi0 = gate.kron4(spin_up, spin_down, spin_up, spin_down)
        elif nq==3:
            psi0 = gate.kron(spin_up,spin_up,spin_down,spin_up,spin_up,spin_down)
    if J is None:
        if nq==2:
            J = get_J(1,2)[0]
        elif nq==3:
            J = get_J(1,3)[0]

    H0 = multi_NE_H0(Bz, A, J, nq)

    Hw = get_NE_Hw(Bx, By, nq)
    Ix = gate.get_Ix_sum(nq)
    Iy = gate.get_Iy_sum(nq)
    Sx = gate.get_Sx_sum(nq)
    Sy = gate.get_Sy_sum(nq)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy) - get_pulse_hamiltonian(Bx, By, gamma_n, 2*Ix, 2*Iy)

    # H0 has shape (d,d), Hw has shape (N,d,d). H0+Hw automatically adds H0 to all N Hw timestep values.
    H=sum_H0_Hw(H0, Hw)
    dt=tN/N
    U = pt.matrix_exp(-1j*H*dt)
    X=forward_prop(U)
    psi = pt.matmul(X,psi0)

    fig,ax=plt.subplots(1,1)
    plot_spin_states(psi,tN,ax, label_getter=get_NE_label)

    return


def run_NE_sim(tN,N,nq,Bz,A,J, psi0=None):
    tN_locked = lock_to_coupling(get_A(1,1),tN)
    Bx, By = NE_CX_pulse(tN_locked, N, A, Bz)
    nuclear_electron_sim(Bx,By,tN_locked,nq,A,J,psi0)


if __name__ == '__main__':

    run_NE_sim(
        tN = 10.0*nanosecond,
        N = 50000,
        nq = 2, 
        Bz = 2*tesla,
        A = get_A(1,1), 
        J = get_J(1,2)[0],
        psi0 = gate.spin_1001
        )


    plt.show()