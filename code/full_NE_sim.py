
import torch as pt 
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from GRAPE import *
import gates as gate
from gates import spin_0010
from atomic_units import *
from data import get_A, get_J
from visualisation import plot_spin_states
from pulse_maker import square_pulse
from utils import lock_to_coupling
from NE_swap import NE_swap_pulse, NE_CX_pulse
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, multi_NE_H0, multi_NE_Hw

from transition_visualisation import visualise_allowed_transitions


from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']




spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)






def plot_nuclear_electron_wf(psi,tN, ax=None, label_getter=multi_NE_label_getter):
    N,dim = psi.shape
    nq = get_nq(dim)
    T = pt.linspace(0,tN/nanosecond,N)
    
    if ax is None: ax=plt.subplot()
    for j in range(dim):
        ax.plot(T,pt.abs(psi[:,j]), label = label_getter(j))
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






def nuclear_electron_sim(Bx,By,tN,nq,A=get_A(1,1)*Mhz, J=None, psi0=None, deactivate_exchange=False, ax=None, label_getter = multi_NE_label_getter):
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

    H0 = multi_NE_H0(Bz, A, J, nq, deactivate_exchange=deactivate_exchange)

    Hw = multi_NE_Hw(Bx, By, nq)
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

    if ax is None: ax=plt.subplot()
    plot_spin_states(psi,tN,ax, label_getter=label_getter)

    return


def run_NE_sim(tN,N,nq,Bz,A,J, psi0=None):
    tN_locked = lock_to_coupling(get_A(1,1),tN)
    Bx, By = NE_CX_pulse(tN_locked, N, A, Bz)
    nuclear_electron_sim(Bx,By,tN_locked,nq,A,J,psi0)


def double_NE_swap_with_exchange(Bz=2*tesla, A=get_A(1,1), J=get_J(1,2), tN=500*nanosecond, N=1000000, ax=None, deactivate_exchange=False, label_states=None):
    print("Simulating attempted nuclear-electron spin swap for 2 electron system")
    print(f"J = {J/Mhz} MHz")
    print(f"Bz = {Bz/tesla} T")
    print(f"A = {A/Mhz} MHz")
    if ax is None: ax = plt.subplot()
    Bx,By = NE_swap_pulse(tN, N, A, Bz)
    def label_getter(j):
        return multi_NE_label_getter(j, label_states=label_states)
    nuclear_electron_sim(Bx, By, tN, 2, A=A, J=J, psi0=spin_0010, deactivate_exchange=deactivate_exchange,ax=ax, label_getter=label_getter)


def nuclear_spin_CNOT():
    pass


def graph_full_NE_H0_transitions():

    H0 = multi_NE_H0(Bz=2*tesla)
    rf = get_resonant_frequencies(H0)
    visualise_allowed_transitions(H0)



if __name__ == '__main__':



    #double_NE_swap_with_exchange(N=500000)

    graph_full_NE_H0_transitions()

    plt.show()