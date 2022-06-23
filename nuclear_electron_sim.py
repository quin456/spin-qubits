

from pathlib import Path
import os
dir = os.path.dirname(__file__)
os.chdir(dir)


from GRAPE import *
import gates as gate
from atomic_units import *
from data import get_A, get_J
from visualisation import plot_spin_states
from pulse_maker import square_pulse


import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']




spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)




def get_n_e_label(j):
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
        ax.plot(T,pt.abs(psi[:,j]), label = get_n_e_label(j))
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

def nuclear_electron_sim(Bx,By,tN,nq,A,J, psi0=None):
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

    # unit conversions
    A,J=convert_Mhz(A,J)
    tN*=nanosecond
    Bx*=tesla; By*=tesla
    A_mag = np.abs(A[0])


    o_n1e1 = gate.o4_13 
    o_n2e2 = gate.o4_24 
    o_e1e2 = gate.o4_34


    nspin1,nspin2 = get_nuclear_spins(A)

    if psi0 is None:
        if nq==2:
            psi0 = gate.kron4(spin_up, spin_down, spin_up, spin_down)
        elif nq==3:
            psi0 = gate.kron(spin_up,spin_up,spin_down,spin_up,spin_up,spin_down)
    #H0 = 0.5*gamma_e*Bz*oze + A[0]*gate.IIZI + A[1]*gate.IIIZ + J*o_e1e2 


    ozn = gate.ZIIIII + gate.IZIIII + gate.IIZIII 
    oze = gate.IIIZII + gate.IIIIZI + gate.IIIIIZ

    o_n1e1 = gate.o6_14
    o_n2e2 = gate.o6_25
    o_n3e3 = gate.o6_36
    o_e1e2 = gate.o6_45 
    o_e2e3 = gate.o6_56 

    X_nuc = gate.get_nuclear_ox(nq)
    X_elec = gate.get_electron_ox(nq)
    Y_nuc = gate.get_nuclear_oy(nq)
    Y_elec = gate.get_electron_oy(nq)

    if psi0==None:
        H0 = 0.5*gamma_e*Bz*oze - 0.5*gamma_P*Bz*ozn + A_mag*o_n1e1 + A_mag*o_n2e2 + J*o_e1e2 
    elif nq==3:
        H0 = 0.5*gamma_e*Bz*oze - 0.5*gamma_P*Bz*ozn + A_mag*(o_n1e1+o_n2e2+o_n3e3) + J[0]*o_e1e2 + J[1]*o_e2e3


    else:
        raise Exception("Invalid nq")

    Hw_nuc = 0.5*gamma_P * (pt.einsum('j,ab->jab',Bx,X_nuc) + pt.einsum('j,ab->jab',By,Y_nuc))
    Hw_elec = 0.5*gamma_e * (pt.einsum('j,ab->jab',Bx,X_elec) + pt.einsum('j,ab->jab',By,Y_elec))
    Hw = Hw_elec+Hw_nuc
    # H0 has shape (d,d), Hw has shape (N,d,d). H0+Hw automatically adds H0 to all N Hw timestep values.
    H=Hw+H0
    dt=tN/N
    U = pt.matrix_exp(-1j*H*dt)
    X=forward_prop(U)
    psi = pt.matmul(X,psi0)

    fig,ax=plt.subplots(1,1)
    plot_spin_states(psi,tN,ax)

    return


def run_NE_sim(tN,N,nq,A,J, psi0=None):

    

    nuclear_electron_sim(Bx,By,tN,N,nq,A,J,psi0)


if __name__ == '__main__':
    tN=2.0
    N=50000
    nq=2
    A=get_A(1,2)[0]
    J=get_J(1,2)[0]
    run_NE_sim(2.0,5000,2, get_A(1,2)[0], get_J(1,2)[0])


    plt.show()