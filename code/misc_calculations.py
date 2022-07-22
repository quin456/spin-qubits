


import numpy as np 
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from hamiltonians import get_X, get_U0, get_1S_HA, get_1S_HJ
from data import *
from utils import dagger, get_resonant_frequencies, fidelity, wf_fidelity
import gates as gate
from gates import spin_101, spin_10

from pdb import set_trace


def get_t_fidelity(J,A, tN, N, fid_min):
    nq = len(A)

    HA = get_1S_HA(A)
    HJ = get_1S_HJ(J)

    UA = get_U0(HA, tN, N)
    H = dagger(UA)@HJ@UA 

    X = get_X(H,tN,N)
    Id = gate.II if nq==2 else gate.III
    fid = pt.tensor([fidelity(X[j], Id) for j in range(N)])

    j = 0
    while fid[j] > fid_min:
        j+=1
    return j*tN/N
   
def get_t_wf(J,A, tN, N, fid_min, psi0=spin_101):
    nq = len(A)

    HA = get_1S_HA(A)
    HJ = get_1S_HJ(J)

    UA = get_U0(HA, tN, N)
    H = dagger(UA)@HJ@UA 

    X = get_X(H,tN,N)
    psi = X@psi0
    fid = pt.abs(psi[:,5])**2
    #fid = pt.tensor([wf_fidelity(psi[j], psi0) for j in range(N)])

    j = 0
    try:
        while fid[j] > fid_min:
            j+=1
    except:
        return -1
    return j*tN/N

def plot_load_time_vs_J(fid_min=0.999, Jmin = 5*Mhz, Jmax=50*Mhz, A=get_A(1,3), tN_max=10*nanosecond, n=100, N=10000, max_time=10*nanosecond, ax=None, fp=None, get_t=get_t_fidelity):

    optimal_J=None
    nq=len(A)
    T = pt.zeros(n)
    J = pt.linspace(Jmin, Jmax, n)
    if nq==3:
        J = pt.einsum('i,j->ij',J, pt.ones(2))
    i_min=0
    for i in range(n):
        time = get_t(J[i],A,tN_max,N, fid_min)
        if time==-1:
            i_min = i+1
            optimal_J = J[i]
        T[i] = time

    if optimal_J is not None:
        print(f"No loss of fidelity for J<{optimal_J/Mhz} MHz")
    if ax is None: ax = plt.subplot()

    set_trace()

    ax.plot(J[i_min:]/Mhz, T[i_min:]/nanosecond)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    # ax.axhline(max_time/nanosecond, linestyle = '--', color = 'red')
    # ax.annotate(f'{max_time/nanosecond} ns', (30,max_time/nanosecond+0.3))
    # ax.axvline(J[i,0]/Mhz, linestyle = '--', color='red')
    # ax.annotate(f'{J[i,0]/Mhz:.0f} MHz', (J[i]/Mhz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)
    




    




def plot_load_time_vs_J_2q(fidelity_min = 0.999, N=2000, J=get_J(1,2), A=get_A(1,1), max_time=10*nanosecond, ax=None, fp=None):
    # MISTAKEY

    def get_t(J):
        return np.arccos(np.sqrt(fidelity_min)) / ( (np.sqrt(4*A**2 + 4*J**2) - 2*A) )

    J = np.linspace(2*Mhz, 50*Mhz, N)
    T = get_t(J)



    if ax is None: ax = plt.subplot()
    ax.plot(J/Mhz, T/nanosecond)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    ax.axhline(max_time/nanosecond, linestyle = '--', color = 'red')
    ax.annotate(f'{max_time/nanosecond} ns', (30,max_time/nanosecond+0.3))
    ax.axvline(J[i]/Mhz, linestyle = '--', color='red')
    ax.annotate(f'{J[i]/Mhz:.0f} MHz', (J[i]/Mhz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)



if __name__ == '__main__':
    plot_load_time_vs_J(fid_min=0.99, Jmin=1*Mhz, Jmax=50*Mhz, tN_max=100*nanosecond, A=get_A(1,3), n=100, get_t=get_t_wf)

    #plot_load_time_vs_J_2q(fidelity_min=0.99)

    #print(f"{get_t_wf(get_J(1,3), get_A(1,3), 10*nanosecond,1000,0.99)/nanosecond} ns")


    plt.show()