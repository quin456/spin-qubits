


import numpy as np
import torch as pt 
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from hamiltonians import get_X_from_H, get_U0, get_1S_HA, get_1S_HJ, multi_NE_H0, get_H0
from data import *
from utils import dagger, get_resonant_frequencies, fidelity, wf_fidelity, get_rec_min_N
import gates as gate
from gates import spin_101, spin_10

from pdb import set_trace


def get_t_fidelity(J,A, tN, N, fid_min):
    nq = len(A)

    HA = get_1S_HA(A)
    HJ = get_1S_HJ(J)

    UA = get_U0(HA, tN, N)
    H = dagger(UA)@HJ@UA 

    X = get_X_from_H(H,tN,N)
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

    X = get_X_from_H(H,tN,N)
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

def plot_load_time_vs_J(fid_min=0.999, Jmin = 5*unit.MHz, Jmax=50*unit.MHz, A=get_A(1,3), tN_max=10*unit.ns, n=100, N=10000, max_time=10*unit.ns, ax=None, fp=None, get_t=get_t_fidelity):

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
        print(f"No loss of fidelity for J<{optimal_J/unit.MHz} MHz")
    if ax is None: ax = plt.subplot()

    set_trace()

    ax.plot(J[i_min:]/unit.MHz, T[i_min:]/unit.ns)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    # ax.axhline(max_time/unit.ns, linestyle = '--', color = 'red')
    # ax.annotate(f'{max_time/unit.ns} ns', (30,max_time/unit.ns+0.3))
    # ax.axvline(J[i,0]/unit.MHz, linestyle = '--', color='red')
    # ax.annotate(f'{J[i,0]/unit.MHz:.0f} unit.MHz', (J[i]/unit.MHz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)
    

def approximate_full_NE_optimisation_time():
    '''
    3 nuclear, 3 electron spin system optimisation time approximation based on 40 second 99.5% 
    fidelity 3 electron CNOT optimisation time.
    '''
    tN_3E = 100 * unit.ns
    tN_3NE = 1000 * unit.ns
 
    t_3E = 40 # time to optimise 3 electron system
    
    dim_3E = 2**3 
    dim_3NE = 2**6

    H0_3E = get_H0(get_A(1,2), get_J(1,2))
    rf_3E = get_resonant_frequencies(H0_3E)
    N_3E = get_rec_min_N(rf_3E, tN_3E)

    H0_3NE = multi_NE_H0(Bz=2*unit.T)
    rf_3NE = get_resonant_frequencies(H0_3NE)
    N_3NE = get_rec_min_N(rf_3NE, tN_3NE)
    #N_3NE=N_3E

    n_fields_3E = 15
    n_fields_3NE = 637

    print(f"w_max_3E = {pt.max(pt.real(rf_3E))/unit.MHz} MHz")
    print(f"w_max_3NE = {pt.max(pt.real(rf_3NE))/unit.MHz} MHz")
    print(f"N_3E = {N_3E}")
    print(f"N_3NE = {N_3NE}")


    t_3NE = ((dim_3NE/dim_3E)**2) * (n_fields_3NE/n_fields_3E) * (N_3NE/N_3E) * (tN_3NE/tN_3E) * t_3E

    print(f"t_3NE = {t_3NE} s")

    


def plot_load_time_vs_J_2q(fidelity_min = 0.999, N=2000, J=get_J(1,2), A=get_A(1,1), max_time=10*unit.ns, ax=None, fp=None):
    # MISTAKEY

    def get_t(J):
        return np.arccos(np.sqrt(fidelity_min)) / ( (np.sqrt(4*A**2 + 4*J**2) - 2*A) )

    J = np.linspace(2*unit.MHz, 50*unit.MHz, N)
    T = get_t(J)



    if ax is None: ax = plt.subplot()
    ax.plot(J/unit.MHz, T/unit.ns)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    ax.axhline(max_time/unit.ns, linestyle = '--', color = 'red')
    ax.annotate(f'{max_time/unit.ns} ns', (30,max_time/unit.ns+0.3))
    ax.axvline(J[i]/unit.MHz, linestyle = '--', color='red')
    ax.annotate(f'{J[i]/unit.MHz:.0f} MHz', (J[i]/unit.MHz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)



if __name__ == '__main__':
    #plot_load_time_vs_J(fid_min=0.99, Jmin=1*unit.MHz, Jmax=50*unit.MHz, tN_max=100*unit.ns, A=get_A(1,3), n=100, get_t=get_t_wf)

    #plot_load_time_vs_J_2q(fidelity_min=0.99)

    #print(f"{get_t_wf(get_J(1,3), get_A(1,3), 10*unit.ns,1000,0.99)/unit.ns} ns")

    approximate_full_NE_optimisation_time()

    plt.show()