

import torch as pt 
import numpy as np
import matplotlib



if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

import gates as gate
import atomic_units as unit
from utils import *
from eigentools import *
from data import get_A, get_J
from visualisation import plot_psi, plot_fields, plot_fidelity, multi_NE_label_getter
from pulse_maker import square_pulse
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, multi_NE_H0, multi_NE_Hw, get_X_from_H, get_U0, get_NE_H0, get_NE_Hw
from GRAPE import Grape
from pulse_maker import pi_pulse_square

from pdb import set_trace


color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']




spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'





def plot_nuclear_electron_wf(psi,tN, ax=None, label_getter=multi_NE_label_getter):
    N,dim = psi.shape
    nq = get_nq_from_dim(dim)
    T = pt.linspace(0,tN/unit.ns,N)
    
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



def print_NE(psi, pmin=0.1):
    '''
    Prints triple donor nuclear-electron spin state as a1|000000> + ... + a63|111111>,
    ignoring all a's for which |a|^2 < pmin.
    '''
    print(psi_to_string(psi, pmin=pmin))


def double_NE_swap_with_exchange(Bz=2*unit.T, A=get_A(1,1), J=get_J(1,2), tN=500*unit.ns, N=1000000, ax=None, deactivate_exchange=False, label_states=None):
    print("Simulating attempted nuclear-electron spin swap for 2 electron system")
    print(f"J = {J/unit.unit.MHz} MHz")
    print(f"Bz = {Bz/unit.T} T")
    print(f"A = {A/unit.MHz} MHz")
    if ax is None: ax = plt.subplot()
    Bx,By = NE_swap_pulse(tN, N, A, Bz)
    def label_getter(j):
        return multi_NE_label_getter(j, label_states=label_states)
    multi_NE_evol(Bx, By, tN, 2, A=A, J=J, psi0=gate.spin_0010, deactivate_exchange=deactivate_exchange,ax=ax, label_getter=label_getter)


def get_nuclear_spin_ordered_eigensystem(Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3)):
    '''
    Returns triple donor nuclear-electron spin eigensystem with states ordered by nuclear spin computational basis. 
    '''
    def get_3NE_H0_for_ordering(Bz, A, J):
        '''
        Returns H0 with gamma_n made big and negative for control and target nuclear spins, so that eigenstates
        ordered by energy will go in order of nuclear computational basis states.
        '''
        Iz = gate.get_Iz_sum(3)
        Sz = gate.get_Sz_sum(3)
        o_n1e1 = gate.o6_14
        o_n2e2 = gate.o6_25
        o_n3e3 = gate.o6_36
        o_e1e2 = gate.o6_45 
        o_e2e3 = gate.o6_56 
        gamma_n_big = -10*gamma_e
        gamma_n3 = 10*gamma_n_big
        gamma_n1 = 1*gamma_n_big
        H0 = gamma_e*Bz*Sz - 0.5*gamma_n*Bz*gate.IZIIII - 0.5*gamma_n1*Bz*gate.ZIIIII - 0.5*gamma_n3*gate.IIZIII + A*(o_n1e1+o_n2e2+o_n3e3) + J[0]*o_e1e2 + J[1]*o_e2e3
        return H0


    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)
    H0_order = get_3NE_H0_for_ordering(Bz, A, J)

    S,D = get_ordered_eigensystem(H0, H0_order, ascending=True)

    return S,D


def nuclear_spin_projection(psi):
    '''
    Determines nuclear spin state probability distribution from nuclear-electron wave function.
    Essentially projects probabilities from 64 dimension 3 nuclear 3 electron spin wave function
    onto 8 dimensional 3 nuclear spin probabiltiy dist.
    '''
    psi_sq_nuc = pt.zeros(8, dtype=cplx_dtype)
    for i in range(8):
        # iterate over nuclear spin register state
        for j in range(8):
            # iterate over electron spin register states
            psi_sq_nuc[i] += pt.abs(psi[8*i+j])**2

    return psi_sq_nuc


def group_eigenstate_indices_by_nuclear_spins(S):

    pmin = 0.99

    nuc_spin_state_indices = [[] for _ in range(8)]

    for j in range(64):
        evec = S[:,j]
        evec_nuc_proj_sq = nuclear_spin_projection(evec)
        for i in range(8):
            if pt.abs(evec_nuc_proj_sq[i])**2 > pmin:
                nuc_spin_state_indices[i].append(j)
    return nuc_spin_state_indices



def analyse_3NE_eigensystem(S,D):
    dim = 64


    print("============================================================================================================================================")
    print("Triple donor NE eigenstates")
    for i in range(dim):
        print(f"|E{i}> = {psi_to_string(S[:,i])}")





def nuclear_spin_CNOT():
    pass




def map_3NE_transitions():



    H0 = multi_NE_H0(J=J_1s3q)

    S,D = get_ordered_eigensystem(H0, ascending = True)  
    E=pt.diag(D)
    nucspin_indices = group_eigenstate_indices_by_nuclear_spins(S)
    idxs_100 = nucspin_indices[4]
    idxs_101 = nucspin_indices[5]

    trans = get_allowed_transitions(S=S, D=D)



    analyse_3NE_eigensystem(S, D)
    C = get_triple_NE_couplings(S)

    print("Printing out 100 <--> 101 transitions")
    for i in idxs_100:
        for j in idxs_101:
            if (i,j) in trans:
                print(f"{i}<-->{j} == {psi_to_string(S[:,i])} <--> {psi_to_string(S[:,j])} transition frequency = {pt.real(E[i]-E[j])/unit.MHz} MHz, coupling = {pt.real(C[i,j] * unit.T/unit.MHz)} MHz/T")
            else:
                print(f"{i}<-->{j} transition not allowed")
    #visualise_triple_donor_NE_transitions(H0, nucspin_indices)


def get_psi_projections(psi, states):

    nstates = len(states)
    N,dim = psi.shape
    psi_projected = pt.zeros(N, nstates)
    for j in range(N):
        for m in range(nstates):
            psi_projected[j,m] = pt.dot(psi[j],pt.conj(states[m]))
    return psi_projected



def get_triple_NE_couplings(S):
    Hw_mag = 2*gamma_e*gate.get_Sx_sum(3) - 2*gamma_n*gate.get_Ix_sum(3)
    return get_couplings(S,Hw_mag)


def all_triple_NE_basis_transitions(tN=4000*unit.ns, N=10000, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3)):

    fig,ax = plt.subplots(2,2)
    basis = [0,12,25,28]

    H0 = multi_NE_H0(Bz, A, J)
    

    allowed_transitions = get_allowed_transitions(H0)

    for j, transition in enumerate(allowed_transitions):
        if transition[0] in basis and transition[1] in [i0, i1_target]:
            omega = pt.real(E[transition[0]]-E[transition[1]])
            print(f"Transition |E{transition[0]}> <--> |E{transition[1]}> is allowed, w_res = {omega/unit.MHz} MHz")
    for i in basis:
        triple_NE_estate_transition(i, tN, N, Bz, A, J, ax[i//2,i%2])


def get_NE_estate_transition(S, D, i_psi0=25, i_psi1=28, tN=4000*unit.ns, N=10000, ax=None):
    couplings = get_triple_NE_couplings(S)
    E = pt.diag(D)

    allowed_transitions = get_allowed_transitions(S=S, D=D)
    if (i_psi0,i_psi1) in allowed_transitions:
        omega = E[i_psi0] - E[i_psi1]
    elif (i_psi1,i_psi0) in allowed_transitions:
        omega = E[i_psi1] - E[i_psi0]
    else:
        raise Exception(f"Transition |{i_psi0}> <--> |{i_psi1}> not allowed.")
        

    coupling = couplings[i_psi0,i_psi1]
    print(f"Target nuclear spin flip: w_res = {pt.real(omega)/unit.MHz} MHz, coupling = {pt.real(coupling)*unit.T/unit.MHz} MHz/tesla")
    Bx,By = pi_pulse_square(omega, coupling, tN, N)
    return Bx,By


def triple_NE_estate_transition(i_psi0=27, i_psi1=28, tN=14000*unit.ns, N=4000, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3,J2=J_1s3q/1.3), ax=None):
    H0 = multi_NE_H0(J=J, A=A, Bz=Bz)
    S,D = get_ordered_eigensystem(H0, ascending=True) 


    nucspin_indices = group_eigenstate_indices_by_nuclear_spins(S)
    idxs_100 = nucspin_indices[4]
    idxs_101 = nucspin_indices[5]


    Bx, By = get_NE_estate_transition(S, D, i_psi0=i_psi0, i_psi1=i_psi1, tN=tN, N=N, ax=ax)
    #Bx = None; By = None
    # Hw = multi_NE_Hw(Bx, By, 3)
    # H = sum_H0_Hw(H0, Hw)
    # X = get_X_from_H(H, tN, N)

    X = get_multi_NE_X(tN, N, Bz=Bz, A=A, J=J, Bx=Bx, By=By)

    print(f"psi0 = |{i_psi0}>")
    
    basis = [0,26,27,28]
    def label_getter(i):
        if i in basis: return f"|E{i}>"
    
    psi0 = S[:,i_psi0]
    print(f"psi0 = {psi_to_string(psi0)}")
    psi = X@psi0 
    print(f"psi_f = {psi_to_string(psi[-1])}")
    print(f"psi_f_target = {psi_to_string(S[:,i_psi1])}")
    plot_psi(pt.einsum('ab,jb->ja', S.T, psi), tN, ax, label_getter=label_getter)







def triple_NE_free_evolution(tN=15000*unit.ns, N=5000, A=get_A(1,1), J=get_J(1,3)):
    H0 = multi_NE_H0(J=J, A=A)
    S,D = get_ordered_eigensystem(H0, ascending=True) 

    psi0 = pt.zeros_like(S[:,0])
    psi0[15]=1

    set_trace()

    U0 = get_U0(H0, tN=tN, N=N)
    #psi = U0@psi0 

    Hw = pt.zeros(N,64,64)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)
    psi=X@psi0
    plot_psi(psi, tN)

def multi_NE_evol(Bx, By, Bz=2*unit.T, A=get_A(1,1), J=get_A(1,3), tN=1000*unit.ns, psi0=pt.kron(gate.spin_100,gate.spin_111)):
    N = len(Bx)
    Hw = multi_NE_Hw(Bx, By, 3)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    psi = X@psi0 
    return psi

def get_multi_NE_X(tN, N, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3), Bx=None, By=None):

    # Bx and By set to None is free evolution
    if Bx is None:
        Bx = pt.zeros(N)
    if By is None:
        By = pt.zeros(N)


    Hw = multi_NE_Hw(Bx, By, 3)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    return X

def get_multi_NE_X_low_mem(Bx, By, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3), tN=1000*unit.ns, basis=[7, 15, 39, 47], X0=None):
    N = len(Bx)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    nq=3
    dim=2**(2*nq)
    Sx = gate.get_Sx_sum(nq)
    Ix = gate.get_Ix_sum(nq)
    Sy = gate.get_Sy_sum(nq)
    Iy = gate.get_Iy_sum(nq)
    if X0==None:
        Xj = pt.eye(dim, dtype=cplx_dtype, device=default_device)
        return_XN = False
    else:
        Xj = X0
        return_XN = True
    X = pt.zeros(N,dim,dim, dtype=cplx_dtype, device=default_device)
    X[0] = pt.eye(dim, dtype=cplx_dtype, device=default_device)
    for j in range(1,N):
        Hj = H0 + gamma_e * (Bx[j]*Sx + By[j]*Sy) - gamma_n * (Bx[j]*Ix + By[j]*Iy)
        Uj = pt.matrix_exp(-1j*Hj*tN/N)
        X[j] = Uj@X[j-1]
    if return_XN:
        return X, Xj
    return X


def get_multi_NE_reduced_X_low_mem(Bx, By, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3), tN=1000*unit.ns, basis=[7, 15, 39, 47], X0=None):
    N = len(Bx)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    nq=3
    dim=2**(2*nq)
    Sx = gate.get_Sx_sum(nq)
    Ix = gate.get_Ix_sum(nq)
    Sy = gate.get_Sy_sum(nq)
    Iy = gate.get_Iy_sum(nq)
    if X0==None:
        Xj = pt.eye(dim, dtype=cplx_dtype, device=default_device)
        return_XN = False
    else:
        Xj = X0
        return_XN = True
    dim_reduced = len(basis)
    X_reduced = pt.zeros(N,dim_reduced,dim_reduced, dtype=cplx_dtype, device=default_device)
    for j in range(1,N):
        Hj = H0 + gamma_e * (Bx[j]*Sx + By[j]*Sy) - gamma_n * (Bx[j]*Ix + By[j]*Iy)
        Uj = pt.matrix_exp(-1j*Hj*tN/N)
        Xj = Uj@Xj 
        for a in range(dim_reduced):
            for b in range(dim_reduced):
                X_reduced[j,a,b] = Xj[basis[a],basis[b]]
    if return_XN:
        return X_reduced, Xj
    return X_reduced

def get_X_reduced_from_X(X, basis=[7, 15, 39, 47]):
    N = len(X)
    dim_reduced = len(basis)
    X_reduced = pt.zeros(N, dim_reduced, dim_reduced)
    for j in range(N):
        for a in range(dim_reduced):
            for b in range(dim_reduced):
                X_reduced[j,a,b] = X[basis[a],basis[b]]
    return X_reduced

def multi_NE_evol_low_mem(Bx, By, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,3), tN=1000*unit.ns, psi0=pt.kron(gate.spin_100,gate.spin_111)):
    '''
    Evolves wave function of nuclear + electron spin system. Same input and output as to multi_NE_evol, but 
    uses less memory at the cost of being slower.
    '''


    N = len(Bx)
    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)
    dim = H0.shape[-1]

    Sx = gate.get_Sx_sum(3)
    Ix = gate.get_Ix_sum(3)
    Sy = gate.get_Sy_sum(3)
    Iy = gate.get_Iy_sum(3)
    Xj = pt.eye(dim, dtype=cplx_dtype, device=default_device)
    psi = pt.zeros(N,64, dtype=cplx_dtype, device=default_device)
    for j in range(N):
        Hj = H0 + gamma_e * (Bx[j]*Sx + By[j]*Sy) - gamma_n * (Bx[j]*Ix + By[j]*Iy)
        Uj = pt.matrix_exp(-1j*Hj*tN/N)
        Xj = Uj@Xj 
        psi[j] = Xj@psi0
    return psi


class MultiNuclearElectronGrape(Grape):

    def __init__(self, tN, N, Bz=2*unit.T, A=get_A(1,1), J=get_J(1,2), target=gate.CX_3NE, rf=None, u0=None, max_time=9999999):
        self.Bz=Bz 
        self.A=A
        self.J=J
        super().__init__(tN, N, target, rf, max_time=max_time)


    def get_H0(self):
        return multi_NE_H0(Bz=self.Bz, )

    def get_Hw(self):
        return multi_NE_Hw()



def NE_swap_with_exchange(J=0):
    A = get_A(1,1)
    B0 = 2*unit.T 

    Bx_By_T = pt.load("fields/swap_fields")
    Bx = Bx_By_T[:,0]
    By = Bx_By_T[:,1]
    T = Bx_By_T[:,2]

    H0 = multi_NE_H0(B0, A, J)
    H_ac = multi_NE_Hw(Bx, By, 2)

    H = sum_H0_Hw(H0, H_ac)
    X = get_X_from_H(H, T=T)
    

    swaps = gate.NE_swap_2q
    Phi = fidelity(swaps, X[-1])

    return Phi


def NE_swap_fidelity_vs_J(n=4):
    Jmin = 0*unit.kHz 
    Jmax = 1*unit.kHz
    J = pt.linspace(Jmin, Jmax, n)
    Phi = pt.zeros(n)
    for i in range(n):
        Phi[i] = NE_swap_with_exchange(J[i])

    ax=plt.subplot()
    ax.plot(J/unit.kHz, Phi)
    ax.set_xlabel('J (kHz)')

if __name__ == '__main__':



    # print_rank2_tensor(S)
    # #print_rank2_tensor(D)




    #map_3NE_transitions()
    
    
    triple_NE_estate_transition()
    #triple_NE_free_evolution()
    
    #NE_swap_fidelity_vs_J()
    
    plt.show()