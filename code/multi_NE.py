
from calendar import c
import torch as pt 
import numpy as np
import matplotlib

from transition_visualisation import visualise_triple_donor_NE_transitions


matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from GRAPE import *
import gates as gate
from gates import spin_0010
from atomic_units import *
from data import get_A, get_J
from visualisation import plot_spin_states, plot_fields, plot_fidelity
from pulse_maker import square_pulse
from utils import lock_to_coupling, get_couplings_over_gamma_e, psi_to_string
from single_NE import NE_swap_pulse, NE_CX_pulse
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, multi_NE_H0, multi_NE_Hw, get_X, get_U0

from transition_visualisation import visualise_allowed_transitions
from pulse_maker import pi_rot_square_pulse

from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']




spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'





def plot_nuclear_electron_wf(psi,tN, ax=None, label_getter=multi_NE_label_getter):
    N,dim = psi.shape
    nq = get_nq_from_dim(dim)
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



def print_NE(psi, pmin=0.1):
    '''
    Prints triple donor nuclear-electron spin state as a1|000000> + ... + a63|111111>,
    ignoring all a's for which |a|^2 < pmin.
    '''
    print(psi_to_string(psi, pmin=pmin))





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


def get_nuclear_spin_ordered_eigensystem(Bz=2*tesla, A=get_A(1,1), J=get_J(1,3)):
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

    S,D = get_ordered_eigensystem(H0, H0_order)

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



def analyse_3NE_eigensystem(A=get_A(1,1), J=get_J(1,3), Bz=2*tesla):
    dim = 64

    H0 = multi_NE_H0(J=J, A=A)
    S,D = get_ordered_eigensystem(H0) 

    print("============================================================================================================================================")
    print("Triple donor NE eigenstates")
    for i in range(dim):
        print(f"|E{i}> = {psi_to_string(S[:,i])}")

    return S,D

def nuclear_spin_CNOT():
    pass


def graph_full_NE_H0_transitions(Bz=2*tesla, A=get_A(1,1), J=get_J(1,3)):

    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)
    rf = get_resonant_frequencies(H0)
    visualise_allowed_transitions(H0)


def map_3NE_transitions():



    H0 = multi_NE_H0(J=get_J(1,3))

    S,D = get_ordered_eigensystem(H0)  
    E=pt.diag(D)
    nucspin_indices = group_eigenstate_indices_by_nuclear_spins(S)
    idxs_100 = nucspin_indices[4]
    idxs_101 = nucspin_indices[5]

    trans = get_allowed_transitions(H0, S=S, E=E)



    analyse_3NE_eigensystem(S)
    C = get_triple_NE_couplings(S)

    print("Printing out 100 <--> 101 transitions")
    for i in idxs_100:
        for j in idxs_101:
            if (i,j) in trans:
                print(f"{i}<-->{j} == {psi_to_string(S[:,i])} <--> {psi_to_string(S[:,j])} transition frequency = {pt.real(E[i]-E[j])/Mhz} MHz, coupling = {pt.real(C[i,j])}")
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
    Hw_mag = gamma_e*gate.get_Sx_sum(3) - gamma_n*gate.get_Ix_sum(3)
    return get_couplings(S,Hw_mag)



def triple_NE_estate_transition(i0=53, i1_target=49, tN=4000*nanosecond, N=10000, A=get_A(1,1), J=get_J(1,3)):
    H0 = multi_NE_H0(J=J, A=A)
    S,D = get_ordered_eigensystem(H0) 
    couplings = get_triple_NE_couplings(S)
    E = pt.diag(D)

    allowed_transitions = get_allowed_transitions(H0)
    #i0, i1_target = allowed_transitions[9]


    # determine target unitary for transition
    target = pt.eye(64, dtype=cplx_dtype)
    target[i0,i0]=0; target[i0,i1_target]=1
    target[i1_target,i1_target]=0; target[i1_target,i0]=1


    print(f"Attempting to drive transitions {i0}<-->{i1_target} == {psi_to_string(S[:,i0])} <--> {psi_to_string(S[:,i1_target])}")



    if (i0,i1_target) in allowed_transitions:
        omega = E[i0] - E[i1_target]
        print(f"Found resonant frequency E{i0}-E{i1_target} = {omega/Mhz} MHz")
    elif (i1_target,i0) in allowed_transitions:
        omega = E[i1_target] - E[i0]
        print(f"Found resonant frequency E{i1_target}-E{i0} = {omega/Mhz} MHz")
    else:
        print("Transition not allowed.")
    

    rf_hack = [E[T[0]]-E[T[1]] for T in allowed_transitions]




    omega=omega

    coupling = couplings[i0,i1_target]
    print(f"coupling = {coupling}")

    Bx,By = pi_rot_square_pulse(omega, coupling, tN, N)
    Hw = multi_NE_Hw(Bx, By, 3)
    #Hw = pt.zeros_like(Hw)
    
    H = sum_H0_Hw(H0, Hw)
    X = get_X(H, tN, N)

    nucspin_indices = group_eigenstate_indices_by_nuclear_spins(S)
    idxs_100 = nucspin_indices[4]
    idxs_101 = nucspin_indices[5]

    i_psi = 51
    print(f"psi0 = evec {i_psi}")

    psi0 = S[:,i_psi]



    print(f"psi0 = {psi_to_string(psi0)}")

    psi = X@psi0 
    print(f"psi_f = {psi_to_string(psi[-1])}")
    print(f"psi_f_target = {psi_to_string(S[:,i1_target])}")

    fig,ax = plt.subplots(2)

    plot_fields(Bx, By, tN, ax[0])
    psi_projected = get_psi_projections(psi, pt.stack((S[:,i0], S[:,i1_target])))
    def label_getter(i):
        if i==i0: return f"psi0 = |{i}>"
        elif i==i1_target: return f"psi_target = |{i}>"
        #else: return f"|{i}>"
    plot_spin_states(pt.einsum('ab,jb->ja', S.T, psi), tN, ax[1], label_getter=label_getter)

    fids = fidelity_progress(X,target)
    target_narrowed = gate.X 
    X_narrowed = pt.zeros(len(X),2,2, dtype=cplx_dtype)
    for j in range(len(X)):
        X_narrowed[j,0,0] = X[j,i0,i0]
        X_narrowed[j,0,1] = X[j,i0,i1_target]
        X_narrowed[j,1,0] = X[j,i1_target,i0]
        X_narrowed[j,1,1] = X[j,i1_target,i1_target]
    #fids_narrowed = fidelity_progress(X_narrowed,target_narrowed)
    #plot_fidelity(ax[2], fids_narrowed, tN)

    Uf = X[-1]


def triple_NE_free_evolution(tN=50*nanosecond, N=500, A=get_A(1,1), J=get_J(1,3)):
    H0 = multi_NE_H0(J=J, A=A)
    S,D = get_ordered_eigensystem(H0) 

    psi0 = S[:,53]

    U0 = get_U0(H0, tN, N)
    #psi = U0@psi0 

    Hw = pt.zeros(N,64,64)

    H = sum_H0_Hw(H0, Hw)
    X = get_X(H, tN, N)
    psi=X@psi0
    plot_spin_states(psi, tN)



class MultiNuclearElectronGrape(Grape):

    def __init__(self, tN, N, Bz=2*tesla, A=get_A(1,1), J=get_J(1,2), target=gate.CX_3NE, rf=None, u0=None, max_time=9999999):
        self.Bz=Bz 
        self.A=A
        self.J=J
        super().__init__(tN, N, target, rf, max_time=max_time)


    def get_H0(self):
        return multi_NE_H0(Bz=self.Bz, )

    def get_Hw(self):
        return get_pulse_hamiltonian()




if __name__ == '__main__':



    # print_rank2_tensor(S)
    # #print_rank2_tensor(D)

    # rf = get_resonant_frequencies(H0)


    #analyse_3NE_eigensystem()
    #map_3NE_transitions()
    
    
    triple_NE_estate_transition()
    #triple_NE_free_evolution()
    
    
    plt.show()