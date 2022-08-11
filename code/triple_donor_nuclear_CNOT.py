

from email.policy import default
from numpy import flip
import torch as pt
import matplotlib



if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 




import gates as gate
from utils import get_ordered_eigensystem, psi_to_string, print_eigenstates, get_couplings, get_allowed_transitions, map_psi
from data import get_A, get_J, cplx_dtype, gamma_e, gamma_n, default_device
from atomic_units import *
from hamiltonians import get_H0, multi_NE_H0, get_pulse_hamiltonian
from electrons import analyse_3E_system, simulate_electrons, electron_wf_evolution, get_electron_X
from multi_NE import *
from visualisation import uparrow, downarrow, Uparrow, Downarrow, plot_psi, eigenstate_label_getter, spin_state_label_getter
from pulse_maker import pi_rot_square_pulse


comp00 = pt.kron(gate.spin_000, gate.spin_111)
comp01 = pt.kron(gate.spin_001, gate.spin_111)
comp10 = pt.kron(gate.spin_100, gate.spin_111) 
comp11 = pt.kron(gate.spin_101, gate.spin_111)

comp_basis = [7,15,39,47]

from pdb import set_trace


def get_transition(i0, i1, allowed_transitions):
    if (i0, i1) in allowed_transitions:
        print(f"Found transition ({i0}, {i1})")
        return (i0,i1)
    elif (i1,i0) in allowed_transitions:
        print(f"Found transition ({i1}, {i0})")
        return (i1,i0)
    else:
        raise Exception(f"Unable to find transitions for |{i0}> <--> |{i1}>")



def match_E_estate_to_NE_estate(evec_E, i_n, S_NE):

    # first need to project evec_3E into 6-spin space with 
    dim_E=8
    dim_NE=64
    evec_NE = pt.zeros(dim_NE, dtype=cplx_dtype, device=default_device)
    evec_NE[i_n*8:(i_n+1)*dim_E] = evec_E 

    for j in range(dim_NE):
        if (pt.abs(evec_NE - S_NE[:,j])<1e-3).to(int).all():
            return j


def get_E_transition(evec1_idx, evec0_idx, tN, N, A, NucSpin, J, Bz, S_NE):

    A_E = pt.tensor([(-n*2+1)*A for n in NucSpin], dtype=cplx_dtype)
    H0 = get_H0(A=A_E, J=J, Bz=Bz)
    S, D = get_ordered_eigensystem(H0)
    E = pt.diag(D)
    allowed_transitions = get_allowed_transitions(H0, S=S, E=E)
    print_eigenstates(S)
    couplings = get_couplings(S)
    transition = get_transition(evec0_idx, evec1_idx, allowed_transitions)
    omega = E[transition[0]] - E[transition[1]]
    coupling = couplings[evec1_idx,evec0_idx]
    print(f"|{downarrow}{downarrow}{downarrow}> <--> |E{evec1_idx}>: omega = {pt.real(omega)/Mhz} MHz, coupling = {pt.real(coupling)*tesla/Mhz} MHz/tesla")

    
    NucSpin_str = ''
    for n in NucSpin: NucSpin_str += str(n)
    nuc_dec_state = int(NucSpin_str,2)
    evec1_NE_idx = match_E_estate_to_NE_estate(S[:,evec1_idx],nuc_dec_state,S_NE)
    print(f"Electron eigenstate |E{evec1_idx}> = {psi_to_string(S[:,evec1_idx])} matches to nuclear-electron eigenstate |E{evec1_NE_idx}> = {psi_to_string(S_NE[:,evec1_NE_idx])}")

    Bx, By = pi_rot_square_pulse(omega, coupling, tN, N)

    return Bx, By, evec1_NE_idx


def get_triple_donor_psi(i):
    dim = 64
    psi = pt.zeros(dim, dtype=cplx_dtype, device=default_device)
    psi[i] = 1
    return psi


def get_composition_vectors(comp_idx, S, pmin = 0.001):
    '''
    Accepts integer comp_idx which regers to a computational basis vector, and returns eigenvectors from which it is composed,
    where the eigenvectors are the columns of S.
    '''

    psi_eig = S.T[:,comp_idx]

    composition = []
    for a in range(len(psi_eig)):
        if pt.abs(psi_eig[a])**2 > pmin:
            composition.append(a)
            
    return composition


def get_comp_basis_eigen_composition(comp_basis, S):
    '''
    Accepts list of indices which identify computational basis vectors, comp_basis_idxs, and returns indices of eigenvectors from 
    which these basis vectors are composed.
    '''
    eigs = []
    for idx in comp_basis:
        eigs += get_composition_vectors(idx, S)
    return list(set(eigs))


def get_triple_donor_CX_T(tN_e, tN_n, N_e, N_n):
    T1 = pt.linspace(0, tN_e, N_e)
    T2 = pt.linspace(tN_e, tN_e+tN_n, N_n)
    T3 = pt.linspace(tN_e+tN_n, 2*tN_e+tN_n, N_e)
    T = pt.cat((T1,T2,T3))
    return T

def triple_donor_CX_fields(tN_e, tN_n, N_e, N_n, A,J, Bz, eigen_basis):

    # specify electron eigenstates to transition to 
    flip_e_idx = 3 # if n1=|1>
    idle_e_idx = 1 # if n1=|0>
    e0_idx = 0 # |111> state

    #H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    # H0_e100
    # S_e100, D_e = get_ordered_eigensystem(H0_e)

    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)
    S,D = get_ordered_eigensystem(H0) 
    analyse_3NE_eigensystem(S,D)
    couplings = get_triple_NE_couplings(S)
    allowed_transitions = get_allowed_transitions(H0)
    E = pt.diag(D)
    H0 = S@D@S.T

    #print(f"NucSpin=[0,0,0]")
    #w000, c000 = get_E_transition_info(idle_e_idx, e0_idx, A, [0,0,0], J, Bz, S)
    eig00 = 0
    print(f"NucSpin=[0,0,1]")
    Bx_01, By_01, eig01 = get_E_transition(idle_e_idx, e0_idx, tN_e, N_e, A, [0,0,1], J, Bz, S)
    print(f"NucSpin=[1,0,0]")
    Bx_10, By_10, eig10 = get_E_transition(flip_e_idx, e0_idx, tN_e, N_e, A, [1,0,0], J, Bz, S)
    print(f"NucSpin=[1,0,1]")
    Bx_11, By_11, eig11 = get_E_transition(flip_e_idx, e0_idx, tN_e, N_e, A, [1,0,1], J, Bz, S)
    eigen_basis += [eig00, eig01, eig10, eig11]
    for j, transition in enumerate(allowed_transitions):
        if transition[0] in eigen_basis and transition[1] in eigen_basis:
            print(f"Transition |E{transition[0]}> <--> |E{transition[1]}> is allowed, w_res = {pt.real(E[transition[0]]-E[transition[1]])/Mhz} MHz")

    comp_basis = [7,15,39,47]
    transition_states = list(set(get_comp_basis_eigen_composition(comp_basis, S) + eigen_basis))
    print("Eigenstates involved in CNOT:")
    for i in transition_states:
        print(f"|E{i}> = {psi_to_string(S[:,i])}")


    Bx_n, By_n = get_NE_estate_transition(eig10, eig11, tN=tN_n, N=N_n, Bz=Bz, A=A, J=J)
    Bx_e = Bx_01 + Bx_10 + Bx_11 
    By_e = By_01 + By_10 + By_11 
    return Bx_e, By_e, Bx_n, By_n

def triple_donor_CX_psi_evol(tN_e = 500*nanosecond, tN_n=5000*nanosecond, N_e = 200, N_n=10, A=get_A(1,1),J=get_J(1,3), Bz=2*tesla, psi0=comp11, ax=None):

    eigen_basis = []
    Bx_e, By_e, Bx_n, By_n = triple_donor_CX_fields(tN_e, tN_n, N_e, N_n, A, J, Bz, eigen_basis)

    H0 = multi_NE_H0(Bz=Bz, A=A, J=J)
    S,D = get_ordered_eigensystem(H0) 
    transition_states = list(set(get_comp_basis_eigen_composition(comp_basis, S) + eigen_basis))

    psi1 = multi_NE_evol_low_mem(Bx_e, By_e, Bz=Bz, A=A, J=J, tN=tN_e, psi0=psi0)
    psi2 = multi_NE_evol_low_mem(Bx_n, By_n, Bz=Bz, A=A, J=J, tN=tN_n, psi0=psi1[-1])
    psi3 = multi_NE_evol_low_mem(Bx_e, By_e, Bz=Bz, A=A, J=J, tN=tN_e, psi0=psi2[-1])
    T = get_triple_donor_CX_T(tN_e, tN_n, N_e, N_n)

    psi = pt.cat((psi1, psi2, psi3))
    psi_eig = map_psi(S.T,psi)
    label_getter = lambda i: eigenstate_label_getter(i, transition_states)
    plot_psi(psi_eig,tN_n, T=T, ax=ax,label_getter=label_getter)



    
def triple_donor_CX_X(tN_e, tN_n, N_e, N_n, A,J, Bz, fp=None, reduced=False):

    Bx_e, By_e, Bx_n, By_n = triple_donor_CX_fields(tN_e, tN_n, N_e, N_n, A, J, Bz, [])

    if reduced:
        X1,XN = get_multi_NE_reduced_X_low_mem(Bx_e, By_e, Bz, A, J, tN_e, basis=comp_basis, X0=pt.eye(64, dtype=cplx_dtype, device=default_device))
        X2,XN = get_multi_NE_reduced_X_low_mem(Bx_n, By_n, Bz, A, J, tN_n, basis=comp_basis, X0=XN)
        X3,XN = get_multi_NE_reduced_X_low_mem(Bx_e, By_e, Bz, A, J, tN_e, basis=comp_basis, X0=XN)
    else:
        X1,XN = get_multi_NE_X_low_mem(Bx_e, By_e, Bz, A, J, tN_e, basis=comp_basis, X0=pt.eye(64, dtype=cplx_dtype, device=default_device))
        X2,XN = get_multi_NE_X_low_mem(Bx_n, By_n, Bz, A, J, tN_n, basis=comp_basis, X0=XN)
        X3,XN = get_multi_NE_X_low_mem(Bx_e, By_e, Bz, A, J, tN_e, basis=comp_basis, X0=pt.eye(64, dtype=cplx_dtype, device=default_device))

    X = pt.cat((X1,X2,X3), 0)
    T = get_triple_donor_CX_T(tN_e, tN_n, N_e, N_n)
    return X,T
    

def triple_donor_CX_on_comp_basis_states(tN_e, tN_n, N_e, N_n, A,J, Bz, fp=None):
    X,T = triple_donor_CX_X(tN_e, tN_n, N_e, N_n, A,J, Bz, fp=None, reduced=False)

    S,_D = get_ordered_eigensystem(H0=multi_NE_H0(Bz,A,J))

    fig,ax = plt.subplots(1)
    plot_psi(map_psi(S.T,X@comp10), T=T, ax=ax)

    set_trace()


def apply_phase():

    def remove_arb_phase(A):
        return A * pt.abs(A[0,0])/A[0,0]

    Uf = pt.tensor([
        [-0.26-0.96j,  -0.00-0.00j,  -0.00-0.00j,  0.00-0.00j ],
        [0.00+0.00j,  0.91-0.35j,  0.00-0.00j,  0.00-0.00j  ],
        [0.00-0.00j,  0.00+0.00j,  -0.05-0.00j,  0.29-0.94j  ],
        [0.00-0.00j,  -0.00+0.00j,  -0.90+0.41j,  0.02+0.05j  ]
    ], dtype=cplx_dtype, device=default_device)

    Uf = remove_arb_phase(Uf)

    U_phi = remove_arb_phase(gate.CX @ pt.inverse(Uf))

    # remove arbitrary phase 
    phi2 = pt.angle(U_phi[1,1])
    phi3 = pt.angle(U_phi[2,2])
    phi4 = pt.angle(U_phi[3,3])

    U_corr1 = remove_arb_phase(pt.matrix_exp(-1j*phi4/4 * gate.get_Zn(2)))

    CX_c1 = remove_arb_phase(U_corr1 @ Uf)

    set_trace()

def triple_donor_CX(tN_e, tN_n, N_e, N_n, A,J, Bz, fp=None, reduced=False):

    X,T = triple_donor_CX_X(tN_e, tN_n, N_e, N_n, A,J, Bz, fp=None, reduced=reduced)

    if reduced:
        psi0=gate.spin_10
    else:
        psi0 = comp10

    plot_psi(X@psi0, T=T)


    U_phi = gate.CX @ pt.inverse(Uf)
    # can we use GRAPE to find pulses which act U_phi on triple donor system?
    # Will then achieve CX = U_phi @ Uf!

    set_trace()

   



if __name__=='__main__':

    #triple_donor_CX_psi_evol()
    #triple_donor_CX_X(tN_e = 500*nanosecond, tN_n=5000*nanosecond, N_e = 200000, N_n=10000, A=get_A(1,1),J=get_J(1,3), Bz=2*tesla)
    #visualise_E_transitions_for_NucSpins()
    #triple_donor_CX_on_comp_basis_states(tN_e = 500*nanosecond, tN_n=5000*nanosecond, N_e = 200000, N_n=10000, A=get_A(1,1),J=get_J(1,3), Bz=2*tesla)
    #triple_donor_CX(tN_e = 500*nanosecond, tN_n=5000*nanosecond, N_e = 200000, N_n=10000, A=get_A(1,1),J=get_J(1,3), Bz=2*tesla, reduced=True)
    apply_phase()
    plt.show()




