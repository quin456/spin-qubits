

from numpy import flip
import torch as pt
import matplotlib


matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 




import gates as gate
from utils import get_ordered_eigensystem, psi_to_string, print_eigenstates, get_couplings, get_allowed_transitions
from data import get_A, get_J, cplx_dtype, gamma_e, gamma_n
from atomic_units import *
from hamiltonians import get_H0, multi_NE_H0, get_pulse_hamiltonian
from electrons import analyse_3E_system, simulate_electrons
from multi_NE import analyse_3NE_eigensystem, get_triple_NE_couplings
from visualisation import uparrow, downarrow, Uparrow, Downarrow
from transition_visualisation import visualise_E_transitions
from pulse_maker import pi_rot_square_pulse

from pdb import set_trace


def visualise_E_transitions_for_NucSpins(J=get_J(1,3)):

    #fig,ax = plt.subplots(2,2)
    ax=plt.subplot()
    visualise_E_transitions(J=get_J(1,3),A=get_A(1,3,NucSpin=[0,0,0]), Bz=2*tesla, ax=ax)


    #visualise_E_transitions(J=get_J(1,3),A=get_A(1,3,NucSpin=[0,0,1]), Bz=2*tesla, ax=ax[0,1])
    #visualise_E_transitions(J=get_J(1,3),A=get_A(1,3,NucSpin=[1,0,0]), Bz=2*tesla, ax=ax[1,0])
    #visualise_E_transitions(J=get_J(1,3),A=get_A(1,3,NucSpin=[1,0,1]), Bz=2*tesla, ax=ax[1,1])


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
    evec_NE = pt.zeros(dim_NE, dtype=cplx_dtype)
    evec_NE[i_n*8:(i_n+1)*dim_E] = evec_E 
    for j in range(dim_NE):
        if (pt.abs(evec_NE - S_NE[:,j])<1e-3).to(int).all():
            return j


def get_E_transition_info(evec1_idx, evec0_idx, A, NucSpin, J, Bz, S_NE):

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
    print(f"Electron eigenstate |E{evec1_idx}> = {psi_to_string(S[:,evec1_idx])} matches to nuclear-electron eigenstate |E{evec1_NE_idx}> = {psi_to_string(S_NE[:,evec1_idx])}")

    return omega, coupling 


def analyse_transitions(tN_E = 100*nanosecond, N_E = 50000, A=get_A(1,1),J=get_J(1,3), Bz=2*tesla):

    # specify electron eigenstates to transition to 
    flip_e_idx = 5 # if n1=|1>
    idle_e_idx = 4 # if n1=|0>
    e0_idx = 7 # |111> state

    #H0 = multi_NE_H0(Bz=Bz, A=A, J=J)

    S,D = analyse_3NE_eigensystem()

    # print(f"NucSpin=[0,0,0]")
    # w000, c000 = get_E_transition_info(idle_e_idx, e0_idx, A, [0,0,0], J, Bz, S)
    # print(f"NucSpin=[0,0,1]")
    # w001, c001 = get_E_transition_info(flip_e_idx, e0_idx, A, [0,0,1], J, Bz, S)
    print(f"NucSpin=[1,0,0]")
    w100, c100 = get_E_transition_info(flip_e_idx, e0_idx, A, [1,0,0], J, Bz, S)
    # print(f"NucSpin=[1,0,1]")
    # w101, c101 = get_E_transition_info(flip_e_idx, e0_idx, A, [1,0,1], J, Bz, S)


    psi0 = gate.spin_111
    Bx, By = pi_rot_square_pulse(-w100, c100, tN_E, N_E)

    simulate_electrons(psi0, tN_E, N_E, Bz, get_A(1,3,[1,0,0]), J, Bx, By)


    





    H0_e = get_H0(A=get_A(1,3,NucSpin=[0,0,1]), J=get_J(1,3))
    S_e, D_e = get_ordered_eigensystem(H0_e)

    # print("Electron eigenstates:")
    # for j in range(8):
    #     print(f"|e{j}> = {psi_to_string(S[:,j])}")






if __name__=='__main__':

    analyse_transitions()
    #visualise_E_transitions_for_NucSpins()
    plt.show()




