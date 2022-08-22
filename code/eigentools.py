


import torch as pt 
import numpy as np




import atomic_units as unit
import gates as gate
from utils import *
from data import default_device
from pulse_maker import pi_pulse_field_strength








def get_allowed_transitions(H0, Hw_shape=None, S=None, E=None, device=default_device):
    if Hw_shape is None:
        nq = get_nq_from_dim(H0.shape[-1])
        Hw_shape = (gate.get_Xn(nq) + gate.get_Yn(nq))/np.sqrt(2)

    if S is None:
        eig = pt.linalg.eig(H0)
        E=eig.eigenvalues
        S = eig.eigenvectors
        S=S.to(device)

    S_T = pt.transpose(S,0,1)
    d = len(E)

    # transform shape of control Hamiltonian to basis of energy eigenstates
    Hw_trans = matmul3(S_T,Hw_shape,S)
    Hw_nz = (pt.abs(Hw_trans)>1e-8).to(int)
    Hw_angle = pt.angle(Hw_trans**2)


    allowed_transitions = []
    for i in range(d):
        for j in range(d):
            if Hw_nz[i,j] and Hw_angle[i,j] < 0:
                allowed_transitions.append((i,j))

    return allowed_transitions

def get_resonant_frequencies(H0,Hw_shape=None, E=None, device=default_device):
    '''
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0. 
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. 
    '''
    if E is None:
        eig = pt.linalg.eig(H0)
        E=eig.eigenvalues
    allowed_transitions = get_allowed_transitions(H0, Hw_shape=Hw_shape, device=device)
    print(allowed_transitions)

    return get_transition_freqs(allowed_transitions, E, device=device)

def get_transition_freqs(allowed_transitions, E, device=default_device):
    freqs = []
    for transition in allowed_transitions:
        freqs.append((pt.real(E[transition[0]]-E[transition[1]])).item())
    freqs = pt.tensor(remove_duplicates(freqs), dtype = real_dtype, device=device)
    return freqs


def get_low_J_rf_u0(S, D, tN, N):
    '''
    Gets resonant frequencies and u0 in low J limit, where transition is driven mostly by a single frequency.
    u0 is chosen to give appropriate pi-pulse.
    '''

    E = pt.diag(D)
    H0 = S@D@S.T
    allowed_transitions = get_allowed_transitions(H0, S=S, E=E)
    target_transition = (2,3)
    if target_transition in allowed_transitions:
        idx = allowed_transitions.index(target_transition)
    elif target_transition[::-1] in allowed_transitions:
        idx = allowed_transitions.index(target_transition[::-1])
    else:
        raise Exception("Failed to find low J resonant frequencies and u0: target transition not in allowed transitions.")
    
    target_transition = allowed_transitions[idx]
    allowed_transitions[idx] = allowed_transitions[0]
    allowed_transitions[0] = target_transition
    freqs = get_transition_freqs(allowed_transitions, E)

    couplings = get_couplings()
    m = len(freqs)
    u0 = pt.zeros(m, N, dtype=cplx_dtype, device=default_device)
    u0[0] = pi_pulse_field_strength(couplings[target_transition], tN)

    
    



def get_multi_system_resonant_frequencies(H0s, device=default_device):
    rf = pt.tensor([], dtype = real_dtype, device=device)
    nS = len(H0s); nq = get_nq_from_dim(H0s.shape[-1])
    Hw_shape = (gate.get_Xn(nq) + gate.get_Yn(nq)) / np.sqrt(2)
    for q in range(nS):
        rf_q=get_resonant_frequencies(H0s[q], Hw_shape)
        rf=pt.cat((rf,rf_q))
    return rf




    
def get_couplings_over_gamma_e(S,D):
    '''
    Takes input S which has eigenstates of free Hamiltonian as columns.

    Determines the resulting coupling strengths between these eigenstates which arise 
    when a transverse control field is applied.

    The coupling strengths refer to the magnitudes of the terms in the eigenbasis ctrl Hamiltonian.

    Couplings are unitless.
    '''

    nq = get_nq_from_dim(len(S[0]))
    couplings = pt.zeros_like(S)
    Xn = gate.get_Xn(nq)
    couplings = S.T @ Xn @ S
    return couplings

def get_couplings(S, Hw_mag=None):
    nq = get_nq_from_dim(len(S[0]))
    couplings = pt.zeros_like(S)
    if Hw_mag is None: 
        print("No Hw_mag specified, providing couplings for electron spin system.")
        Hw_mag = gamma_e * gate.get_Xn(nq)
    return gamma_e*S.T @ Hw_mag @ S

def get_pi_pulse_tN_from_field_strength(B_mag, coupling, coupling_lock = None):
    tN = np.pi/(coupling*B_mag)
    print(f"Pi-pulse duration for field strength {B_mag/unit.mT} mT with coupling {pt.real(coupling)*unit.T/unit.MHz} MHz/unit.T is {pt.real(tN)/unit.ns} ns.")
    if coupling_lock is not None:
        return lock_to_frequency(coupling_lock, tN)
    return tN

def get_ordered_eigensystem(H0, H0_phys=None, ascending=True):
    '''
    Gets eigenvectors and eigenvalues of Hamiltonian H0 corresponding to hyperfine A, exchange J.
    Orders from lowest energy to highest. Zeeman splitting is accounted for in ordering, but not 
    included in eigenvalues, so eigenvalues will likely not appear to be in order.
    '''
    if H0_phys is None:
        H0_phys=H0
    
    # ordering is always based of physical energy levels (so include_HZ always True)
    E_phys = pt.real(pt.linalg.eig(H0_phys).eigenvalues)

    E,S = order_eigensystem(H0,E_phys, ascending=ascending)
    D = pt.diag(E)
    return S,D

def order_eigensystem(H0, E_order, ascending=True):

    idx_order = pt.topk(E_order, len(E_order), largest = not ascending).indices

    # get unsorted eigensystem
    eig = pt.linalg.eig(H0)
    E_us=eig.eigenvalues
    S_us = eig.eigenvectors

    E = pt.zeros_like(E_us)
    S = pt.zeros_like(S_us)
    for i,j in enumerate(idx_order):
        E[i] = E_us[j]
        S[:,i] = S_us[:,j]
    return E,S

def get_max_allowed_coupling(H0, p=0.9999):
    
    rf = get_resonant_frequencies(H0)

    # first find smallest difference in rf's
    min_delta_rf = 1e30
    for i in range(len(rf)):
        for j in range(i+1, len(rf)):
            if pt.abs(rf[i]-rf[j]) < min_delta_rf:
                min_delta_rf = pt.abs(rf[i]-rf[j])
    return min_delta_rf / np.pi * np.arccos(np.sqrt(p))




def lock_to_frequency(c, tN):
    t_HF = 2*np.pi/c
    tN_locked = int(tN/t_HF) * t_HF
    if tN_locked == 0:
        tN_locked=t_HF
        #print(f"tN={pt.real(tN)/unit.ns:.2f}ns too small to lock to coupling period {t_HF/unit.ns:.2f}ns.")
        #return tN_locked
    print(f"Locking tN={tN/unit.ns:.2f} ns to coupling period {t_HF/unit.ns} ns. New tN={tN_locked/unit.ns:.2f} ns.")
    return tN_locked