import torch as pt

from data import gamma_e, gamma_n, default_device, cplx_dtype, get_A, get_J
import gates as gate
from atomic_units import *
from utils import dagger, forward_prop, get_nS_nq_from_A

from pdb import set_trace


#####################################################################################
###########        Single electron        ###########################################
#####################################################################################

def single_electron_H0(Bz, A):
    return ( 0.5*gamma_e*Bz + A ) * gate.Z


#####################################################################################
###########        Multi electron        ############################################
#####################################################################################


def get_H0(A,J, Bz=0, device=default_device):
    '''
    Free hamiltonian of each system. Reduced because it doesn't multiply by N timesteps, which is a waste of memory.
    
    Inputs:
        A: (nS,nq), J: (nS,) for 2 qubit or (nS,2) for 3 qubits
    '''


    nS, nq = get_nS_nq_from_A(A)
    d = 2**nq

    
    if nS==1:
        A = A.reshape(1,*A.shape)
        J = J.reshape(1,*J.shape)
        reshaped=True

    HZ = 0.5 * gamma_e * Bz * gate.get_Zn(nq)


    if nq==3:
        H0 = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('sc,cab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device)) + pt.einsum('s,ab->sab',pt.ones(nS),HZ)
    elif nq==2:
        H0 = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('s,ab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device)) + pt.einsum('s,ab->sab',pt.ones(nS),HZ)
    
    H0=H0.to(device)
    if reshaped:
        return H0[0]
    return H0


def get_1S_HA(A):
    nq = len(A)
    if nq==2:
        return A[0]*gate.ZI + A[1]*gate.IZ
    elif nq==3:
        return A[0]*gate.ZII + A[1]*gate.IZI + A[2]*gate.IIZ 
    else:
        raise Exception(f"Invalid A: {A}")

def get_1S_HJ(J):
    try:
        nq=len(J)+1
    except:
        nq=2 
    if nq==2:
        return J*gate.sigDotSig
    elif nq==3:
        return J[0]*gate.o12 + J[1]*gate.o23


def get_U0(H0,tN,N):
    if len(H0.shape) == 2:
        H0=H0.reshape(1,*H0.shape)
        reshaped=True
    U0 = pt.matrix_exp(-1j* pt.einsum('j,sab->sjab',pt.linspace(0,tN,N, dtype=cplx_dtype),H0))
    if reshaped: return U0[0]
    return U0 


# def get_Hw(J,A,tN,N):
#     '''
#     Gets Hw and transforms to interaction picture
#     '''
#     omega = get_transition_frequency(A,J,-2,-1)
#     phase = pt.zeros_like(omega)
#     x_cf,y_cf=get_control_fields(omega,phase,tN,N)
#     nq = len(A[0])
#     ox = gate.get_Xn(nq)
#     oy = gate.get_Yn(nq)
#     Hw = pt.einsum('kj,ab->kjab',x_cf,ox) + pt.einsum('kj,ab->kjab',y_cf,oy)
#     return Hw





#####################################################################################
###########        Nuclear-electron        ##########################################
#####################################################################################

def H_zeeman(Bz):
    Iz = gate.get_Iz_sum(1)
    Sz = gate.get_Sz_sum(1)
    return gamma_e*Bz*Sz - gamma_n*Bz*Iz

def H_hyperfine(A):
    return A * gate.sigDotSig

def get_NE_H0(A,Bz):
    return H_zeeman(Bz) + H_hyperfine(A)


#####################################################################################
###########        Multi nuclear-electron        ####################################
#####################################################################################

def multi_NE_Hw(Bx, By, nq):
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


def multi_NE_H0(Bz=2*tesla, A=get_A(1,1), J=get_J(1,3), deactivate_exchange=False, gamma_e=gamma_e, gamma_n=gamma_n):
    """
    Returns free evolution Hamiltonian of nq==2 or nq==3 electron-nucleus pairs. Each electron interacts with its
    nucleus via hyperfine term 4AS.I, each neighboring electron interacts via exchange 4JS.S, and nulear and electrons
    experience Zeeman splitting due to background static field, Bz.
    """
    try:
        nq = len(J)+1
    except:
        nq=2
    Iz = gate.get_Iz_sum(nq)
    Sz = gate.get_Sz_sum(nq)
    if nq==2:
        o_n1e1 = gate.o4_13 
        o_n2e2 = gate.o4_24 
        o_e1e2 = gate.o4_34
        H0 = gamma_e*Bz*Sz - gamma_n*Bz*Iz + A*o_n1e1 + A*o_n2e2
        if not deactivate_exchange: 
            H0 += J*o_e1e2
    elif nq==3:
        o_n1e1 = gate.o6_14
        o_n2e2 = gate.o6_25
        o_n3e3 = gate.o6_36
        o_e1e2 = gate.o6_45 
        o_e2e3 = gate.o6_56 
        H0 = gamma_e*Bz*Sz - gamma_n*Bz*Iz + A*(o_n1e1+o_n2e2+o_n3e3)
        if not deactivate_exchange: 
            H0 += J[0]*o_e1e2 + J[1]*o_e2e3
    return H0



#####################################################################################
###########       Tools        ######################################################
#####################################################################################


def get_pulse_hamiltonian(Bx, By, gamma, X=gate.X, Y=gate.Y):
    '''
    Inputs:
        Bx: (N,) tensor describing magnetic field in x direction
        By: (N,) tensor describing magnetic field in y direction
        gamma: gyromagnetic ratio
    Returns Hamiltonian corresponding to magnetic field pulse (Bx,By,0)
    '''
    reshaped=False
    if len(Bx.shape) == 1:
        Bx = Bx.reshape(1,*Bx.shape)
        By = By.reshape(1,*By.shape)
        reshaped=True 

    Hw = 0.5 * gamma * ( pt.einsum('kj,ab->kjab', Bx, X) + pt.einsum('kj,ab->kjab', By, Y) )

    if reshaped:
        return Hw[0]
    return Hw

def sum_H0_Hw(H0, Hw):
    '''
    Inputs
        H0: (d,d) tensor describing free Hamiltonian (time indep)
        Hw: (N,d,d) tensor describing control Hamiltonian at each timestep
    '''
    N = len(Hw)
    H = pt.einsum('j,ab->jab',pt.ones(N),H0) + Hw
    return H

def get_U0(H0, tN, N):
    T = pt.linspace(0,tN,N)
    H0T = pt.einsum('j,ab->jab',T,H0)
    U0 = pt.matrix_exp(-1j*H0T)
    return U0

def get_X(H, tN, N, H0_IP=None):
    
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    if H0_IP is not None:
        U0 = get_U0(H0_IP,tN,N)
        X = dagger(U0)@X
    return X


def get_IP_X(X,H0,tN,N):
    U0 = get_U0(H0, tN, N)
    return pt.matmul(dagger(U0),X)

def get_IP_eigen_X(X, H0, tN, N):
    U0 = get_U0(H0, tN, N)
    eig = pt.linalg.eig(H0)
    S = eig.eigenvectors 
    D = pt.diag(eig.eigenvalues)
    return dagger(S) @ dagger(U0) @ X