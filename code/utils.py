
import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 



import gates as gate
from gates import default_device, cplx_dtype
from data import *
from atomic_units import *




from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']





def convert_Mhz(A,J):
    ''' A,J need to be multiplied by plancks constant to be converted to Joules (which are equal to Hz given hbar=1) '''
    return h_planck*A*Mhz, h_planck*J*Mhz


def get_nq(d):
    ''' Takes the dimension of the Hilbert space as input, and returns the number of qubits. '''
    return int(np.log2(d))

def forward_prop(U,device=default_device):
    '''
    Forward propagates U suboperators. U has shape (N,d,d) or (nS,N,d,d)
    '''
    sys_axis=True
    if len(U.shape)==3: 
        U=U.reshape(1,*U.shape)
        sys_axis=False

    nS,N,dim,dim=U.shape
    nq = get_nq(dim)
    X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator
    
    for j in range(1,N):
        X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
    
    if sys_axis:
        return X
    return X[0]


def get_pulse_hamiltonian(Bx, By, gamma, X=gate.X, Y=gate.Y):
    '''
    Inputs:
        Bx: (N,) tensor describing magnetic field in x direction
        By: (N,) tensor describing magnetic field in y direction
        gamma: gyromagnetic ratio
    Returns Hamiltonian corresponding to magnetic field pulse (Bx,By,0)
    '''
    Hw = 0.5 * gamma * ( pt.einsum('j,ab->jab', Bx, X) + pt.einsum('j,ab->jab', By, Y) )

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


def get_dimensions(A):
    ''' Returns (number of systems, number of qubits in each system) '''
    return len(A), len(A[0])





def normalise(v):
    ''' Normalises 1D tensor '''
    return v/pt.norm(v)

def innerProd(A,B):
    '''  Calculates the inner product <A|B>=Phi(A,B) of two matrices A,B.  '''
    return pt.trace(pt.matmul(dagger(A),B)).item()/len(A)

def dagger(A):
    '''  Returns the conjugate transpose of a matrix or batch of matrices.  '''
    return pt.conj(pt.transpose(A,-2,-1))

def commutator(A,B):
    '''  Returns the commutator [A,B]=AB-BA of matrices A and B.  '''
    return pt.matmul(A,B)-pt.matmul(B,A)

def matmul3(A,B,C):
    '''  Returns multiple of three matrices A,B,C.  '''
    return pt.matmul(A,pt.matmul(B,C))



def fidelity(A,B):
    ''' Calculates fidelity of operators A and B '''
    IP = innerProd(A,B)
    return np.real(IP*np.conj(IP))


def fidelity_progress(X, target):
    '''
    For each system, determines the fidelity of unitaries in P with the target over the time of the pulse
    '''
    multisys = True
    if len(X.shape)==3:
        X = X.reshape(1,*X.shape)
        target = target.reshape(1,*target.shape)
        multisys = False
    nS=len(X); N = len(X[0])
    fid = pt.zeros(nS,N)
    for q in range(nS):
        for j in range(N):
            IP = innerProd(target[q],X[q,j])
            fid[q,j] = np.real(IP*np.conj(IP))

    if not multisys:
        fid = fid[0]
    return fid


def get_U0(H0, tN, N):
    T = pt.linspace(0,tN,N)
    H0T = pt.einsum('j,ab->jab',T,H0)
    U0 = pt.matrix_exp(-1j*H0T)
    return U0


def get_IP_X(X,H0,tN,N):
    U0 = get_U0(H0, tN, N)
    return pt.matmul(dagger(U0),X)

def get_IP_eigen_X(X, H0, tN, N):
    U0 = get_U0(H0, tN, N)
    eig = pt.linalg.eig(H0)
    S = eig.eigenvectors 
    D = pt.diag(eig.eigenvalues)
    return dagger(S) @ dagger(U0) @ X