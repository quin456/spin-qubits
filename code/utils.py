
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

def get_nq(d):
    ''' Takes the dimension of the Hilbert space as input, and returns the number of qubits. '''
    return int(np.log2(d))

def psi_from_polar(theta,phi):
    if not pt.is_tensor(theta):
        theta = pt.tensor([theta]); phi = pt.tensor([phi])
    return pt.stack((pt.cos(theta/2), pt.einsum('j,j->j',pt.exp(1j*phi),pt.sin(theta/2)))).T


def get_single_qubit_angles(psi):
    '''
    input psi: (N,2) complex array describing single qubit wave function over N timesteps.
    output: (N,2) real array (probably still complex dtype) describing theta, phi over N timesteps.
    '''
    reshaped=False
    if len(psi.shape)==1:
        psi=psi.reshape(1,*psi.shape)
        reshaped=True
    theta = 2*pt.arctan(pt.abs(psi[:,1]/psi[:,0]))
    phi = pt.angle(psi[:,1])-pt.angle(psi[:,0])
    if reshaped:
        return theta[0], phi[0]
    return theta,phi

def psi_to_cartesian(psi):
    '''
    psi: (N,2) or (2,) array describing single qubit wave funtion over N timesteps.
    Function returns (N,3) or (3,) array describing cartesian coordinates on the Bloch sphere
    '''
    reshaped=False
    if len(psi.shape)==1:
        psi=psi.reshape(1,*psi.shape)
        reshaped=True

    theta,phi = get_single_qubit_angles(psi)
    x = pt.sin(theta)*pt.cos(phi)
    y = pt.sin(theta)*pt.sin(phi)
    z = pt.cos(theta)
    r = pt.stack((x,y,z)).T

    if reshaped:
        return r[0]
    return r

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


def get_IP_X(X,H0,tN,N):
    U0 = get_U0(H0, tN, N)
    return pt.matmul(dagger(U0),X)

def get_IP_eigen_X(X, H0, tN, N):
    U0 = get_U0(H0, tN, N)
    eig = pt.linalg.eig(H0)
    S = eig.eigenvectors 
    D = pt.diag(eig.eigenvalues)
    return dagger(S) @ dagger(U0) @ X

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


def lock_to_coupling(c, tN):
    t_HF = 2*np.pi/c
    tN_locked = int(tN / (t_HF) ) * t_HF
    if tN_locked == 0:
        tN_locked=t_HF
        print(f"tN={tN/nanosecond}ns too small to lock to coupling period {t_HF/nanosecond}ns.")
        return tN
    else:
        print(f"Locking tN={tN/nanosecond}ns to coupling period {t_HF/nanosecond}ns. New tN={tN_locked/nanosecond}ns.")
    return tN_locked
