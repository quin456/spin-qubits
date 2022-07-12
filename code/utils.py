
import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import math
import itertools


import gates as gate
from gates import default_device, cplx_dtype
from data import *
from atomic_units import *
from electrons import get_ordered_2E_eigensystem



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
        multisys = False
    if len(target.shape)==2:
        target = target.reshape(1,*target.shape)
    nS=len(X); N = len(X[0])
    fid = pt.zeros(nS,N)
    for q in range(nS):
        for j in range(N):
            IP = innerProd(target[q],X[q,j])
            fid[q,j] = np.real(IP*np.conj(IP))

    if not multisys:
        fid = fid[0]
    return fid



def remove_duplicates(A):
    '''
    Removes duplicates from A where equivalence is required to 9 decimal places
    '''
    i=0
    while i<len(A):
        j=i+1
        while j<len(A):
            if math.isclose(A[i],A[j],rel_tol=1e-9):
                A.pop(j)
                continue 
            j+=1
        i+=1
    return A


def get_resonant_frequencies(H0,Hw_shape,device=default_device):
    '''
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0. 
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. 
    '''
    eig = pt.linalg.eig(H0)
    E=eig.eigenvalues
    S = eig.eigenvectors
    S=S.to(device)

    #S,D = get_ordered_2E_eigensystem(get_A(1,1), get_J(1,2))
    #E = pt.diagonal(D)
    #S = pt.transpose(pt.stack((S[:,2],S[:,1],S[:,0],S[:,3])),0,1)
    #evals = pt.stack((evals[2],evals[1],evals[0],evals[3]))
    S_T = pt.transpose(S,0,1)
    d = len(E)
    pairs = list(itertools.combinations(pt.linspace(0,d-1,d,dtype=int),2))

    # transform shape of control Hamiltonian to basis of energy eigenstates

    Hw_trans = matmul3(S_T,Hw_shape,S)
    Hw_nz = (pt.abs(Hw_trans)>1e-9).to(int)
    Hw_angle = pt.angle(Hw_trans)
    freqs = []
    for i in range(d):
        for j in range(d):
            # The difference between energy levels i,j will be a resonant frequency if the control field Hamiltonian
            # has a non-zero (i,j) element.
            #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
            if Hw_nz[i,j] and Hw_angle[i,j] < 0:
                freqs.append((pt.real(E[i]-E[j])).item())
            #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
            # if Hw_nz[idx2,idx1]:
            #     freqs.append((pt.real(evals[pair[0]]-evals[pair[1]])).item())
    freqs = pt.tensor(remove_duplicates(freqs), dtype = real_dtype, device=device)

    return freqs

def lock_to_coupling(c, tN):
    t_HF = 2*np.pi/c
    tN_locked = int(tN/t_HF) * t_HF
    if tN_locked == 0:
        tN_locked=t_HF
        print(f"tN={tN/nanosecond}ns too small to lock to coupling period {t_HF/nanosecond}ns.")
        return tN
    else:
        print(f"Locking tN={tN/nanosecond}ns to coupling period {t_HF/nanosecond}ns. New tN={tN_locked/nanosecond}ns.")
    return tN_locked
