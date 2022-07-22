
from regex import P
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



from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_nS_nq_from_A(A):
    ''' Returns (number of systems, number of qubits in each system) '''
    try:
        if len(A.shape)==2:
            return len(A), len(A[0])
        elif len(A.shape)==1:
            return 1, len(A)
    except:
        # A is not an array
        return 1,1

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

def wf_fidelity(u,v):
    ''' Calculates vector fidelity of u and v '''
    return pt.real(pt.dot(u,pt.conj(v)) * pt.dot(pt.conj(u),v))

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

def get_allowed_transitions(H0, Hw_shape=None, S=None, E=None, device=default_device):
    if Hw_shape is None:
        nq = get_nq(H0.shape[-1])
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
    Hw_nz = (pt.abs(Hw_trans)>1e-9).to(int)
    Hw_angle = pt.angle(Hw_trans)

    allowed_transitions = []
    for i in range(d):
        for j in range(d):
            if Hw_nz[i,j] and Hw_angle[i,j] < 0:
                allowed_transitions.append((i,j))
    return allowed_transitions

def get_resonant_frequencies(H0,Hw_shape=None,device=default_device):
    '''
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0. 
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. 
    '''
    freqs = []
    eig = pt.linalg.eig(H0)
    E=eig.eigenvalues
    allowed_transitions = get_allowed_transitions(H0, Hw_shape=Hw_shape, device=device)
    for transition in allowed_transitions:
        freqs.append((pt.real(E[transition[0]]-E[transition[1]])).item())

    freqs = pt.tensor(remove_duplicates(freqs), dtype = real_dtype, device=device)

    return freqs


def get_ordered_eigensystem(H0, H0_phys=None):
    '''
    Gets eigenvectors and eigenvalues of Hamiltonian H0 corresponding to hyperfine A, exchange J.
    Orders from lowest energy to highest. Zeeman splitting is accounted for in ordering, but not 
    included in eigenvalues, so eigenvalues will likely not appear to be in order.
    '''
    if H0_phys is None:
        H0_phys=H0
    
    # ordering is always based of physical energy levels (so include_HZ always True)
    E_phys = pt.real(pt.linalg.eig(H0_phys).eigenvalues)

    E,S = order_eigensystem(H0,E_phys)
    D = pt.diag(E)
    return S,D

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

def order_eigensystem(H0, E_order):

    idx_order = pt.topk(E_order, len(E_order)).indices

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

def print_rank2_tensor(T):
    m,n = T.shape
    for i in range(m):
        if i==0:
            print("⌈", end="")
        elif i==n-1:
            print("⌊", end="")
        else:
            print("|", end="")
        for j in range(n):
            print(f"{pt.real(T[i,j]).item():>6.2f}", end="  ")


        if i==0:
            print("⌉")    
        elif i==n-1:
            print("⌋")
        else:
            print("|")

def label_axis(ax, label, x_offset=-0.05, y_offset=-0.05):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1]-xlim[0]
    dy = ylim[1]-ylim[0]

    x = xlim[0]+x_offset*dx 
    y = ylim[0]+y_offset*dy 

    # if offset<0:
    #     ax.set_xlim(offset, xlim[1])
    #     ax.set_ylim(offset, ylim[1])

    ax.text(x, y, label, fontsize=16, fontweight="bold", va="bottom", ha="left")




def multi_NE_label_getter(j, label_states=None):
    ''' Returns state label corresponding to integer j\in[0,dim] '''
    if label_states is not None:
        if j not in label_states:
            return ''
    uparrow = u'\u2191'
    downarrow = u'\u2193'
    b = np.binary_repr(j,4)
    if b[2]=='0':
        L2 = uparrow 
    else:
        L2=downarrow
    if b[3]=='0':
        L3 = uparrow
    else:
        L3 = downarrow
    
    return b[0]+b[1]+L2+L3


if __name__ == '__main__':
    label_axis(plt.subplot(), offset=-0.1, label='A')