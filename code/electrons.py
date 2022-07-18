
import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 


from GRAPE import *
import gates as gate
from atomic_units import *
from utils import get_nS_nq_from_A
from hamiltonians import get_H0, get_U0

from pdb import set_trace


from data import gamma_e, gamma_n, cplx_dtype, default_device


spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)


def get_K(A,J):
    dA=A[0]-A[1]
    return (2*(4*J**2 + dA**2 + dA*np.sqrt(4*J**2+dA**2)))**(-1/2)

def get_alpha(A,J):
    dA=A[0]-A[1]
    return get_K(A,J)*(np.sqrt(4*J**2+dA**2)+dA)

def get_beta(A,J):
    return 2*get_K(A,J)*J



def plot_energy_spectrum(E, ax=None):
    dim = E.shape[-1]
    if ax is None: ax=plt.subplot()
    for sys in E:
        for i in range(len(sys)):
            ax.axhline(pt.real(sys[i]/Mhz), label=f'E{dim-1-i}', color=colors[i])
    ax.legend()
    
def get_lowest_energy_indices(D):

    # singlet is at 0, T- is at 3 (for 2q system)
    return 3,0

def put_diagonal(E):
    dim=E.shape[-1]
    D = pt.zeros(dim,dim,dtype=cplx_dtype)
    for j in range(4):
        D[j,j] = E[0,j]
    return D


def get_ordered_2E_eigensystem(A,J, Bz=0):
    '''
    Gets eigenvectors and eigenvalues of Hamiltonian H0 corresponding to hyperfine A, exchange J.
    Orders from lowest energy to highest. Zeeman splitting is accounted for in ordering, but not 
    included in eigenvalues, so eigenvalues will likely not appear to be in order.
    '''
    
    # ordering is always based of physical energy levels (so include_HZ always True)
    E_phys = pt.real(pt.linalg.eig(get_H0(A,J,Bz=B0)).eigenvalues)

    # Eigensystem to be returned usually in rotating frame (so include_HZ usually False)
    H0 = get_H0(A,J,Bz=Bz)

    E,S = order_eigensystem(H0,E_phys)
    D = pt.diag(E)
    return S,D

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

def print_E_system_info(A, J, tN, N):
    nS,nq = get_nS_nq_from_A(A)
    print(f"Simulating {nq} electron system.")
    print(f"A = {A/Mhz} MHz")
    print(f"J = {J/Mhz} MHz")
    print(f"tN = {tN/nanosecond} ns")
    print(f"N = {N}")

def get_free_electron_evolution(tN, N, A=get_A(1,3), J=get_J(1,3), psi0 = None):
    print_E_system_info(A, J, tN, N)
    nS,nq = get_nS_nq_from_A(A)

    print("Getting electron free evolution.")
    print(f"J = {J/Mhz} MHz")
    print(f"A = {A/Mhz} MHz")
    print(f"tN = {tN/nanosecond} ns")


    if psi0 is None:
        if nq==2:
            psi0 = pt.tensor([0,0,1,0], dtype=cplx_dtype)
        elif nq==3:
            psi0 = pt.tensor([0,0,0,0,1,0,0,0], dtype=cplx_dtype)

    H0 = get_H0(A,J)
    U0 = get_U0(H0, tN, N)
    psi = U0 @ psi0 
    return psi

def plot_free_electron_evolution(tN, N, A, J, psi0 = None, ax=None, label_getter=None):
    '''
    Simulates and plots the free evolution of a system of 2 or 3 electrons coupled via exchange.
    '''
    psi = get_free_electron_evolution(A, J, tN, N, psi0=psi0)
    plot_spin_states(psi, tN, ax, label_getter=label_getter)





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


def get_couplings(A,J):
    '''
    Takes input S which has eigenstates of free Hamiltonian as columns.

    Determines the resulting coupling strengths between these eigenstates which arise 
    when a transverse control field is applied.

    The coupling strengths refer to the magnitudes of the terms in the eigenbasis ctrl Hamiltonian.

    Couplings are unitless.
    '''

    E,S = get_ordered_eigensystem(A,J)
    nS = len(S)
    nq = get_nq(len(S[0]))
    couplings = pt.zeros_like(S)
    Xn = gate.get_Xn(nq)
    for s in range(nS):
        couplings[s] = S[s].T @ Xn @ S[s]
    return couplings

def project_psi(psi, n, S):
    return pt.abs(pt.einsum('ja,a->j', psi, S[0,:,n]))

def plot_2q_eigenstates(psi,S,tN, ax=None):
    if ax is None: ax = plt.subplot()
    N = len(psi)
    T = pt.linspace(0,tN/nanosecond,N)
    ax.plot(T,project_psi(psi, 0, S), label = 'T+')
    ax.plot(T,project_psi(psi, 1, S), label = 'T0')
    ax.plot(T,project_psi(psi, 2, S), label = 'S')
    ax.plot(T,project_psi(psi, 3, S), label = "T-")


def plot_eigenstates(psi,S,tN,ax=None):
    dim=S.shape[-1]
    if dim==4:
        plot_2q_eigenstates(psi,S,tN,ax)
        return
        
    if ax is None: ax = plt.subplot()


    N = len(psi)
    T = pt.linspace(0,tN/nanosecond,N)
    for i in range(dim):
        ax.plot(T,project_psi(psi, i, S), label = f'E{dim-1-i}')


def visualise_hamiltonian(H,tN):
    '''
    Takes (N,d,d) tensor H which is Hamiltonian at each of N timesteps. 
    Plots real and imaginary parts of each cell.
    '''
    N,d,d = H.shape 
    T=pt.linspace(0,tN/nanosecond,N)
    fig,ax = plt.subplots(d,d)
    maxval=0
    for a in range(d):
        for b in range(d):
            localmax = max(pt.real(H[:,a,b]))
            if localmax > maxval:
                maxval = localmax 
            ax[a,b].plot(T,pt.real(H[:,a,b]))
            ax[a,b].plot(T,pt.imag(H[:,a,b]))
    
    for a in range(d):
        for b in range(d):
            ax[a,b].set_ylim(-1.2*maxval, 1.2*maxval)

def get_transition_frequency(A,J,i,j,include_HZ=False):
    H0 = get_H0(A,J, include_HZ=include_HZ)
    E,S = get_ordered_eigensystem(A,J,include_HZ=include_HZ)
    return E[:,i]-E[:,j]

def ground_state(nq):
    if nq==2:
        return pt.tensor([0,0,0,1], dtype=cplx_dtype)
    elif nq==3:
        return pt.tensor([0,0,0,0,0,0,0,1], dtype=cplx_dtype)

def visualise_resonant_Hw(J,A,tN,N):
    J*=Mhz 
    A*=Mhz 
    tN*=nanosecond
    E,S=get_ordered_eigensystem(A,J)
    H0=get_H0(A,J)


    # eig = pt.linalg.eig(H0)
    # D=put_diagonal(eig.eigenvalues)
    # S = eig.eigenvectors

    U0=get_U0(H0,tN,N)
    Hw = get_Hw(J,A,tN,N)
    Hw = evolve_Hw(Hw,dagger(U0))
    Hw = transform_Hw(Hw,S)

    visualise_hamiltonian(Hw[0,0],tN)

#visualise_resonant_Hw(get_J(1,3),get_A(1,3),15,300)

def get_Bw_field(tN,N,J,A,multisys=True, include_HZ=False):
    '''
    Accepts pulse time tN (nanoseconds) and A,J couplngs (Mhz), and returns control pulse (tesla) to achieve
    desired eigenstate transition.
    '''
    A*=Mhz; J*=Mhz; tN*=nanosecond
    if not multisys:
        J=J.reshape(1,*J.shape)
        A=A.reshape(1,*A.shape)

    C = get_couplings(A,J)
    omega = get_transition_frequency(A,J,-2,-1,include_HZ=include_HZ)
    phase = pt.zeros_like(omega)

    T = pt.linspace(0,tN,N)
    wt = pt.einsum('k,j->kj', omega,T)

    # get unitless field direction vectors
    x_cf, y_cf = get_unit_CFs(omega,phase,tN,N)

    # get magnetic field strength (teslas)
    Bw = np.pi / (C[0][-2,-1]*g_e*mu_B*tesla*tN)

    x_field = Bw*x_cf
    y_field = Bw*y_cf

    if not multisys:
        return x_field[0], y_field[0]
    return x_field,y_field

def forward_prop(U,device=default_device):
    '''
    Forward propagates U suboperators. U has shape (N,d,d) or (nS,N,d,d)
    '''
    if len(U.shape)==3: 
        U=U.reshape(1,*U.shape)
        sys_axis=False 
    else:
        sys_axis=True

    nS,N,dim,dim=U.shape
    nq = get_nq(dim)
    X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator

    

    for j in range(1,N):
        X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
    
    if sys_axis:
        return X
    else:
        return X[0]


def excite_electrons(tN,N,nS,nq, include_HZ=False):

    A = get_A(nS,nq) 
    J = get_J(nS,nq) 
    Bx,By = get_Bw_field(tN,N,J,A,include_HZ=include_HZ)

    
    A,J=convert_Mhz(A,J)
    tN*=nanosecond; Bx*=tesla; By*=tesla

    C = get_couplings(A,J)

    x_field = 0.5*gamma_e*Bx 
    y_field = 0.5*gamma_e*By
    X_elec=gate.XI+gate.IX 
    Y_elec=gate.YI+gate.IY
    Hw = 0.5*gamma_e * (pt.einsum('j,ab->jab',Bx[0],X_elec) + pt.einsum('j,ab->jab',By[0],Y_elec))
    print(f"Hw = {Hw}")

    m=len(x_field)

    u = pt.ones(m*N)

    #A[0,0]*=-1
    #A[0,2]*=-1
    H0 = get_H0(A,J,include_HZ=include_HZ)
    U0=get_U0(H0,tN,N)
    E,S=get_ordered_eigensystem(A,J,include_HZ=include_HZ)
    H = Hw + H0
    global H0g,Hwg 
    H0g=H0[0]; Hwg=Hw
    dt=tN/N
    U = pt.matrix_exp(-1j*H*dt)
    X=forward_prop(U)

    # U = time_evol(u,H0,x_field,y_field,tN)
    # X = forward_prop(U)[0]


    psi0 = pt.kron(spin_up,spin_down)
    psi = pt.matmul(X,psi0)
    #psi = pt.cat((psi, pt.matmul(X_free[0],psi[-1,:])))
    #psi = pt.einsum('jab,jb->ja',dagger(U0[0]),psi)

    
    #fig,ax = plt.subplots(1,3)
    #plot_energy_spectrum(E,ax[0])
    fig,ax=plt.subplots(1,2)
    plot_spin_states(psi,tN,ax[0])
    plot_eigenstates(psi,S,tN,ax[1])
    #plt.axhline(0.1, label='Negligible amplitude', linestyle='dashed')
    plt.legend()

#excite_electrons(2.0, 50000,1,2, True)




def get_n_e_label(j):
    ''' Returns state label corresponding to integer j\in[0,dim] '''
    uparrow = u'\u2191'
    downarrow = u'\u2193'
    b= np.binary_repr(j,4)
    if b[2]=='0':
        L2 = uparrow 
    else:
        L2=downarrow
    if b[3]=='0':
        L3 = uparrow
    else:
        L3 = downarrow
    
    return b[0]+b[1]+L2+L3


def plot_nuclear_electron_wf(psi,tN, ax=None):
    N,dim = psi.shape
    nq = get_nq(dim)
    T = pt.linspace(0,tN/nanosecond,N)
    
    if ax is None: ax=plt.subplot()
    for j in range(dim):
        ax.plot(T,pt.abs(psi[:,j]), label = get_n_e_label(j))
    ax.legend()


def plot_nuclear_electron_spectrum(H0):
    get_ordered_eigensystem()

def get_nuclear_spins(A):
    nq=len(A)
    spins = [0]*nq
    for i in range(nq):
        if pt.real(A[i])>0:
            spins[i]=spin_up 
        else:
            spins[i]=spin_down 
    return spins

def nuclear_electron_sim(Bx,By,tN,N,nq,A,J, psi0=None):
    '''
    Simulation of nuclear and electron spins for single CNOT system.

    Order Hilbert space as |n1,...,e1,...>

    Inputs:
        (Bx,By): Tensors describing x and y components of control field at each timestep (in teslas).
        A: hyperfines
        J: Exchange coupling (0-dim or 2 element 1-dim)
    '''

    Bz = 2*tesla

    # unit conversions
    A,J=convert_Mhz(A,J)
    tN*=nanosecond
    Bx*=tesla; By*=tesla
    A_mag = np.abs(A[0])



    if nq==2:

        ozn = gate.ZIII + gate.IZII 
        oze = gate.IIZI + gate.IIIZ

        o_n1e1 = gate.o4_13 
        o_n2e2 = gate.o4_24 
        o_e1e2 = gate.o4_34

        X_nuc = gate.XIII+gate.IXII
        X_elec = gate.IIXI+gate.IIIX
        Y_nuc = gate.YIII+gate.IYII
        Y_elec = gate.IIYI+gate.IIIY

        nspin1,nspin2 = get_nuclear_spins(A)

        if psi0==None:
            psi0 = gate.kron4(nspin1, nspin2, spin_up, spin_down)
        print(f"psi0={psi0}")
        H0 = 0.5*gamma_e*Bz*oze - 0.5*gamma_n*Bz*ozn + A_mag*o_n1e1 + A_mag*o_n2e2 + J*o_e1e2 
        #H0 = 0.5*gamma_e*Bz*oze + A[0]*gate.IIZI + A[1]*gate.IIIZ + J*o_e1e2 
    
    elif nq==3:

        ozn = gate.ZIIIII + gate.IZIIII + gate.IIZIII 
        oze = gate.IIIZII + gate.IIIIZI + gate.IIIIIZ

        o_n1e1 = gate.o6_14
        o_n2e2 = gate.o6_25
        o_n3e3 = gate.o6_36
        o_e1e2 = gate.o6_45 
        o_e2e3 = gate.o6_56 

        H0 = 0.5*gamma_e*Bz*oze - 0.5*gamma_n*Bz*ozn + A_mag*(o_n1e1+o_n2e2+o_n3e3) + J[0]*o_e1e2 + J[1]*o_e2e3


    else:
        raise Exception("Invalid nq")

    Hw_nuc = 0.5*gamma_n * (pt.einsum('j,ab->jab',Bx,X_nuc) + pt.einsum('j,ab->jab',By,Y_nuc))
    Hw_elec = 0.5*gamma_e * (pt.einsum('j,ab->jab',Bx,X_elec) + pt.einsum('j,ab->jab',By,Y_elec))
    Hw = Hw_elec+Hw_nuc
    # H0 has shape (d,d), Hw has shape (N,d,d). H0+Hw automatically adds H0 to all N Hw timestep values.
    H=Hw+H0
    dt=tN/N
    U = pt.matrix_exp(-1j*H*dt)
    X=forward_prop(U)
    psi = pt.matmul(X,psi0)

    fig,ax=plt.subplots(1,1)
    plot_spin_states(psi,tN,ax)

    return


def run_NE_sim(tN,N,nq,A,J, psi0=None):
    Bx,By = get_Bw_field(tN,N,J,A,multisys=False,include_HZ=True)
    nuclear_electron_sim(Bx,By,tN,N,nq,A,J,psi0)


if __name__ == '__main__':
    tN=2.0
    N=50000
    nq=2
    A=get_A(1,2)
    J=get_J(1,2)
    run_NE_sim(2.0,50000,2, get_A(1,2)[0], get_J(1,2)[0])
    excite_electrons(tN,N,1,nq, include_HZ=True)


    plt.show()


