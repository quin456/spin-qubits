
import torch as pt 
import numpy as np
import matplotlib


if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 


import gates as gate
from atomic_units import *
from utils import *
from hamiltonians import get_H0, get_U0, get_pulse_hamiltonian, sum_H0_Hw, get_X_from_H
from pulse_maker import pi_rot_square_pulse
from visualisation import plot_psi, eigenstate_label_getter, visualise_Hw


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
    plot_psi(psi, tN, ax, label_getter=label_getter)









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

def get_Bw_field(tN,N,J,A,multisys=True, include_HZ=False):
    '''
    Accepts pulse time tN (nanoseconds) and A,J couplngs (Mhz), and returns control pulse (tesla) to achieve
    desired eigenstate transition.
    '''
    A*=Mhz; J*=Mhz; tN*=nanosecond
    if not multisys:
        J=J.reshape(1,*J.shape)
        A=A.reshape(1,*A.shape)

    C = get_couplings_over_gamma_e(A,J)
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
    nq = get_nq_from_dim(dim)
    X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator

    

    for j in range(1,N):
        X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
    
    if sys_axis:
        return X
    else:
        return X[0]


def get_electron_X(tN, N, Bz, A, J, Bx, By):
    if Bx is None:
        Bx = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    if By is None:
        By = pt.zeros(N, dtype=cplx_dtype, device=default_device)

    _nS,nq = get_nS_nq_from_A(A)
    H0 = get_H0(A, J, Bz)

    # print("Printing RF's:")
    # rf = get_resonant_frequencies(H0)
    # for w in rf:
    #     print(f"{w/Mhz} MHz")

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=gate.get_Xn(nq), Y=gate.get_Yn(nq))
    H = sum_H0_Hw(H0, Hw)

    X = get_X_from_H(H, tN, N)
    return X


def electron_wf_evolution(tN, N, Bz, A, J, Bx=None, By=None, psi0=gate.spin_111):
    '''
    Simulates electron evolution. Bx and By = None for free evolution.
    '''
    X = get_electron_X(tN, N, Bz, A, J, Bx, By)
    psi = X@psi0 
    return psi


def simulate_electrons(psi0, tN, N, Bz, A, J, Bx, By, ax=None):

    psi = electron_wf_evolution(tN, N, Bz, A, J, Bx, By, psi0)

    label_getter=None

    H0 = get_H0(A, J, Bz)
    S,D = get_ordered_eigensystem(H0)
    if S is not None: 
        psi = pt.einsum('ab,jb->ja', S.T, psi)
        label_getter = lambda j: f"E{j}"
    plot_psi(psi, tN, label_getter=label_getter, ax=ax)


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




def get_nuclear_spins(A):
    nq=len(A)
    spins = [0]*nq
    for i in range(nq):
        if pt.real(A[i])>0:
            spins[i]=spin_up 
        else:
            spins[i]=spin_down 
    return spins








def analyse_3E_system(Bz=2*tesla, A=get_A(1,3), J=get_J(1,3)):

    H0 = get_H0(A=A, J=J, Bz=Bz)
    S,D = get_ordered_eigensystem(H0)

    dim = len(S)

    print("Electron eigenstates:")
    for j in range(dim):
        print(f"|e{j}> = {psi_to_string(S[:,j], pmin=0.001)}")


def drive_electron_transition(S, D, transition, tN=100*nanosecond, N=10000, ax=None, label_getter = eigenstate_label_getter):
    H0 = S@D@S.T

    nq = get_nq_from_dim(H0.shape[-1])
    E = pt.diag(D)

    w_res = E[transition[0]] - E[transition[1]]
    couplings = get_couplings(S)
    coupling = couplings[transition]
    print(f"Driving electron transition: |E{transition[0]}> <--> |E{transition[1]}> with frequency {pt.real(w_res)/Mhz} MHz, coupling {pt.real(coupling)*tesla/Mhz} MHz/tesla.")

    Bx, By = pi_rot_square_pulse(w_res, coupling, tN, N)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(nq), gate.get_Yn(nq))

    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, tN, N)

    psi0 = S[:,transition[0]]
    psi_target = S[:,transition[1]]

    psi = X@psi0

    psi_eig = pt.einsum('ab,jb->ja', S.T, psi)
    plot_psi(psi_eig, tN, label_getter=label_getter, ax=ax)
    label_axis(ax, transition, x_offset=0, y_offset=0)



def investigate_3E_resfreqs(tN=1000*nanosecond, N=10000, fp=None):
    nq=3
    H0 = get_H0(get_A(1,nq), get_J(1,nq))
    S,D = get_ordered_eigensystem(H0); E=pt.diag(D)

    fig,ax=plt.subplots(5,3)

    allowed_transitions = get_allowed_transitions(H0, S=S, E=E)

    for j,transition in enumerate(allowed_transitions):
        #if j+1 in [5,6,7,8,9,10]:
        #    drive_electron_transition(S, D, (transition[1], transition[0]), tN=tN)
        label_getter = lambda i: eigenstate_label_getter(i, states_to_label=transition)
        drive_electron_transition(S, D, transition, tN=tN, N=N, ax=ax[j//3, j%3], label_getter=label_getter)
    fig.set_size_inches(16,12)


    if fp is not None:
        fig.savefig(fp)

def visualise_3E_Hw(A=get_A(1,3), J=get_J(1,3), Bz=0, tN=10*nanosecond, N=1000):
    
    H0 = get_H0(A=A, J=J, Bz=Bz)
    S,D = get_ordered_eigensystem(H0); E=pt.diag(D)
    trans_idx=8
    transitions = get_allowed_transitions(H0, S=S, E=E)
    transition = transitions[trans_idx]
    omega = E[transition[0]]-E[transition[1]]

    Bx, By = pi_rot_square_pulse(omega, gamma_e, tN, N)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, X=gate.get_Xn(nq), Y=gate.get_Yn(nq))

    dim = H0.shape[-1]

    Hw_eig = pt.einsum('ab,jbd->jad', S.T, pt.einsum('jbc,cd->jbd', Hw, S))


    print(f"w_res = {pt.real(omega)/Mhz:.2f} MHz for transition |E{transition[0]}> <--> |E{transition[1]}>")


    for a in range(dim):
        for b in range(dim):
            if (pt.max(pt.real(Hw_eig[:,a,b])) + pt.max(pt.imag(Hw_eig[:,a,b])))/Mhz < 1e-9:
                Hw_eig[:,a,b] = pt.zeros_like(Hw_eig[:,a,b])


    exp_iDt = get_U0(D, tN, N)
    Hw_eig_IP = pt.einsum('jab,jbd->jad', dagger(exp_iDt), pt.einsum('jbc,jcd->jbd', Hw_eig, exp_iDt))

    visualise_Hw(Hw_eig_IP,tN)



def dressed_spins(Bz=0, A=get_A(1,2), J=get_J(1,2)/71, tN=150*nanosecond, N=10000):

    field_mult = 5

    psi0 = gate.spin_01 

    Bx = pt.zeros(N, dtype=cplx_dtype, device=default_device)
    By = pt.zeros(N, dtype=cplx_dtype, device=default_device)

    omega1 = 2*get_A(1,1)
    omega2 = -2*get_A(1,1)
    coupling = 1/2
    Bx1,By1 = pi_rot_square_pulse(omega1, coupling, tN, N)
    Bx2,By2 = pi_rot_square_pulse(omega2, coupling, tN, N)
    #Bx = Bx1+Bx2; By=By1+By2
    psi = electron_wf_evolution(tN, N, Bz, A, J, field_mult*Bx, field_mult*By, psi0)
    plot_psi(psi, tN)






if __name__ == '__main__':
    
    dressed_spins()

    # investigate_3E_resfreqs(tN=1000*nanosecond)
    # nq=3
    #visualise_3E_Hw(A=get_A(1,nq), J=get_J(1,nq))
    plt.show()


