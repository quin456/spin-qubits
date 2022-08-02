

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt 
from atomic_units import *
from utils import get_nq_from_dim, dagger, fidelity_progress, psi_to_cartesian, get_resonant_frequencies, get_ordered_eigensystem, print_rank2_tensor
from hamiltonians import get_H0, multi_NE_H0
from data import get_A, get_J, gamma_n, gamma_e

import qiskit
from qiskit.visualization import plot_bloch_vector


from pdb import set_trace


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


double_long_width = 10
single_long_height = 2.3
double_long_height = 2.6
square_size = 10/2.6

annotate=False
y_axis_labels = False

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'

def spin_state_label_getter(i, nq):
    return np.binary_repr(i,nq)

def eigenstate_label_getter(i):
    return f"E{i}"

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

def plot_spin_states(psi, tN, ax=None, label_getter =  None, squared=True, fp=None, legend_loc='upper center'):
    '''
    Plots the evolution of each component of psi.

    Inputs
        psi: (N,d) tensor where d is Hilbert space dimension and N is number of timesteps.
        tN: duration spanned by N timesteps
        ax: axis on which to plot
    '''
    if ax is None: ax = plt.subplot()
    if label_getter is None: label_getter = lambda i: spin_state_label_getter(i, nq)
    N,dim=psi.shape
    nq=get_nq_from_dim(dim)
    T=pt.linspace(0,tN/nanosecond,N)
    for i in range(dim):
        if squared:
            y = pt.abs(psi[:,i])**2
        else:
            y = pt.abs(psi[:,i])
        label = f"Pr({label_getter(i)})" if squared else label_getter(i)
        ax.plot(T,y, label = label)
    ax.legend(loc=legend_loc)
    ax.set_xlabel("time (ns)")
    if y_axis_labels: ax.set_ylabel("$|\psi|$")

    if fp is not None: plt.savefig(fp)

    return ax

def plot_phases(psi, tN, ax=None):

    if ax is None: ax = plt.subplot() 
    N,dim = psi.shape 
    nq=get_nq_from_dim(dim)
    T=pt.linspace(0,tN/nanosecond,N)
    phase = pt.zeros_like(psi)
    for i in range(dim):
        phase[:,i] = pt.angle(psi)[:,i]-pt.angle(psi)[:,0]
        ax.plot(T,phase[:,i], label = f'$\phi_{i}$')
    ax.legend()
    ax.set_xlabel("time (ns)")
    print(f"Final phase = {phase[-1,:]}")
    return ax


def plot_fields(Bx,By,tN,ax=None):
    '''
    Inputs
        Bx: Magnetic field in x direction over N timesteps (atomic units) 
        By: Magnetic field in y direction over N timesteps (atomic units)
        tN: Duration of pulse
    '''
    N = len(Bx)
    T_axis = pt.linspace(0,tN/nanosecond, N)
    if ax==None: ax = plt.subplot()
    ax.plot(T_axis,Bx*1e3/tesla, label = 'X field (mT)')
    ax.plot(T_axis,By*1e3/tesla, label = 'Y field (mT)')
    ax.set_xlabel('time (ns)')
    if y_axis_labels: ax.set_ylabel('$B_\omega(t)$ (mT)')
    ax.legend()
    return ax 

def plot_psi_and_fields(psi, Bx, By, tN):
    fig,ax = plt.subplots(1,2)
    plot_spin_states(psi, tN, ax[0])
    plot_fields(Bx, By, tN, ax[1])
    return ax

    
def visualise_Hw(Hw,tN, eigs=None):
    '''
    Generates an array of plots, one for each matrix element of the Hamiltonian Hw, which
    shows the evolution of Hw through time.

    Inputs
        Hw: (N,d,d) tensor describing d x d dimensional Hamiltonian over N timesteps
        tN: duration spanned by Hw.
    '''
    N,dim,dim = Hw.shape
    T = pt.linspace(0,tN/nanosecond,N)
    if eigs is not None:
        D = pt.diag(eigs.eigenvalues)
        U0_e = pt.matrix_exp(-1j*pt.einsum('ab,j->jab',D,T))
        S = eigs.eigenvectors
        Hw = dagger(U0_e) @ dagger(S) @ Hw @ S @ U0_e
    fig,ax = plt.subplots(dim,dim)
    for i in range(dim):
        for j in range(dim):
            y = Hw[:,i,j]/Mhz
            ax[i,j].plot(T,pt.real(y))
            ax[i,j].plot(T,pt.imag(y))

            
def plot_fidelity(ax,fids,tN, legend=True, printfid=True):
    if len(fids.shape)==1:
        fids = fids.reshape(1,*fids.shape)
    nS=len(fids); N = len(fids[0])
    T = pt.linspace(0,tN/nanosecond, N)
    if nS==1:
        ax.plot(T,pt.real(fids[0]), label=f"Fidelity")
    else:
        for q in range(nS):
            ax.plot(T,pt.real(fids[q]), label=f"System {q+1} fidelity")
    if legend: ax.legend()
    ax.set_xlabel("time (ns)")
    if y_axis_labels: ax.set_ylabel("Fidelity")
    if annotate: ax.annotate("Fidelity progress", (0,0.95))
    if printfid: print(f"Achieved fidelity = {pt.real(fids[0,-1]):.4f}")
    return ax


def plot_multi_sys_energy_spectrum(E, ax=None):
    dim = E.shape[-1]
    if ax is None: ax=plt.subplot()
    for sys in E:
        for i in range(len(sys)):
            ax.axhline(pt.real(sys[i]/Mhz), label=f'E{dim-1-i}', color=colors[i])
    ax.legend()

def plot_energy_spectrum(E, ax=None):
    if ax is None: ax=plt.subplot()
    dim = len(E)
    for i in range(dim):
        ax.axhline(pt.real(E[i]/Mhz), label=f'E{dim-1-i}', color=colors[i%len(colors)])

def plot_energy_spectrum_from_H0(H0):
    rf = get_resonant_frequencies(H0)
    S,D = get_ordered_eigensystem(H0)
    plot_energy_spectrum(pt.diagonal(D))




def show_fidelity(X, tN, target, ax=None):
    print(f"Final unitary:")
    print_rank2_tensor(X[-1]/(X[-1,0,0]/pt.abs(X[-1,0,0])))
    fids = fidelity_progress(X,target)
    print(f"Final fidelity = {fids[-1]}")
    
    if ax is None: ax = plt.subplot()
    plot_fidelity(ax,fids,tN)
    return fids








def bloch_sphere(psi, fp=None):
    blochs = psi_to_cartesian(psi).numpy()
    plot_bloch_vector(blochs)
    if fp is not None: plt.savefig(fp)




if __name__=='__main__':
    H0_3E = get_H0(A=get_A(1,3, NucSpin=[1,1,1]), J=get_J(2,3)[1], Bz=0.1*tesla)
    H0_3NE = multi_NE_H0(A=get_A(1,1), J=get_J(1,3), Bz=0.2*tesla, gamma_n=10*gamma_n, gamma_e=gamma_e/10)
    plot_energy_spectrum_from_H0(H0_3NE)


    plt.show()
