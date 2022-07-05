

from cProfile import label
from turtle import shape
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt 
from atomic_units import *
from utils import get_nq, dagger


from pdb import set_trace


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

annotate=True 

def plot_spin_states(psi, tN, ax=None, label_getter = None):
    '''
    Plots the evolution of each component of psi.

    Inputs
        psi: (N,d) tensor where d is Hilbert space dimension and N is number of timesteps.
        tN: duration spanned by N timesteps
        ax: axis on which to plot
    '''
    if ax is None: ax = plt.subplot()
    if label_getter is None:
        label_getter = lambda i: np.binary_repr(i,nq)
    N,dim=psi.shape
    nq=get_nq(dim)
    T=pt.linspace(0,tN/nanosecond,N)
    for i in range(dim):
        ax.plot(T,pt.abs(psi[:,i]), label = label_getter(i))
    ax.legend()
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("Wave function amplitude")
    return ax

def plot_phases(psi, tN, ax=None):

    if ax is None: ax = plt.subplot() 
    N,dim = psi.shape 
    nq=get_nq(dim)
    T=pt.linspace(0,tN/nanosecond,N)
    for i in range(dim):
        ax.plot(T,pt.angle(psi[:,i]-pt.angle(psi[:,0])), label = np.binary_repr(i,nq))
    ax.legend()
    ax.set_xlabel("time (ns)")
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
    ax.plot(T_axis,Bx/Mhz, label = 'X field (mT)')
    ax.plot(T_axis,By/Mhz, label = 'Y field (mT)')
    ax.set_xlabel('time (ns)')
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
    N,d,d = Hw.shape
    T = pt.linspace(0,tN,N)
    if eigs is not None:
        D = pt.diag(eigs.eigenvalues)
        U0_e = pt.matrix_exp(-1j*pt.einsum('ab,j->jab',D,T))
        S = eigs.eigenvectors
        Hw = dagger(U0_e) @ dagger(S) @ Hw @ S @ U0_e
    fig,ax = plt.subplots(4,4)
    for i in range(d):
        for j in range(d):
            y = Hw[:,i,j]
            ax[i,j].plot(T,pt.real(y))
            ax[i,j].plot(T,pt.imag(y))

            
def plot_fidelity_progress(ax,fids,tN, legend=True):
    if len(fids.shape)==1:
        fids = fids.reshape(1,*fids.shape)
    nS=len(fids); N = len(fids[0])
    T = pt.linspace(0,tN/nanosecond, N)
    for q in range(nS):
        ax.plot(T,pt.real(fids[q]), label=f"System {q+1} fidelity")
    if legend: ax.legend()
    ax.set_xlabel("time (ns)")
    if annotate: ax.annotate("Fidelity progress", (0,0.95))
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
        ax.axhline(pt.real(E[i]/Mhz), label=f'E{dim-1-i}', color=colors[i])
