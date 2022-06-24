

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt 
from atomic_units import *
from utils import get_nq


def plot_spin_states(psi, tN, ax=None):
    '''
    Plots the evolution of each component of psi.

    Inputs
        psi: (N,d) tensor where d is Hilbert space dimension and N is number of timesteps.
        tN: duration spanned by N timesteps
        ax: axis on which to plot
    '''
    if ax is None: ax = plt.subplot()
    N,dim=psi.shape
    nq=get_nq(dim)
    T=pt.linspace(0,tN/nanosecond,N)
    for i in range(dim):
        ax.plot(T,pt.abs(psi[:,i]), label = np.binary_repr(i,nq))
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