

import torch as pt 
import matplotlib
import numpy as np
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 



import gates as gate
from pulse_maker import pi_pulse_square
import atomic_units as unit
from gates import spin_up, spin_down
from visualisation import plot_fields, plot_psi, show_fidelity, plot_phases
from data import gamma_e, dir, cplx_dtype
from utils import forward_prop
from eigentools import lock_to_frequency
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, get_U0
from data import get_A
from GRAPE import Grape
from hamiltonians import single_electron_H0
from visualisation import *

from pdb import set_trace


def label_getter(j):
    if j==0: return '$|<0|\psi>|$'
    return '$|<1|\psi>|$'



def show_single_spin_evolution(Bz = 0*unit.T, A=get_A(1,1), tN = 500*unit.ns, N=100000, target = gate.X, psi0=spin_up, fp=None):

    tN = lock_to_frequency(A, tN)

    w_res = gamma_e*Bz + 2*A
    #fig,ax = plt.subplots(2,2)
    fig,ax = plt.subplots(3,1)
    Bx,By = pi_pulse_square(w_res, gamma_e/2, tN, N)
    plot_fields(Bx,By,tN,ax[0])

    H0 = single_electron_H0(Bz, A)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    U0 = get_U0(H0, tN=tN, N=N)

    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    show_fidelity(X,tN=tN,target=gate.X, ax=ax[1])
    psi = X@psi0 

    plot_psi(psi, tN=tN, ax=ax[2], label_getter=label_getter)
    #plot_phases(psi, tN, ax[1,1])
    plt.tight_layout()

    if fp is not None:
        plt.savefig(fp)


class SingleElectronGRAPE(Grape):
    def __init__(self, tN, N, target, rf=None, u0=None, hist0=[], max_time=60, save_data=False, Bz=0, A=get_A(1,1), lam=0):
        self.nq = 1
        self.nS = 1
        self.Bz=Bz
        self.A=A
        super().__init__(tN,N,target, rf=rf, nS=1, u0=u0, max_time=max_time, kappa=1e12, lam=lam)
        self.rf = self.get_all_resonant_frequencies() if rf is None else rf
        self.Hw=self.get_Hw()
        self.fun = self.cost



    def get_H0(self, Bz=0):
        H0 = single_electron_H0(Bz, self.A)
        return H0.reshape(1,*H0.shape)

    def get_Hw(self):
        return get_pulse_hamiltonian(self.x_cf, self.y_cf, gamma_e)

    def get_all_resonant_frequencies(self):
        return pt.tensor([gamma_e*self.Bz + 2*self.A])


    def plot_result(self, psi0 = spin_up):
        fig,ax = plt.subplots(2,2)

        psi = self.X[0]@psi0 

        self.plot_u(ax[0,0])
        self.plot_control_fields(ax[0,1])

        #plot_psi(psi, tN=self.tN, ax=ax[1,1], label_getter=label_getter)
        Bx, By = self.sum_XY_fields()
        self.plot_XY_fields(ax[1,0], Bx, By)
        #plot_phases(psi, self.tN, ax[0])
        show_fidelity(self.X[0], tN=self.tN, target=self.target, ax=ax[1,1])
        fig.set_size_inches(1.1*fig_width_double, 1.1*0.8*fig_height_double_long)
        fig.tight_layout()

        x_offset=-0.11; y_offset=-0.25
        label_axis(ax[0,0], '(a)', x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[0,1], '(b)', x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[1,0], '(c)', x_offset=x_offset, y_offset=y_offset)
        label_axis(ax[1,1], '(d)', x_offset=x_offset, y_offset=y_offset)

        return ax



def run_single_electron_grape(fp=None):
    target = gate.H
    N = 300
    Bz=0
    A = get_A(1,1)
    tN = 50*unit.ns 
    u0 = 1.5*np.pi/( (gamma_e*unit.T)*tN) * pt.ones(2,N, dtype=cplx_dtype)*0.2

    grape = SingleElectronGRAPE(tN,N,target, Bz=Bz,u0=u0, lam=1e5)
    grape.run()
    grape.print_result()
    ax=grape.plot_result()

    if fp is not None: ax[0,0].get_figure().savefig(fp)



if __name__ == '__main__':

    #show_single_spin_evolution(N=500, tN=100*unit.ns); plt.show()
    run_single_electron_grape()
