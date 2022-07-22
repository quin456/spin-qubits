

import torch as pt 
import matplotlib





matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

import gates as gate
from pulse_maker import pi_rot_square_pulse
from atomic_units import *
from gates import spin_up, spin_down
from visualisation import plot_fields, plot_spin_states, show_fidelity, plot_phases
from data import gamma_e, dir, cplx_dtype
from utils import forward_prop, lock_to_coupling
from hamiltonians import get_pulse_hamiltonian, sum_H0_Hw, get_U0
from data import get_A
from GRAPE import Grape
from hamiltonians import single_electron_H0


from pdb import set_trace


def label_getter(j):
    if j==0: return '$|<0|\psi>|$'
    return '$|<1|\psi>|$'



def show_single_spin_evolution(Bz = 0*tesla, A=get_A(1,1), tN = 500*nanosecond, N=100000, target = gate.X, psi0=spin_up, fp=None):

    tN = lock_to_coupling(A, tN)

    w_res = gamma_e*Bz + 2*A
    #fig,ax = plt.subplots(2,2)
    fig,ax = plt.subplots(3,1)
    Bx,By = pi_rot_square_pulse(w_res, gamma_e/2, tN, N)
    plot_fields(Bx,By,tN,ax[0])

    H0 = single_electron_H0(Bz, A)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    U0 = get_U0(H0, tN, N)

    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    show_fidelity(X,tN,gate.X, ax[1])
    psi = X@psi0 

    plot_spin_states(psi, tN, ax[2], label_getter=label_getter)
    #plot_phases(psi, tN, ax[1,1])
    plt.tight_layout()

    if fp is not None:
        plt.savefig(fp)


class SingleElectronGRAPE(Grape):
    def __init__(self, tN, N, target, rf=None, u0=None, hist0=[], max_time=60, save_data=False, Bz=0, A=get_A(1,1)):
        self.nq = 1
        self.nS = 1
        self.Bz=Bz
        self.A=A
        super().__init__(tN,N,target, rf, u0, hist0, max_time, save_data)
        self.Hw = self.get_Hw()
        self.rf = self.get_all_resonant_frequencies() if rf is None else rf

        self.fun = self.cost


    def get_H0(self):
        return single_electron_H0(self.Bz, self.A)

    def get_Hw(self):
        return get_pulse_hamiltonian(self.x_cf, self.y_cf, gamma_e)

    def get_all_resonant_frequencies(self):
        return pt.tensor([gamma_e*self.Bz + 2*self.A])


    def plot_result(self, u, X, psi0 = spin_up, show_plot=True):

        fig,ax = plt.subplots(3,1)

        psi = X[0]@psi0 


        plot_spin_states(psi, self.tN, ax[2], label_getter=label_getter)
        Bx, By = self.sum_XY_fields(self.u_mat())
        self.plot_XY_fields(ax[0], Bx, By)
        #plot_phases(psi, self.tN, ax[0])
        show_fidelity(X,self.tN,self.target,ax[1])
        plt.tight_layout()

        plt.savefig("thesis-plots/Ch1-numerical-example.pdf")
        plt.show()



def run_single_electron_grape():
    target = gate.X 
    N = 1000
    Bz=0
    A = get_A(1,1)
    tN = 100*nanosecond 
    tN = lock_to_coupling(A,tN)
    u0 = 1.5*np.pi/( (gamma_e*tesla)*tN) * pt.ones(2,N, dtype=cplx_dtype)

    grape = SingleElectronGRAPE(tN,N,target, Bz=Bz,u0=u0)
    #grape.run()
    grape.result()





if __name__ == '__main__':

    #show_single_spin_evolution(N=500, tN=100*nanosecond); plt.show()
    run_single_electron_grape()
