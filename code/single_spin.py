
from tokenize import Single
import torch as pt 
import matplotlib



matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

import gates as gate
from pulse_maker import pi_rot_square_pulse
from atomic_units import *
from gates import spin_up, spin_down
from visualisation import plot_fields, plot_spin_states, show_fidelity
from data import gamma_e, dir
from utils import get_pulse_hamiltonian, sum_H0_Hw, get_U0, forward_prop
from GRAPE import GRAPE

plots_folder = f"{dir}thesis-plots/"

def single_spin_H0(Bz, gamma=gamma_e):
    return 0.5*gamma_e*Bz*gate.Z


def show_single_spin_evolution(Bz = 2*tesla, tN = 1*nanosecond, N=100000, target = gate.X, psi0=spin_up, fn=None):

    w_res = gamma_e*Bz
    fig,ax = plt.subplots(3,1)
    Bx,By = pi_rot_square_pulse(w_res, gamma_e/2, tN, N)
    plot_fields(Bx,By,tN,ax[0])

    H0 = single_spin_H0(Bz)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    U0 = get_U0(H0, tN, N)

    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    show_fidelity(X,tN,gate.X, ax[1])
    psi = X@psi0 

    def label_getter(j):
        if j==0: return '$|<0|\psi>|$'
        return '$|<1|\psi>|$'
    plot_spin_states(psi, tN, ax[2], label_getter=label_getter)
    plt.tight_layout()

    if fn is not None:
        plt.savefig(f"{plots_folder}{fn}")


class SingleElectronGRAPE(GRAPE):

    def __init__(self, tN,N,target):

        self.fun = lambda u: self.cost()


    def cost(u):
        pass



def run_single_electron_grape():
    target = gate.X 
    tN = 10*nanosecond 
    N = 1000

    grape = SingleElectronGRAPE(tN,N,target)





if __name__ == '__main__':
    show_single_spin_evolution(fn = "Ch1-single-spin-flip.pdf"); plt.show()
