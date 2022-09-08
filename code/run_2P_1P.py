

from email.policy import default
import torch as pt
import numpy as np
import matplotlib
import pickle


if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


import gates as gate
import atomic_units as unit
from data import *
from utils import *
from eigentools import *
from data import get_A, get_J, J_100_18nm, J_100_14nm, cplx_dtype, default_device, gamma_e, gamma_n
from visualisation import visualise_Hw, plot_E_A_J, plot_psi, show_fidelity, fidelity_bar_plot
from hamiltonians import get_U0, get_H0, get_X_from_H
from GRAPE import GrapeESR, CNOT_targets, GrapeESR_AJ_Modulation, load_grape
from electrons import get_electron_X

from pulse_maker import get_smooth_E
from pdb import set_trace

    


def grape_48S_fid_bars():
    grape = load_grape('fields-gadi/fields/g228_48S_2q_5000ns_10000step', Grape=GrapeESR_AJ_Modulation)
    fids = grape.fidelity()[0]
    fidelity_bar_plot(fids)

def run_2P_1P_CNOTs(tN,N, nS=15, Bz=0, max_time = 24*3600, save_data=False, lam=0, prev_grape_fn=None):

    nq = 2


    E = get_smooth_E(tN, N)

    A = get_A_1P_2P(nS, NucSpin=[0,0])
    J = get_J_1P_2P(nS)

    # T = linspace(0,tN,N)
    # plot_E_A_J(T,E,A,J);plt.show()

    target = CNOT_targets(nS,nq)
    if prev_grape_fn is None:
        grape = GrapeESR_AJ_Modulation(J,A,tN,N, Bz=Bz, target=target, max_time=max_time, save_data=save_data, lam=lam, verbosity=1)
    else:
        grape = load_grape(prev_grape_fn, max_time=max_time, Grape=GrapeESR_AJ_Modulation)
    grape.run()
    #grape.plot_result()
    grape.print_result()
    grape.save()
    
    
if __name__ == '__main__':

    lam=1e11

    #run_2P_1P_CNOTs(5000*unit.ns, 10000, nS=2, max_time = 60, lam=1e8, save_data=True, prev_grape_fn='fields-gadi/fields/g228_48S_2q_5000ns_10000step')
    #run_2P_1P_CNOTs(200*unit.ns, 500, nS=1, max_time = 10, lam=0)
    #grape_48S_fid_bars(); plt.show()
    



