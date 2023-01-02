

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
from visualisation import visualise_Hw, plot_psi, show_fidelity, fidelity_bar_plot
from hamiltonians import get_U0, get_H0, get_X_from_H
from GRAPE import GrapeESR, CNOT_targets, CXr_targets, GrapeESR_AJ_Modulation, load_grape
from electrons import get_electron_X

from pulse_maker import get_smooth_E
from pdb import set_trace

    


def grape_48S_fid_bars():
    grape = load_grape('fields-gadi/fields/g228_48S_2q_5000ns_10000step', Grape=GrapeESR_AJ_Modulation)
    fids = grape.fidelity()[0]
    fidelity_bar_plot(fids)

def run_2P_1P_CNOTs(tN,N, nS=15, Bz=0, max_time = 24*3600, lam=0, prev_grape_fn=None, verbosity=2, reverse_CX=False, kappa=1):

    nq = 2
    A = get_A_1P_2P(nS)
    J = get_J_1P_2P(nS)


    target = CNOT_targets(nS,nq)
    if reverse_CX:
        target = CXr_targets(nS)
    if prev_grape_fn is None:
        grape = GrapeESR_AJ_Modulation(J,A,tN,N, Bz=Bz, target=target, max_time=max_time, lam=lam, verbosity=verbosity)
    else:
        grape = load_grape(prev_grape_fn, max_time=max_time, Grape=GrapeESR_AJ_Modulation, verbosity=verbosity, kappa=kappa)
    #grape.plot_E_A_J();plt.show()
    grape.run()
    grape.plot_result()
    grape.print_result()
    grape.save()
    
    
if __name__ == '__main__':

    lam=1e11

    run_2P_1P_CNOTs(5000*unit.ns, 10000, nS=48, max_time = 23.5*3600, lam=1e8, prev_grape_fn='fields/g254_48S_2q_5000ns_10000step', reverse_CX=False, kappa=1e2)
    #run_2P_1P_CNOTs(500*unit.ns, 1000, nS=2, max_time = 10, lam=0, reverse_CX=True); plt.show()
    #grape_48S_fid_bars(); plt.show()
    



