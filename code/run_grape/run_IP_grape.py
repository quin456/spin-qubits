

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
from GRAPE import GrapeESR, CNOT_targets, GrapeESR_AJ_Modulation, load_grape, GrapeESR_IP
from electrons import get_electron_X

from pulse_maker import get_smooth_E
from pdb import set_trace





def run_IP_CNOTs(tN,N, nq=3,nS=15, Bz=0, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, prev_grape_fn=None, kappa=1, minprint=False, mergeprop=False, lam=0, alpha=0):

    div=1

    if A is None: A = get_A(nS, nq)
    if J is None: J = get_J(nS, nq, J1=J_100_18nm, J2=J_100_14nm[8:])/div

    H0 = get_H0(A, J)
    H0_phys = get_H0(A, J, B0)
    S,D = get_ordered_eigensystem(H0, H0_phys)

    #rf,u0 = get_low_J_rf_u0(S, D, tN, N)
    rf=None; u0=None

    target = CNOT_targets(nS,nq, native=True)
    if prev_grape_fn is None:
        grape = GrapeESR_IP(J,A,tN,N, Bz=Bz, target=target,rf=rf,u0=u0, max_time=max_time, lam=lam, alpha=alpha)
    else:
        grape = load_grape(prev_grape_fn, max_time=max_time)
    grape.run()
    grape.plot_result()
    if save_data:
        grape.save()
    
if __name__ == '__main__':

    lam=1e11

    #run_CNOTs(tN = 100.0*unit.ns, N = 500, nq = 2, nS = 1, max_time = 18000, save_data = True, lam=1e8, prev_grape_fn=None)
    run_IP_CNOTs(tN = 200.0*unit.ns, N = 500, nq = 3, nS = 2, max_time = 10, save_data = True, lam=0, alpha=0, prev_grape_fn=None)
    plt.show()
    
    











