

import torch as pt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle

import GRAPE
from GRAPE import GrapeESR, CNOT_targets
import gates as gate 
from atomic_units import *
from data import get_A, get_J, J_100_18nm, J_100_14nm, cplx_dtype, default_device, gamma_e, gamma_n

from pdb import set_trace





def run_CNOTs(tN,N, nq=3,nS=15, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False):

    if A is None: A = get_A(nS,nq)
    if J is None: J = get_J(nS,nq)

    target = CNOT_targets(nS,nq)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None
   
    grape = GrapeESR(J,A,tN,N,target,rf,u0,hist0, max_time=max_time, save_data=save_data)
    grape.run()
    grape.result(show_plot=show_plot,minprint=minprint)



if __name__ == '__main__':
    run_CNOTs(
        tN = 100.0*nanosecond, 
        N = 500, 
        nq = 3, 
        nS = 2, 
        max_time = 9999, 
        kappa = 1, 
        rf = None, 
        save_data = True, 
        init_u_fn = None, 
        mergeprop = False
        )










