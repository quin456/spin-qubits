

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


def count_RFs(nS,nq):
    return len(grape.get_RFs(get_A(nS,nq), get_J(nS,nq)))


def memory_requirement(nS,nq,N):
    rf = grape.get_RFs(get_A(nS,nq), get_J(nS,nq))
    m = 2*len(rf)
    bytes_per_cplx = 16

    # x_cf, y_cf are (nS,m,N) arrays of 128 bit 
    print("> {:.5e} bytes required".format(nS*m*N * bytes_per_cplx))



################################################################################################################
################        EXPRIMENTS        ######################################################################
################################################################################################################

def sim2q_3q():

    J1 = get_J(2,2)
    A = get_A(2,2)
    A[1]*=-1
    print(A)

    CX = gate.CX 
    Id = pt.eye(4)
    target1 = pt.stack((CX,Id))
    target2 = pt.stack((Id,CX))


    N=2000
    tN=90

    grape.optimise2(target1, N, tN, J1, A, show_plot=True, save_data=True, max_time=None)
    grape.optimise2(target2, N, tN, J1, A, show_plot=True, save_data=True, max_time=None)





def run_CNOTs(tN,N, nq=3,nS=15, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False):

    if A is None: A = get_A(nS,nq)
    if J is None: J = get_J(nS,nq)

    target = CNOT_targets(nS,nq)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None
   
    grape = GrapeESR(J,A,tN,N,rf,target,u0,hist0, save_data=save_data)
    grape.run()
    grape.result(show_plot=show_plot,minprint=minprint)



if __name__ == '__main__':
    run_CNOTs(
        tN = 100.0*nanosecond, 
        N = 500, 
        nq = 2, 
        nS = 1, 
        max_time = 5, 
        kappa = 1, 
        rf = None, 
        save_data = True, 
        init_u_fn = None, 
        mergeprop = False
        )










