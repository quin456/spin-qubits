

import GRAPE as grape
from GRAPE import cplx_dtype, default_device, J_100_18nm, J_100_14nm, get_J
import torch as pt
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle

import gates as gate 
from atomic_units import *


A=grape.A_kane; A1=A[0]; A2=A[1]
A = pt.tensor(len(J_100_14nm)*[[A1, A2]], dtype=gate.real_dtype, device=default_device)
Mhz=1e6*grape.hz

#gyromagnetic ratios (MHz/T)
gamma_p = 17.2 
gamma_e = 28025


def get_rec_min_N(A,J,tN, printFreqs=False, printPeriods=False):

    N_period=40 # recommended min number of timesteps per period
    rf=grape.get_RFs(A,J)
    T=1e3/rf
    max_w=pt.max(rf).item()
    rec_min_N = int(np.ceil(N_period*max_w*Mhz*tN*nanosecond/(2*np.pi)))
    
    if printFreqs: print(f"resonant freqs = {rf}")
    if printPeriods: print(f"T = {T}")
    print(f"Recommened min N = {rec_min_N}")

    return rec_min_N

def count_RFs(nS,nq):
    return len(grape.get_RFs(grape.get_A(nS,nq), grape.get_J(nS,nq)))

def recommended_steps(nS,nq,tN):
    A=grape.get_A(nS,nq)
    J=get_J(nS,nq)
    return get_rec_min_N(A,J,tN)

def memory_requirement(nS,nq,N):
    rf = grape.get_RFs(grape.get_A(nS,nq), grape.get_J(nS,nq))
    m = 2*len(rf)
    bytes_per_cplx = 16

    # x_cf, y_cf are (nS,m,N) arrays of 128 bit 
    print("> {:.5e} bytes required".format(nS*m*N * bytes_per_cplx))



################################################################################################################
################        EXPRIMENTS        ######################################################################
################################################################################################################
def run_CNOTs(tN,N, nq=3,nS=15, max_time = 8*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, minprint=False):

    if A is None: A = grape.get_A(nS,nq)
    if J is None: J = get_J(nS,nq)
    target = grape.CNOT_targets(nq,nS)
    get_rec_min_N(A,J,tN)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None

    grape.optimise2(target, N, tN, J, A, u0=u0, rf=rf, show_plot=show_plot, save_data=save_data, max_time=max_time,NI_qub=True,hist0=hist0, minprint=minprint)




if __name__ == '__main__':

    #memory_requirement(225,3,2000)

    nS=225; nq=3

    #print(count_RFs(225,3))
    init_u_fn="g212_225S_3q_150ns_6000step"
    run_CNOTs(1000.0, 6000, nq=nq, nS=nS, max_time = 23.5*3600, rf=None, save_data=True, init_u_fn=None)


    #fn='g113_15S_2q_220ns_400step'
    #grape.process_u_file(fn)









def cull_freqs(rf):
    pdb.set_trace()


