

import torch as pt

if not pt.cuda.is_available():
    # Set directory for vscode. Not necessary on gadi/pawsey and upsets pawsey for some reason.
    from pathlib import Path
    import os
    dir = os.path.dirname(__file__)
    os.chdir(dir)

import GRAPE as grape
from GRAPE import cplx_dtype, default_device, J_100_18nm, J_100_14nm, get_J
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

def sim2q_3q():

    J1 = get_J(2,2)
    A = grape.get_A(2,2)
    A[1]*=-1
    print(A)

    CX = gate.CX 
    Id = pt.eye(4)
    target1 = pt.stack((CX,Id))
    target2 = pt.stack((Id,CX))


    N=2000
    tN=90
    get_rec_min_N(A,J1,tN)   


    grape.optimise2(target1, N, tN, J1, A, show_plot=True, save_data=True, max_time=None)
    grape.optimise2(target2, N, tN, J1, A, show_plot=True, save_data=True, max_time=None)





def run_CNOTs(tN,N, nq=3,nS=15, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False):

    if A is None: A = grape.get_A(nS,nq)
    if J is None: J = get_J(nS,nq)
    target = grape.CNOT_targets(nS,nq)
    if not minprint: get_rec_min_N(A,J,tN)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None

    grape.optimise2(target, N, tN, J, A, u0=u0, rf=rf, show_plot=show_plot, save_data=save_data, max_time=max_time,NI_qub=True,hist0=hist0, minprint=minprint, mergeprop=mergeprop)




if __name__ == '__main__':

    #memory_requirement(225,3,2000)

    nS=25; nq=3
    rf = grape.get_RFs(grape.get_A(nS,nq), get_J(nS,nq))

    #print(count_RFs(225,3))
    #get_rec_min_N(grape.get_A(225,3),get_J(225,3),500)
    init_u_fn="p32_225S_3q_300ns_3000step"
    run_CNOTs(500.0, 5000, nq=nq, nS=nS, max_time = 60, kappa=1, rf=None, save_data=True, init_u_fn=None, mergeprop=False)


    #fn='g113_15S_2q_220ns_400step'
    #grape.process_u_file(fn)









def cull_freqs(rf):
    pdb.set_trace()


