

import torch as pt
import numpy as np
import matplotlib

if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle

import GRAPE
from GRAPE import GrapeESR, CNOT_targets, GrapeESR_AJ_Modulation
import gates as gate 
from atomic_units import *
from data import get_A, get_J, J_100_18nm, J_100_14nm, cplx_dtype, default_device, gamma_e, gamma_n
from visualisation import visualise_Hw
from utils import dagger
from hamiltonians import get_U0

from pdb import set_trace






def inspect_system():
    J = get_J(3,3)[2:3]
    J[0,0]/=5
    tN = 200.0*nanosecond
    N = 1500
    nq = 3
    nS = 1
    max_time = 15
    kappa = 1
    rf = None
    save_data = True
    init_u_fn = None
    mergeprop = False
    A = get_A(1,3, NucSpin=[-1,-1,-1])*0

    grape = GrapeESR(J,A,tN,N, Bz=0.02*tesla, max_time=max_time, save_data=save_data)
    Hw=grape.get_Hw()

    eigs = pt.linalg.eig(grape.H0)
    E = eigs.eigenvalues[0] 
    S = eigs.eigenvectors[0] 
    D = pt.diag(E)
    UD = get_U0(D, tN, N)
    Hwd = dagger(UD)@S.T@Hw[0]@UD
    visualise_Hw(Hw[0],tN)
    set_trace()

    plt.show()
    

def run_CNOTs(tN,N, nq=3,nS=15, Bz=0, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False):

    if A is None: A = get_A(nS,nq, N=N, donor_composition=[1,2])
    if J is None: J = get_J(nS,nq, N=N)

    target = CNOT_targets(nS,nq)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None
   
    grape = GrapeESR_AJ_Modulation(J,A,tN,N, Bz=Bz, target=target,rf=rf,u0=u0, max_time=max_time, save_data=save_data)




    grape.run()
    grape.plot_result()


if __name__ == '__main__':
    run_CNOTs(
        tN = 100.0*nanosecond, 
        N = 3000, 
        nq = 2, 
        nS = 3, 
        max_time = 14, 
        kappa = 1, 
        rf = None, 
        save_data = True, 
        init_u_fn = None, 
        mergeprop = False
        )
    












