

from email.policy import default
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
import atomic_units as unit
from data import get_A, get_J, J_100_18nm, J_100_14nm, cplx_dtype, default_device, gamma_e, gamma_n
from visualisation import visualise_Hw, plot_E_A_J
from utils import *
from hamiltonians import get_U0

from pdb import set_trace






def inspect_system():
    J = get_J(3,3)[2:3]
    J[0,0]/=5
    tN = 200.0*unit.ns
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

    grape = GrapeESR(J,A,tN,N, Bz=0.02*unit.T, max_time=max_time, save_data=save_data)
    Hw=grape.get_Hw()

    eigs = pt.linalg.eig(grape.H0)
    E = eigs.eigenvalues[0] 
    S = eigs.eigenvectors[0] 
    D = pt.diag(E)
    UD = get_U0(D, tN, N)
    Hwd = dagger(UD)@S.T@Hw[0]@UD
    visualise_Hw(Hw[0],tN)

    plt.show()
    

def run_CNOTs(tN,N, nq=3,nS=15, Bz=0, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False):

    if A is None: A = get_A(nS,nq, N=1, donor_composition=[1,1])
    if J is None: J = get_J(nS,nq, N=1)

    target = CNOT_targets(nS,nq)
    if init_u_fn is not None:
        u0,hist0 = grape.load_u(init_u_fn); hist0=list(hist0)
    else:
        u0=None; hist0=None
   
    grape = GrapeESR(J,A,tN,N, Bz=Bz, target=target,rf=rf,u0=u0, max_time=max_time, save_data=save_data)

    grape.run()
    grape.plot_result()


def sigmoid(z):
    return 1 / (1 + pt.exp(-z))


def get_smooth_E(tN, N, rise_time = 1*unit.ns):
    E_mag = 0.7*unit.MV/unit.m
    T = pt.linspace(0,tN,N,device=default_device)
    E = E_mag * (sigmoid((T-10*unit.ns)/rise_time) - sigmoid((T-T[-1]+10*unit.ns)/rise_time))
    return E




def run_2P_1P_CNOTs(tN,N, nS=15, Bz=0, max_time = 24*3600, save_data=False):

    nq = 2

    E = get_smooth_E(tN, N)

    A = get_A_1P_2P(nS, N, E)
    J = get_J_1P_2P(nS, N, E)/50

    # T = linspace(0,tN,N)
    # plot_E_A_J(T,E,A,J);plt.show()

    target = CNOT_targets(nS,nq)
    grape = GrapeESR_AJ_Modulation(J,A,tN,N, Bz=Bz, target=target, max_time=max_time, save_data=save_data)

    grape.run()
    grape.plot_result() 


if __name__ == '__main__':


    run_2P_1P_CNOTs(3000*unit.ns, 7500, nS=1, max_time = 10)



    # run_CNOTs(
    #     tN = 200.0*unit.ns, 
    #     N = 400, 
    #     nq = 2, 
    #     nS = 1, 
    #     max_time = 14, 
    #     kappa = 1, 
    #     rf = None, 
    #     save_data = True, 
    #     init_u_fn = None, 
    #     mergeprop = False
    #     )
    












