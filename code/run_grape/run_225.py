

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
from GRAPE import GrapeESR, CNOT_targets, GrapeESR_AJ_Modulation, load_grape
from electrons import get_electron_X

from pulse_maker import get_smooth_E
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
    






def sum_grapes(grapes):

    grape=grapes[0].copy()
    nG = len(grapes)

    m = sum([grape.m for grape in grapes])
    N = grape.N
    u_sum = zeros_like_reshape(grapes[0].u, (m, N))
    rf = zeros_like_reshape(grapes[0].rf, (m//2,))

    k0 = 0
    for grape in grapes:
        kmid = grape.m//2
        k1 = k0 + kmid
        u_sum[k0:k1] = grape.u_mat()[:kmid]
        u_sum[m//2+k0 : m//2+k1] = grape.u_mat()[kmid:]
        rf[k0:k1] = grape.rf
        k0=k1
    u_sum = uToVector(u_sum)


    grape.u = u_sum 
    grape.rf = rf
    grape.m=m; grape.N=N
    grape.initialise_control_fields()

    
    return grape

    
def run_CNOTs(tN,N, nq=3,nS=15, Bz=0, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, prev_grape_fn=None, kappa=1, minprint=False, mergeprop=False, lam=0, alpha=0):

    div=1
    Jmax = 1.87*unit.MHz
    J1_low = J_100_18nm/10
    J2_low = J_100_18nm * Jmax / pt.max(pt.real(J_100_18nm))
    if A is None: A = get_A(nS, nq)
    if J is None: J = get_J(nS, nq, J1=J1_low, J2=J2_low)/div
    H0 = get_H0(A, J)
    H0_phys = get_H0(A, J, B0)
    S,D = get_ordered_eigensystem(H0, H0_phys)

    #rf,u0 = get_low_J_rf_u0(S, D, tN, N)
    rf=None; u0=None

    target = CNOT_targets(nS,nq, native=True)
    if prev_grape_fn is None:
        grape = GrapeESR(J,A,tN,N, Bz=Bz, target=target,rf=rf,u0=u0, max_time=max_time, lam=lam, alpha=alpha)
    else:
        grape = load_grape(prev_grape_fn, max_time=max_time)
    #grape.run()
    grape.print_result(2)
    grape.plot_result()
    if save_data:
        grape.save()



    
if __name__ == '__main__':

    lam=1e11


    #run_CNOTs(tN = 1000.0*unit.ns, N = 500, nq = 3, nS = 45, max_time = 60, save_data = True, lam=0, prev_grape_fn='fields/g251_225S_3q_5000ns_5000step')
    run_CNOTs(tN = 5000.0*unit.ns, N = 5000, nq = 3, nS = 225, max_time = 23.5*3600, save_data = True, lam=1e9, alpha=0, prev_grape_fn='fields/g251_225S_3q_5000ns_5000step')
    plt.show()
    
    











