

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
from visualisation import visualise_Hw, plot_E_A_J, plot_psi, show_fidelity
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
    

def run_CNOTs(tN,N, nq=3,nS=15, Bz=0, max_time = 24*3600, J=None, A=None, save_data=True, show_plot=True, rf=None, init_u_fn=None, kappa=1, minprint=False, mergeprop=False, lam=0):

    div=10

    if A is None: A = get_A(nS, nq)
    if J is None: J = get_J(nS, nq)/div

    H0 = get_H0(A, J)
    H0_phys = get_H0(A, J, B0)
    S,D = get_ordered_eigensystem(H0, H0_phys)

    rf,u0 = get_low_J_rf_u0(S, D, tN, N)
    rf=None; u0=None

    target = CNOT_targets(nS,nq, native=True)
   
    grape = GrapeESR(J,A,tN,N, Bz=Bz, target=target,rf=rf,u0=u0, max_time=max_time, save_data=save_data, lam=lam)
    #fig,ax = plt.subplots(1,2)
    print_rank2_tensor(S)
    print_rank2_tensor(grape.X[0,-1])
    #show_fidelity(grape.X[0], tN=tN, ax=ax[0])


    
    grape.run()
    grape.plot_result()









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
        grape = load_grape(prev_grape_fn, max_time=max_time)
    grape.run()
    set_trace()
    grape.save()
    grape.plot_result() 




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

def save_grapes(J=get_J_1P_2P(48), A=get_A_1P_2P(48), tN=5000*unit.ns, N=5000, idxs = np.linspace(0,47,48).astype(int), max_time=60, lam=1e8):

    for i in idxs:
        grape = GrapeESR_AJ_Modulation(J[i], A[i], tN, N, Bz=0, target=CNOT_targets(1,2), max_time=max_time, lam=lam)
        grape.run()
        grape.save(f"grape_bunch/grape{i}")


def test_sum(tN = 5000*unit.ns, N=5000, nS=2, max_time=15, div=1, lam=0, save_grapes=False):
    save_data=False
    nq=2
    target_single = CNOT_targets(1, nq)
    J = get_J_1P_2P(48)
    A = get_A_1P_2P(48)

    grapes = []

    if save_grapes:
        grape = GrapeESR_AJ_Modulation(J[0], A[0], tN, N, Bz=0, target=target_single, max_time=max_time, save_data=save_data, lam=lam)
        grape.run()
        grape.save("grape_bunch/grape0")
        grape = GrapeESR_AJ_Modulation(J[1], A[1], tN, N, Bz=0, target=target_single, max_time=max_time, save_data=save_data, lam=lam)
        grape.run()
        grape.save("grape_bunch/grape1")
    
    grape0 = load_grape("grape_bunch/grape0", GrapeESR_AJ_Modulation)
    grape0.print_result()
    grape1 = load_grape("grape_bunch/grape1", GrapeESR_AJ_Modulation)
    grapes=[grape0, grape1]

    grape = sum_grapes(grapes)
    grape.plot_result()

    for i in range(48):
        grape.J = J[i]/div
        grape.A = A[i]
        grape.H0=grape.get_H0()
        #X_free = get_electron_X(grape.tN, grape.N, 0, grape.A, grape.J)
        X_free = get_X_from_H(grape.H0, tN, N)
        grape.propagate()


        print(f"CX, free fids: {fidelity(grape.X[0,-1], gate.CX):.3f} {fidelity(grape.X[0,-1], X_free[0,-1]):.3f}")



if __name__ == '__main__':

    lam=1e11

    run_2P_1P_CNOTs(5000*unit.ns, 5000, nS=2, max_time = 1, lam=1e8, save_data=True, prev_grape_fn='fields/c813_2S_2q_5000ns_5000step')
    #run_2P_1P_CNOTs(200*unit.ns, 500, nS=1, max_time = 10, lam=0)


    #run_CNOTs(tN = 5000.0*unit.ns, N = 10000, nq = 2, nS = 1, max_time = 10, save_data = True, lam=lam)
    
    #test_sum(tN = 3000.0*unit.ns, N = 5000, lam=1e8, max_time=45)
    












