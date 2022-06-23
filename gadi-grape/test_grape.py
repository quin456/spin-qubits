gp=None

from GRAPE import ignore_tensor, grad, default_device


# set directory
from pathlib import Path
import os
dir = os.path.dirname(__file__)
os.chdir(dir)

import GRAPE as grape
import torch as pt
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
import pickle
import math 
from datetime import datetime
from atomic_units import *
import gates as gate
from run_grape import get_J
device = 'cuda' if pt.cuda.is_available() else 'cpu'
cplx_dtype=grape.cplx_dtype
exch_filename = "exchange_data_updated.p"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18']) 
J_100_14nm = pt.tensor(exch_data['100_14']) 





def test_gpu_fid():
    ngpus = pt.cuda.device_count()
    N=1000
    tN=90.0
    nS=2
    nq=2
    A = grape.get_A(nS,nq)
    J = get_J(nS,nq)
    target = grape.CNOT_targets(nq,nS)

    omega,phase = grape.get_CF_params(J,A)
    m=len(omega)
    u = grape.init_u(tN,m,N,device=default_device)
    x_cf,y_cf = grape.get_control_fields(omega,phase,tN,N)
    H0 = grape.multi_gpu_H0(A,J) 

    MP = grape.Multi_Processes(ngpus, grape.fidelity_batch)

    t1=time.time()
    fid=grape.flexi_fid(u,H0,nS,J,A,x_cf,y_cf,tN,target,MP)
    print(fid)
    print(f"Time taken = {time.time()-t1}, gpus requested = {pt.cuda.device_count()}")

if  __name__=='__main__':
    test_gpu_fid()




def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def get_J(nS,nq,J1=J_100_18nm,J2=J_100_18nm):
    if nq==2:
        return J1[:nS]
    elif nq==3:
        return grape.all_J_pairs(J1,J2)[:nS]


def test_multi_gpu():
    print(grape.get_gpu_systems(8, 225))


#test_multi_gpu()


def compare_grapes():
    nq=2
    nS=15
    N=420
    tN=220
    A = grape.get_A(nS,nq)
    J = get_J(nS,nq)
    target = grape.CNOT_targets(nq,nS)
    max_time=60

    omega,phase = grape.get_CF_params(J,A)
    m=len(omega)

    u0=None 
    rf=None 

    grape.optimise2(target, N, tN, J, A, u0=u0, rf=rf, show_plot=True, save_data=False, max_time=max_time)
    gp.optimiseFields(target,N,tN,J,A,save_data=False,max_time=max_time)

    x_cf,y_cf = grape.get_control_fields(omega,phase,tN,N)
    H0 = grape.reduced_H0_array(A,J)
    u=pt.rand(m*N)

    Jn,dJn=grape.fast_cost(u,target,H0,x_cf,y_cf,tN,[[],[]])

    H0p = grape.make_H0_array(A,J,N)
    Hw = grape.make_Hw_array(omega,nq,N,nS,tN,phase)
    Jp,dJp=grape.cost(u,target,Hw,tN,H0p)

    print(dJn/dJp)




def test_grad():
    N=50;m=8
    tN=20
    nS=1; nq=2
    J = J_100_18nm[0:1]
    A = grape.get_A(nS)
    u0=pt.ones((m,N))* np.pi/(tN*J[0])/4
    u0=pt.rand((m,N))* np.pi/(tN*J[0])/4
    u0=grape.uToVector(u0)
    target = grape.CNOT_targets(nS)

    S=grape.get_S_matrix(J,A)
    H0 = grape.make_H0_matrix(A*Mhz,J*Mhz,N,S=S)
    rf = grape.get_2q_freqs(J,A)
    omega,phase = grape.config_90deg_phase_fields(rf)

    Hw = grape.make_Hw_matrix(omega, nq, N, nS, tN, phase,H0=H0,S=S)


    fun = lambda u: grape.cost(u,target,Hw,tN,H0,jac=True, kap=1)
    ga = fun(u0)[1]

    fun = lambda u: grape.cost(u,target,Hw,tN,H0,jac=False)
    gn = grad(fun,u0,5e-2)

    print(gn)

    print(ga/gn)


#test_grad()


def test_Hw():
    N=1000
    tN=50
    nq=2;nS=1
    J=pt.tensor([1e2])
    A = grape.get_A(nS)
    rf=grape.get_2q_freqs(J*Mhz,A*Mhz)
    omega,phase = grape.config_90deg_phase_fields(rf)
    S=grape.get_S_matrix(J,A)
    H0 = grape.make_H0_matrix(A*Mhz,J*Mhz,N,S=S)

    ignore=[['13','34'],['12','24','34'],['12','13','24'],['13','34']]

    Hw = grape.make_Hw_matrix(omega,nq,N,nS,tN*nanosecond,phase=phase,S=S,H0=H0, ignore_transitions=ignore)

    grape.visualise_Hw(Hw,tN,N)

#test_Hw()

def get_rec_min_N(A,J,tN):
    N_period=40
    H0 = grape.make_H0(A[0],J[0])
    omega=grape.getFreqs(H0,gate.X2)
    max_w=pt.max(pt.real(omega)).item()
    rec_min_N = int(np.ceil(N_period*max_w*tN/(2*np.pi)))
    print(f"Recommened min N = {rec_min_N}")
    return rec_min_N


A = grape.A_kane.reshape((1,2)) 
J = J_100_18nm[0:1]
#print(get_2q_freqs(J,A[0]))
#print(grape.multiSystemFreqs(A,J))


def pleasepleasepleasework():
    J = pt.tensor([J_100_18nm[0],J_100_14nm[0]])
    A1 = grape.A_kane[0]; A2=grape.A_kane[1]
    A = pt.tensor([[A1,A2],[A2,A1]])
    target = pt.stack((gate.CX, gate.Id2))
    N=6000
    #tN=1.5705e+03
    tN=625.88
    rf = grape.get_2q_freqs(J[0:2],A[0:2])
    #rf=pt.cat((rf,1.3*rf,1.55*rf))
    T = 1e3/rf 
    print(f"T={T} (ns)")
    get_rec_min_N(A*Mhz,J*Mhz,tN*nanosecond)
    #grape.optimiseFields(target[0:2],N,tN,J[0:2],A[0:2], rf=rf*Mhz)

#pleasepleasepleasework()


def freebie():
    nS=1
    A=grape.get_A(nS)*0
    #A=pt.zeros(*A.shape)
    J=J_100_18nm[0:nS]
    tN=1e3*1/(8*J[0])*20
    print(f"J={J}")
    target=grape.CNOT_targets(nS)
    finalfid=grape.free_evolution_fidelity(J,A,tN,1000,target)    
    print(f"finalfid={finalfid[-1]}")
#freebie()

def freebie3():
    tN=10
    A1 = grape.A_kane[0]; A2=grape.A_kane[1]
    A = pt.tensor([[A1,A2,A1]])
    J=pt.tensor([[J_100_18nm[0],J_100_14nm[0]]])
    finalfid=grape.free_evolution_fidelity(J,A,tN,1000,pt.kron(gate.root_swap,gate.Id).reshape(1,8,8))    
    print(f"finalfid={finalfid[-1]}")
#freebie3()


def freepsi3():
    tN=30
    N=1000
    d=8
    A1 = grape.A_kane[0]; A2=grape.A_kane[1]
    A = pt.tensor([[A1,A2,A1]])
    J=pt.tensor([[J_100_18nm[0],J_100_18nm[1]]])
    H0=grape.make_H0(A[0]*Mhz,J[0]*Mhz)
    print(f"J={J}")
    psi0 = grape.normalise(pt.tensor([0,1,0,0,0,0,0,0], dtype=grape.cplx_dtype))
    T = pt.linspace(0,tN,N)
    psi=grape.wf_freeEvolution(H0,J,A,psi0,tN,N)
    for i in range(d):
        plt.plot(T,pt.multiply(psi[:,i],pt.conj(psi[:,i])),label='|<'+np.binary_repr(i,3)+'|$\psi$>|$^2$')
    plt.legend()
    plt.xlabel("time (ns)")
    plt.show()

#freepsi3()


def testfids():
    nq=2 ; nS=1
    target = gate.CX
    tN = 100*nanosecond 
    J = pt.tensor([97*Mhz]); A=grape.get_A(nS)*Mhz
    omega = grape.get_2q_freqs(J,A)
    m=len(omega)
    N=1000
    u = grape.uToVector(grape.init_u(tN,m,N))
    Hw = grape.make_Hw_matrix(omega,nq,N,nS,tN)
    H0 = grape.make_H0_matrix(A,J,N)
    fid,dfid=grape.fidelity(u,m,nq,target,Hw,tN,H0)
    #print(fid)

    Hwp = gp.make_Hw(omega,nq,tN,N)
    H0p=gp.make_H0(A[0],J[0])
    fidp,dfidp = gp.fidelity(u,m,nq,target,Hwp,tN,H0p)
    dfidp = gp.uToMatrix(dfidp,m)

    print((dfid/dfidp))


#testfids()




def test_freqs():
    N=1
    nS=len(J_100_14nm)
    J = J_100_18nm[0:nS]
    A = nS*[grape.A_kane]
    H0 = grape.make_H0_matrix(A,J,N)
    #w = grape.multiSystemFreqs(A,J)
    w=grape.get_2q_freqs(J[0:1],A[0])
    T=2*np.pi/w/grape.second

    print(T)

    w_sorted=pt.sort(w*1e9).values
    #print(w_sorted[len(w_sorted)//2:])
    #print(T[0]*26)
    
    #print(w*grape.second)
    #print(2*np.pi/w/grape.second)
    tints = list(set(pt.round(pt.abs(T)*1e10).int().tolist()))
    #print(tints)
    #print(lcm(tints[0],tints[1]))
    
#test_freqs()


def whyeigdif():

    J = J_100_18nm[0:1]
    
    A_kane = pt.tensor([58.5/2, -58.5/2], device=device) * 1e6*grape.hz
    A_kanet = pt.tensor([58.5, 0], device=device) * 1e6*grape.hz

    H0 = grape.make_H0(A_kane,J)
    #print(pt.eig(H0).eigenvalues)
    H0t = grape.make_H0(A_kanet,J)
    #print(pt.eig(H0t).eigenvalues)
    #print(H0)
    #print(H0t)
    print(grape.get_2q_freqs(J,A[0]))
    print(grape.multiSystemFreqs(A,J))

#whyeigdif()


def plot_evol():
    J = J_100_18nm[2]
    A = grape.A_kane

    H0 = grape.make_H0(A,J)
    w = grape.getFreqs(H0, grape.get_Xn(2))
    n=30000
    tN=200e-9*grape.second
    T=pt.linspace(0,tN,n)
    psi0 = grape.normalise(pt.tensor([1,4,2,5], dtype=grape.cplx_dtype))
    psi = grape.freeEvolution(H0,psi0,tN,n)

    plt.plot(T,psi[:,0], label='<00|$\psi$>')
    plt.plot(T,psi[:,1], label='<01|$\psi$>')
    plt.plot(T,psi[:,2], label='<10|$\psi$>')
    plt.plot(T,psi[:,3], label='<11|$\psi$>')
    plt.plot(T,pt.einsum('iq,iq->i',psi, pt.conj(psi)), label='$\psi^2$')

    # plt.plot(T,pt.sin(w[0]*T), label='w1')
    # plt.plot(T,pt.sin(w[2]*T), label='w2')

    #plt.plot(T,pt.sin(np.pi*A[0]*T), label='A1')
    #plt.plot(T,pt.sin(2*np.pi*A[1]*T), label='A2')

    plt.legend()
    plt.show()
    







def test_3q_CNOT():
    nq=3 
    nS=1
    target = grape.CX3q_gate
    N=400
    A=[grape.A_kane3 ]
    J = [pt.tensor([15, 30]) *1e6*grape.hz]
    tN=500e-9 * grape.second 
    w = grape.multiSystemFreqs(A,J)
    m=len(w)
    u = grape.init_u(tN,m,N)
    H0 = gp.make_H0(A[0],J[0])
    Hw = gp.make_Hw(w,nq,tN,N)
    histp=2*[[]]
    Jp,dJp = gp.cost(u,m,nq,target,Hw,tN,H0,histp)

    '''
    max_w=pt.max(pt.real(omega))
    rec_min_N = N_period*max_w*tN/(2*np.pi)
    print(f"Recommened min N = {rec_min_N}")
    '''
    
    hist=2*[[]]
    H0_matrix = grape.make_H0_matrix(A,J,N)
    Hw_matrix = grape.make_Hw_matrix(w,nq,N,nS,tN)

    J,dJ = grape.cost(u,m,nq,target,Hw_matrix,tN,H0_matrix,hist)

    print(Jp,dJp)
    print(J,dJ)

    #grape.optimiseFields(target, N, tN, [J], [A],kap=1, alpha=0, lam=0)
    #gp.optimiseFields(target, N, tN, J, A,alpha=0, lam=0)


#test_3q_CNOT()


def test_freqs(nS=225, max_time = 8*3600, J=None, A=None):

    if A is None: A = grape.get_A3(nS)
    if J is None: J = grape.all_J_pairs(J_100_18nm,J_100_18nm)[:nS]
    rf = grape.multiSystemFreqs(A,J)
    omega,_phase = grape.get_rotating_CFs(J,A); m=len(omega)
    print(f"m={m}")
    grape.visualise_frequencies(A,J)
#test_freqs()












def OG_test():
    nS=2
    N=200
    tN=90.0
    J=pt.tensor([J_100_18nm[0], J_100_18nm[1]])
    A=grape.get_A(nS)
    target=grape.CNOT_targets(nS)
    grape.optimise2(target, N, tN, J, A, u0=None, save_data=True,show_plot=True, max_time=8)



    tN=tN*nanosecond
    A = A*Mhz 
    J = J*Mhz 


    H0 = grape.make_H0_array(A,J,N)
    rf=grape.multiSystemFreqs(A,J)
    omega,phase=grape.config_90deg_phase_fields(rf)
    Hw=grape.make_Hw_array(omega,2,N,nS,tN,phase)
    #print(grape.cost(u,target,Hw,tN,H0))

    x_cf,y_cf = grape.get_control_fields(omega,phase,tN,N)
    #print(grape.fast_cost(u,target,grape.reduced_H0_array(A,J),x_cf,y_cf,tN))




def test_2q_CNOT():
    nq=2
    nS=2
    target = grape.CX_gate
    N=123
    A=nS*[grape.A_kane ]
    J = pt.tensor([[30],[40]]) *1e6*grape.hz
    tN=500e-9 * grape.second 
    w = grape.multiSystemFreqs(A,J)
    m=len(w)
    u = grape.init_u(tN,m,N)
    H0 = [gp.make_H0(A[0],J[0]),gp.make_H0(A[1],J[1])]
    Hw = gp.make_Hw(w,nq,tN,N)
    histp=2*[[]]
    Jp,dJp = gp.multiCost(u,m,nq,target,nS,H0,Hw,tN,N)

    '''
    max_w=pt.max(pt.real(omega))
    rec_min_N = N_period*max_w*tN/(2*np.pi)
    print(f"Recommened min N = {rec_min_N}")
    '''
    
    hist=2*[[]]
    H0_matrix = grape.make_H0_matrix(A,J,N)
    Hw_matrix = grape.make_Hw_matrix(w,nq,N,nS,tN)

    J,dJ = grape.cost(u,m,nq,target,Hw_matrix,tN,H0_matrix,hist)

    print(Jp,dJp)
    print(J,dJ)

    #grape.optimiseFields(target, N, tN, [J], [A],kap=1, alpha=0, lam=0)
    #gp.optimiseFields(target, N, tN, J, A,alpha=0, lam=0)


#test_2q_CNOT()

