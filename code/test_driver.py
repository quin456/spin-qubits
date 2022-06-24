

# set directory
from pathlib import Path
import os
dir = os.path.dirname(__file__)
os.chdir(dir)



device = 'cpu'
import pickle
from pdb import set_trace 
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import GRAPE as grape
import ast
from GRAPE import g,mu, tesla, A_kane, load_system_data
import gates as gate
from gates import cplx_dtype,get_Xn,get_Yn, CX, Id
from atomic_units import Mhz
exch_filename = "exchange_data_updated.p"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18']) 
J_100_14nm = pt.tensor(exch_data['100_14']) 
nanosecond = 1e-9*grape.second

#J=20e6*grape.hz
#A_arr = pt.tensor([[A1, A2],[A1,A2],[A1,A2], [A1, A2], [A1, A2], [A1,A2], [A1,A2], [A1,A2], [A1,A2], [A1,A2]], dtype=grape.real_dtype, device=device)
#J_arr=pt.tensor([10e6,15e6,20e6,25e6,30e6,35e6,40e6,45e6,50e6,55e6], dtype=grape.real_dtype, device=device)*grape.hz



################################################################################################################
################        TEST DRIVER FUNCTIONS        ###########################################################
################################################################################################################

def get_fields(filename):
    u = pt.load(f"fields/{filename}_XY", map_location=pt.device('cpu'))
    X = u[0,:]
    Y = u[1,:]
    return X,Y


def plot_fields(X_field,Y_field,tN,ax=None):
    N = len(X_field)
    T_axis = pt.linspace(0,tN/nanosecond, N)
    if ax==None: ax = plt.subplot()
    ax.plot(T_axis,X_field*1e3, label = 'X field (mT)')
    ax.plot(T_axis,Y_field*1e3, label = 'Y field (mT)')
    ax.set_xlabel('time (ns)')
    ax.legend()
    return ax 

def plot_ft_fields(X,Y):
    fX=pt.fft.fft(X)
    fY = pt.fft.fft(Y)
    ax=plt.subplot()
    ax.plot(fX)
    ax.plot(fY)



def simulate_field(tN,X_field,Y_field,J,A):
    N = len(X_field)
    dt=tN/N
    if len(A.shape)>1:
        nS,nq=A.shape
    else:
        nq=len(A)
        nS=1 #actually 1 system, but zero more useful
    dim=2**nq
    X = pt.zeros(nS,N,dim,dim, dtype=cplx_dtype)
    for q in range(nS):    
        H0 = grape.get_H0(A,J)[q]
        U = pt.eye(2**nq, dtype = grape.cplx_dtype)
        for j in range(N):
            H = H0 + 0.5*g*mu*tesla*(X_field[j]*get_Xn(nq) + Y_field[j]*get_Yn(nq))
            U = pt.matmul(pt.matrix_exp(-1j*dt*H),U)
            X[q][j]=U  
    return X

def fidelity(X,target):
    
    Uf = X[:,-1]
    # fidelity of resulting unitary with target
    IP = grape.batch_IP(target,Uf)
    Phi = pt.real(IP*pt.conj(IP))
    return Phi

def evaluate_XY_field(fn):

    J,A,tN,N,target,fid = load_system_data(fn)
    X_field,Y_field = get_fields(fn)


    print(f"A={A}, J={J}")

    plot_fields(X_field,Y_field,tN)
    X=simulate_field(tN,X_field,Y_field,J,A)
    fid_progress = grape.get_fidelity_progress(X,tN,target)
    print(f"Fidelities = {fid_progress[:,-1]}")
    _fig,ax = plt.subplots(1,2)
    grape.plot_fidelity_progress(ax[1],fid_progress,tN,False)
    plot_fields(X_field,Y_field,tN,ax[0]); plt.show()


def display_results(X, X_field, Y_field, tN, target):
    fig,ax = plt.subplots(2,1)
    fid_progress = grape.get_fidelity_progress(X,3*tN,target)
    print(f"Final fidelities = {fid_progress[:,-1]}")
    grape.plot_fidelity_progress(ax[1],fid_progress,tN,False)
    plot_fields(X_field,Y_field,tN,ax[0]); plt.show()

def wf_evolution(Xs, psi0=pt.tensor([0,1,0,0,0,0,0,0], dtype=cplx_dtype)):
    psi = pt.matmul(Xs,psi)
    ax=plt.usbplot()
    for a in range(len(psi)):
        ax.plot(psi[a,:])
################################################################################################################
################################################################################################################
################################################################################################################


def frequency_collisions(fn):

    def single_CNOT_target(CNOT_sys, nS):
        '''
        System CNOT_sys assigned CNOT target, remaining systems assigned identity target.
        '''
        target = grape.ID_targets(nS,3)
        target[CNOT_sys] = gate.CX3q
        return target
    
    J,A,tN,N,target,fid=load_system_data(fn)
    X_field,Y_field = get_fields(fn)

    nS=225; nq=3
    all_J = grape.get_J(nS,nq)*Mhz 
    all_A = grape.get_A(nS,nq)*Mhz
    all_target = single_CNOT_target(0,nS)
    X = simulate_field(tN, X_field, Y_field, all_J, all_A)
    print(max(fidelity(X,all_target)[1:]))

#frequency_collisions('c302_1S_3q_190ns_500step')

def three_q_from_2q():

    fn1='c277_2S_2q_90ns_2000step'
    fn2='c278_2S_2q_90ns_2000step'


    J1,A1,tN,N,target1,fid=load_system_data(fn1)
    X_field1,Y_field1 = get_fields(fn1)

    J2,A2,tN,N,target2,fid = load_system_data(fn2)
    X_field2,Y_field2 = get_fields(fn2)


    # X1=simulate_field(tN,target1,X_field1,Y_field1,J1,A)
    # fig,ax = plt.subplots(2,1)
    # fid_progress = grape.get_fidelity_progress(X1,tN,target1)
    # print(f"Final fidelities = {fid_progress[:,-1]}")
    # grape.plot_fidelity_progress(ax[1],fid_progress,tN,False)
    # plot_fields(X_field1,Y_field1,tN,ax[0]); plt.show()

    J_3q = grape.get_J(2,3)[1:]*Mhz
    target_3q = grape.CNOT_targets(1,3)
    A_3q = grape.get_A(1,3)*Mhz
    A_3q[0][1:]*=-1
    print(f"A={A_3q}, \nA1={A1}, \nA2={A2}")
    print(f"J={J_3q}, \nJ1={J1}, \nJ2={J2}")
    X_field = pt.cat((X_field1,X_field2,X_field1))
    Y_field = pt.cat((Y_field1,Y_field2,Y_field1))
    
    X=simulate_field(3*tN,target_3q,X_field,Y_field,J_3q,A_3q)

    wf_evolution(X)
    #display_results(X, X_field, Y_field, 3*tN, target_3q); plt.show()






def compare_fields(times,Ns,Js,As,IDs):
    n=len(times)
    nq = len(As[0][0])
    fig,ax = plt.subplots(n,2)
    for i in range(n):
        filename = grape.get_field_filename(As[i],times[i]*nanosecond,Ns[i], IDs[i])
        _J,_A,tN,target,fid = load_system_data(filename)
        target=grape.CNOT_targets(len(_J),nq)
        X_field,Y_field = get_fields(filename)

        X = simulate_field(tN,target,X_field,Y_field,_J,_A)
        fids = grape.get_fidelity_progress(X,tN,target)
        grape.plot_fidelity_progress(ax[i][0],fids,tN)
        plot_fields(X_field,Y_field,tN,ax[i][1])
    
    plt.show()





def run():

    filename = 'c453_1S_2q_90ns_200step'

    SD,target,_fid = load_system_data(filename)
    X_field,Y_field = get_fields(filename)


    print(f"A={SD.A}, J={SD.J}")

    X=simulate_field(SD.tN,X_field,Y_field,SD.J,SD.A)
    display_results(X,X_field,Y_field,SD.tN,SD.target)


def all_2q():
    fig,ax = plt.subplots(2,1)
    filename = 'g121_15S_2q_180ns_1000step'
    J,A,tN,N,target,fid = load_system_data(filename)
    start=0;nS=15
    J=J_100_18nm[start:start+nS]*Mhz; A=grape.get_A(nS,2)*Mhz; target=grape.CNOT_targets(nS,2)
    X_field,Y_field = get_fields(filename)
    plot_fields(X_field,Y_field,tN,ax[0])
    X=simulate_field(tN,X_field,Y_field,J,A)
    fids=grape.get_fidelity_progress(X,tN,target)
    print(f"Fidelities = {fids[:,-1]}")
    grape.plot_fidelity_progress(ax[1],fids,tN, legend=False)
    plt.show()



