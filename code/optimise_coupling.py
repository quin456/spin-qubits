import torch as pt
from scipy.optimize import minimize

from atomic_units import *
from utils import get_nS_nq_from_A 


################################################################################################################
################        COUPLING STRENGTH VARIATION        #####################################################
################################################################################################################
'''
Here GRAPE is used to optimise coupling strenths (J in the case of e-e CNOT, A in the case of e-n swap).
'''
def make_Hc(A,N,m):
    '''
    Generates tensor of shape (nS,N,d,d) corresponding to the coupling Hamiltonian term c*(sig1.sig2) for each system at each timestep.

    Inputs:
        nS: number of systems
        N: number of timesteps
    '''
    nS = len(A)
    return pt.einsum('j,sab->sjab', pt.ones(N),pt.einsum('s,ab->sab',A,gate.o2))
    return pt.einsum('skj,ab->skjab', pt.ones(nS,m,N),gate.o2)
    
def make_Hz(delta_w,N,m=1):
    nS = delta_w.shape[0]
    return pt.einsum('k,sjab->skjab',pt.ones(m),make_H0_array(delta_w,pt.zeros(nS,device=default_device),N))

    
def optimise_coupling(delta_w,A,target,tN,N,u0=None,kap=1,L=0, alpha=0):
    tN=tN*nanosecond
    A=A*Mhz 
    nS,nq=get_nS_nq_from_A(delta_w);m=1
    if nq!=2 or nS not in [1,2]: raise Exception("Not implemented")
    
    Hz = make_Hz(delta_w,N)
    H0 = make_Hc(A,N,m)

    hist=2*[[]]
    fun = lambda u: fast_cost(u,target,Hz,tN,H0,hist,kap=kap,L=L, alpha=alpha)
    if u0==None: u0 = pt.ones(N*m)*0.1*tesla
    opt=minimize(fun,u0,method='CG',jac=True)
    u=opt.x
    print(f"nit = {opt.nfev}, nfev = {opt.nit}")
    print(f"final cost = {opt.fun}")
    T = pt.linspace(0,tN,N)
    plt.plot(T/nanosecond,u)
    plt.xlabel("time (ns)")
    plt.ylabel("Field strength (T)")