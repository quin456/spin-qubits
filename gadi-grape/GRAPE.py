

# coding tools
from pdb import set_trace
import time
import warnings
warnings.filterwarnings("ignore")

# library imports
import torch as pt
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools    
from scipy import linalg as la
from scipy.optimize import minimize
from numpy import random as rand
from numpy import sin, cos, sqrt
from datetime import datetime
import ast
from torch import multiprocessing as mp 
import pickle
from queue import Queue 
from threading import Thread


# Other python files in project
from atomic_units import *
import gates as gate
from gates import cplx_dtype, real_dtype, kron3 

time_exp = 0
time_prop = 0
time_grad=0
time_fid=0




exch_filename = "exchange_data_updated.p"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18'], dtype=cplx_dtype)
J_100_14nm = pt.tensor(exch_data['100_14'], dtype=cplx_dtype)


jac=True # keep as True unless testing cost grad function
save_mem = False # save mem in Hamiltonian arrays. 
work_in_eigenbasis = False # Should be set to False unless testing
interaction_picture= False # Should be set to False unless testing 
''' save_mem incompatible with work_in_eigenbasis and interaction picture '''




ngpus = pt.cuda.device_count()
default_device = 'cuda:0' if ngpus>0 else 'cpu'

np.set_printoptions(4)
pt.set_printoptions(sci_mode=True)
annotate=True 


log_fn = 'log/log.txt'
ID_fn = 'log/ID_tracker.txt'
precision_loss_msg = "Desired error not necessarily achieved due to precision loss."



B0 = 0 #2*tesla      # static background field
g = 2.0023               # electron g-factor


mu = qE*hbar/(2*mE)          # Bohr magneton
omega0 = g*mu*B0/hbar   # Larmor frequency


# exchange values 
J_10nm = 0.1e-3*qE_n*joule # ~10e-23
J_Omin = 15e6 * hz * hbar
J_Omax = 50e6 * hz * hbar
A_BP = 5e6 * hz
A_approx = 1e9 * hz
A_real1 = pt.tensor([183.5e6, 66.5e6]) * hz 
delta_A_kane = 58.5
#A_kane = pt.tensor([58.5, 0], device=device)   ##stoopystoopystoopy
A_kane = pt.tensor([58.5/2, -58.5/2])
A_kane3 = pt.tensor([58.5/2, -58.5/2, 58.5/2])


################################################################################################################
################        HELPER FUNCTIONS        ################################################################
################################################################################################################

def get_A(nS,nq, device=default_device):
    if nq==2:
        return pt.tensor(nS*[[58.5/2, -58.5/2]], device=device, dtype = cplx_dtype)
    elif nq==3:
        return pt.tensor(nS*[[58.5/2, 58.5/2, -58.5/2]], device=device, dtype=cplx_dtype)


def all_J_pairs(J1, J2, device=default_device):
    nJ=15
    J = pt.zeros(nJ**2,2, device=device,dtype=cplx_dtype)
    for i in range(nJ):
        for j in range(nJ):
            J[i*15+j,0] = J1[i]; J[i*15+j,1] = J2[j]
    return J

def get_J(nS,nq,J1=J_100_18nm,J2=J_100_18nm, device=default_device):
    if nq==2:
        return J1[:nS].to(device)
    elif nq==3:
        return all_J_pairs(J1,J2)[:nS]

def CNOT_targets(nS,nq,device=default_device):
    if nq==2:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.CX.to(device))
    elif nq==3:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.CX3q.to(device))

def CNOT3_targets(n, device=default_device):
    return pt.einsum('i,ab->iab',pt.ones(n, device=device),gate.CX3q)

def SWAP_targets(n, device=default_device):
    return pt.einsum('i,ab->iab',pt.ones(n, device=device),gate.swap.to(device))

def RSW_targets(n, device=default_device): 
    return pt.einsum('i,ab->iab',pt.ones(n, device=device),gate.root_swap.to(device))


def normalise(v):
    ''' Normalises 1D tensor '''
    return v/pt.norm(v)

def innerProd(A,B):
    '''  Calculates the inner product <A|B>=Phi(A,B) of two matrices A,B.  '''
    return pt.trace(pt.matmul(dagger(A),B)).item()/len(A)

def dagger(A):
    '''  Returns the conjugate transpose of a matrix or batch of matrices.  '''
    return pt.conj(pt.transpose(A,-2,-1))

def commutator(A,B):
    '''  Returns the commutator [A,B]=AB-BA of matrices A and B.  '''
    return pt.matmul(A,B)-pt.matmul(B,A)

def matmul3(A,B,C):
    '''  Returns multiple of three matrices A,B,C.  '''
    return pt.matmul(A,pt.matmul(B,C))


def grad(f,x0,dx):
    ''' Uses difference method to compute grad of function f which takes vector input. '''
    df = pt.zeros(len(x0),dtype=real_dtype)
    for j in range(len(x0)):
        x1 = x0.clone()
        x2 = x0.clone()
        x2[j]+=dx
        x1[j]-=dx
        df[j] = (f(x2)-f(x1))/(2*dx)
    return df


def get_dimensions(A):
    ''' Returns (number of systems, number of qubits in each system) '''
    return len(A), len(A[0])

'''
The following functions are used to manipulate and access 'u', which contains the control field amplitudes at 
each timestep, the learning parameters of the GRAPE algorithm. The most natural form for 'u' is an m x N matrix,
but it is converted to and from vector form for input into scipy.minimize.
'''
def uToVector(u):
    '''  Takes m x N torch tensor 'u' and converts to 1D tensor in which the columns of u are kept together.  '''
    #return (ptflatten?) (pt.transpose(u,0,1))
    return pt.reshape(pt.transpose(u,0,1),(u.numel(),))

def uToMatrix(u,m):
    '''  
    Inverse of uToVector. Takes m*N length 1D tensor, splits into m N-sized tensors which are stacked together as columns of an
    m x N tensor which is the output of the function.
    '''
    N = int(len(u)/m)
    return pt.transpose(pt.reshape(u,(N,m)),0,1)

def uIdx(u,m,j,k):
    '''  Accesses element (j,k) of vector form u.  '''
    if len(u.shape)==1:
        return u[k*m+j]
    return u[j,k]

def uCol(u,j,m):
    '''  Accepts vector form 'u' as input, and returns what would be column 'j' if 'u' were in matrix form  '''
    return u[j*m:(j+1)*m]

def get_nq(d):
    ''' Takes the dimension of the Hilbert space as input, and returns the number of qubits. '''
    return int(np.log2(d))

################        BATCH OPERATIONS        ################################################################
def makeBatch(H,T):
    '''
    Returns a rank 3 tensor, with the outermost array consisting of each element of 1D tensor T multiplied by matrix H.
    '''
    N=len(T)
    return pt.mm(T.unsqueeze(1), H.view(1, H.shape[0]*H.shape[1])).view(N,*H.shape)  

def batch_trace(T):
    '''
    Takes an array of arbitrary shape containing square matrices (in other words a tensor whose innermost dimensions are of
    equal size), and determines the trace of each matrix. Traces are returned in a tensor having the same shape as the
    aforementioned array.
    '''
    return (T.diagonal(dim1=-2,dim2=-1)).sum(-1)

def batch_IP(A,B):
    ''' 
    Takes two equal sized arrays of square matrices A,B, and returns a bitwise inner product of the matrices of the form
        torch.tensor([<A[0],B[0]>, <A[1],B[1]>, ... , <A[-1],B[-1]>])
    '''
    d = A.shape[-1]
    return (1/d) * batch_trace(pt.matmul(dagger(A),B))


def mergeMatmul(A, backwards=False):
    
    # A has shape Nsys x Nstep x d x d where d = 2^(number of qubits) = dim of H.S.

    if len(A[0])==1:
        return A

    h = len(A[0])//2
    A[:,:h] = mergeMatmul(A[:,:h],backwards)
    A[:,h:] = mergeMatmul(A[:,h:],backwards)

    if backwards:
        #back propagates target to get P matrices
        A[:,:h] = pt.einsum('sjab,sbc->sjac',A[:,:h],A[:,h])
    else:
        # propagates identity to generate X matrices
        A[:,h:] = pt.einsum('sjab,sbc->sjac',A[:,h:],A[:,h-1])
    return A




################################################################################################################
################        CLASSES        #########################################################################
################################################################################################################

class Optimisation_Timer(object):

    def __init__(self,max_time, filename, cost_hist, save_data, sp_distance = 99999*3600):
        self.max_time=max_time
        self.start_time = time.time()
        self.sp_count = 0
        self.sp_distance = sp_distance
        self.filename="fields/"+filename
        self.cost_hist=cost_hist
        self.save_data = save_data

    def check_time(self, xk):
        time_passed = time.time() - self.start_time
        if time_passed > (self.sp_count+1)*self.sp_distance:
            self.sp_count+=1
            print(f"Save point {self.sp_count} reached, time passed = {time_passed}.")
            if self.save_data:
                pt.save(pt.tensor(xk, dtype=real_dtype), self.filename+'_SP'+str(self.sp_count))
                pt.save(np.array(self.cost_hist), self.filename+'_cost_hist')
        if time_passed > self.max_time:
            global u_opt_timeout
            u_opt_timeout = xk
            print(f"Max time of {self.max_time} has been reached. Optimisation terminated.")
            raise TimeoutError






################################################################################################################
################        GRAPE IMPLEMENTATION        ############################################################
################################################################################################################


def time_evol(u,H0,x_cf,y_cf,tN, device=default_device):
    m,N = x_cf.shape; nS,d = H0.shape[:-1]; nq = get_nq(d)
    sig_xn = gate.get_Xn(nq,device); sig_yn = gate.get_Yn(nq,device)
    dt = tN/N
    u_mat = uToMatrix(u,m)
    x_sum = pt.einsum('kj,kj->j',u_mat,x_cf).to(device)
    y_sum = pt.einsum('kj,kj->j',u_mat,y_cf).to(device)

    H = pt.einsum('j,sab->sjab',pt.ones(N,device=device),H0.to(device)) + pt.einsum('s,jab->sjab',pt.ones(nS,device=device),pt.einsum('j,ab->jab',x_sum,sig_xn)+pt.einsum('j,ab->jab',y_sum,sig_yn))
    H=pt.reshape(H, (nS*N,d,d))
    U = pt.matrix_exp(-1j*H*dt)
    del H
    U = pt.reshape(U, (nS,N,d,d))
    return U



def propagate(U,target,device=default_device):
    '''
    Determines total time evolution operators to evolve system from t=0 to time t(j) and stores in array P. 
    Backward propagates target evolution operators and stores in X.
    '''
    nS=len(U)
    N=len(U[0])
    dim = U[0].shape[1]  # forward propagated time evolution operator
    X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator
    P = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    P[:,N-1,:,:] = target    # backwards propagated target 

    
    global mergeprop_g

    # vectorized is slower :) 
    if mergeprop_g:
        X[:,1:] = U[:,1:]
        P[:,:-1] = dagger(U[:,1:])
        X = mergeMatmul(X)
        P = mergeMatmul(P,backwards=True)
    
    else:
        for j in range(1,N):
            X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
            P[:,-1-j,:,:] = pt.matmul(dagger(U[:,-j,:,:]),P[:,-j,:,:])
    
    return X,P


################       COST FUNCTION        ############################################################
def fluc_cost(u,m,alpha):
    '''
    Cost function term which penalises fluctuations in u between adjacent timesteps.
    '''
    N = int(len(u)/m)
    u = uToMatrix(u,m)
    u_aug = pt.cat((u[:,0:1],u, u[:,N-1:N]),dim=1)
    J = alpha  * np.real(pt.sum( (u[:,1:] - u[:,:N-1])**2).item())
    dJ = pt.real(uToVector( 2*alpha*(2*u-u_aug[:,2:]-u_aug[:,0:N]) ))
    return J,dJ


def reg_cost(u,lam):
    '''
    Regularisation cost function term which penalises large values of u.
    '''
    J = lam/(2*len(u)) * np.real(pt.dot(u,u).item())
    dJ = lam/len(u) * pt.real(u)
    return J,dJ

def lock_endpoints(u_mat, u0_L, uf_L):
    '''
    Cost function terms which locks the endpoints of each u_k by applying a large penalty when endpoints
    deviate from specified positions.

    Inputs:
        u_mat: (k,N) tensor - matrix form of u
        u0_L: k element vector representing initial values at which to lock u.  
        uf_L: k element vector representing final values at which to lock u.  
    '''
    L=1e4 # Any number big enough to enforce lock

    J = 0.5 * L * pt.norm(u_mat[:,0]-u0_L)**2 + pt.norm(u_mat[:,-1]-uf_L)**2

    dJ = pt.zeros(*u_mat.shape)
    dJ[:,0] = L * (u_mat[:,0] - u0_L)
    dJ[:,-1] = L * (u_mat[:,-1] - uf_L)

    return J,uToVector(dJ)

def hamShape(H0,Hw):
    if save_mem:
        nS=len(H0)
        m,N,d = Hw.shape[:-1]
    else:
        nS,m,N,d = Hw.shape[:-1]; 
    nq=int(np.log2(d))

    return nS,m,N,d,nq


def new_target(Uf):

    # non-zero target elements
    target_nz = pt.tensor([
        [1,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,1],
        [0,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,0,1],
        [0,0,0,0,1,0,1,0]
    ], dtype = cplx_dtype)

    nS,d,d = Uf.shape
    # accepted U elements
    U_acc = pt.multiply(Uf, pt.einsum('s,ab->sab',pt.ones(nS,dtype=cplx_dtype),target_nz))
    U_sqamp = pt.multiply(U_acc,pt.conj(U_acc))

    # new target will be U_accepted, only normalised so that each row has length 1.
    U_sqlen = pt.sum(U_sqamp,2)

    target_new = pt.sqrt(pt.div(U_sqamp, pt.einsum('sa,b->sab',U_sqlen,pt.ones(d, dtype=cplx_dtype))))
    return target_new


def fast_fid(u,H0,x_cf,y_cf,tN,target, device=default_device):

    global target_g
    target=target_g

    m,N = x_cf.shape; nS,d = H0.shape[:-1]; nq = get_nq(d)
    sig_xn = gate.get_Xn(nq, device); sig_yn = gate.get_Yn(nq, device)

    t0 = time.time()
    if pt.cuda.device_count()==2:
        U2 = time_evol(u,H0,x_cf,y_cf,tN, device='cuda:1')
        U=U2.to('cuda:0')
        del U2
    else:
        U = time_evol(u,H0,x_cf,y_cf,tN, device=device)
    global time_exp 
    time_exp += time.time()-t0

    t0 = time.time()
    X,P = propagate(U,target, device=device)
    global time_prop 
    time_prop += time.time()-t0
    del U

    Ut = P[:,-1]; Uf = X[:,-1]
    # fidelity of resulting unitary with target
    IP = batch_IP(Ut,Uf)
    Phi = pt.real(IP*pt.conj(IP))


    t0 = time.time()
    # calculate grad of fidelity
    XP_IP = batch_IP(X,P)

    ox_X = pt.einsum('ab,sjbc->sjac' , sig_xn, X)
    oy_X = pt.einsum('ab,sjbc->sjac' , sig_yn, X)
    PoxX_IP =  batch_trace(pt.einsum('sjab,sjbc->sjac', dagger(P), 1j*ox_X)) / d
    PoyX_IP =  batch_trace(pt.einsum('sjab,sjbc->sjac', dagger(P), 1j*oy_X)) / d
    del X,P,ox_X, oy_X

    Re_IP_x = -2*pt.real(pt.einsum('sj,sj->sj', PoxX_IP, XP_IP))
    Re_IP_y = -2*pt.real(pt.einsum('sj,sj->sj', PoyX_IP, XP_IP))

    # sum over systems axis
    sum_IP_x = pt.sum(Re_IP_x,0)
    sum_IP_y = pt.sum(Re_IP_y,0)
    del Re_IP_x, Re_IP_y

    dPhi_x = pt.einsum('kj,j->kj', pt.real(x_cf), sum_IP_x)
    dPhi_y = pt.einsum('kj,j->kj', pt.real(y_cf), sum_IP_y)
    del sum_IP_x, sum_IP_y

    dPhi = dPhi_x + dPhi_y
    del dPhi_x, dPhi_y
    global time_grad 
    time_grad += time.time()-t0

    global it 
    it +=1 
    #if it>100 and it%10==0:
    #    target_g = new_target(Uf)

    return Phi, dPhi


def fast_cost(u,H0,nS,J,A,x_cf,y_cf,tN,target,hist,MP=False, kappa=1):
    '''
    Fast cost is faster and more memory efficient but less generalised cost function which can be used for CNOTs.

    INPUTS:
        u: control field amplitudes
        x_cf: 0.5*g*mu*(1T)*cos(wt-phi) - An (m,N) tensor recording coefficients of sigma_x for Hw_kj
        y_cf: 0.5*g*mu*(1T)*sin(wt-phi) - An (m,N) tensor recording coefficients of sigma_y for Hw_kj
            Hw = x_cf * sig_x + y_cf * sig_y
    
    '''
    
    u=pt.tensor(u, dtype=cplx_dtype, device=default_device)

    t0 = time.time()
    Phi,dPhi = fast_fid(u,H0,x_cf,y_cf,tN,target)
    global time_fid 
    time_fid += time.time()-t0
    J = 1 - pt.sum(Phi,0)/nS
    dJ = -uToVector(dPhi)/nS
    J=J.item(); dJ = dJ.cpu().detach().numpy()
    hist.append(J)

    # huge waste of space
    #hist[1].append(dJ)
    
    return J, dJ * tN/nS / kappa






def get_control_fields(omega,phase,tN,N,device=default_device):
    T = pt.linspace(0,tN,N, device=device)
    wt = pt.einsum('k,j->kj', omega,T)
    x_cf = 0.5*g*mu*(1*tesla)*pt.cos(wt-phase)
    y_cf = 0.5*g*mu*(1*tesla)*pt.sin(wt-phase)
    return x_cf.type(cplx_dtype), y_cf.type(cplx_dtype)

def print_info(u_opt,time_taken,H0,nS,J,A,x_cf,y_cf,tN,target, minprint):

    fidelities = fast_fid(u_opt,H0,x_cf,y_cf,tN,target)[0]
    avgfid=sum(fidelities)/len(fidelities)
    minfid = min(fidelities).item()

    print(f"Average fidelity = {avgfid}")
    if not minprint:
        print(f"Fidelities = {fidelities}")
        print(f"Min fidelity = {minfid}")
        print(f"Time taken = {time_taken}")
        print(f"GPUs requested = {ngpus}")
        global time_grad, time_exp, time_prop, time_fid 
        print(f"texp={time_exp}, tprop={time_prop}, tgrad = {time_grad}, tfid={time_fid}")
    

def unit_conversion(J,A,tN):
    return J*Mhz, A*Mhz, tN*nanosecond


def run_optimisation(fun,u0,opt_timer, max_time):
    callback = opt_timer.check_time if max_time is not None else None
    try:
        opt=minimize(fun,u0,method='CG',jac=True, callback=callback)
        print(f"nit = {opt.nfev}, nfev = {opt.nit}")
        opt_timer.u_opt=pt.tensor(opt.x, device=default_device)
        opt_timer.status='C' #complete
    except TimeoutError:
        global u_opt_timeout
        opt_timer.u_opt = pt.tensor(u_opt_timeout, dtype=cplx_dtype, device=default_device)
        opt_timer.status='UC' #uncomplete
    opt_timer.time_taken = time.time() - opt_timer.start_time

    return opt_timer

def get_X(u_opt,H0,x_cf,y_cf,tN,target):
    U = time_evol(u_opt,H0,x_cf,y_cf,tN)
    X,P = propagate(U,target)
    Uf = X[0][-1]
    return X


def optimise2(target, N, tN, J, A, u0=None, rf=None, save_data=False,show_plot=True, max_time=None, NI_qub=False, hist0=None, kappa=1, minprint=False, mergeprop=False):
    '''
    New optimiseFields which uses fast_cost
    '''

    global target_g ; target_g=target
    global it; it=0
    global mergeprop_g; mergeprop_g=mergeprop 

    # unit conversion
    J,A,tN = unit_conversion(J,A,tN)
    # Log job start
    ID,filename = preLog(J,A,tN,N,target,save_data)


    nS,nq=get_dimensions(A)
    if rf is None: rf=get_RFs(A,J)
    omega,phase = config_90deg_phase_fields(rf); m = len(omega) 
    H0 = get_H0(A,J)
    
    if u0 is None: u0=init_u(tN,m,N)

    cost_hist=hist0 if hist0 is not None else []
    opt_timer = Optimisation_Timer(max_time, filename, cost_hist, save_data)
    
    x_cf,y_cf = get_control_fields(omega,phase,tN,N)
    fun = lambda u: fast_cost(u,H0,nS,J,A,x_cf,y_cf,tN,target,cost_hist,kappa)

    result = run_optimisation(fun,u0,opt_timer, max_time)
    target = target_g
    print_info(result.u_opt,result.time_taken,H0,nS,J,A,x_cf,y_cf,tN,target, minprint)

    X = get_X(result.u_opt,H0,x_cf,y_cf,tN,target)


    # Plot fields and process data
    plotFields(result.u_opt,m,tN,omega, target,X,A, cost_hist,save_data,show_plot=show_plot, plotLabel=filename)

    fidelities = fast_fid(result.u_opt,H0,x_cf,y_cf,tN,target)[0]
    avgfid=sum(fidelities)/len(fidelities)
    minfid = min(fidelities).item()
    

    alpha=0
    if save_data:
        # save_system_data will overwrite previous file if job was not terminated by gadi
        log_result(minfid,avgfid,alpha,kappa,nS,nq,tN,N,A,result.status,ID, opt_timer.time_taken)
        save_system_data(tN, J, A, N, target, filename, fid=fidelities, status=result.status)
        save_field(result.u_opt, m, rf,tN, filename,cost_hist)



################################################################################################################
################        VISUALISATION        ###################################################################
################################################################################################################


def plot_XY_fields(ax, X_field, Y_field, tN):
    X_field = X_field.cpu(); Y_field=Y_field.cpu()
    N=len(X_field)
    t_axis = np.linspace(0, tN/nanosecond, N)
    ax.plot(t_axis,X_field*1e3,label='total x-field (mT)')
    ax.plot(t_axis,Y_field*1e3,label='total y-field (mT)')
    ax.set_xlabel("time (ns)")
    ax.legend()
    
def get_fidelity_progress(X, tN, target):
    '''
    For each system, determines the fidelity of unitaries in P with the target over the time of the pulse
    '''
    nS=len(X); N = len(X[0])
    fid = pt.zeros(nS,N)
    for q in range(nS):
        for j in range(N):
            IP = innerProd(target[q],X[q,j])
            fid[q,j] = np.real(IP*np.conj(IP))
    return fid

def plot_fidelity_progress(ax,fids,tN, legend=True):
    nS=len(fids); N = len(fids[0])
    T = pt.linspace(0,tN/nanosecond, N)
    for q in range(nS):
        ax.plot(T,fids[q], label=f"System {q+1} fidelity")
    if legend: ax.legend()
    ax.set_xlabel("time (ns)")
    if annotate: ax.annotate("Fidelity progress", (0,0.95))
    return ax


def plot_cost_hist(cost_hist, ax):
    ax.plot(cost_hist, label='cost')
    ax.set_xlabel('Iterations')
    ax.legend()
    return ax

def plotFields(u,m,tN,omegas, target,X,A, cost_hist, save_plot,show_plot=True,coupled_XY=True, plotLabel=None):
    rf=omegas[:len(omegas)//2]
    N = int(len(u)/m)
    u_m = uToMatrix(u,m).cpu()
    w_np = omegas.cpu().detach().numpy()
    T = np.linspace(0, tN, N)
    t_axis = np.linspace(0, tN/nanosecond, N)
    fig,ax = plt.subplots(2,2)
    for i in range(len(w_np)):
        if i<len(w_np)//2:
            B1 = np.cos(w_np[i]*T)
        else:
            B1=np.sin(w_np[i]*T)
        #ax[0,1].plot(t_axis,B1, label="Hw"+str(i))
        if not coupled_XY:
            B1uc = np.sin(w_np[i]*T)
            #ax[0,1].plot(t_axis,B1uc,label="Hw"+str(i)+"y")

    plot_cost_hist(cost_hist, ax[1,1])
        

    for i in range(len(w_np)):
        if i<len(w_np)//2:
            B1 = np.cos(w_np[i]*T)
        else:
            B1=np.sin(w_np[i]*T)
        prod=np.multiply(u_m[i],B1)
        
        ax[0,0].plot(t_axis,u_m[i], label='u'+str(i))
        #ax[1,0].plot(t_axis,1e3*prod,label='u'+str(i)+'*Hw'+str(i)+" (mT)")
        if not coupled_XY:
            ax[0,0].plot(t_axis,u_m[len(omegas)+i], label='u'+str(i)+'y')
            #ax[1,0].plot(t_axis,pt.multiply(u_m[len(omegas)+i],B1uc),label='u'+str(i)+'*By'+str(i))
            
    X_field, Y_field = sum_XY_fields(uToMatrix(u,m),m,rf,tN)
    plot_XY_fields(ax[0,1],X_field,Y_field,tN)
    transfids = get_fidelity_progress(X,tN,target)
    plot_fidelity_progress(ax[1,0],transfids,tN,legend=False)
    ax[0,0].set_xlabel("time (ns)")
    #ax[0,j].legend()

    if save_plot:
        fig.savefig("plots/"+plotLabel)
    if show_plot: plt.show()


def visualise_Hw(Hw,tN,N):
    T = pt.linspace(0,tN,N)
    for k in range(Hw.shape[1]):
        fig,ax = plt.subplots(4,4)
        for i in range(4):
            for j in range(4):
                y = Hw[0,k,:,i,j]
                ax[i,j].plot(T,pt.real(y))
                ax[i,j].plot(T,pt.imag(y))
        fig.suptitle(f"k={k}")
        plt.show()
################################################################################################################
################        Hamiltonians        ####################################################################
################################################################################################################

def transform_H0(H,S):
    '''
    Applies transformation S.T @ H0 @ S for each system, to transform H0 into frame whose basis vectors are the columns of S.
    H0: (nS,N,d,d)
    S: (nS,d,d)
    '''
    HS = pt.einsum('sjab,sbc->sjac', H, S)
    H_trans = pt.einsum('sab,sjbc->sjac', dagger(S),HS)
    return H_trans

def transform_Hw(H,S):
    '''
    Applies transformation S.T @ Hw @ S for each system, to transform Hw into frame whose basis vectors are the columns of S.
    Hw: (nS,m,N,d,d)
    S: (nS,d,d)
    '''
    HS = pt.einsum('skjab,sbc->skjac', H, S)
    H_trans = pt.einsum('sab,skjbc->skjac', dagger(S),HS)
    return H_trans


def evolve_Hw(Hw,U):
    '''
    Evolves Hw by applying U @ H0 @ U_dagger to every control field at each timestep for every system.
    Hw: Tensor of shape (nS,m,N,d,d), describing m control Hamiltonians having dimension d for nS systems at N timesteps.
    U: Tensor of shape (nS,N,d,d), ""
    '''
    UH = pt.einsum('sjab,skjbc->skjac',U,Hw)
    return pt.einsum('skjab,sjbc->skjac',UH,dagger(U))

def make_Hw(omegas, nq, tN,N,phase = pt.zeros(1),coupled_XY=True):
    '''
    Takes input array omegas containing m/2 frequencies.
    Returns an array containing m oscillating fields transverse fields, 
    with one in the x direction and one in the y direction for each omega
    '''
    mw=len(omegas)
    Xn=gate.get_Xn(nq); Yn=gate.get_Yn(nq); Zn=gate.get_Zn(nq)
    T = pt.linspace(0,tN,N, device=default_device)

    wt_tensor = pt.kron(omegas,T).reshape((mw,N))
    X_field_ham = pt.einsum('ij,kl->ijkl', pt.cos(wt_tensor-phase), Xn)
    Y_field_ham = pt.einsum('ij,kl->ijkl', pt.sin(wt_tensor-phase), Yn)
    if coupled_XY:
        Hw = 0.5*g*mu * 1*tesla * ( X_field_ham + Y_field_ham )
    else:
        Hw = 0.5*g*mu * 1*tesla * pt.cat((X_field_ham, Y_field_ham),0)
    return Hw

def ignore_tensor(trans,d):
    nS=len(trans)
    ignore = pt.ones(len(trans),d,d)
    for s in range(nS):
        for i in range(len(trans[s])):
            ignore[s,int(trans[s][i][0])-1, int(trans[s][i][1])-1] = 0
            ignore[s,int(trans[s][i][1])-1, int(trans[s][i][0])-1] = 0
    return pt.cat((ignore,ignore))



def get_H0(A,J, device=default_device):
    '''
    Free hamiltonian of each system. Reduced because it doesn't multiply by N timesteps, which is a waste of memory.
    
    Inputs:
        A: (nS,nq), J: (nS,) for 2 qubit or (nS,2) for 3 qubits
    '''
    nS, nq = get_dimensions(A)
    d = 2**nq
    if nq==3:
        H0 = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('sc,cab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device))
    elif nq==2:
        H0 = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('s,ab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device))
    return H0.to(device)
    



def get_S_matrix(J,A, device=default_device):
    nS,nq = get_dimensions(A); d=2**nq
    if nq != 2: raise Exception("Not implemented")
    S = pt.zeros(nS,d,d, dtype=cplx_dtype, device=device)
    for s in range(nS):
        dA = (A[s][0]-A[s][1]).item()
        d = pt.sqrt(8*J[s]**2 + 2*dA**2 - 2*dA*pt.sqrt(dA**2+4*J[s]**2))
        alpha = 2*J[s]/d
        beta = (-dA + pt.sqrt(dA**2+4*J[s]**2))/d
        S[s] = pt.tensor([[1,0,0,0],[0,alpha,-beta,0],[0,beta,alpha,0],[0,0,0,1]])
    return S



################################################################################################################
################        Resonant Frequencies        ############################################################
################################################################################################################
def remove_duplicates(A):
    '''
    Removes duplicates from A where equivalence is required to 9 decimal places
    '''
    i=0
    while i<len(A):
        j=i+1
        while j<len(A):
            if math.isclose(A[i],A[j],rel_tol=1e-9):
                A.pop(j)
                continue 
            j+=1
        i+=1
    return A


def getFreqs(H0,Hw_shape,device=default_device):
    '''
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0. 
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. Probably won't 
    end up being used as 3 qubit CNOTs will be performed as sequences of 2 qubit CNOTs.
    '''
    eig = pt.linalg.eig(H0)
    evals=eig.eigenvalues
    S = eig.eigenvectors
    S=S.to(device)
    #S = pt.transpose(pt.stack((S[:,2],S[:,1],S[:,0],S[:,3])),0,1)
    #evals = pt.stack((evals[2],evals[1],evals[0],evals[3]))
    S_T = pt.transpose(S,0,1)
    d = len(evals)
    pairs = list(itertools.combinations(pt.linspace(0,d-1,d,dtype=int),2))

    # transform shape of control Hamiltonian to basis of energy eigenstates

    Hw_trans = matmul3(S_T,Hw_shape,S)
    Hw_nz = (pt.abs(Hw_trans)>1e-9).to(int)
    freqs = []
    for i in range(len(pairs)):
        # The difference between energy levels i,j will be a resonant frequency if the control field Hamiltonian
        # has a non-zero (i,j) element.
        pair = pairs[i]
        idx1=pair[0].item()
        idx2 = pair[1].item()
        #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
        if Hw_nz[idx1,idx2]:
            freqs.append((pt.real(evals[pair[1]]-evals[pair[0]])).item())
        #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
        if Hw_nz[idx2,idx1]:
            freqs.append((pt.real(evals[pair[0]]-evals[pair[1]])).item())
    freqs = pt.tensor(remove_duplicates(freqs), dtype = real_dtype, device=device)
    return freqs

def get_RFs(A,J, device=default_device):
    nS,nq= get_dimensions(A)
    rf = pt.tensor([], dtype = real_dtype, device=device)
    H0 = get_H0(A,J)
    Hw_shape = gate.get_Xn(nq)
    for q in range(nS):
        rf_q=getFreqs(H0[q], Hw_shape)
        rf=pt.cat((rf,rf_q))
    return rf

def config_90deg_phase_fields(rf, device=default_device):
    '''
    Takes array of control/resonant frequencies (rf). For each frequency, sets up a pair of circular fields rotating pi/2 out of phase.
    First half of omegas are assumed to be more important to desired transition.
    For the love of g*d make sure each omega has a 0 and a pi/2 phase
    '''
    zero_phase = pt.zeros(len(rf),device=device)
    piontwo_phase = pt.ones(len(rf),device=device)*np.pi/2
    phase = pt.cat((zero_phase, piontwo_phase))
    omega = pt.cat((rf,rf))
    return omega, phase.reshape(len(phase),1)

def get_CF_params(J,A):
    '''
    Returns control field frequencies and phase, generating an x field and a y field for each resonant frequency (rf).
    Frequencies will be in same units as J,A.
    '''
    rf = get_RFs(A,J)
    omega,phase = config_90deg_phase_fields(rf)
    return omega,phase


def get_2q_freqs(J,A, all_freqs=True, device=default_device):
    '''
    Analytically determines resonant frequencies for a collection of systems.
    '''
    dA = A[0][0]-A[0][1]

    # if True:
    #     return pt.tensor([-2*J[0]-pt.sqrt(dA**2+4*J[0]**2)])

    if all_freqs:
        w = pt.zeros(4*len(J), device=device)
        for i in range(len(J)):
            w[4*i+2]=-2*J[i]-pt.sqrt(dA**2+4*J[i]**2)
            w[4*i+3]=-2*J[i]+pt.sqrt(dA**2+4*J[i]**2)
            w[4*i] = 2*J[i]-pt.sqrt(dA**2+4*J[i]**2)
            w[4*i+1]=2*J[i]+pt.sqrt(dA**2+4*J[i]**2)
    else:
        w = pt.zeros(2*len(J), device=device)
        for i in range(len(J)):
            w[2*i+0]= (-2*J[i]-pt.sqrt(dA**2+4*J[i]**2))
            w[2*i+1]=(-2*J[i]+pt.sqrt(dA**2+4*J[i]**2))

    return w

################################################################################################################
################        Save Data        #######################################################################
################################################################################################################




def sum_XY_fields(u,m,rf,tN):
    '''
    Sum contributions of all control fields along x and y axes
    '''
    N = u.shape[1] 
    T = pt.linspace(0,tN,N,device=default_device)
    wt = pt.einsum('k,j->kj', rf, T)
    cos_wt = pt.cos(wt)
    sin_wt = pt.sin(wt)
    X_field = pt.sum(pt.einsum('kj,kj->kj', u[:m//2,:],cos_wt),0) + pt.sum(pt.einsum('kj,kj->kj', u[m//2:,:],sin_wt),0)
    Y_field = pt.sum(pt.einsum('kj,kj->kj', u[:m//2:,:],sin_wt),0) - pt.sum(pt.einsum('kj,kj->kj', u[m//2:,:],cos_wt),0)
    return X_field, Y_field

def save_field(u, m, rf, tN, filename, cost_hist):
    '''
    Saves total control field. Assumes each omega has x field and y field part, so may need to send only half of 
    omega tensor to this function while pi/2 offset is being handled manually.
    '''

    pt.save(u, f"fields/{filename}")
    X_field, Y_field = sum_XY_fields(uToMatrix(u,m),m,rf,tN)
    fields = pt.stack((X_field,Y_field))
    pt.save(fields, f"fields/{filename}_XY")
    pt.save(np.array(cost_hist), f"fields/{filename}_cost_hist")
    


def save_system_data(tN, J, A, N, target, filename, fid=None, status='T'):
    '''
    Saves system data to a file.
    
    status carries information about how the optimisation exited, and can take 3 possible values:
        C = complete
        UC = uncomplete - ie reached maximum time and terminated minimize
        T = job terminated by gadi, which means files are not saved (except savepoints)

    '''
    J=pt.real(J); A=pt.real(A)
    if type(J)==pt.Tensor: J=J.tolist()
    with open("fields/"+filename+".txt", 'w') as f:
        f.write("J = "+str(J)+"\n")
        f.write("A = "+str(A.tolist())+"\n")
        f.write("tN = "+str(tN)+"\n")
        f.write("N = "+str(N)+"\n")
        f.write("target = "+str(target.tolist())+"\n")
        f.write("Completion status = "+status+"\n")
        if fid is not None: f.write("fidelity = "+str(fid.tolist())+"\n")

def get_next_ID():
    with open(ID_fn, 'r') as f:
        prev_ID = f.readlines()[-1].split(',')[-1]
    if prev_ID[-1]=='\n': prev_ID = prev_ID[:-1]
    ID = prev_ID[0] + str(int(prev_ID[1:])+1)
    return ID

def get_field_filename(A,tN,N,ID):
    nSys,nq=A.shape
    filename = "{}_{}S_{}q_{}ns_{}step".format(ID,nSys,nq,int(round(tN/nanosecond,0)),N)
    return filename


def log_result(minfid,avgfid,alpha,kap,nS,nq,tN,N,A,status,ID, time_taken):
    '''
    Logs key details of result:
        field filename, minfid, J (MHz), A (MHz), avgfid, fidelities, time date, alpha, kappa, nS, nq, tN (ns), N, ID
    '''

    field_filename = get_field_filename(A,tN,N,ID)
    #fids_formatted = [round(elem,4) for elem in fidelities.tolist()]
    #A_formatted = [round(elem,2) for elem in (pt.real(A)[0]/Mhz).tolist()]
    #J_formatted = [round(elem,2) for elem in (pt.real(J).flatten()/Mhz).tolist()]
    now = datetime.now().strftime("%H:%M:%S %d-%m-%y")
    with open(log_fn, 'a') as f:
       f.write("{},{:.4f},{:.4f},{},{:.3e},{:.3e},{},{},{:.1f},{},{},{}\n".format(field_filename,avgfid,minfid, now, alpha, kap, nS,nq,tN/nanosecond,N,status,time_taken))


def preLog(J,A,tN,N,target,save_data):
    '''
    Logs some data prior to running job incase job is terminated and doesn't get a chance to log at the end.
    If job completes, system_data will be rewritten with more info (eg fidelities).
    '''
    ID = get_next_ID()
    filename = get_field_filename(A,tN,N,ID)
    if save_data: 
        save_system_data(tN, J, A, N, target, filename)
        with open(ID_fn, 'a') as f:
            f.write(f"{ID}\n")
    return ID,filename


################################################################################################################
################        Access Data        #####################################################################
################################################################################################################



def load_system_data(filename):
    '''
    Retrieves system data (exchange, hyperfine, pulse length, target, fidelity) from file.
    '''
    with open("fields/"+filename+'.txt','r') as f:
        lines = f.readlines()
        J = pt.tensor(ast.literal_eval(lines[0][4:-1]), dtype=cplx_dtype)
        A = pt.tensor(ast.literal_eval(lines[1][4:-1]), dtype=cplx_dtype)
        tN = float(lines[2][4:-1])
        N = int(lines[3][3:-1])
        target = pt.tensor(ast.literal_eval(lines[4][9:-1]),dtype=cplx_dtype)
        try:
            fid = ast.literal_eval(lines[6][11:-1])
        except: fid=None

    return J,A,tN,N,target,fid

def load_u(filename, SP=None):
    '''
    Loads saved 'u' tensor and cost history from files
    '''
    u_path = f"fields/{filename}" if SP is None else f"fields/{filename}_SP{SP}"
    u = pt.load(u_path, map_location=pt.device('cpu')).type(cplx_dtype)
    cost_hist = pt.load(f"fields/{filename}_cost_hist")
    return u,cost_hist

def process_u(u,J,A,tN,N,target,cost_hist,save_SP=False):
    '''
    Takes vector form 'u' and system data as input.
    '''
    nS,nq = get_dimensions(A)
    rf = get_RFs(A,J)
    omega,phase = config_90deg_phase_fields(rf)
    m = len(omega)
    x_cf,y_cf = get_control_fields(omega,phase,tN,N)
    H0 = get_H0(A,J)
    fid=fast_fid(u,H0,x_cf,y_cf,tN,target)[0]
    print(f"Fidelities: {fid}")


    U = time_evol(u,H0,x_cf,y_cf,tN)
    X,_P = propagate(U,target)

    if save_SP:
        preLog(J,A,tN,N,target,True)
        ID = get_next_ID()
        filename = get_field_filename(A,tN,N,ID)
        avgfid=sum(fid)/len(fid) 
        log_result(min(fid), avgfid,0,1,nS,nq,tN,N,A,'SP',ID)
        save_system_data(tN,J,A,N,target,filename,fid)
        save_field(u,m,rf,tN,filename,cost_hist)
    else:
        filename = None

    plotFields(u,m,tN,omega,target,X,A,cost_hist,save_SP,plotLabel=filename)

def process_u_file(filename,SP=None, save_SP = False):
    '''
    Gets fidelities and plots from files. save_SP param can get set to True if a save point needs to be logged.
    '''
    J,A,tN,N,target,_fid = load_system_data(filename)
    u,cost_hist = load_u(filename,SP=SP)
    process_u(u,J,A,tN,N,target,cost_hist,save_SP=save_SP)



################################################################################################################
################        Run GRAPE        #######################################################################
################################################################################################################


def init_u(tN,m,N, device='cpu'):
    '''
    Generates u0. Less important freuencies are placed in the second half of u0, and can be experimentally initialised to lower values.
    Initialised onto the cpu by default as it will normally be passed to scipy minimize.
    '''
    mult=1
    u0 = hbar/(g*mu*tN) * pt.ones(m,N,dtype=cplx_dtype, device=device)/tesla * mult
    u0[2::3] /= mult 
    return uToVector(u0)

def interpolate_u(u,m, N_new):
    '''
    Takes an m x N matrix, u_mat, and produces an m x N_new matrix whose rows are found by connecting the dots between the points
    give in the rows of u_mat. This allows us to produce a promising initial guess for u with a large number of timesteps using
    a previously optimised 'u' with a lower number of timesteps. 
    '''
    u_mat = uToMatrix(u,m)
    m,N = u_mat.shape 
    u_new = pt.zeros(m,N_new, dtype=cplx_dtype)
    for j in range(N_new):
        u_new[:,j] = u_mat[:, j*N//N_new]


    return uToVector(u_new)

def refine_u(filename, N_new):
    '''
    Uses a previous optimization performed with N timesteps to generate an initial guess u0 with 
    N_new > N timesteps, which is then optimised.
    '''
    J,A,tN,N,target,_fid = load_system_data(filename)
    u,cost_hist = load_u(filename)
    m=2*len(get_RFs(A,J))
    u0 = interpolate_u(u,m,N_new)
    optimise2(target,N_new,tN,J,A,u0=u0)


def single_qubit_unitary(target,tN,N,alpha=0,w0=0):
    '''
    very broken
    
    '''
    m=2
    u0 = 1*np.pi/(g*mu*tN) * pt.ones(m,N) * 1/tesla
    omega=pt.ones(2)*w0

    H0_single = w0/2 * gate.Z
    U0_N = pt.matrix_exp(-1j*tN*H0_single)
    target = matmul3(U0_N,target,dagger(U0_N))
    H0 = pt.einsum('qj,ab->qjab',pt.ones(1,N),H0_single)

    fun = lambda u: fast_cost(u,m,1,target,tN,H0,alpha=alpha)
    opt = minimize(fun,u0,method='CG',jac=jac)
    u=pt.tensor(opt.x, device=default_device, dtype=cplx_dtype)
    U = time_evol(u,H0,tN)
    X,P = propagate(U,target)


    print(opt.message)
    print(f"final average cost = {opt.fun}")
    print(f"Fidelity = {fast_fid(u,m,1,target,tN,H0)[0]}")
    print(f"Uf = {X[0][-1]}")
    plotFields(u,m,tN,omega)
    
################################################################################################################
################        Frequency plots        ########################################################
################################################################################################################
def visualise_frequencies(A, J):
    nq = len(A[0])
    nSys = len(J)
    freqs = get_RFs(A,J)
    colours = ['plum','red', 'salmon', 'orange', 'yellow', 'green', 'turquoise', 'blue','purple', 'pink','chocolate']
    for i in range(len(freqs)):
        plt.vlines(freqs[i],0,1,linewidth=0.5)
        #plt.xscale('logit')
    plt.show()


################################################################################################################
################        FREE EVOLUTION SIMULATION        #######################################################
################################################################################################################
def free_evolution_fidelity(J,A,tN,N,target):
    '''
    Simulates evolution by observing the change in fidelity of unitary U against target 
    as time passes with no applied control fields.
    '''
    tN=tN*nanosecond; J=J*Mhz; A=A*Mhz
    nS,nq=get_dimensions(A)
    dim=2**nq
    m=1; u = pt.zeros(m*N)
    Hw = pt.zeros(nS,m,N,dim,dim, dtype=cplx_dtype)
    H0 = get_H0(A,J,N)
    S = get_S_matrix(J,A) if work_in_eigenbasis else None
    U = time_evol(u,Hw,H0,tN, simSteps=1,S=S)
    X,_P = propagate(U,target)

    ax = plt.subplot()
    fids = get_fidelity_progress(X,tN,target)
    plot_fidelity_progress(ax,fids,tN)
    plt.show()
    return fids[-1]

################################################################################################################
################        QUANTUM STATE SIMULATION        ########################################################
################################################################################################################
'''
For visualisation and debugging purposes
'''

def psiEvol(psi0,X):
    '''
    Evolves initial state psi0 through N time steps, returning array Psi
    which contains the state at each time step.
    '''
    return pt.matmul(X,psi0)

def init_1q_Psi(theta,phi):
    return pt.tensor([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], device=default_device)

def init_rand_psi(nq):
    d=2**nq 
    return normalise(pt.rand(d, dtype=cplx_dtype)+1j*pt.rand(d, dtype=cplx_dtype))



def psiAngles(psi):
    theta = 2*np.arccos(np.abs(psi[0]))
    phi = np.angle(psi[1])
    return theta,phi

def plotPsi(psi,tN, ax=None):
    N,d = psi.shape
    nq = int(np.log2(d))
    T = pt.linspace(0,tN,N)
    if ax==None: ax=plt.subplot()
    ax.set_xlabel('t (ns)')
    squamp =pt.einsum('ja,ja->ja',psi,pt.conj(psi))
    for a in range(d):
        if not squamp[:,a].any(): continue
        state = np.binary_repr(a,nq)
        ax.plot(T/nanosecond,squamp[:,a],label='|<'+state+'|$\psi>|^2$')
    ax.legend() 


def plotPsi_1q(Psi,tN):
    N=len(Psi)
    theta = pt.zeros(N); phi = pt.zeros(N)
    for i in range(N):
        theta[i],phi[i] = psiAngles(Psi[i])
    T = np.linspace(0,tN,N)
    plt.plot(T,theta, label='theta')
    plt.plot(T,phi, label='phi')
    plt.legend()
    plt.show()

def wf_freeEvolution(J,A,psi0,tN,N):
    H0 = get_H0(A,J)[0]
    T = pt.linspace(0,tN,N, dtype=cplx_dtype)
    H0T = pt.einsum('ab,j->jab',H0,T)
    U0 = pt.matrix_exp(-1j*H0T)
    psi = pt.matmul(U0,psi0)
    return psi



