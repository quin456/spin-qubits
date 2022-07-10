

# coding tools
from abc import abstractmethod
from pdb import set_trace
import time
import warnings
warnings.filterwarnings("ignore")

# library imports
import torch as pt
import numpy as np
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import itertools    
from scipy import linalg as la
from scipy.optimize import minimize
from numpy import random as rand
from numpy import sin, cos, sqrt
from datetime import datetime
import ast


# Other python files in project
from atomic_units import *
import gates as gate
from utils import *
from data import *
from visualisation import plot_fidelity_progress

time_exp = 0
time_prop = 0
time_grad=0
time_fid=0

mergeprop_g=False




jac=True # keep as True unless testing cost grad function
save_mem = False # save mem in Hamiltonian arrays. 
work_in_eigenbasis = False # Should be set to False unless testing
interaction_picture= False # Should be set to False unless testing 
''' save_mem incompatible with work_in_eigenbasis and interaction picture '''




ngpus = pt.cuda.device_count()

np.set_printoptions(4)
pt.set_printoptions(sci_mode=True)


log_fn = dir+'logs/log.txt'
ID_fn = dir+'logs/ID_tracker.txt'
precision_loss_msg = "Desired error not necessarily achieved due to precision loss."




################################################################################################################
################        HELPER FUNCTIONS        ################################################################
################################################################################################################


def CNOT_targets(nS,nq,device=default_device):
    if nq==2:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.CX.to(device))
    elif nq==3:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.CX3q.to(device))

def ID_targets(nS, nq, device=default_device):
    if nq==2:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.Id2.to(device))
    elif nq==3:
        return pt.einsum('i,ab->iab',pt.ones(nS, device=device),gate.Id3.to(device))

def SWAP_targets(n, device=default_device):
    return pt.einsum('i,ab->iab',pt.ones(n, device=device),gate.swap.to(device))

def RSW_targets(n, device=default_device): 
    return pt.einsum('i,ab->iab',pt.ones(n, device=device),gate.root_swap.to(device))




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
################        GRAPE IMPLEMENTATION        ############################################################
################################################################################################################





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





def get_unit_CFs(omega,phase,tN,N,device=default_device):
    T = pt.linspace(0,tN,N, device=device)
    wt = pt.einsum('k,j->kj', omega,T)
    x_cf = pt.cos(wt-phase)
    y_cf = pt.sin(wt-phase)
    return x_cf.type(cplx_dtype), y_cf.type(cplx_dtype)


def get_control_fields(omega,phase,tN,N,device=default_device):
    '''
    Returns x_cf, y_cf, which relate to transverse control field, and have units of joules so that 'u' can be unitless.
    '''
    x_cf,y_cf = get_unit_CFs(omega,phase,tN,N,device=device)
    x_cf*=0.5*g_e*mu_B*(1*tesla)
    y_cf*=0.5*g_e*mu_B*(1*tesla)
    return x_cf, y_cf


    
        



class Grape:
    '''
    General GRAPE class.
    '''
    def __init__(self, tN, N, target, rf, u0=None, hist0=[], max_time=9999999, save_data=False, sp_distance = 99999*3600):
        self.tN=tN
        self.N=N
        self.target=target
        self.rf = self.get_RFs() if rf is None else rf
        self.omega,self.phase = config_90deg_phase_fields(self.rf); self.m = len(self.omega) 
        self.m = len(self.omega)
        self.x_cf,self.y_cf = get_control_fields(self.omega,self.phase,self.tN,N)

        if u0 is None: u0=self.init_u()
        self.u0=u0
        self.max_time=max_time
        self.start_time = time.time()
        self.save_data = save_data
        self.H0 = self.get_H0()

        # allow for H0 with no systems axis
        self.reshaped=False
        if len(self.H0.shape)==2:
            self.H0 = self.H0.reshape(1,*self.H0.shape)
            self.target = self.target.reshape(1,*self.target.shape)
            self.reshaped=True
            
        self.nS = len(self.H0)

        self.sp_count = 0
        self.sp_distance = sp_distance

        self.cost_hist=hist0 if hist0 is not None else []  
        self.filename = None

        
        # Log job start
        if save_data: self.preLog(save_data)  

    @abstractmethod
    def get_RFs(self):
        pass
    
    def u_mat(self):
        return uToMatrix(self.u, self.m)


    def get_rec_min_N(self, printFreqs=False, printPeriods=False):
        
        N_period=40 # recommended min number of timesteps per period
        rf=self.get_RFs()
        T=1e3/rf
        max_w=pt.max(rf).item()
        rec_min_N = int(np.ceil(N_period*max_w*Mhz*self.tN*nanosecond/(2*np.pi)))
        
        if printFreqs: print(f"resonant freqs = {rf}")
        if printPeriods: print(f"T = {T}")
        print(f"Recommened min N = {rec_min_N}")

        return rec_min_N

    def time_evolution(self, u, simSteps=1):
        '''
        Calculates the time-evolution operators corresponding to each of the N timesteps,
        and returns as an array, U.
        '''
        d=2**self.nq
        nS=len(self.H0)
        simSteps=1
        N = int(len(u)/self.m)
        dt = self.tN/self.N

        # add axes
        Hw = pt.einsum('s,kjab->skjab', pt.ones(nS), self.Hw)
        H0 = pt.einsum('j,sab->sjab', pt.ones(N), self.H0)

        H_control = pt.einsum('kj,skjab->sjab', uToMatrix(u,self.m), Hw)


        if interaction_picture:
            H = H_control
        else:
            H = H_control + H0
            
        if simSteps==1:
            U_flat = pt.matrix_exp(-1j*pt.reshape(H, (nS*N,d,d))*(dt/hbar))
            U = pt.reshape(U_flat, (nS,N,d,d))
            return U
        else:
            # scrap this?
            U = pt.matrix_exp(pt.zeros(N,2**self.nq,2**self.nq, dtype=cplx_dtype, device=default_device))
            U_s = pt.matrix_exp(-1j*H*((dt/simSteps)/hbar))
            for i in range(simSteps):
                U = pt.matmul(U_s, U)
            return U

    def propagate(self, U, device=default_device):
        '''
        Determines total time evolution operators to evolve system from t=0 to time t(j) and stores in array P. 
        Backward propagates target evolution operators and stores in X.

        This function is sufficiently general to work on all GRAPE implementations.
        '''
        target=self.target
        nS=len(U)
        N=len(U[0])
        dim = U[0].shape[1]  # forward propagated time evolution operator
        nq = get_nq(dim)
        if target is None: target = CNOT_targets(nS,nq)
        X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
        X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator
        P = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
        P[:,N-1,:,:] = target    # backwards propagated target 

        for j in range(1,N):
            X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
            P[:,-1-j,:,:] = pt.matmul(dagger(U[:,-j,:,:]),P[:,-j,:,:])
        
        return X,P
        
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


    def fidelity(self, u):

        '''
        Determines the time evolution resulting from control field amplitudes in 'u', and 
        calculates the corresponding fidelity Phi=<Uf,Ut> of the final time evolution operator 
        Uf with the target operator Ut, and its gradient with respect to 'u'.

        Inputs:

            u: vector containing learning parameters, which are magnetic field amplitudes
                for each control field at each timestep.

            m: number of control fields

            target: target unitary for time evolution operator

            Hw: function of time which returns vector of control Hamiltonian values
                at each time.
            
            H0: Unperturbed Hamiltonian due to static background z-field

            tN: duration of gate

        '''


        # determine time evolution resulting from control fields
        nq=self.nq
        target=self.target
        tN=self.tN
        N = self.N
        nS=self.nS
        H0 = self.H0
        Hw = self.Hw
        d=2**nq
        m = self.m

        # add axes
        Hw = pt.einsum('s,kjab->skjab', pt.ones(nS), Hw)
        H0 = pt.einsum('j,sab->sjab', pt.ones(N), H0)


        U = self.time_evolution(u)

        X,P = self.propagate(U)
        
        Uf = P[:,-1]; Ut = X[:,-1]
        # fidelity of resulting unitary with target
        Phi = batch_IP(Ut,Uf); Phi = pt.real(Phi*pt.conj(Phi))

        # calculate grad of fidelity

        XP_IP = batch_IP(X,P)
        #PHX_IP = batch_IP(P,1j*dt*pt.matmul(Hw,X))
        HwX=pt.einsum('skjab,sjbc->skjac' , Hw, X)
        dt = tN/len(X)
        PHX_IP =  batch_trace(pt.einsum('sjab,skjbc->skjac', dagger(P), 1j*dt*HwX)) / d


        dPhi = -2*pt.real( pt.einsum('skj,sj->skj', PHX_IP, XP_IP) ) / hbar
        return Phi, dPhi


    def cost(self, u):
        '''
        Generalised GRAPE cost function, taking arbitrary Hamiltonians as input. 
        More optimised cost functions may be needed for specific implementations
        (ie running many CNOTs in parallel, where Hw is constructed on the fly).
        '''
        nS=self.nS
        u = pt.tensor(u,dtype=cplx_dtype, device=default_device)

        Phi, dPhi = self.fidelity(u)


        J = 1 - pt.sum(Phi,0)/nS
        dJ = -uToVector(pt.sum(dPhi,0))/nS



        
        J=J.item(); dJ = dJ.cpu().detach().numpy() 
        return J, dJ

    def init_u(self, device='cpu'):
        '''
        Generates u0. Less important freuencies are placed in the second half of u0, and can be experimentally initialised to lower values.
        Initialised onto the cpu by default as it will normally be passed to scipy minimize.
        '''
        mult=1
        u0 = hbar/(g_e*mu_B*self.tN) * pt.ones(self.m,self.N,dtype=cplx_dtype, device=device)/tesla * mult
        u0[2::3] /= mult 
        return uToVector(u0)*0.1

    def run(self):
        callback = self.check_time if self.max_time is not None else None
        try:
            opt=minimize(self.fun,self.u0,method='CG',jac=True, callback=callback)
            print(f"nit = {opt.nfev}, nfev = {opt.nit}")
            self.u=pt.tensor(opt.x, device=default_device)
            self.status='C' #complete
        except TimeoutError:
            global u_opt_timeout
            self.u = pt.tensor(u_opt_timeout, dtype=cplx_dtype, device=default_device)
            self.status='UC' #uncomplete
        self.time_taken = time.time() - self.start_time


    def result(self, show_plot=True, minprint=False):


        self.print_info(minprint)
        # Plot fields and process data
        self.plotFields(show_plot=show_plot)

        fidelities = self.fidelity(self.u)[0]
        avgfid=sum(fidelities)/len(fidelities)
        minfid = min(fidelities).item()
        

        alpha=0
        if self.save_data:
            # save_system_data will overwrite previous file if job was not terminated by gadi
            self.log_result(minfid,avgfid,alpha)
            self.save_system_data(self, self.target, self.filename, fid=fidelities, status=self.status)
            self.save_field()

    def print_info(self, minprint):

        fidelities = self.fidelity(self.u)[0]
        avgfid=sum(fidelities)/len(fidelities)
        minfid = min(fidelities).item()

        print(f"Average fidelity = {avgfid}")
        if not minprint:
            print(f"Fidelities = {fidelities}")
            print(f"Min fidelity = {minfid}")
            print(f"Time taken = {self.time_taken}")
            print(f"GPUs requested = {ngpus}")
            global time_grad, time_exp, time_prop, time_fid 
            print(f"texp={time_exp}, tprop={time_prop}, tgrad = {time_grad}, tfid={time_fid}")


    def get_X(self):
        U = self.time_evolution(self.u)
        X,P = self.propagate(U)
        Uf = X[0][-1]
        return X


    def sum_XY_fields(self):
        '''
        Sum contributions of all control fields along x and y axes
        '''
        T = pt.linspace(0, self.tN, self.N, device=default_device)
        wt = pt.einsum('k,j->kj', self.rf, T)
        cos_wt = pt.cos(wt)
        sin_wt = pt.sin(wt)
        X_field = pt.sum(pt.einsum('kj,kj->kj', self.u_mat()[:self.m//2,:],cos_wt),0) + pt.sum(pt.einsum('kj,kj->kj', self.u_mat()[self.m//2:,:],sin_wt),0)
        Y_field = pt.sum(pt.einsum('kj,kj->kj', self.u_mat()[:self.m//2:,:],sin_wt),0) - pt.sum(pt.einsum('kj,kj->kj', self.u_mat()[self.m//2:,:],cos_wt),0)
        return X_field, Y_field


    ################################################################################################################
    ################        VISUALISATION        ###################################################################
    ################################################################################################################

    def plotFields(self, show_plot=True):
        u=self.u 
        X = self.get_X()
        omegas=self.omega
        m=self.m
        tN=self.tN
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

        self.plot_cost_hist(ax[1,1])
            

        for i in range(len(w_np)):
            if i<len(w_np)//2:
                B1 = np.cos(w_np[i]*T)
            else:
                B1=np.sin(w_np[i]*T)
            prod=np.multiply(u_m[i],B1)
            
            ax[0,0].plot(t_axis,u_m[i], label='u'+str(i))
            #ax[1,0].plot(t_axis,1e3*prod,label='u'+str(i)+'*Hw'+str(i)+" (mT)")
                
        X_field, Y_field = self.sum_XY_fields()
        self.plot_XY_fields(ax[0,1],X_field,Y_field)
        transfids = fidelity_progress(X,self.target)
        plot_fidelity_progress(ax[1,0],transfids,tN,legend=False)
        ax[0,0].set_xlabel("time (ns)")


        if self.save_data:
            fig.savefig(f"{dir}plots/{self.filename}")
        if show_plot: plt.show()


    def plot_XY_fields(self, ax, X_field, Y_field):
        X_field = X_field.cpu(); Y_field=Y_field.cpu()
        N=len(X_field)
        t_axis = np.linspace(0, self.tN/nanosecond, N)
        ax.plot(t_axis,X_field*1e3,label='total x-field (mT)')
        ax.plot(t_axis,Y_field*1e3,label='total y-field (mT)')
        ax.set_xlabel("time (ns)")
        ax.legend()
        
    def plot_cost_hist(self, ax):
        ax.plot(self.cost_hist, label='cost')
        ax.set_xlabel('Iterations')
        ax.legend()
        return ax


    def save_field(self):
        '''
        Saves total control field. Assumes each omega has x field and y field part, so may need to send only half of 
        omega tensor to this function while pi/2 offset is being handled manually.
        '''
        filepath = f"{dir}fields/{self.filename}"
        pt.save(self.u, filepath)
        X_field, Y_field = self.sum_XY_fields()
        fields = pt.stack((X_field,Y_field))
        pt.save(fields, f"{filepath}_XY")
        pt.save(np.array(self.cost_hist), f"{filepath}_cost_hist")



    # LOGGING

    def log_result(self, minfid,avgfid,alpha):
        '''
        Logs key details of result:
            field filename, minfid, J (MHz), A (MHz), avgfid, fidelities, time date, alpha, kappa, nS, nq, tN (ns), N, ID
        '''

        field_filename = self.get_field_filename()
        #fids_formatted = [round(elem,4) for elem in fidelities.tolist()]
        #A_formatted = [round(elem,2) for elem in (pt.real(A)[0]/Mhz).tolist()]
        #J_formatted = [round(elem,2) for elem in (pt.real(J).flatten()/Mhz).tolist()]
        now = datetime.now().strftime("%H:%M:%S %d-%m-%y")
        with open(log_fn, 'a') as f:
            f.write("{},{:.4f},{:.4f},{},{:.3e},{},{},{:.1f},{},{},{}\n".format(
                field_filename, avgfid, minfid, now, alpha, self.nS, self.nq, self.tN/nanosecond, self.N, self.status, self.time_taken))


    def preLog(self,save_data):
        '''
        Logs some data prior to running job incase job is terminated and doesn't get a chance to log at the end.
        If job completes, system_data will be rewritten with more info (eg fidelities).
        '''
        self.ID = self.get_next_ID()
        self.filename = self.get_field_filename()
        if save_data: 
            self.save_system_data(self.filename)
            with open(ID_fn, 'a') as f:
                f.write(f"{self.ID}\n")





class GrapeESR(Grape):

    def __init__(self, J, A, tN, N, target=None, rf=None, u0=None, hist0=[], max_time=9999999, save_data=False, alpha=0):

        self.nS,self.nq=get_dimensions(A)
        self.J=J 
        self.A=A 
        self.tN=tN
        self.N=N
        self.target=target if target is not None else CNOT_targets(self.nS, self.nq)
        self.rf=self.get_RFs() if rf is None else rf
        super().__init__(tN, N, self.target, rf, u0, hist0, max_time, save_data)
        

        fun = self.cost


        self.fun=fun
        self.alpha=alpha

    def get_H0(self, include_HZ=False, device=default_device):
        '''
        Free hamiltonian of each system. Reduced because it doesn't multiply by N timesteps, which is a waste of memory.
        
        Inputs:
            A: (nS,nq), J: (nS,) for 2 qubit or (nS,2) for 3 qubits
        '''
        nS, nq = get_dimensions(self.A)
        d = 2**nq

        # Zeeman splitting term is generally rotated out and does not need to be simulated.
        if include_HZ:
            gamma_e = g_e*mu_B
            HZ = 0.5 * gamma_e * B0 * gate.get_Zn(nq)
        else:
            HZ = pt.zeros(d,d)


        if nq==3:
            H0 = pt.einsum('sq,qab->sab', self.A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('sc,cab->sab', self.J.to(device), gate.get_coupling_matrices(nq).to(device)) + pt.einsum('s,ab->sab',pt.ones(nS),HZ)
        elif nq==2:
            H0 = pt.einsum('sq,qab->sab', self.A.to(device), gate.get_PZ_vec(nq).to(device)) + pt.einsum('s,ab->sab', self.J.to(device), gate.get_coupling_matrices(nq).to(device)) + pt.einsum('s,ab->sab',pt.ones(nS),HZ)
        
        
        return H0.to(device)

    def get_RFs(self, device=default_device):
        A = self.A; J=self.J
        nS,nq= get_dimensions(A)
        rf = pt.tensor([], dtype = real_dtype, device=device)
        H0 = self.get_H0()
        Hw_shape = gate.get_Xn(nq)
        for q in range(nS):
            rf_q=getFreqs(H0[q], Hw_shape)
            rf=pt.cat((rf,rf_q))
        return rf

        

    def time_evolution(self, u, device=default_device):
        m,N = self.x_cf.shape; nS,d = self.H0.shape[:-1]; nq = get_nq(d)
        sig_xn = gate.get_Xn(nq,device); sig_yn = gate.get_Yn(nq,device)
        dt = self.tN/N
        u_mat = uToMatrix(u,m)
        x_sum = pt.einsum('kj,kj->j',u_mat,self.x_cf).to(device)
        y_sum = pt.einsum('kj,kj->j',u_mat,self.y_cf).to(device)
        H = pt.einsum('j,sab->sjab',pt.ones(N,device=device),self.H0.to(device)) + pt.einsum('s,jab->sjab',pt.ones(nS,device=device),pt.einsum('j,ab->jab',x_sum,sig_xn)+pt.einsum('j,ab->jab',y_sum,sig_yn))
        H=pt.reshape(H, (nS*N,d,d))
        U = pt.matrix_exp(-1j*H*dt)
        del H
        U = pt.reshape(U, (nS,N,d,d))
        return U

    def fidelity(self,u, device=default_device):
        '''
        Adapted grape fidelity function designed specifically for multiple systems with transverse field control Hamiltonians.
        '''

        H0=self.H0
        x_cf=self.x_cf
        y_cf=self.y_cf
        tN=self.tN
        target=self.target

        m,N = x_cf.shape; nS,d = H0.shape[:-1]; nq = get_nq(d)
        sig_xn = gate.get_Xn(nq, device); sig_yn = gate.get_Yn(nq, device)

        t0 = time.time()
        if pt.cuda.device_count()==2:
            U2 = self.time_evolution(u, device='cuda:1')
            U=U2.to('cuda:0')
            del U2
        else:
            U = self.time_evolution(u, device=device)
        global time_exp 
        time_exp += time.time()-t0

        t0 = time.time()
        X,P = self.propagate(U,device=device)
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


        return Phi, dPhi


    def cost(self,u):
        '''
        Fast cost is faster and more memory efficient but less generalised cost function which can be used for CNOTs.

        INPUTS:
            u: control field amplitudes
            x_cf: 0.5*g*mu*(1T)*cos(wt-phi) - An (m,N) tensor recording coefficients of sigma_x for Hw_kj
            y_cf: 0.5*g*mu*(1T)*sin(wt-phi) - An (m,N) tensor recording coefficients of sigma_y for Hw_kj
                Hw = x_cf * sig_x + y_cf * sig_y
        
        '''
        MP=False; kappa=1; alpha=0
        H0=self.H0
        nS=self.nS
        J=self.J
        A=self.A
        x_cf=self.x_cf
        y_cf=self.y_cf
        tN=self.tN
        target=self.target
        cost_hist=self.cost_hist

        u=pt.tensor(u, dtype=cplx_dtype, device=default_device)
        m=len(x_cf)
        t0 = time.time()
        Phi,dPhi = self.fidelity(u)
        global time_fid 
        time_fid += time.time()-t0
        J = 1 - pt.sum(Phi,0)/nS
        dJ = -uToVector(dPhi)/nS
        J=J.item(); dJ = dJ.cpu().detach().numpy()
        cost_hist.append(J)

        # J_fluc,dJ_fluc = fluc_cost(u,m,alpha)
        # J+=J_fluc; dJ+=dJ_fluc
        
        return J, dJ * tN/nS / kappa


    #class FieldOptimiser(object):
    '''
    Optimisation object which passes fidelity function to scipy optimiser, and handles termination of 
    optimisation once predefined time limit is reached.
    '''

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

    def run(self):
        callback = self.check_time if self.max_time is not None else None
        try:
            opt=minimize(self.fun,self.u0,method='CG',jac=True, callback=callback)
            print(f"nit = {opt.nfev}, nfev = {opt.nit}")
            self.u=pt.tensor(opt.x, device=default_device)
            self.status='C' #complete
        except TimeoutError:
            global u_opt_timeout
            self.u = pt.tensor(u_opt_timeout, dtype=cplx_dtype, device=default_device)
            self.status='UC' #uncomplete
        self.time_taken = time.time() - self.start_time


        
    ################################################################################################################
    ################        Save Data        #######################################################################
    ################################################################################################################
    def save_system_data(self, filename, fid=None, status='T'):
        '''
        Saves system data to a file.
        
        status carries information about how the optimisation exited, and can take 3 possible values:
            C = complete
            UC = uncomplete - ie reached maximum time and terminated minimize
            T = job terminated by gadi, which means files are not saved (except savepoints)

        '''
        J=pt.real(self.J)/Mhz; A=pt.real(self.A)/Mhz; tN = self.tN/nanosecond
        if type(J)==pt.Tensor: J=J.tolist()
        with open(f"{dir}fields/{filename}.txt", 'w') as f:
            f.write("J = "+str(J)+"\n")
            f.write("A = "+str((A).tolist())+"\n")
            f.write("tN = "+str(tN)+"\n")
            f.write("N = "+str(self.N)+"\n")
            f.write("target = "+str(self.target.tolist())+"\n")
            f.write("Completion status = "+status+"\n")
            if fid is not None: f.write("fidelity = "+str(fid.tolist())+"\n")

    @staticmethod
    def get_next_ID():
        with open(ID_fn, 'r') as f:
            prev_ID = f.readlines()[-1].split(',')[-1]
        if prev_ID[-1]=='\n': prev_ID = prev_ID[:-1]
        ID = prev_ID[0] + str(int(prev_ID[1:])+1)
        return ID

    def get_field_filename(self):
        filename = "{}_{}S_{}q_{}ns_{}step".format(self.ID,self.nS,self.nq,int(round(self.tN/nanosecond,0)),self.N)
        return filename





def load_u(filename, SP=None):
    '''
    Loads saved 'u' tensor and cost history from files
    '''
    u_path = f"{dir}fields/{filename}" if SP is None else f"fields/{filename}_SP{SP}"
    u = pt.load(u_path, map_location=pt.device('cpu')).type(cplx_dtype)
    cost_hist = pt.load(f"{dir}fields/{filename}_cost_hist")
    return u,cost_hist






def process_u_file(filename,SP=None, save_SP = False):
    '''
    Gets fidelities and plots from files. save_SP param can get set to True if a save point needs to be logged.
    '''
    grape = load_system_data(filename)
    u,cost_hist = load_u(filename,SP=SP)
    grape.u=u 
    grape.hist=cost_hist
    grape.plotFields()





def load_system_data(filename):
    '''
    Retrieves system data (exchange, hyperfine, pulse length, target, fidelity) from file.
    '''
    with open(dir+"fields/"+filename+'.txt','r') as f:
        lines = f.readlines()
        J = pt.tensor(ast.literal_eval(lines[0][4:-1]), dtype=cplx_dtype)
        A = pt.tensor(ast.literal_eval(lines[1][4:-1]), dtype=cplx_dtype)
        tN = float(lines[2][4:-1])
        N = int(lines[3][3:-1])
        target = pt.tensor(ast.literal_eval(lines[4][9:-1]),dtype=cplx_dtype)
        try:
            fid = ast.literal_eval(lines[6][11:-1])
        except: fid=None
    grape = GrapeESR(J,A,tN,N,None,target)
    return grape






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

def transform_Hw(Hw,S):
    '''
    Applies transformation S.T @ Hw @ S for each system, to transform Hw into frame whose basis vectors are the columns of S.
    Hw: (m,N,d,d)
    S: (nS,d,d)
    '''
    if len(Hw.shape)==4:
        HS = pt.einsum('kjab,sbc->skjac', Hw, S)
    else:
        HS = pt.einsum('skjab,sbc->skjac', Hw, S)
    H_trans = pt.einsum('sab,skjbc->skjac', dagger(S),HS)
    return H_trans


def evolve_Hw(Hw,U):
    '''
    Evolves Hw by applying U @ H0 @ U_dagger to every control field at each timestep for every system.
    Hw: Tensor of shape (m,N,d,d), describing m control Hamiltonians having dimension d for nS systems at N timesteps.
    U: Tensor of shape (nS,N,d,d), ""
    '''
    UH = pt.einsum('sjab,kjbc->skjac',U,Hw)
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
        Hw = 0.5*g_e*mu_B * 1*tesla * ( X_field_ham + Y_field_ham )
    else:
        Hw = 0.5*g_e*mu_B * 1*tesla * pt.cat((X_field_ham, Y_field_ham),0)
    return Hw

def ignore_tensor(trans,d):
    nS=len(trans)
    ignore = pt.ones(len(trans),d,d)
    for s in range(nS):
        for i in range(len(trans[s])):
            ignore[s,int(trans[s][i][0])-1, int(trans[s][i][1])-1] = 0
            ignore[s,int(trans[s][i][1])-1, int(trans[s][i][0])-1] = 0
    return pt.cat((ignore,ignore))



def get_HA(A, device=default_device):
    nS, nq = get_dimensions(A)
    d = 2**nq
    if nq==3:
        HA = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device))
    elif nq==2:
        HA = pt.einsum('sq,qab->sab', A.to(device), gate.get_PZ_vec(nq).to(device))
    return HA.to(device)

def get_HJ(J,nq, device=default_device):
    nS=len(J)
    d = 2**nq
    if nq==3:
        HJ = pt.einsum('sc,cab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device))
    elif nq==2:
        HJ =pt.einsum('s,ab->sab', J.to(device), gate.get_coupling_matrices(nq).to(device))
    return HJ.to(device)





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


def getFreqs_broke(H0,Hw_shape,device=default_device):
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

    for i in range(len(H0)):
        for j in range(i+1,len(H0)):
            if Hw_nz[i,j]:
                freqs.append((pt.real(evals[i]-evals[j])).item())

    # for i in range(len(pairs)):
    #     # The difference between energy levels i,j will be a resonant frequency if the control field Hamiltonian
    #     # has a non-zero (i,j) element.
    #     pair = pairs[i]
    #     idx1=pair[0].item()
    #     idx2 = pair[1].item()
    #     #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
    #     if Hw_nz[idx1,idx2]:
    #         freqs.append((pt.real(evals[pair[1]]-evals[pair[0]])).item())
    #     #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
    #     # if Hw_nz[idx2,idx1]:
    #     #     freqs.append((pt.real(evals[pair[0]]-evals[pair[1]])).item())
    freqs = pt.tensor(remove_duplicates(freqs), dtype = real_dtype, device=device)
    return freqs


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
################        Run GRAPE        #######################################################################
################################################################################################################



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
    run_optimisation(target,N_new,tN,J,A,u0=u0)







################################################################################################################
################        Run GRAPE        #######################################################################
################################################################################################################