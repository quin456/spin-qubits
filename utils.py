

from pathlib import Path
import os
dir = os.path.dirname(__file__)
os.chdir(dir)


from gates import default_device, cplx_dtype
from data import *
from atomic_units import *



import torch as pt 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 

from pdb import set_trace

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']





def convert_Mhz(A,J):
    ''' A,J need to be multiplied by plancks constant to be converted to Joules (which are equal to Hz given hbar=1) '''
    return h_planck*A*Mhz, h_planck*J*Mhz


def get_nq(d):
    ''' Takes the dimension of the Hilbert space as input, and returns the number of qubits. '''
    return int(np.log2(d))

def forward_prop(U,device=default_device):
    '''
    Forward propagates U suboperators. U has shape (N,d,d) or (nS,N,d,d)
    '''
    if len(U.shape)==3: 
        U=U.reshape(1,*U.shape)
        sys_axis=False 
    else:
        sys_axis=True

    nS,N,dim,dim=U.shape
    nq = get_nq(dim)
    X = pt.zeros((nS,N,dim,dim), dtype=cplx_dtype, device=device)
    X[:,0,:,:] = U[:,0]       # forward propagated time evolution operator
    
    for j in range(1,N):
        X[:,j,:,:] = pt.matmul(U[:,j,:,:],X[:,j-1,:,:])
    
    if sys_axis:
        return X
    else:
        return X[0]