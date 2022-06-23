
import torch as pt
import numpy as np


ngpus = pt.cuda.device_count()
default_device = 'cuda:0' if ngpus>0 else 'cpu'

cplx_dtype = pt.complex128
real_dtype = pt.float64
def kron3(A,B,C):
    '''  Returns kronecker product of 3 matrices A,B,C.  '''
    return pt.kron(pt.kron(A,B),C)
    

X = pt.tensor([
    [0,1],
    [1,0]], dtype=cplx_dtype, device=default_device)
Y = pt.tensor([
    [0,-1j],
    [1j,0]], dtype=cplx_dtype, device=default_device)
Z = pt.tensor([
    [1,0],
    [0,-1]], dtype=cplx_dtype, device=default_device)

# spin matrices / hbar
Ix = X/2; Iy = Y/2; Iz = Z/2

Ip = pt.tensor([
    [0,1],
    [0,0]], dtype=cplx_dtype, device=default_device)
Im = pt.tensor([
    [0,0],
    [1,0]], dtype=cplx_dtype, device=default_device)

Id = pt.tensor([
    [1,0],
    [0,1]], dtype=cplx_dtype, device=default_device)

Id2 = pt.kron(Id,Id)

# Quantum computing gates
H = pt.tensor([
    [1,1],
    [1,-1]], dtype=cplx_dtype, device=default_device)/np.sqrt(2)
X1=X; Y1=Y; Z1=Z
SX=pt.tensor([[1+1j, 1-1j],[1-1j,1+1j]], dtype=cplx_dtype, device=default_device)/2
CX = pt.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]], dtype=cplx_dtype, device=default_device)
CXr = pt.tensor([
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
], dtype = cplx_dtype, device=default_device)

CX3q = pt.tensor([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0]], dtype=cplx_dtype, device=default_device)

sigDotSig = pt.tensor([
    [1,0,0,0],
    [0,-1,2,0],
    [0,2,-1,0],
    [0,0,0,1]], dtype=cplx_dtype, device=default_device)

swap = pt.tensor([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]], dtype=cplx_dtype, device=default_device)

root_swap = pt.tensor([
    [1,0,0,0],
    [0,(1+1j)/2,(1-1j)/2,0],
    [0,(1-1j)/2,(1+1j)/2,0],
    [0,0,0,1]],dtype = cplx_dtype, device=default_device)

XI = pt.kron(X,Id)
IX = pt.kron(Id,X)
YI = pt.kron(Y,Id)
IY = pt.kron(Id,Y)
ZI = pt.kron(Z,Id)
IZ = pt.kron(Id,Z)
XII = kron3(X,Id,Id)
IXI = kron3(Id,X,Id)
IIX = kron3(Id,Id,X)
YII = kron3(Y,Id,Id)
IYI = kron3(Id,Y,Id)
IIY = kron3(Id,Id,Y)
ZII = kron3(Z,Id,Id)
IZI = kron3(Id,Z,Id)
IIZ = kron3(Id,Id,Z)
X2 = XI+IX
Y2=YI+IY
Z2=ZI+IZ
X3 = XII+IXI+IIX
Y3= YII + IYI + IIY
Z3 = ZII + IZI + IIZ
o2 = pt.kron(X,X) + pt.kron(Y,Y) + pt.kron(Z,Z)
o12 = kron3(X,X,Id)+kron3(Y,Y,Id)+kron3(Z,Z,Id)
o23 = kron3(Id,X,X)+kron3(Id,Y,Y)+kron3(Id,Z,Z)

def get_coupling_matrices(nq,device=default_device):
    if nq==3:
        return pt.stack((o12,o23)).to(device)
    elif nq==2:
        return o2.to(device)
    else: raise Exception("Invalid qubit number")

def get_PZ_vec(nq, device=default_device):
    if nq==3:
        return pt.stack((ZII,IZI,IIZ)).to(device)
    elif nq==2:
        return pt.stack((ZI,IZ)).to(device)
    else: raise Exception("Invalid qubit number")

''' 
Get matrices corresponding to X,Y and Z fields for 1,2 or 3 qubit systems. Xn = X1 + ... + Xn.
'''
def get_Xn(nq, device=default_device):
    if nq==1:
        Xn = X1
    elif nq==2:
        Xn = X2
    elif nq==3:
        Xn = X3
    return Xn.to(device)

def get_Yn(nq, device=default_device):
    if nq==1:
        Yn = Y1
    elif nq==2:
        Yn = Y2
    elif nq==3:
        Yn = Y3
    return Yn.to(device)
    
        
def get_Zn(nq, device=default_device):
    if nq==1:
        Zn = Z1
    elif nq==2:
        Zn = Z2
    elif nq==3:
        Zn = Z3
    return Zn.to(device)