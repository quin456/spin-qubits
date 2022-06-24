
import torch as pt
import numpy as np
from torch import kron

from data import cplx_dtype, real_dtype

ngpus = pt.cuda.device_count()
default_device = 'cuda:0' if ngpus>0 else 'cpu'


def kron3(A,B,C):
    '''  Returns kronecker product of 3 matrices A,B,C.  '''
    return pt.kron(pt.kron(A,B),C)

def kron4(A,B,C,D):
    return pt.kron(pt.kron(A,B), pt.kron(C,D))

def kron6(A,B,C,D,E,F):
    return pt.kron(kron3(A,B,C),kron3(D,E,F))
    
spin_up = pt.tensor([1,0], dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)

X = pt.tensor([
    [0,1],
    [1,0]], dtype=cplx_dtype, device=default_device)
Y = pt.tensor([
    [0,-1j],
    [1j,0]], dtype=cplx_dtype, device=default_device)
Z = pt.tensor([
    [1,0],
    [0,-1]], dtype=cplx_dtype, device=default_device)

sigma = pt.cat((X,Y,Z))

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

Id3 = pt.kron(Id2,Id)

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

II = kron(Id,Id)
XI = kron(X,Id)
IX = kron(Id,X)
YI = kron(Y,Id)
IY = kron(Id,Y)
ZI = kron(Z,Id)
IZ = kron(Id,Z)
XII = kron3(X,Id,Id)
IXI = kron3(Id,X,Id)
IIX = kron3(Id,Id,X)
YII = kron3(Y,Id,Id)
IYI = kron3(Id,Y,Id)
IIY = kron3(Id,Id,Y)
ZII = kron3(Z,Id,Id)
IZI = kron3(Id,Z,Id)
IIZ = kron3(Id,Id,Z)
ZIII = kron(ZII,Id)
IZII = kron(IZI,Id)
IIZI = kron(IIZ,Id)
IIIZ = kron(Id,IIZ)
YIII = kron(YII,Id)
IYII = kron(IYI,Id)
IIYI = kron(IIY,Id)
IIIY = kron(Id,IIY)
XIII = kron(XII,Id)
IXII = kron(IXI,Id)
IIXI = kron(IIX,Id)
IIIX = kron(Id,IIX)

ZIIIII = kron(ZIII,II)
IZIIII = kron(IZII,II)
IIZIII = kron(IIZI,II)
IIIZII = kron(II,IZII)
IIIIZI = kron(II,IIZI)
IIIIIZ = kron(II,IIIZ)

YIIIII = kron(YIII,II)
IYIIII = kron(IYII,II)
IIYIII = kron(IIYI,II)
IIIYII = kron(II,IYII)
IIIIYI = kron(II,IIYI)
IIIIIY = kron(II,IIIY)

XIIIII = kron(XIII,II)
IXIIII = kron(IXII,II)
IIXIII = kron(IIXI,II)
IIIXII = kron(II,IXII)
IIIIXI = kron(II,IIXI)
IIIIIX = kron(II,IIIX)




# maybe bad

# sig4_1 = pt.stack((XIII,YIII,ZIII))
# sig4_2 = pt.stack((IXII,IYII,IZII))
# sig4_3 = pt.stack((IIXI,IIYI,IIZI))
# sig4_4 = pt.stack((IIIX,IIIY,IIIZ))

# def sigdot(sig_i, sig_j):
#     return pt.einsum('iab,ibc->ac')




def get_nuclear_oz(nq):
    if nq==2:
        return ZIII + IZII 
    elif nq==3:
        return ZIIIII + IZIIII + IIZIII

def get_nuclear_ox(nq):
    if nq==2:
        return XIII + IXII 
    elif nq==3:
        return XIIIII + IXIIII + IIXIII
        
def get_nuclear_oy(nq):
    if nq==2:
        return YIII + IYII 
    elif nq==3:
        return YIIIII + IYIIII + IIYIII

def get_electron_oz(nq):
    if nq==2:
        return IIZI+IIIZ 
    elif nq==3:
        return IIIZII + IIIIZI + IIIIIZ

def get_electron_oy(nq):
    if nq==2:
        return IIYI+IIIY 
    elif nq==3:
        return IIIYII + IIIIYI + IIIIIY

def get_electron_ox(nq):
    if nq==2:
        return IIXI+IIIX 
    elif nq==3:
        return IIIXII + IIIIXI + IIIIIX


X2 = XI+IX
Y2=YI+IY
Z2=ZI+IZ
X3 = XII+IXI+IIX
Y3= YII + IYI + IIY
Z3 = ZII + IZI + IIZ
o2 = pt.kron(X,X) + pt.kron(Y,Y) + pt.kron(Z,Z)
o12 = kron3(X,X,Id)+kron3(Y,Y,Id)+kron3(Z,Z,Id)
o23 = kron3(Id,X,X)+kron3(Id,Y,Y)+kron3(Id,Z,Z)

o4_13 = kron4(X,Id,X,Id)+kron4(Y,Id,Y,Id)+kron4(Z,Id,Z,Id)
o4_24 = kron4(Id,X,Id,X)+kron4(Id,Y,Id,Y)+kron4(Id,Z,Id,Z)
o4_34 = kron4(Id,Id,X,X)+kron4(Id,Id,Y,Y)+kron4(Id,Id,Z,Z)


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