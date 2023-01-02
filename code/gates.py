
import torch as pt
import numpy as np
from torch import kron

from data import cplx_dtype, real_dtype
from utils import print_rank2_tensor, sqrtm, dagger

ngpus = pt.cuda.device_count()
default_device = 'cuda:0' if ngpus>0 else 'cpu'

alternate_NEs = False

def kron3(A,B,C):
    '''  Returns kronecker product of 3 matrices A,B,C.  '''
    return kron(kron(A,B),C)

def kron4(A,B,C,D):
    return kron(kron(A,B), kron(C,D))

def kron6(A,B,C,D,E,F):
    return kron(kron3(A,B,C),kron3(D,E,F))

def UJ(t, J):
    return pt.diag(pt.tensor([np.exp(-1j*np.pi*J*t/2), np.exp(1j*np.pi*J*t/2), np.exp(1j*np.pi*J*t/2), np.exp(-1j*np.pi*J*t/2)], dtype=cplx_dtype))


    
spin_up = pt.tensor([1,0], dtype=cplx_dtype, device=default_device)
spin_down = pt.tensor([0,1], dtype=cplx_dtype, device=default_device)
spin_0 = spin_up 
spin_1 = spin_down 
spin_00 = kron(spin_0, spin_0)
spin_01 = kron(spin_0, spin_1)
spin_10 = kron(spin_1, spin_0)
spin_11 = kron(spin_1, spin_1)
spin_000 = kron(spin_00,spin_0)
spin_001 = kron(spin_00,spin_1)
spin_010 = kron(spin_01,spin_0)
spin_011 = kron(spin_01,spin_1)
spin_100 = kron(spin_10,spin_0)
spin_101 = kron(spin_10,spin_1)
spin_110 = kron(spin_11,spin_0)
spin_111 = kron(spin_11,spin_1)
spin_0000 = kron(spin_00, spin_00)
spin_0001 = kron(spin_00, spin_01)
spin_0010 = kron(spin_00, spin_10)
spin_0011 = kron(spin_00, spin_11)
spin_0100 = kron(spin_01, spin_00)
spin_0101 = kron(spin_01, spin_01)
spin_0110 = kron(spin_01, spin_10)
spin_0111 = kron(spin_01, spin_11)
spin_1000 = kron(spin_10, spin_00)
spin_1001 = kron(spin_10, spin_01)
spin_1010 = kron(spin_10, spin_10)
spin_1011 = kron(spin_10, spin_11)
spin_1100 = kron(spin_11, spin_00)
spin_1101 = kron(spin_11, spin_01)
spin_1110 = kron(spin_11, spin_10)
spin_1111 = kron(spin_11, spin_11)

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

SX = pt.tensor([[1+1j, 1-1j],[1-1j,1+1j]], dtype=cplx_dtype, device=default_device)/2
SY = pt.sqrt(Y)

CX = pt.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]], dtype=cplx_dtype, device=default_device)
CXr = pt.tensor([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
], dtype = cplx_dtype, device=default_device)

CX_native = pt.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,-1j],
    [0,0,-1j,0]], dtype=cplx_dtype, device=default_device)
CXr_native = pt.tensor([
    [1,0,0,0],
    [0,0,0,1j],
    [0,0,1,0],
    [0,1j,0,0]
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
III = kron(Id,II)
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


sw2_12 = kron(swap, II)
sw2_23 = kron3(Id, swap, Id)
sw2_34 = kron(II, swap)
NE_swap_2q = sw2_23 @ sw2_34 @ sw2_12 @ sw2_23


def get_Iz_sum(nq):
    if nq==1:
        return 0.5 * ZI
    elif nq==2:
        return 0.5 * (ZIII + IZII)
    elif nq==3:
        return 0.5 * (ZIIIII + IZIIII + IIZIII)

def get_Ix_sum(nq):
    if nq==1:
        return 0.5 * XI 
    elif nq==2:
        return 0.5 * (XIII + IXII)
    elif nq==3:
        return 0.5 * (XIIIII + IXIIII + IIXIII)
        
def get_Iy_sum(nq):
    if nq==1:
        return 0.5 * YI
    elif nq==2:
        return 0.5 * (YIII + IYII)
    elif nq==3:
        return 0.5* (YIIIII + IYIIII + IIYIII)

def get_Sz_sum(nq):
    if nq==1:
        return 0.5 * IZ
    elif nq==2:
        return 0.5 * (IIZI+IIIZ )
    elif nq==3:
        return 0.5 * (IIIZII + IIIIZI + IIIIIZ)

def get_Sy_sum(nq):
    if nq==1:
        return 0.5*IY
    elif nq==2:
        return 0.5 * (IIYI+IIIY)
    elif nq==3:
        return 0.5 * (IIIYII + IIIIYI + IIIIIY)

def get_Sx_sum(nq):
    if nq==1:
        return 0.5*IX
    elif nq==2:
        return 0.5 * (IIXI+IIIX)
    elif nq==3:
        return 0.5 * (IIIXII + IIIIXI + IIIIIX)


X2 = XI+IX
Y2=YI+IY
Z2=ZI+IZ
X3 = XII+IXI+IIX
Y3= YII + IYI + IIY
Z3 = ZII + IZI + IIZ
X4 = XIII + IXII + IIXI + IIIX
X4 = XIII + IXII + IIXI + IIIX
Y4 = YIII + IYII + IIYI + IIIY
Z4 = ZIII + IZII + IIZI + IIIZ
X6 = XIIIII + IXIIII + IIXIII + IIIXII + IIIIXI + IIIIIX 
Y6 = YIIIII + IYIIII + IIYIII + IIIYII + IIIIYI + IIIIIY 
Z6 = ZIIIII + IZIIII + IIZIII + IIIZII + IIIIZI + IIIIIZ


o2 = pt.kron(X,X) + pt.kron(Y,Y) + pt.kron(Z,Z)
o12 = kron3(X,X,Id)+kron3(Y,Y,Id)+kron3(Z,Z,Id)
o23 = kron3(Id,X,X)+kron3(Id,Y,Y)+kron3(Id,Z,Z)
o13 = kron3(X,Id,X)+kron3(Y,Id,Y)+kron3(Z,Id,Z)

o4_13 = kron4(X,Id,X,Id)+kron4(Y,Id,Y,Id)+kron4(Z,Id,Z,Id)
o4_24 = kron4(Id,X,Id,X)+kron4(Id,Y,Id,Y)+kron4(Id,Z,Id,Z)
o4_34 = kron4(Id,Id,X,X)+kron4(Id,Id,Y,Y)+kron4(Id,Id,Z,Z)

o6_14 = kron6(X,Id,Id,X,Id,Id) + kron6(Y,Id,Id,Y,Id,Id) + kron6(Z,Id,Id,Z,Id,Id)
o6_25 = kron6(Id,X,Id,Id,X,Id) + kron6(Id,Y,Id,Id,Y,Id) + kron6(Id,Z,Id,Id,Z,Id)
o6_36 = kron6(Id,Id,X,Id,Id,X) + kron6(Id,Id,Y,Id,Id,Y) + kron6(Id,Id,Z,Id,Id,Z)
o6_45 = kron6(Id,Id,Id,X,X,Id) + kron6(Id,Id,Id,Y,Y,Id) + kron6(Id,Id,Id,Z,Z,Id)
o6_56 = kron6(Id,Id,Id,Id,X,X) + kron6(Id,Id,Id,Id,Y,Y) + kron6(Id,Id,Id,Id,Z,Z)

CX_3NE = kron(CX3q, III)

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
    elif nq==4:
        Xn = X4 
    elif nq==6:
        Xn = X6
    return Xn.to(device)

def get_Yn(nq, device=default_device):
    if nq==1:
        Yn = Y1
    elif nq==2:
        Yn = Y2
    elif nq==3:
        Yn = Y3
    elif nq==4:
        Yn = Y4 
    elif nq==6:
        Yn = Y6
    return Yn.to(device)
    
        
def get_Zn(nq, device=default_device):
    if nq==1:
        Zn = Z1
    elif nq==2:
        Zn = Z2
    elif nq==3:
        Zn = Z3
    elif nq==4:
        Zn = Z4 
    elif nq==6:
        Zn = Z6
    return Zn.to(device)


def get_2E_H0(A,J):
    return A*ZI-A*IZ + J*sigDotSig


if __name__ == '__main__':
    from pdb import set_trace
    set_trace()