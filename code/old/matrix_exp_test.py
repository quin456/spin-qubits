


import torch as pt 
from hamiltonians import get_H0 
from data import get_J, get_A, cplx_dtype, default_device
from pdb import set_trace





nS = 225

N=5000

while True:
    A = get_A(nS, 3)
    J = get_J(nS,3)
    H0 = get_H0(A, J)
    H = pt.einsum('j,sab->sjab', pt.ones(N,dtype=cplx_dtype, device=default_device), H0)
    U=pt.matrix_exp(H)

    print(f"N={N} successful")
    set_trace()
    N+=1000