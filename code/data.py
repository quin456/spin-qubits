

import pickle 
import torch as pt 
from atomic_units import *

cplx_dtype = pt.complex128
real_dtype = pt.float64

default_device = 'cuda:0' if pt.cuda.is_available() else 'cpu'

dir = './'
exch_filename = f"exchange_data_updated.p"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18'], dtype=cplx_dtype) * Mhz
J_100_14nm = pt.tensor(exch_data['100_14'], dtype=cplx_dtype) * Mhz


#gyromagnetic ratios (Mrad/T)
gamma_P = 17.235 * Mhz/tesla
gamma_e = 28025 * Mhz/tesla


B0 = 2*tesla      # static background field
g_e = 2.0023               # electron g-factor


mu_B = qE*hbar/(2*mE)          # Bohr magneton
omega0 = g_e*mu_B*B0/hbar   # Larmor frequency


# exchange values 
A_mag = 58.5/2 * Mhz
delta_A_kane = 2*A_mag
A_kane = pt.tensor([A_mag, -A_mag])
A_kane3 = pt.tensor([A_mag, -A_mag, A_mag])



def get_A(nS,nq, device=default_device):
    if nq==2:
        return pt.tensor(nS*[[A_mag, -A_mag]], device=device, dtype = cplx_dtype)
    elif nq==3:
        return pt.tensor(nS*[[A_mag, A_mag, -A_mag]], device=device, dtype=cplx_dtype)
    elif nq==1:
        return A_mag

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


# junk?
J_10nm = 0.1e-3*qE_n*joule # ~10e-23
J_Omin = 15e6 * hz * hbar
J_Omax = 50e6 * hz * hbar
A_BP = 5e6 * hz
A_approx = 1e9 * hz
A_real1 = pt.tensor([183.5e6, 66.5e6]) * hz 