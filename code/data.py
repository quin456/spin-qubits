

import pickle 
import torch as pt 
from atomic_units import *
import atomic_units as unit
from pdb import set_trace

cplx_dtype = pt.complex128
real_dtype = pt.float64

VERBOSE = False

default_device = 'cuda:0' if pt.cuda.is_available() else 'cpu'

dir = './'
exch_filename = f"exchange_data_updated.p"
exch_data_folder = "exchange_data_fab/"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18'], dtype=cplx_dtype, device=default_device) * unit.MHz
J_100_14nm = pt.tensor(exch_data['100_14'], dtype=cplx_dtype, device=default_device) * unit.MHz
J_110_18nm = pt.tensor(exch_data['110_18'], dtype=cplx_dtype, device=default_device) * unit.MHz
J_110_14nm = pt.tensor(exch_data['110_14'], dtype=cplx_dtype, device=default_device) * unit.MHz
J_2P_1P_fab = pt.load(f"{exch_data_folder}J").to(cplx_dtype).to(device=default_device) * unit.MHz
A_2P_1P_fab = pt.load(f"{exch_data_folder}A_2P_1P").to(cplx_dtype).to(default_device) * unit.MHz
A_2P_fab = pt.load(f"{exch_data_folder}A_2P") * unit.MHz


J_extended = pt.cat((J_100_18nm, J_100_18nm*1.1, J_100_18nm*1.2, J_100_18nm*1.3, J_100_18nm*1.4, J_100_18nm*1.5, J_100_18nm*1.6, J_100_18nm*1.7, J_100_18nm*1.8, J_100_18nm*1.9))


#gyromagnetic ratios (Mrad/T)
gamma_n = 17.235 * unit.MHz/unit.T
gamma_e = 28025 * unit.MHz/unit.T


B0 = 2*unit.T      # static background field
g_e = 2.0023               # electron g-factor


mu_B = qE*hbar/(2*mE)          # Bohr magneton
omega0 = g_e*mu_B*B0/hbar   # Larmor frequency


# exchange values 
A_mag = 58.5/2 * unit.MHz
delta_A_kane = 2*A_mag
A_kane = pt.tensor([A_mag, -A_mag])
A_kane3 = pt.tensor([A_mag, -A_mag, A_mag])
A_2P_mag = 262*unit.MHz


def get_A_from_num_P_donors(num_P_donors):
    if num_P_donors == 1:
        return A_mag 
    elif num_P_donors == 2:
        return A_2P_mag

def get_A(nS,nq, NucSpin=None, A_mags=None, device=default_device, E_rise_time = 1*unit.ns):
    if NucSpin is not None:
        # map 0->1, 1->-1
        NucSpin = [1-2*ns for ns in NucSpin]
    if nq==1:
        return A_mag
    if A_mags is None: A_mags = nq*[A_mag]
    if nq==2:
        if NucSpin is None: NucSpin = [1, -1]
        A = pt.tensor(nS*[[NucSpin[0]*A_mags[0], NucSpin[1]*A_mags[1]]], device=device, dtype = cplx_dtype)
    elif nq==3:
        if NucSpin is None: NucSpin = [1, -1, 1]
        A = pt.tensor(nS*[[NucSpin[0]*A_mags[0], NucSpin[1]*A_mags[1], NucSpin[2]*A_mags[2]]], device=device, dtype=cplx_dtype)

    if nS==1:
        return A[0]
    return A


def all_J_pairs(J1, J2, device=default_device):
    nJ=15
    J = pt.zeros(nJ**2,2, device=device,dtype=cplx_dtype)
    for i in range(nJ):
        for j in range(nJ):
            J[i*15+j,0] = J1[i]; J[i*15+j,1] = J2[j]
    return J

def get_J(nS,nq,J1=J_100_18nm,J2=J_100_18nm/2.3, N=1, device=default_device, E_rise_time=1*unit.ns):

    # default to small exchange for testing single triple donor
    # if nS==1 and nq==3:
    #     return pt.tensor([0.37*unit.MHz, 0.21*unit.MHz], dtype=cplx_dtype)

    if nq==2:
        J = J1[:nS].to(device)

    elif nq==3:
        J = all_J_pairs(J1,J2)[:nS]


    if nS==1:
        return J[0]
    return J

def get_A_1P_2P(nS, NucSpin=[0,0]):
    A = A_2P_1P_fab[:nS] if nS>1 else A_2P_1P_fab[0]
    return A


def get_J_1P_2P(nS):
    return get_J(nS, 2, J1=J_2P_1P_fab)




if __name__ == '__main__':
    print("============================================")
    print("Printing data used in spin-qubit simulations")
    print("============================================\n")
    print(f"gamma_e = {gamma_e*unit.T/unit.MHz:.1f} MHz, gamma_n = {gamma_n*unit.T/unit.MHz:.1f} MHz")
    print(f"\nHyperfine coupling: A_1P = {get_A(1,1)/unit.MHz:.2f} MHz")
    print("\nExchange valiues for (100) 18nm separation:")
    for i in range(15):
        print(f"J_18nm_100_{i} = {pt.real(J_100_18nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (100) 18nm separation:")
    for i in range(15):
        print(f"J_18nm_110_{i} = {pt.real(J_110_18nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (100) 14nm separation:")
    for i in range(15):
        print(f"J_14nm_{i} = {pt.real(J_100_14nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for (110) 14nm separation:")
    for i in range(15):
        print(f"J_14nm_{i} = {pt.real(J_110_14nm[i]).item()/unit.MHz:.1f} MHz")

    print(f"\nHyperfine coupling 2P-1P (fabricated):")
    for i in range(len(A_2P_1P_fab)):
        print(f"A_2P_{i} = {A_2P_fab[i].item()/unit.MHz:.2f} MHz")
    print(f"Exchange 2P-1P (fabricated)")
    for i in range(len(A_2P_1P_fab)):
        print(f"J_2P_1P_{i} = {J_2P_1P_fab[i].item()/unit.MHz:.2f} MHz")