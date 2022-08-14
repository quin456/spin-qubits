

import pickle 
import torch as pt 
from atomic_units import *
import atomic_units as unit

cplx_dtype = pt.complex128
real_dtype = pt.float64

default_device = 'cuda:0' if pt.cuda.is_available() else 'cpu'

dir = './'
exch_filename = f"exchange_data_updated.p"
exch_data = pickle.load(open(exch_filename,"rb"))
J_100_18nm = pt.tensor(exch_data['100_18'], dtype=cplx_dtype) * unit.MHz
J_100_14nm = pt.tensor(exch_data['100_14'], dtype=cplx_dtype) * unit.MHz


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

def get_A(nS,nq, NucSpin=None, N=1, device=default_device, donor_composition = [1,1]):
    if NucSpin is not None:
        # map 0->1, 1->-1
        if nS==1:
            NucSpin = [1-2*ns for ns in NucSpin]
    if nq==1:
        return A_mag
    elif nq==2:
        if NucSpin is None: NucSpin = [1, -1]
        A0 = get_A_from_num_P_donors(donor_composition[0])
        A1 = get_A_from_num_P_donors(donor_composition[1])
        A = pt.tensor(nS*[[NucSpin[0]*A0, NucSpin[1]*A1]], device=device, dtype = cplx_dtype)
    elif nq==3:
        if NucSpin is None: NucSpin = [1, -1, 1]
        A = pt.tensor(nS*[[NucSpin[0]*A_mag, NucSpin[1]*A_mag, NucSpin[2]*A_mag]], device=device, dtype=cplx_dtype)




    if N>1:
        # signals hyperfine modulation
        modulator = pt.linspace(1, 1.2, N)
        A = pt.einsum('j,sq->sjq', modulator, A)

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

def get_J(nS,nq,J1=J_100_18nm,J2=J_100_18nm/2.3, N=1, device=default_device):

    # default to small exchange for testing single triple donor
    # if nS==1 and nq==3:
    #     return pt.tensor([0.37*unit.MHz, 0.21*unit.MHz], dtype=cplx_dtype)

    if nq==2:
        J = J1[:nS].to(device)
    elif nq==3:
        J = all_J_pairs(J1,J2)[:nS]

    if N>1:
        # signals exchange modulation
        if nq==3:
            raise Exception("Three qubits not supported with exchange modulation")
        rise_prop = 3
        modulator = pt.cat((pt.linspace(0.01, 1, N//rise_prop), pt.ones(N-2*N//rise_prop), pt.linspace(1, 0.01, N//rise_prop)))
        J = pt.einsum('j,s->sj', modulator, J)

    if nS==1:
        return J[0]
    return J






if __name__ == '__main__':
    print("============================================")
    print("Printing data used in spin-qubit simulations")
    print("============================================\n")
    print(f"gamma_e = {gamma_e*unit.T/unit.MHz:.1f} MHz, gamma_n = {gamma_n*unit.T/unit.MHz:.1f} MHz")
    print("\nExchange valiues for 18nm separation:")
    for i in range(15):
        print(f"J_18nm_{i} = {pt.real(J_100_18nm[i]).item()/unit.MHz:.1f} MHz")
    print("\nExchange valiues for 14nm separation:")
    for i in range(15):
        print(f"J_14nm_{i} = {pt.real(J_100_14nm[i]).item()/unit.MHz:.1f} MHz")