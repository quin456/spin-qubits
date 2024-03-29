


import numpy as np
import torch as pt 
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
from itertools import combinations
import networkx as nx



from hamiltonians import *
from data import *
from utils import dagger, fidelity, wf_fidelity, get_rec_min_N
from eigentools import get_resonant_frequencies
from single_NE import *
import gates as gate
from gates import spin_101, spin_10
from eigentools import *
from single_NE import NE_couplings
from pulse_maker import *

from pdb import set_trace

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'

atom_spacing = 0.384

pt.manual_seed(0)

def get_t_fidelity(J,A, tN, N, fid_min):
    nq = len(A)

    HA = get_1S_HA(A)
    HJ = get_1S_HJ(J)

    UA = get_U0(HA, tN, N)
    H = dagger(UA)@HJ@UA 

    X = get_X_from_H(H,tN,N)
    Id = gate.II if nq==2 else gate.III
    fid = pt.tensor([fidelity(X[j], Id) for j in range(N)])

    j = 0
    while fid[j] > fid_min:
        j+=1
    return j*tN/N
   
def get_t_wf(J,A, tN, N, fid_min, psi0=spin_101):
    nq = len(A)

    HA = get_1S_HA(A)
    HJ = get_1S_HJ(J)

    UA = get_U0(HA, tN, N)
    H = dagger(UA)@HJ@UA 

    X = get_X_from_H(H,tN,N)
    psi = X@psi0
    fid = pt.abs(psi[:,5])**2
    #fid = pt.tensor([wf_fidelity(psi[j], psi0) for j in range(N)])

    j = 0
    try:
        while fid[j] > fid_min:
            j+=1
    except:
        return -1
    return j*tN/N

def plot_load_time_vs_J(fid_min=0.999, Jmin = 5*unit.MHz, Jmax=50*unit.MHz, A=get_A(1,3), tN_max=10*unit.ns, n=100, N=10000, max_time=10*unit.ns, ax=None, fp=None, get_t=get_t_fidelity):

    optimal_J=None
    nq=len(A)
    T = pt.zeros(n)
    J = pt.linspace(Jmin, Jmax, n)
    if nq==3:
        J = pt.einsum('i,j->ij',J, pt.ones(2))
    i_min=0
    for i in range(n):
        time = get_t(J[i],A,tN_max,N, fid_min)
        if time==-1:
            i_min = i+1
            optimal_J = J[i]
        T[i] = time

    if optimal_J is not None:
        print(f"No loss of fidelity for J<{optimal_J/unit.MHz} MHz")
    if ax is None: ax = plt.subplot()

    ax.plot(J[i_min:]/unit.MHz, T[i_min:]/unit.ns)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    # ax.axhline(max_time/unit.ns, linestyle = '--', color = 'red')
    # ax.annotate(f'{max_time/unit.ns} ns', (30,max_time/unit.ns+0.3))
    # ax.axvline(J[i,0]/unit.MHz, linestyle = '--', color='red')
    # ax.annotate(f'{J[i,0]/unit.MHz:.0f} unit.MHz', (J[i]/unit.MHz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)
    

def approximate_full_NE_optimisation_time():
    '''
    3 nuclear, 3 electron spin system optimisation time approximation based on 40 second 99.5% 
    fidelity 3 electron CNOT optimisation time.
    '''
    tN_3E = 100 * unit.ns
    tN_3NE = 1000 * unit.ns
 
    t_3E = 40 # time to optimise 3 electron system
    
    dim_3E = 2**3 
    dim_3NE = 2**6

    H0_3E = get_H0(get_A(1,2), get_J(1,2))
    rf_3E = get_resonant_frequencies(H0_3E)
    N_3E = get_rec_min_N(rf_3E, tN_3E)

    H0_3NE = multi_NE_H0(Bz=2*unit.T)
    rf_3NE = get_resonant_frequencies(H0_3NE)
    N_3NE = get_rec_min_N(rf_3NE, tN_3NE)
    #N_3NE=N_3E

    n_fields_3E = 15
    n_fields_3NE = 637

    print(f"w_max_3E = {pt.max(pt.real(rf_3E))/unit.MHz} MHz")
    print(f"w_max_3NE = {pt.max(pt.real(rf_3NE))/unit.MHz} MHz")
    print(f"N_3E = {N_3E}")
    print(f"N_3NE = {N_3NE}")


    t_3NE = ((dim_3NE/dim_3E)**2) * (n_fields_3NE/n_fields_3E) * (N_3NE/N_3E) * (tN_3NE/tN_3E) * t_3E

    print(f"t_3NE = {t_3NE} s")

    


def plot_load_time_vs_J_2q(fidelity_min = 0.999, N=2000, J=get_J(1,2), A=get_A(1,1), max_time=10*unit.ns, ax=None, fp=None):
    # MISTAKEY

    def get_t(J):
        return np.arccos(np.sqrt(fidelity_min)) / ( (np.sqrt(4*A**2 + 4*J**2) - 2*A) )

    J = np.linspace(2*unit.MHz, 50*unit.MHz, N)
    T = get_t(J)



    if ax is None: ax = plt.subplot()
    ax.plot(J/unit.MHz, T/unit.ns)
    ax.set_ylabel("Max loading window (ns)")
    ax.set_xlabel("Exchange strength (MHz)")


    i=0; 
    while T[i]>max_time: i+=1
    ax.axhline(max_time/unit.ns, linestyle = '--', color = 'red')
    ax.annotate(f'{max_time/unit.ns} ns', (30,max_time/unit.ns+0.3))
    ax.axvline(J[i]/unit.MHz, linestyle = '--', color='red')
    ax.annotate(f'{J[i]/unit.MHz:.0f} MHz', (J[i]/unit.MHz+0.2, 15))

    if fp is not None:
        plt.savefig(fp)



def exchange_HF(R):

    a = 1.8*unit.nm 
    J = (R/a)**(5/2) * np.exp(-2*R/a)
    return J
    print(f" Exchange: J = {J/unit.MHz} MHz")



def distance(r1, r2):
    return np.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2)


def draw_square(x, y, L, ax, color='black'):
    ax.plot([x-L/2, x+L/2], [y-L/2, y-L/2], color=color)
    ax.plot([x+L/2, x+L/2], [y-L/2, y+L/2], color=color)
    ax.plot([x+L/2, x-L/2], [y+L/2, y+L/2], color=color)
    ax.plot([x-L/2, x-L/2], [y+L/2, y-L/2], color=color)
    

def plot_squares(sites, color='gray', L=1, ax=None):
    if ax is None: ax=plt.subplot()
    n = len(sites)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        draw_square(*sites[i], L, ax)




def get_rabi_prob(w_ac, w_res, B_ac, c):
    '''
    Inputs
        w_ac: frequency to be applied to system
        w_res: resonant frequency of transition
        B_ac: strength of applied field
        c: coupling of B field to transition

    Determines disruption to system caused by w_ac using Rabi solution.
    '''
    W = c*B_ac/2
    dw = w_ac - w_res
    Omega = np.sqrt(dw**2 + W**2)
    Pr = np.abs(W**2 / Omega**2)
    return Pr


def HS_SQR_rabi_probs():
    '''
    Calculates the rabi probs for single qubit rotations on 1P. 2P and As.
    '''
    w_res = [2*A_P, 2*A_2P, 2*A_As]
    names = ['P', '2P', 'As']
    c = gamma_e 
    B_ac = 0.1*unit.mT
    for i in range(len(w_res)):
        for j in range(i+1,len(w_res)):
            print(f"{names[i]}, {names[j]}: Pr(|{uparrow}> <-> |{downarrow}> = {get_rabi_prob(w_res[i], w_res[j], B_ac, c)}, duration = {pi_pulse_duration(c, B_ac)/unit.ns} ns")



def frequencies_are_similar(w_ac, w_res, B_ac, c, maxp=0.01, print_info = True):
    p=get_rabi_prob(w_ac, w_res, B_ac, c)
    if print_info:
        print(f"Field B_ac = {B_ac/unit.mT} mT, w_ac = {pt.real(w_ac/unit.MHz):.1f} MHz has probability p={p:.4f} of exciting transition with w_res = {pt.real(w_res/unit.MHz):.1f} MHz, coupling = {pt.real(c*unit.T/unit.MHz):.1f} MHz/T")

    if p > maxp:
        return False 
    return True


def system_comparison(w_ac, rf,  B_ac, C):
    Pr = []
    
    transitions = [(0,1), (0,2), (1,3), (2,3)]
    for T1 in transitions:
        for T2 in transitions:

            Pr.append(get_rabi_prob(w_ac[T1], rf[T2], B_ac, C[T1]))
    Pr = pt.tensor(Pr)
    return pt.max(Pr)


def get_similar_systems(RF, B_ac, C, pmax):
    nS, dim, dim = RF.shape
    similar_systems = []
    for i in range(nS):
        for j in range(i+1, nS):
            if system_comparison(RF[i], RF[j], B_ac, C[i]) > pmax:
                similar_systems.append((i,j))
    return similar_systems


def graph_system_similarity(RF, B_ac, C, pmax=0.01, ax=None):
    nS = 48
    
    if ax is None: ax = plt.subplot()
    edges = get_similar_systems(RF, B_ac, C, pmax)

    nodes = np.linspace(0, nS, nS, dtype=int)

    G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    G = nx.Graph(name="transitions")
    G.add_nodes_from(np.linspace(0,nS-1,nS))
    G.add_edges_from(edges)
    #G = nx.Graph(edges)
    #nx.draw_networkx_nodes(G, node_color='red')
    nx.draw(G, ax=ax)




def graph_frequency_similarity(rf_mat, C, B_ac=1*unit.T, pmax=0.01):
    edges=[]
    rf = flatten_rf_mat(rf_mat)
    C_vec = flatten_rf_mat(C)
    for i in range(len(rf)):
        for j in range(0,len(rf)):
            if frequencies_are_similar(rf[i], rf[j], B_ac, C_vec[j]) or frequencies_are_similar(rf[j], rf[i], B_ac, C_vec[i], maxp=pmax):
                edges.append((i,j))
    G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    G = nx.Graph(name="transitions")
    G.add_nodes_from(np.linspace(0,len(rf)-1,len(rf)))
    G.add_edges_from(edges)
    nx.draw(G)

def flatten_rf_mat(M):
    m,n = M.shape
    vec = []
    for i in range(m):
        for j in range(i+1,n):
            if pt.abs(M[i,j])/pt.max(pt.abs(M[i,j])) > 1e-10:
                vec.append(M[i,j])
    return pt.tensor(vec, dtype=M.dtype, device=M.device)
            


def check_frequency_overlap(H0, B_ac=1*unit.mT, pmax=0.01, Hw_mag=None):

    S,D = get_ordered_eigensystem(H0)
    RF = get_rf_matrix(S, D)
    C = get_couplings(S, Hw_mag=Hw_mag)
    graph_frequency_similarity(RF, C, B_ac=B_ac)

def check_system_overlap(H0, B_ac=1*unit.mT, pmax=0.01):

    S,D = get_ordered_eigensystem(H0)
    RF = get_rf_matrix(S, D)
    C = get_couplings(S)

    sys1=1
    sys2=3


    w_res = RF[sys1,0,1]
    c = C[0,0,1]
    w_ac = RF[sys2,0,1]

    B_ac = 5e-3*unit.mT

    #Pmax = system_comparison(RF[sys1], RF[sys2], B_ac, C[0])
    
    graph_system_similarity(RF, B_ac, C, pmax=pmax)


def compare_1P_2P_frequencies():
    
    J = get_J_1P_2P(48)
    A = get_A_1P_2P(48)

    H0 = get_H0(A,J)
    check_system_overlap(H0)

def split_rf(rf):
    rf1=[]; rf2=[]
    midfreq = pt.max(pt.real(rf))/2 
    for w in rf:
        if pt.abs(w)<midfreq:
            rf1.append(w)
        else:
            rf2.append(w)
    return pt.tensor(rf1), pt.tensor(rf2)


def compare_3E_frequencies():
    J = get_J(225,3)
    A = get_A(225,3)
    H0 = get_H0(A,J)
    S,D = get_ordered_eigensystem(H0)

    rf = get_multi_system_resonant_frequencies(H0)
    rf1, rf2 = split_rf(rf)
    #set_trace()

    ax=plt.subplot()
    #for w in rf:
        #ax.axhline(w/unit.MHz)
    C = get_couplings(S)
    dw = pt.linspace(0,1000*unit.MHz,1000)
    P=get_rabi_prob(0, dw, 1*unit.mT, 2*gamma_e)
    i=0
    while P[i]>0.01: i+=1
    print(f"P({dw[i]/unit.MHz} MHz) = {P[i]}")
    ax.plot(dw/unit.MHz,P)
    check_system_overlap(H0, pmax=0.3)


# Question 1: How low does J have to be?



#Question 2: For the most closely spaced pair of resonant frequencies, what pulse time is required to avoid large off resonant effects?



# Question 3: How much does this improve GRAPE optimisation?
def check_NE_frequency_overlap():
    H0 = get_NE_H0(get_A(1,1), 2*unit.T)
    Hw_mag = -gamma_n*gate.XI + gamma_e*gate.IX
    check_frequency_overlap(H0, Hw_mag=Hw_mag)

def swap_exchange_rabi_prob(ax=None, B_ac=1*unit.mT):
    A = get_A(1,1)
    H0 = get_NE_H0(A, B0)
    S,D = NE_eigensystem(H0)

    t_pulse = np.pi / (gamma_e*B_ac)

    couplings = NE_couplings(H0)
    c = couplings[2,3]
    w_res = D[2,2]-D[3,3]

    J = pt.linspace(0,2*unit.MHz, 10000)
    prob = get_rabi_prob(w_res, w_res+J, B_ac, c)

    i=0
    while prob[i]>0.99:
        i+=1
    print(f"Prob has fallen to {prob[i]} at exchange J={J[i]/unit.kHz} kHz")
    print(f"Pulse time at {B_ac/unit.mT} mT is {t_pulse/unit.ns} ns")

    if ax is None: ax=plt.subplot()
    ax.plot(J/unit.kHz, prob, label='Rabi probability')


def hyperfine_calculation(evec_0_sq):
    return (2/3) * evec_0_sq * mu_B * unit.mu_0 * hbar * gamma_n


def visualise_frequency_overlap(fp=None):
    Pr=0.001
    B_ac = 0.1*unit.mT
    Jmax = 1.87*unit.MHz
    J1_low = J_100_18nm/10
    J2_low = J_100_18nm * Jmax / pt.max(pt.real(J_100_18nm))
    H0 = get_H0(get_A(225,3, NucSpin=[1,0,1]), get_J(225, 3, J1=J1_low, J2=J2_low))

    gbac = gamma_e*B_ac/2
    dw = gbac*np.sqrt(1/Pr - 1)/unit.MHz

    rf = pt.real(get_multi_system_resonant_frequencies(H0))/unit.MHz
    print(f"Total number of frequencies = {len(rf)}")
    print(f"dw = {dw} MHz")
    ax = plt.subplot()
    for freq in rf:
        ax.axvline(freq, 0,1)
    ax.set_yticks([])
    ax.set_xlabel('Frequency (MHz)')
    ax.annotate(f"$p_{{max}}={Pr}, B_{{ac}}=1$ mT\n $⟹\Delta\omega = {dw:.1f}$ MHz", [-170,0.4])
    ax.get_figure().set_size_inches(fig_width_double, 0.2*fig_width_double)
    ax.get_figure().tight_layout()
    if fp is not None: ax.get_figure().savefig(fp)


if __name__ == '__main__':
    #plot_load_time_vs_J(fid_min=0.99, Jmin=1*unit.MHz, Jmax=50*unit.MHz, tN_max=100*unit.ns, A=get_A(1,3), n=100, get_t=get_t_wf)

    #plot_load_time_vs_J_2q(fidelity_min=0.99)

    #print(f"{get_t_wf(get_J(1,3), get_A(1,3), 10*unit.ns,1000,0.99)/unit.ns} ns")

    #approximate_full_NE_optimisation_time()

    #placement_symmetries_1P_2P(delta_x = 52, delta_y = 10, save=True)
    #swap_exchange_rabi_prob(B_ac = 1*unit.mT)
    #compare_3E_frequencies()

    #check_NE_frequency_overlap()

    #HS_SQR_rabi_probs()

    #print(f"A = {hyperfine_calculation(8.69e-12+2.3820e-11)/unit.MHz} MHz")

    visualise_frequency_overlap()
    #check_system_overlap(H0, 10*unit.mT)
    plt.show()