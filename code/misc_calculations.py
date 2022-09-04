


import numpy as np
import torch as pt 
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
from itertools import combinations
import networkx as nx



from hamiltonians import get_X_from_H, get_U0, get_1S_HA, get_1S_HJ, multi_NE_H0, get_H0
from data import *
from utils import dagger, fidelity, wf_fidelity, get_rec_min_N
from eigentools import get_resonant_frequencies
import gates as gate
from gates import spin_101, spin_10
from eigentools import *

from pdb import set_trace


atom_spacing = 0.384

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

def plot_sites(sites, ax, color='red'):
    x = [] 
    y = []
    for site in sites:
        x.append(site[0])
        y.append(site[1])
    ax.scatter(x, y, color=color)

def plot_donor_placement(sites_2P_upper, sites_2P_lower, sites_1P, delta_x, delta_y, d=1, i=3, j=1, k=4):

    fig,ax=plt.subplots(1)
    ax.set_aspect(1)
    plot_squares(sites_2P_upper, ax=ax, L=d)
    plot_squares(sites_2P_lower, ax=ax, L=d)
    plot_squares(sites_1P, ax=ax, L=d)
    plot_sites((sites_2P_lower[i], sites_2P_upper[j], sites_1P[k]), ax)

    if delta_y==0:
        head_length = 0.3
        head_width = 0.15
        ax.arrow(delta_x-3, 0, 0,1, head_length=head_length, head_width=head_width)
        ax.arrow(delta_x-3, 0, 1,0, head_length=head_length, head_width=head_width)
        ax.text(delta_x-1.5,0, '[1,-1,0]', rotation=0)
        ax.text(delta_x-3.6,1.5, '[1,1,0]', rotation=0)
        #ax.set_xlabel('[1,1,0]')
        #ax.set_ylabel('[-1,1,0]')
        ax.axis('off')
    elif delta_x==0:
        ax.set_xlabel('[1,-1,0]')
        ax.set_ylabel('[1,1,0]')

def plot_select_donor_placement(delta_x = 10, delta_y = 0):

    sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites(delta_x, delta_y)
    plot_donor_placement(sites_2P_upper, sites_2P_lower, sites_1P, delta_x, delta_y)

def classify_2P_pairs(donor_2P_pairs):
    distances = []
    for pair in donor_2P_pairs:
        d = distance(*pair)
        if d not in distances:
            distances.append(d)
    print(f"Unique 2P hyperfines = {len(distances)}")


def midpoint(r1, r2):
    return ((r1[0]+r2[0])/2, (r1[1]+r2[1])/2)

def get_unique_distance_triples(sites_2P_upper, sites_2P_lower, sites_1P):
    '''
    Gets list of distances of the form(d(2P_1, 2P_2), d(2P_1, 1P), d(2P_2, 1P))
    '''
    donor_2P_pairs = []
    donor_triples = []
    for r1 in sites_2P_upper:
        for r2 in sites_2P_lower:
            donor_2P_pairs.append((r1,r2))
            for r3 in sites_1P:
                donor_triples.append((r1,r2,r3))


    classify_2P_pairs(donor_2P_pairs)
    #classify_triplets()



    #def classify_triplets(triplets)
    distance_tups = []
    distance_tups_unique = []

    # define distances between 2Ps and from 2P midpoint to 1P
    d_pairs = []
    d_pairs_unique = []
    for pair in donor_2P_pairs:
        for site in sites_1P:
            dtup = (distance(*pair), distance(pair[0],site), distance(pair[1],site))
            distance_tups.append(dtup)
            d_pair = (distance(*pair), distance(midpoint(*pair), site))
            d_pairs.append(d_pair)
            if (dtup not in distance_tups_unique) and ((dtup[0], dtup[2], dtup[1]) not in distance_tups_unique):
                distance_tups_unique.append(dtup)
            if d_pair not in d_pairs_unique:
                d_pairs_unique.append(d_pair)


    print(f"{len(distance_tups_unique)} unique systems using d_trips")
    print(f"{len(d_pairs_unique)} unique systems using d_pairs")
    return d_pairs_unique

def get_AJ_distances(d_pairs):
    d_A = []
    d_J = [] 
    for d_pair in d_pairs:
        if d_pair[0] not in d_A:
            d_A.append(d_pair[0])
        if d_pair[1] not in d_J:
            d_J.append(d_pair[1])
    return d_A, d_J


def get_donor_sites(delta_x, delta_y, d=1, separation_2P = 2):

    delta_x*=d; delta_y*=d

    sites_2P_xs = np.linspace(1,3,3) * d
    sites_2P_ys_lower = np.linspace(-1,0,2) * d
    sites_2P_ys_upper = np.linspace(1+separation_2P,2+separation_2P,2) * d
    sites_1P_xs = np.linspace(1,3,3) * d
    sites_1P_ys = np.linspace(0+separation_2P/2,1+separation_2P/2,2) * d
    sites_2P_lower = []
    sites_2P_upper = []
    sites_1P = []

    for x in sites_2P_xs:
        for y in sites_2P_ys_upper:
            sites_2P_upper.append((x,y))
        for y in sites_2P_ys_lower:
            sites_2P_lower.append((x,y))
    for x in sites_1P_xs:
        for y in sites_1P_ys:
            sites_1P.append((x+delta_x,y+delta_y))

    return sites_2P_upper, sites_2P_lower, sites_1P


def placement_symmetries(delta_x=50, delta_y=0, separation_2P = 2, verbose=False, save=False):

    d = 1
    J_2P_1P_fab = pt.cat((J_100_18nm, J_110_18nm[:-3]))/5
    A_2P_fab = pt.tensor([87.3, 90.2, 87.0, 88.9, 74.7, 78.5,  69.2, 69.1, 63.3], dtype=cplx_dtype) * unit.MHz 

    sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites(delta_x, delta_y, d=d, separation_2P=separation_2P)

    print(f"dx = {delta_x}, dy = {delta_y}")

    d_pairs = get_unique_distance_triples(sites_2P_upper, sites_2P_lower , sites_1P)
    nS = len(d_pairs)

    d_A, d_J = get_AJ_distances(d_pairs)
    print(f"{len(d_A)} unique hyperfines, {len(d_J)} unique exchange values.")


    J_map = {}; A_map = {}

    for i in range(len(J_2P_1P_fab)):
        J_map[d_J[i]] = J_2P_1P_fab[i]
    for i in range(len(A_2P_fab)):
        A_map[d_A[i]] = A_2P_fab[i]


    A_2P = pt.zeros(nS, dtype=real_dtype, device=default_device)
    A_2P_1P = pt.zeros(nS, 2, dtype=real_dtype)
    A_1P_2P = pt.zeros(nS, 2, dtype=real_dtype)
    J = pt.zeros(nS, dtype=real_dtype, device=default_device)
    for q in range(nS):
        A_2P[q] = A_map[d_pairs[q][0]]
        A_1P_2P[q,0] = A_mag; A_1P_2P[q,1] = A_2P[q]
        A_2P_1P[q,0] = A_2P[q]; A_2P_1P[q,1] = A_mag
        J[q] = J_map[d_pairs[q][1]]


    if verbose:
        print("2P separations:")
        for i in range(len(d_A)):
            print(f"{d_A[i]*atom_spacing:.3f} nm")
        print("Exchange:")
        for i in range(len(d_J)):
            print(f"{d_J[i]*atom_spacing:.3f} nm, {pt.real(J[i])/unit.MHz:.2f} MHz")

    
    for q in range(nS):
        print(f"A = {A_2P[q]/unit.MHz:.2f} MHz, J = {J[q]/unit.MHz:.2f} MHz")

    if save:
        pt.save(A_2P/unit.MHz, f"{exch_data_folder}A_2P")
        pt.save(A_2P_1P/unit.MHz, f"{exch_data_folder}A_2P_1P")
        pt.save(A_1P_2P/unit.MHz, f"{exch_data_folder}A_1P_2P")
        pt.save(J/unit.MHz, f"{exch_data_folder}J")
    return A_2P, J



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
    Pr = pt.real(W**2 / Omega**2)
    return Pr

def compare_frequency_pair(w_ac, w_res, B_ac, c, maxp=0.01):
    if get_rabi_prob(w_ac, w_res, B_ac, c) > maxp:
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
    nx.draw(G)
    #G = nx.Graph(edges)
    #nx.draw_networkx_nodes(G, node_color='red')
    nx.draw(G, ax=ax)

def compare_1P_2P_frequencies():
    
    J = get_J_1P_2P(48)
    A = get_A_1P_2P(48)

    H0 = get_H0(A,J)

    S,D = get_ordered_eigensystem(H0)
    RF = get_multisys_rf_tensor(S, D)
    C = get_multi_system_couplings(S)

    sys1=1
    sys2=3


    w_res = RF[sys1,0,1]
    c = C[0,0,1]
    w_ac = RF[sys2,0,1]

    B_ac = 5e-3*unit.mT

    #Pmax = system_comparison(RF[sys1], RF[sys2], B_ac, C[0])
    
    graph_system_similarity(RF, B_ac, C)

# Question 1: How low does J have to be?



#Question 2: For the most closely spaced pair of resonant frequencies, what pulse time is required to avoid large off resonant effects?



# Question 3: How much does this improve GRAPE optimisation?




if __name__ == '__main__':
    #plot_load_time_vs_J(fid_min=0.99, Jmin=1*unit.MHz, Jmax=50*unit.MHz, tN_max=100*unit.ns, A=get_A(1,3), n=100, get_t=get_t_wf)

    #plot_load_time_vs_J_2q(fidelity_min=0.99)

    #print(f"{get_t_wf(get_J(1,3), get_A(1,3), 10*unit.ns,1000,0.99)/unit.ns} ns")

    #approximate_full_NE_optimisation_time()

    #placement_symmetries(delta_x = 52, delta_y = 0, save=True)

    compare_1P_2P_frequencies()
    plt.show()