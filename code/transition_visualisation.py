
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import plotly.graph_objects as go
import networkx as nx



from matplotlib import pyplot as plt 
import torch as pt
from scipy.optimize import minimize

from GRAPE import Grape
import gates as gate 
from atomic_units import *
from visualisation import plot_spin_states, plot_psi_and_fields, visualise_Hw, plot_fidelity, plot_fields, plot_phases, plot_energy_spectrum, show_fidelity
from utils import get_resonant_frequencies, get_allowed_transitions, print_rank2_tensor, get_nS_nq_from_A
from pulse_maker import pi_rot_square_pulse
from data import get_A, get_J, gamma_e, gamma_n, cplx_dtype, J_100_18nm

from pdb import set_trace


from electrons import get_H0, get_ordered_2E_eigensystem
from single_NE import NE_swap



def visualise_allowed_transitions(H0):
    

    dim = H0.shape[-1]
    edges = get_allowed_transitions(H0)
    rf = get_resonant_frequencies(H0)

    nodes = np.linspace(1,dim, dim, dtype=int)

    G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    G = nx.Graph(name="transitions")
    G = nx.Graph(edges)
    #nx.draw_networkx_nodes(G, node_color='red')
    nx.draw(G)
    plt.show()


def get_node_label(node,nq):
    if nq==3:
        return f"|E{str(node)}⟩"
    elif nq==2:
        if node==0:
            return "$|T_+⟩$"
        elif node==1:
            return "$|T_0⟩$"
        elif node==2:
            return "$|S_0⟩$"
        elif node==3:
            return "$|T_-⟩$"

def label_transitions(transitions,nq):
    labelled_transitions = [[] for _ in range(len(transitions))]
    for i,transition in enumerate(transitions):
        labelled_transitions[i] = (get_node_label(transition[0],nq), get_node_label(transition[1],nq))
    return labelled_transitions



def visualise_E_transitions(A=get_A(1,3), J=get_J(1,3), Bz=0, ax=None, label=None):
    _nS,nq = get_nS_nq_from_A(A)
    H0 = get_H0(A, J, Bz=Bz)

    S,D = get_ordered_2E_eigensystem(A, J, Bz=Bz)
    d = len(D) #dimension, number of states / nodes
    print_rank2_tensor(S)
    print_rank2_tensor(D/Mhz)


    nlabels = [get_node_label(i,nq) for i in range(d)]

    transitions = get_allowed_transitions(H0, S=S, E=pt.diag(D))
    labelled_transitions = label_transitions(transitions, nq)


    G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    G = nx.Graph(name="transitions")
    node_list = [get_node_label(i, nq) for i in range(d)]
    G.add_nodes_from(node_list)
    G.add_edges_from(labelled_transitions)


    if nq==3:
        G.nodes[nlabels[0]]['pos'] = (-1,10)
        G.nodes[nlabels[7]]['pos'] = (1,-10)
        G.nodes[nlabels[1]]['pos'] = (-6,4)
        G.nodes[nlabels[2]]['pos'] = (-1,4)
        G.nodes[nlabels[3]]['pos'] = (4,4)
        G.nodes[nlabels[4]]['pos'] = (-4,-4)
        G.nodes[nlabels[5]]['pos'] = (1,-4)
        G.nodes[nlabels[6]]['pos'] = (6,-4)
    elif nq==2:
        G.nodes[nlabels[0]]['pos'] = (0,10)
        G.nodes[nlabels[1]]['pos'] = (3,1)
        G.nodes[nlabels[2]]['pos'] = (-3,-1)
        G.nodes[nlabels[3]]['pos'] = (0,-10)

    color = ['']*d

    if nq==2:
        for i in range(d):
            if node_list[i] == get_node_label(0,nq):
                color[i]='blue'
            elif node_list[i] in [get_node_label(j,nq) for j in [1,2]]:
                color[i] = 'green'
            elif node_list[i]==get_node_label(3,nq):
                color[i] = 'red'
    elif nq==3:
        for i in range(d):
            if node_list[i] == get_node_label(0,nq):
                color[i]='blue'
            elif node_list[i] in [get_node_label(j,nq) for j in [1,2,3]]:
                color[i]='green'
            elif node_list[i] in [get_node_label(j,nq) for j in [4,5,6]]:
                color[i]='orange'
            elif node_list[i]==get_node_label(7,nq):
                color[i] = 'red'



    # set_trace()
    # for i in range(n):
    #     G.nodes[i]['state'] = f'$|E_{i}\rangle$'
    pos={node: G.nodes[node]['pos'] for node in G.nodes}
    nx.draw(G, with_labels=True, pos=pos, node_size=700, node_color=color, font_color='white', width=1.5, ax=ax)
    #nx.draw_networkx_nodes(G, label=True, pos=pos, node_size=700, node_color=color, font_color='white', width=1.5, ax=ax, margins=(10,10))
    #nx.draw_networkx_labels(G,pos=pos)
    
    #nx.draw_networkx_nodes()
    if label is not None:
        print(label)
        ax.annotate(label, [-4,-10])




if __name__=='__main__':
    #visualise_allowed_transitions()
    visualise_E_transitions(A=get_A(1,3, NucSpin=[1,0,1]),J=get_J(1,3, J1=J_100_18nm, J2=J_100_18nm)); plt.show()