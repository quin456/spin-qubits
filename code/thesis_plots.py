
import matplotlib


matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure 
import numpy as np 
import torch as pt

from GRAPE import GrapeESR
import gates as gate
from utils import psi_from_polar, get_A, get_J, get_U0, normalise, get_pulse_hamiltonian, get_resonant_frequencies
from visualisation import bloch_sphere, plot_spin_states
from visualisation import *
from single_spin import show_single_spin_evolution
from data import dir, cplx_dtype, gamma_e, gamma_n, J_100_14nm, J_100_18nm
from atomic_units import *
from architecture_design import plot_cell_array, plot_annotated_cell, generate_CNOTs, numbered_qubits_cell, plot_single_cell
from electrons import get_2E_H0, get_ordered_2E_eigensystem, plot_free_electron_evolution, get_free_electron_evolution

plots_folder = f"{dir}thesis-plots/"

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'

max_time = 10

################################################################################################################
################        ALL PLOTS TO BE USED IN THESIS WILL BE GENERATED HERE        ###########################
################################################################################################################




################################################################################################################
################        CHAPTER 1 INTRODUCTION        ##########################################################
################################################################################################################

def energy_level_picture(energies, spin_labels, energy_labels, ax=None, fp=None):

    x0=0.05
    y0=(max(energies)-min(energies))/3 *0.05

    if ax is None: fig,ax = plt.subplots(1)

    for j,energy in enumerate(energies):
        ax.axhline(energy, 0, 0.84)
        plt.annotate(spin_labels[j], (0.85, energy-0.08))
        plt.annotate(energy_labels[j], (x0,energy+y0))

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')

    fig.set_size_inches(2.5, 4.5)
    ax.set_ylim(min(energies)-3*y0, max(energies)+6*y0)
    ax.set_xlim(0, 1)
    
    if fp is not None:
        plt.savefig(fp)

def two_electron_energy_level_picture(ax=None, fp = None):

    E0 = 10
    E1 = 4.7
    E2 = 3.2
    E3 = 0

    energies = [E0, E1, E2, E3]
    labels = [f"$\omega_z+J$", f"$2A-J$",  f"$-2A-J$", f"$-\omega_z+J$"]
    energy_level_picture(energies, labels, ax, fp)




def NE_energy_level_picture(ax=None, fp=None):
    

    energies = [10, 0, 9, 3]
    energy_labels = [f"$A + (\omega_e-\omega_n)/2$", f"$-A - (\omega_e-\omega_n)/2$", f"$-A + (\omega_e+\omega_n)/2$" , f"$A - (\omega_e+\omega_n)/2$"]
    spin_labels = [f"{Uparrow}{uparrow}", f"{Uparrow}{downarrow}",f"{Downarrow}{uparrow}",f"{Downarrow}{downarrow}"]
    energy_level_picture(energies, spin_labels, energy_labels, ax=ax)

    plt.arrow(0.7,energies[2], 0, energies[3]-energies[2], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[3], 0, energies[2]-energies[3], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[1], 0, energies[3]-energies[1], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[3], 0, energies[1]-energies[3], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')

    plt.annotate("$\omega_e$",(0.71,energies[2] + 0.5*(energies[3]-energies[2])))
    plt.annotate("$\omega_n$",(0.71,energies[1] + 0.5*(energies[3]-energies[1])))

    if fp is not None:
        plt.savefig(fp)


def free_2E_evolution(fp=None):
    A = get_A(1,1); J=get_J(1,2); tN=10*nanosecond
    H0 = get_2E_H0(A,J)
    U0 = get_U0(H0, tN, 1000)
    psi0 = pt.tensor([0.1, 0.3, 0.7, 0.2], dtype=cplx_dtype); normalise(psi0)
    psi0 = normalise(pt.tensor([0.5, 0, 0.5, 0], dtype=cplx_dtype))
    psi = U0@psi0

    plot_spin_states(psi, tN, squared=True, fp=fp)

def exchange_label_getter(i):
    if i in [1,2,4]:
        return f"Pr({np.binary_repr(i,3)})"

def compare_free_3E_evols(fp=None):
    '''
    Compares 3E evolution for 101 vs 001 nuclear spin configurations
    '''
    tN = 10*nanosecond 
    N = 500
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(double_long_width, single_long_height)


    plot_free_electron_evolution(tN, N, get_A(1,3,[0,1,0]), get_J(50,3, J1=J_100_14nm, J2=J_100_18nm)[15], ax=ax[0], label_getter=exchange_label_getter)
    plot_free_electron_evolution(tN, N, get_A(1,3,[0,0,1]), get_J(1,3), ax=ax[1], label_getter=exchange_label_getter)
    plt.tight_layout()
    if fp is not None:
        fig.savefig(fp)

def show_2E_Hw(J,A, tN, N, fp=None):
    H0 = get_2E_H0(A,J)
    rf = get_resonant_frequencies(H0, gate.get_Xn(2)+gate.get_Yn(2))
    T = pt.linspace(0,tN,N)
    w = rf[1]
    Bx=pt.cos(w*T); By=pt.sin(w*T)
    Hw =  get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(2), gate.get_Yn(2))
    S,D = get_ordered_2E_eigensystem(A, J)
    U0 = get_U0(H0, tN, N)
    visualise_Hw(S.T@dagger(U0)@Hw@U0@S, tN)

def chapter_1():
    #bloch_sphere(psi_from_polar(np.pi/2,np.pi/4), fp = f'{plots_folder}Ch1-bloch-sphere.pdf')
    #show_single_spin_evolution(tN=100*nanosecond, fp = f"{plots_folder}Ch1-analytic-example.pdf")
    two_electron_energy_level_picture(fp=f"{plots_folder}Ch1-2E-energy-levels.pdf")


def chapter_2():
    plot_cell_array(4,4, filename=f"{plots_folder}cell_array")
    generate_CNOTs()
    plot_annotated_cell(filename=f"{plots_folder}single_cell")
    numbered_qubits_cell()
    plot_single_cell()
    plt.show()



def chapter_3():

    def grape_1s2q(fp=None, fp1=None):
        grape = GrapeESR(J=get_J(1,2),A=get_A(1,2),tN=20*nanosecond,N=500, max_time=5); grape.run()
        fig,ax = plt.subplots(1,2)
        grape.plot_u(ax[0])
        grape.plot_cost_hist(ax[1])
        fig.set_size_inches(double_long_width, single_long_height)
        fig1,ax1 = grape.plot_field_and_evolution()
        fig1.set_size_inches(double_long_width, double_long_height)
        plt.tight_layout()
        if fp is not None: fig.savefig(fp)
        if fp1 is not None: fig1.savefig(fp1)


    #NE_energy_level_picture(fp=f"{plots_folder}Ch3-NE-energy-levels.pdf")

    #free_2E_evolution(fp = f"{plots_folder}Ch3-2E-free-evolution.pdf")


    #grape_1s2q(fp = f"{plots_folder}Ch3-2E-u-and-cost.pdf", fp1 = f"{plots_folder}Ch3-2E-field-and-evolution.pdf")

    #grape = GrapeESR(get_J(1,3), get_A(1,3), tN=100*nanosecond, N=500, max_time=max_time); grape.run(); grape.plot_field_and_fidelity(f"{plots_folder}Ch3-3E-field-and-evolution.pdf")

    #show_2E_Hw(get_J(1,2),get_A(1,2),30*nanosecond,500, "Ch3-2E-Hw.pdf")

    compare_free_3E_evols(fp = f"{plots_folder}Ch3-3E-free-evol-comparison.pdf")


def no_coupler():
    def plot_exchange_switch(fp=None):
        N=5000
        tN=20*nanosecond
        t0 = 5*nanosecond
        psi0 = pt.tensor([0,0,1,0,0,0,0,0], dtype=cplx_dtype)
        psi1 = pt.einsum('j,a->ja', pt.ones(N//2), psi0)
        psi2 = get_free_electron_evolution(tN=tN/2, N=N//2, psi0=psi0)

        j=0
        while pt.abs(psi2[j,5]) > 0.99:
            j+=1 
        print(f"Pr(100) < 0.99 after t-t0 = {j/N*tN/nanosecond} ns")
        psi = pt.cat((psi1, psi2))
        fig,ax = plt.subplots(1,1)
        plot_spin_states(psi, tN, legend_loc = 'center left', ax=ax, label_getter=exchange_label_getter)
        fig.set_size_inches(double_long_width, double_long_height)
        ax.set_xlabel("time (ns)")
        ax.axvline(10, color='black', linestyle='--', linewidth=1)
        ax.annotate("$t=t_0$",[9,0.5])
        fig.tight_layout()
        if fp is not None:
            fig.savefig(fp)


    plot_exchange_switch(f"{plots_folder}NC-exchange-switch.pdf")

if __name__=='__main__':
    #chapter_3()

    no_coupler()

    plt.show()