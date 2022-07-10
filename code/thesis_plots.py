
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure 
import numpy as np 
import torch as pt


from utils import psi_from_polar
from visualisation import bloch_sphere
from single_spin import show_single_spin_evolution
from data import dir
from atomic_units import *
from architecture_design import plot_cell_array, plot_annotated_cell, generate_CNOTs, numbered_qubits_cell, plot_single_cell

plots_folder = f"{dir}thesis-plots/"

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'

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
    NE_energy_level_picture(fp=f"{plots_folder}Ch3-NE-energy-levels.pdf")


if __name__=='__main__':
    chapter_3()
    plt.show()