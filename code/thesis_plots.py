
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


################################################################################################################
################        ALL PLOTS TO BE USED IN THESIS WILL BE GENERATED HERE        ###########################
################################################################################################################




################################################################################################################
################        CHAPTER 1 INTRODUCTION        ##########################################################
################################################################################################################

def two_electron_energy_level_picture(ax=None, fp = None):
    if ax is None: fig,ax = plt.subplots(1)

    E0 = 10
    E1 = 4.7
    E2 = 3.2
    E3 = 0

    x0=0.1
    y0=(E0-E3)/3 *0.05


    fig.set_size_inches(2.5, 4.5)
    ax.set_ylim(E3-3*y0, E0+6*y0)

    ax.axhline(E0)
    ax.axhline(E1)
    ax.axhline(E2)
    ax.axhline(E3)

    matplotlib.pyplot.annotate(f"<00|$H_0$|00> = $\omega_z+J$", (x0,E0+y0))
    matplotlib.pyplot.annotate(f"<01|$H_0$|01> = $2A-J$", (x0,E1+y0))
    matplotlib.pyplot.annotate(f"<10|$H_0$|10> = $-2A-J$", (x0,E2+y0))
    matplotlib.pyplot.annotate(f"<11|$H_0$|11> = $-\omega_z+J$", (x0,E3+y0))

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    
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




if __name__=='__main__':
    chapter_2()
    plt.show()