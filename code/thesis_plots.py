
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import numpy as np 
import torch as pt


from utils import psi_from_polar
from visualisation import bloch_sphere
from single_spin import show_single_spin_evolution
from data import dir

plots_folder = f"{dir}thesis-plots/"


################################################################################################################
################        ALL PLOTS TO BE USED IN THESIS WILL BE GENERATED HERE        ###########################
################################################################################################################



bloch_sphere(psi_from_polar(np.pi/2,np.pi/4), fp = f'{plots_folder}Ch1-bloch-sphere.pdf')
show_single_spin_evolution(fp = f"{plots_folder}Ch1-single-spin-flip.pdf")





plt.show()