

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt 
import torch as pt
from scipy.optimize import minimize

from GRAPE import Grape
import gates as gate 
from atomic_units import *
from visualisation import plot_spin_states, plot_psi_and_fields, visualise_Hw, plot_fidelity, plot_fields, plot_phases, plot_energy_spectrum, show_fidelity
from utils import forward_prop, get_pulse_hamiltonian, sum_H0_Hw, fidelity, fidelity_progress, get_U0, dagger, get_IP_X, get_IP_eigen_X, lock_to_coupling, get_resonant_frequencies, get_max_allowed_coupling
from pulse_maker import pi_rot_square_pulse
from data import get_A, get_J, gamma_e, gamma_n, cplx_dtype

from pdb import set_trace


from electrons import get_H0



def visualise_allowed_transitions():

    nq=3
    H0 = get_H0(get_A(1,nq), get_J(1,nq))
    rf = get_resonant_frequencies(H0)

    set_trace() 

if __name__=='__main__':
    visualise_allowed_transitions()