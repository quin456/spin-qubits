

from email.policy import default
import numpy as np
import torch as pt
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt


import gates as gate 
from gates import cplx_dtype
import atomic_units as unit
from data import default_device
from utils import linspace

from pdb import set_trace




def square_pulse(B, omega, tN, N, phase=0):

    T = linspace(0,tN,N, device=default_device)
    
    Bx = B*pt.cos(omega*T-phase)
    By = B*pt.sin(omega*T-phase)

    return Bx,By

def pi_pulse_field_strength(coupling, tN):
    return np.pi / (coupling * tN)


def pi_pulse_square(w_res, coupling, tN, N, phase=0):
    Bw = pi_pulse_field_strength(coupling, tN)
    print(f"Generating square pulse for π-rotation with amplitude {Bw/unit.mT} mT, frequency {w_res/unit.MHz} MHz.")
    Bx,By = square_pulse(Bw, w_res, tN, N, phase)
    return Bx,By



def gaussian_pulse():
    return 

def pi_rot_gaussian_pulse():
    return
    

