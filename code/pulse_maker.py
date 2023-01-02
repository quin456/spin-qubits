

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
from utils import *

from pdb import set_trace


def get_smooth_E(tN, N, rise_time = 1*unit.ns):
    E_mag = 0.7*unit.MV/unit.m
    T = pt.linspace(0,tN,N,device=default_device)
    E = E_mag * (sigmoid((T-10*unit.ns)/rise_time) - sigmoid((T-T[-1]+10*unit.ns)/rise_time))
    return E

def get_simple_E(tN, N, rise_time = 10*unit.ns, E_max = 1*unit.MV/unit.m):
    rise_prop = rise_time / tN
    E = E_max * rise_ones_fall(N, rise_prop)
    return E
    


def square_pulse(B, omega, tN, N, phase=0):
    T = linspace(0,tN,N, device=default_device)
    Bx = B*pt.cos(omega*T-phase)
    By = B*pt.sin(omega*T-phase)
    return Bx,By

def pi_pulse_field_strength(coupling, tN):
    return np.pi / (coupling * tN)

def pi_pulse_duration(coupling, B_ac):
    return np.pi / (coupling * B_ac)

def pi_pulse_square(w_res, coupling, tN, N, phase=0):
    Bw = pi_pulse_field_strength(coupling, tN)
    print(f"Generating square pulse for π-rotation with amplitude {Bw/unit.mT} mT, frequency {real(w_res/unit.MHz):.2f} MHz, N = {N}, delta_t = {tN/N/unit.ps:.2f} ps")
    Bx,By = square_pulse(Bw, w_res, tN, N, phase)
    return Bx,By

def gaussian_pulse():
    return 

def pi_rot_gaussian_pulse():
    return
    

