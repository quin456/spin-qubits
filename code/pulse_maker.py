

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
from atomic_units import *
from data import default_device

from pdb import set_trace




def square_pulse(B, omega, tN, N, phase=0):

    T = pt.linspace(0,tN,N, device=default_device)
    
    Bx = B*pt.cos(omega*T-phase)
    By = B*pt.sin(omega*T-phase)

    return Bx,By


def pi_rot_square_pulse(w_res, coupling, tN, N, phase=0):
    Bw = np.pi / (coupling * tN)
    Bx,By = square_pulse(Bw, w_res, tN, N, phase)
    return Bx,By



def gaussian_pulse():
    return 

def pi_rot_gaussian_pulse():
    return
    

