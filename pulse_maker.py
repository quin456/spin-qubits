
from pathlib import Path
import os
from turtle import forward
dir = os.path.dirname(__file__)
os.chdir(dir)

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import torch as pt


from GRAPE import get_A 
import gates as gate 
from gates import cplx_dtype
from atomic_units import *
from excitation import plot_spin_states, forward_prop

from pdb import set_trace




def square_pulse(B, omega, tN, N, phase=0):

    T = pt.linspace(0,tN,N)

    Bx = B*pt.cos((omega-phase)*T)
    By = B*pt.sin((omega-phase)*T)

    return Bx,By
    

