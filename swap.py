
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


import GRAPE as grape
from GRAPE import get_A 
import gates as gate 
from gates import cplx_dtype
from atomic_units import *
from excitation import plot_2q_eigenstates, plot_spin_states, forward_prop
from pulse_maker import square_pulse

from pdb import set_trace

gamma_P = 17.235 * Mhz/tesla
gamma_e = 1.7609e11 * 1/(second*tesla)


Sz = 0.5*gate.IZ; Sy = 0.5*gate.IY; Sx = 0.5*gate.IX 
Iz = 0.5*gate.ZI; Iy = 0.5*gate.YI; Ix = 0.5*gate.XI

spin_up = pt.tensor([1,0],dtype=cplx_dtype)
spin_down = pt.tensor([0,1], dtype=cplx_dtype)

A = get_A(1,1)*Mhz
Bz = 2*tesla

def nuclear_electron_H0(A,Bz):

    return gamma_e*Bz*Sz - gamma_P*Bz*Iz + A * gate.sigDotSig

def nuclear_electron_Hw(omega,tN,N,Bw):
    T = pt.linspace(0,tN,N)



    cos_wt_X = pt.einsum('j,ab->jab', pt.cos(omega*T), (Sx+Ix))
    sin_wt_Y = pt.einsum('j,ab->jab', pt.sin(omega*T), (Sy+Iy))

    return (gamma_e-gamma_P)*Bw * (cos_wt_X + sin_wt_Y)

def NE_swap(Bz=2*tesla,N=20000):

    E_sw = (gamma_P+gamma_e)*Bz
    H0 = nuclear_electron_H0(0,Bz)

    psi0 = pt.kron(spin_up, spin_down)

    Bw = 0.5*tesla

    tN = np.pi / Bw

    Hw = nuclear_electron_Hw(E_sw, tN, N, Bw)
    T=pt.linspace(0,tN,N)
    H = pt.einsum('j,ab->jab', T, H0) + Hw 

    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)

    psi = pt.matmul(X,psi0)


    plot_spin_states(psi,tN)
    return psi


def NE_CX(A, Bz, tN, N, psi0=spin_up):
    '''
    CNOT gate with electron spin as target, nuclear spin as control.
    '''
    # resonant frequency 
    omega = 2*A + gamma_e * Bz
    Bw = np.pi / (gamma_e * tN)
    Bx,By = square_pulse(Bw,omega,tN,N)
    H0 = (A+gamma_e*Bz/2)*gate.Z 
    Hw = 0.5*gamma_e * ( pt.einsum('j,ab->jab',Bx,gate.X) + pt.einsum('j,ab->jab',By,gate.Y) )
    set_trace()
    T = pt.linspace(0,tN,N)
    H = pt.einsum('j,ab->jab',pt.ones(N),H0) + Hw
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)
    plot_spin_states(psi,tN)


def EN_CX(A, Bz, tN, N, psi0=spin_up):
    omega = 2*A-gamma_P*Bz 
    Bw = 1*np.pi / (gamma_P * tN)
    Bx,By = square_pulse(Bw,omega,tN,N)
    H0 = (A-gamma_P*Bz/2)*gate.Z 
    Hw = 0.5*gamma_P * ( pt.einsum('j,ab->jab',Bx,gate.X) + pt.einsum('j,ab->jab',By,gate.Y) )
    T = pt.linspace(0,tN,N)
    H = pt.einsum('j,ab->jab',pt.ones(N),H0) + Hw
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)
    plot_spin_states(psi,tN)



    return 0



def NE_cx_swap():
    return 0




    
#NE_CX(get_A(1,1)*Mhz, 2*tesla, 1*nanosecond, 10000)
EN_CX(get_A(1,1)*Mhz, 2*tesla, 100*nanosecond, 10000)

plt.show()
