
from pathlib import Path
import os
from turtle import forward
dir = os.path.dirname(__file__)
os.chdir(dir)

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt 
import torch as pt
from scipy.optimize import minimize


import gates as gate 
from gates import cplx_dtype
from atomic_units import *
from visualisation import plot_spin_states, plot_psi_and_fields
from utils import forward_prop, get_pulse_hamiltonian, sum_H0_Hw, fidelity
from pulse_maker import square_pulse
from data import get_A, gamma_e, gamma_P

from pdb import set_trace

Sz = 0.5*gate.IZ; Sy = 0.5*gate.IY; Sx = 0.5*gate.IX 
Iz = 0.5*gate.ZI; Iy = 0.5*gate.YI; Ix = 0.5*gate.XI

from gates import spin_up, spin_down



def pi_rot_pulse(w_res, gamma, tN, N):
    Bw = np.pi / (gamma * tN)
    Bx,By = square_pulse(Bw,w_res,tN,N)
    return Bx,By


def NE_CX_pulse(tN,N,A,Bz):
    w_res = 2*A + gamma_e * Bz
    return pi_rot_pulse(w_res, gamma_e, tN, N)


def EN_CX_pulse(tN,N,A,Bz):
    w_res = -(2*A+gamma_P*Bz)
    return pi_rot_pulse(w_res, gamma_P, tN, N)




def NE_CX(A, Bz, tN, N, psi0=spin_up):
    '''
    CNOT gate with electron spin as target, nuclear spin as control.
    '''
    Bx,By = NE_CX_pulse(tN,N,A,Bz)
    H0 = (-A+gamma_e*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx,By,gamma_e)
    T = pt.linspace(0,tN,N)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)
    plot_psi_and_fields(psi,Bx,By,tN)


def EN_CX(A, Bz, tN, N, psi0=spin_up):
    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    H0 = (-A-gamma_P*Bz/2)*gate.Z 
    Hw = get_pulse_hamiltonian(Bx, By, gamma_P)
    T = pt.linspace(0,tN,N)
    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)
    plot_psi_and_fields(psi,Bx,By,tN)



    return 0



def NE_swap_pulse(tN,N,A,Bz):
    
    N_NE = N//10
    N_EN = N-2*N_NE
    tN_NE = N_NE/N * tN 
    tN_EN = N_EN/N * tN

    Bx_NE,By_NE = NE_CX_pulse(tN_NE, N_NE, A, Bz)
    Bx_EN,By_EN = EN_CX_pulse(tN_EN, N_EN, A, Bz)

    Bx = pt.cat((Bx_NE,Bx_EN,Bx_NE))
    By = pt.cat((By_NE,By_EN,By_NE))


    return Bx,By

def H_zeeman(Bz):
    return gamma_e*Bz*Sz - gamma_P*Bz*Iz

def H_hyperfine(A):
    return A * gate.sigDotSig

def NE_H0(A,Bz):
    return H_zeeman(Bz) + H_hyperfine(A)

def NE_swap(A,Bz,tN,N,psi0=pt.kron(spin_down,spin_up)):

    H0 = NE_H0(A,Bz)

    #Bx,By = NE_swap_pulse(tN,N,A,Bz)
    Bx,By = EN_CX_pulse(tN,N,A,Bz)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e, 2*Sx, 2*Sy) - get_pulse_hamiltonian(Bx, By, gamma_P, 2*Ix, 2*Iy)
    #Hw = get_pulse_hamiltonian(Bx, By, -gamma_P, 2*Ix, 2*Iy)

    H = sum_H0_Hw(H0,Hw)
    U = pt.matrix_exp(-1j*H*tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)

    plot_psi_and_fields(psi,Bx,By,tN)




def get_subops(H,dt):
    ''' Gets suboperators for time-independent Hamiltonian H '''
    return pt.matrix_exp(-1j*H*dt)






def linear_Bz_decrease(tN=50*nanosecond, N=10000, psi0=pt.kron(spin_up,spin_down), A=get_A(1,1)*Mhz, Bz=2*tesla):
    H0 = H_hyperfine(A) + H_zeeman(Bz)

    t_sw = np.pi/(4*A)
    N_swap = int(t_sw*N/tN) 
    N_dropoff = N-N_swap
    
    u = pt.cat((pt.linspace(0,-2,N_dropoff), -2*pt.ones(N_swap)))
    H_control = 1*tesla * (gamma_e*Sz - gamma_P*Iz)

    Hw = pt.einsum('j,ab->jab', u, H_control)

    H = sum_H0_Hw(H0,Hw)
    U = get_subops(H,tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)

    plot_spin_states(psi,tN)


def NE_grape_X(u, H0, H_control, tN, N, target = gate.swap):
    
    pulse = gaussian(u, tN, N) # (tesla - as in, if I print it it will be around 2 teslas. NOT in atomic units yet.)
    pulse = sigmoid_pulse(u, tN, N)
    Hw = pt.einsum('j,ab->jab',pulse,H_control)

    H = sum_H0_Hw(H0,Hw)
    U = get_subops(H,tN/N)
    X = forward_prop(U)
    return X

def NE_grape_cost(u, H0, H_control, tN, N, target = gate.swap):
    X = NE_grape_X(u, H0, H_control, tN, N, target = target)
    return 1 - fidelity(X[-1], target)


def gaussian(u, tN, N):
    T = pt.linspace(0,tN,N)
    t_center = tN/2
    return u[0]*pt.exp(-((T-t_center)/(u[1]*nanosecond))**2)    

def sigmoid(x,alpha,A):
    return A / (1 + pt.exp(-alpha*x))


def sigmoid_pulse(u, tN, N):
    alpha,A,w,t0 = u
    T = pt.linspace(0,tN,N) / nanosecond
    return sigmoid(T-t0,alpha,A) - sigmoid(T-w-t0,alpha,A)

def optimise_NE_swap_Bz():

    tN=100*nanosecond
    N=500
    A = get_A(1,1) * Mhz
    Bz = 2*tesla
    sigma0 = 5 # spread of pulse
    mag0 = -2 # maximum magnitude of pulse

    H0 = NE_H0(A,Bz)
    H_control = 1*tesla * (gamma_e*Sz - gamma_P*Iz)
    cost = lambda u: NE_grape_cost(u,H0,H_control,tN,N)

    w0= 1.6*np.pi/(4*A) / nanosecond
    alpha0 = 5
    amp0=-2
    t0_0 = 10
    u0 = pt.tensor([alpha0,amp0,w0, t0_0])


    optimisation = minimize(cost,u0, method = 'Nelder-Mead')
    u = optimisation.x

    X = NE_grape_X(u, H0, H_control, tN, N)
    psi0 = pt.kron(spin_up,spin_down)
    psi = pt.matmul(X,psi0)

    fig,ax=plt.subplots(1,2)
    plot_spin_states(psi,tN,ax[0])



    ax[1].plot(pt.linspace(0,tN/nanosecond,N),sigmoid_pulse(u, tN, N))
    print(optimisation)
    







#NE_CX(get_A(1,1)*Mhz, 2*tesla, 1000*nanosecond, 100000, psi0=spin_down); plt.show()
#EN_CX(get_A(1,1)*Mhz, 2*tesla, 99*nanosecond, 10000, psi0=spin_down); plt.show()

NE_swap(get_A(1,1)*Mhz, 2*tesla, 9999*nanosecond, 10000, psi0=pt.kron(spin_down,spin_down)); plt.show()

