import numpy as np


###   Fundamental constants   ###

mE_n = 9.109e-31
qE_n = 1.6022e-19
hbar_n = 1.0546e-34        
a0_n = 5.2918e-11


# Fundamental constants set to unity for atomic units
mE = 1
qE = 1
hbar = 1
a0 = 1


# remaining constants
h_planck = 2*np.pi*hbar
m = 1/a0_n
kg = 1/mE_n 
C = 1/qE_n 
s = hbar_n*kg*m**2
ns = 1e-9*s


rad = 1
A = C/s
T = kg/(A*s**2)
mT = 1e-3*T
newton = kg*m*s**(-2)
joule = newton*m
hz = 2*np.pi/s # because frequencies are expected in rad/second
MHz = 1e6*hz
GHz = 1e9*hz
rps = 1/s
Mrps = 1e6*rps
