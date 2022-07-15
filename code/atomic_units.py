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
metre = 1/a0_n
kg = 1/mE_n 
coulomb = 1/qE_n 
second = hbar_n*kg*metre**2
nanosecond = 1e-9*second


rad = 1
ampere = coulomb/second
tesla = kg/(ampere*second**2)
mT = 1e-3*tesla
newton = kg*metre*second**(-2)
joule = newton*metre
hz = 2*np.pi/second # because frequencies are expected in rad/second
Mhz = 1e6*hz
Ghz = 1e9*hz
rps = 1/second
Mrps = 1e6*rps
