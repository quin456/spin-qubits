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
ps = 1e-12*s


rad = 1
A = C/s
T = kg/(A*s**2)
newton = kg*m*s**(-2)
J = newton*m
V = J/C
Hz = 2*np.pi/s # because frequencies are expected in rad/second
rps = 1/s
eV = qE*V 
meV = 1e-3*eV
K=1
mK=1e-3*K

mu_0 = 1.25663706e-6 * m*kg*(s**-2)*(A**2)
mu_B = qE*hbar/(2*mE)



MV = 1e6*V
MHz = 1e6*Hz
kHz = 1e3*Hz
GHz = 1e9*Hz
mT = 1e-3*T
Mrps = 1e6*rps
um = 1e-6*m 
nm = 1e-9*m

kB = 1.38e-23 * m**2 * kg / (s**2 * K)


if __name__ == '__main__':
    # Doing some unit conversions 
    from pdb import set_trace
    set_trace()