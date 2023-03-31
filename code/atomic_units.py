"""
atomic_units.py
Defines numerical values for required units using the Hartree-Atomic system
of units, which follows from setting the following fundamental constants to 1:
ℏ: Reduced Planck's constant.
e: Elementary charge.
a0: Bohr radius.
m_e: Electron mass.
"""


import numpy as np


###   Fundamental constants in SI units  ###

mE_SI = np.float64(9.109e-31)
qE_SI = np.float64(1.6022e-19)
hbar_SI = np.float64(1.0546e-34)
a0_SI = np.float64(5.2918e-11)


# Fundamental constants set to unity for atomic units
mE = np.float64(1)
qE = np.float64(1)
hbar = np.float64(1)
a0 = np.float64(1)


# units
m = np.float64(1 / a0_SI)
um = np.float64(1e-6 * m)
nm = np.float64(1e-9 * m)
kg = np.float64(1 / mE_SI)
C = np.float64(1 / qE_SI)
s = np.float64(hbar_SI * kg * m ** 2)
us = np.float64(1e-6 * s)
ns = np.float64(1e-9 * s)
ps = np.float64(1e-12 * s)
A = np.float64(C / s)
T = np.float64(kg / (A * s ** 2))
mT = np.float64(1e-3 * T)
uT = np.float64(1e-6 * T)
newton = np.float64(kg * m * s ** (-2))
J = np.float64(newton * m)
V = np.float64(J / C)
MV = np.float64(1e6 * V)
mV = np.float64(1e-3 * V)
Hz = np.float64(2 * np.pi / s)  # because frequencies are expected in rad/second
kHz = np.float64(1e3 * Hz)
MHz = np.float64(1e6 * Hz)
GHz = np.float64(1e9 * Hz)
rad = np.float64(1)
rps = rad / s
eV = np.float64(qE * V)
meV = np.float64(1e-3 * eV)
K = np.float64(1)
mK = np.float64(1e-3 * K)
Mrps = np.float64(1e6 * rps)


# remaining constants
h_planck = np.float64(2 * np.pi * hbar)
mu_0 = np.float64(1.25663706e-6 * m * kg * (s ** -2) * (A ** 2))
mu_B = np.float64(qE * hbar / (2 * mE))
kB = np.float64(1.38e-23 * m ** 2 * kg / (s ** 2 * K))
eps0 = np.float64(8.85e-12) * m ** (-3) * kg ** (-1) * s ** 4 * A ** 2

if __name__ == "__main__":
    breakpoint()
