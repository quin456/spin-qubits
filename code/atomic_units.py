import numpy as np


###   Fundamental constants   ###

mE_n = np.float64(9.109e-31)
qE_n = np.float64(1.6022e-19)
hbar_n = np.float64(1.0546e-34)
a0_n = np.float64(5.2918e-11)


# Fundamental constants set to unity for atomic units
mE = np.float64(1)
qE = np.float64(1)
hbar = np.float64(1)
a0 = np.float64(1)


# remaining constants
h_planck = np.float64(2 * np.pi * hbar)
m = np.float64(1 / a0_n)
kg = np.float64(1 / mE_n)
C = np.float64(1 / qE_n)
s = np.float64(hbar_n * kg * m ** 2)
us = np.float64(1e-6 * s)
ns = np.float64(1e-9 * s)
ps = np.float64(1e-12 * s)


rad = np.float64(1)
A = np.float64(C / s)
T = np.float64(kg / (A * s ** 2))
newton = np.float64(kg * m * s ** (-2))
J = np.float64(newton * m)
V = np.float64(J / C)
Hz = np.float64(2 * np.pi / s)  # because frequencies are expected in rad/second
rps = np.float64(1 / s)
eV = np.float64(qE * V)
meV = np.float64(1e-3 * eV)
K = np.float64(1)
mK = np.float64(1e-3 * K)

mu_0 = np.float64(1.25663706e-6 * m * kg * (s ** -2) * (A ** 2))
mu_B = np.float64(qE * hbar / (2 * mE))


MV = np.float64(1e6 * V)
MHz = np.float64(1e6 * Hz)
kHz = np.float64(1e3 * Hz)
GHz = np.float64(1e9 * Hz)
mT = np.float64(1e-3 * T)
uT = np.float64(1e-6 * T)
Mrps = np.float64(1e6 * rps)
um = np.float64(1e-6 * m)
nm = np.float64(1e-9 * m)

kB = np.float64(1.38e-23 * m ** 2 * kg / (s ** 2 * K))


if __name__ == "__main__":
    # Doing some unit conversions
    breakpoint()
