from email.policy import default
import numpy as np
import torch as pt
import matplotlib

if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import torch as pt
from torch.fft import fft


import gates as gate
from gates import cplx_dtype
from data import *
import atomic_units as unit
from data import default_device
from utils import *


def get_smooth_E(tN, N, rise_time=1 * unit.ns):
    E_mag = 0.7 * unit.MV / unit.m
    T = linspace(0, tN, N, dtype=real_dtype, device=default_device)
    E = E_mag * (
        sigmoid((T - 10 * unit.ns) / rise_time)
        - sigmoid((T - T[-1] + 10 * unit.ns) / rise_time)
    )
    return E


def get_simple_E(tN, N, rise_time=10 * unit.ns, E_max=1 * unit.MV / unit.m):
    rise_prop = rise_time / tN
    E = E_max * rise_ones_fall(N, rise_prop)
    return E


def square_pulse(B, omega, tN=None, N=None, T=None, phase=0):
    T = evaluate_timestep_inputs(T, tN, N)
    Bx = B * pt.cos(omega * T - phase)
    By = B * pt.sin(omega * T - phase)
    return Bx, By, T


def pi_pulse_field_strength(coupling, tN):
    return np.pi / (coupling * tN)


def pi_pulse_duration(coupling, B_ac):
    return np.pi / (coupling * B_ac)


def pi_pulse_square(w_res, coupling, tN=None, N=None, T=None, phase=0):
    T = evaluate_timestep_inputs(T, tN, N)
    Bw = pi_pulse_field_strength(coupling, T[-1])
    print(
        f"Generating square pulse for π-rotation with amplitude {Bw/unit.mT} mT, frequency {real(w_res/unit.MHz):.2f} MHz, N = {N}, delta_t = {T[-1]/len(T)/unit.ps:.2f} ps"
    )
    Bx, By, T = square_pulse(Bw, w_res, T=T, phase=phase)
    return Bx, By


def gaussian_pulse():
    return


def pi_rot_gaussian_pulse():
    return


def block_interpolate_single_vector(v, N_new):
    v_new = pt.zeros(N_new, dtype=v.dtype)
    N = len(v)
    if N_new % N != 0:
        raise Exception(
            "Pulse interpolate not implemented for N_new not divisible by original N."
        )
    m = int(N_new / N)
    for j in range(N):
        for k in range(m):
            v_new[j * m + k] = v[j]
    return v_new


def block_interpolate_pulse(Bx, By, T, N_new):
    Bx = block_interpolate_single_vector(Bx, N_new)
    By = block_interpolate_single_vector(By, N_new)
    T = linspace(0, T[-1], N_new)
    # T = block_interpolate_single_vector(T, N_new) # NOT THIS, because T = [dt, 2dt, ..., N], so T[0] needs to change.
    return Bx, By, T


def cartesian_to_polar(x, y):
    n = len(x)
    A = np.sqrt(x ** 2 + y ** 2)
    theta = pt.zeros_like(x)
    for j in range(n):
        if real(x[j]) == 0 and real(y[j]) == 0:
            # amplitude is zero, theta value not important.
            theta[j] = 0
        elif real(x[j]) >= 0:
            if real(y[j]) >= 0:
                # first quadrant
                theta[j] = np.arctan(y[j] / x[j])
            else:
                # fourth quadrant
                theta[j] = np.arctan(y[j] / x[j]) + 2 * np.pi
        else:
            # second or third quadrant
            theta[j] = np.arctan(y[j] / x[j]) + np.pi
    return A, theta


def frame_transform_pulse(Bx, By, T, w0):
    """
    Transforms pulse (Bx, By) from frame rotating at frequency w0.
    This is done by adding theta = w0*t at each timestep t.

    Bx = B cos(theta)
    By = B sin(theta)
    """

    B, theta = cartesian_to_polar(Bx, By)
    theta0 = w0 * T
    theta += theta0

    return B * pt.cos(theta), B * pt.sin(theta)


def load_XY(fp):
    XY = pt.load(fp + "_XY")
    Bx = XY[0] * unit.T
    By = XY[1] * unit.T
    T = XY[2] * unit.ns
    
    return Bx, By, T


if __name__ == "__main__":

    # XY_field = pt.load("fields/electron_flip_XY")

    # Bx = XY_field[0]
    # By = XY_field[1]
    # T = XY_field[2]
    # frame_transform_pulse(Bx, By)

    t = linspace(0, 4 * np.pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    A, theta = cartesian_to_polar(x, y)
    plt.plot(t, theta)

    plt.show()
