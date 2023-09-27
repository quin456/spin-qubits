import torch as pt
import numpy as np
import atomic_units as unit

import matplotlib as mpl

mpl.use("Qt5Agg")
from matplotlib import pyplot as plt

detuning_1P_2P = 9 * unit.meV  # energy detuning at activation point


class Wire:
    def __init__(self, lam, x=None, y=None, z=None):
        self.direction = self.get_wire_direction(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.r0 = 5 * unit.nm

    @staticmethod
    def check_wire_direction(wire_x, wire_y, wire_z):
        coords = [wire_x, wire_y, wire_z]
        spec_coords = [int(coord != None) for coord in coords]
        if sum(spec_coords) != 2:
            raise Exception("Need to specify exactly two wire coordinates.")
        return spec_coords.index(0)

    def voltage(self, loc):
        if self.direction == 0:
            delta_r = np.array([loc[1] - self.y, loc[2] - self.z])
        elif self.direction == 1:
            delta_r = np.array([loc[0] - self.x, loc[2] - self.z])
        if self.direction == 2:
            delta_r = np.array([loc[0] - self.x, loc[1] - self.y])
        else:
            raise Exception("Invalid wire direction.")
        r = np.linalg.norm(delta_r)
        return -self.lam / (2 * np.pi * unit.eps0) * np.log(r / self.r0)


def voltage_from_many_gates(
    h=46 * unit.nm, L=36 * unit.nm, delta_e=9 * unit.meV, width=50
):

    N = 500
    x = np.linspace(0, L, N)
    V = np.zeros(N)
    for k in np.arange(-width, width, 1) + 1:
        d = np.sqrt(h ** 2 + (x - k * L) ** 2)

        A = delta_e / np.log(h / np.sqrt(h ** 2 + L ** 2))

        V += A * np.log(d / np.sqrt(h ** 2 + (L / 2) ** 2))

    V = V - max(V)
    ax = plt.subplot()
    ax.plot(x / unit.nm, V / unit.mV)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("E (meV)")
    ax.set_yticks([V[0] / unit.mV, 0, V[-1] / unit.mV])
    ax.set_xticks([0, 18, 36])


def analytic_voltage_from_gates(h=46 * unit.nm, L=36 * unit.nm, delta_e=9 * unit.meV):
    x = np.linspace(0, L, 500)
    d_plus_2 = np.sqrt(h ** 2 + x ** 2)
    d_plus_1 = np.sqrt(h ** 2 + (L - x) ** 2)
    d_minus = np.sqrt(h ** 2 + (L - x) ** 2)

    A = delta_e / np.log(h / np.sqrt(h ** 2 + L ** 2))

    V_minus1 = A * np.log(np.sqrt(h ** 2 + (L / 2) ** 2) / d_minus)
    V_plus2 = A * np.log(d_plus_2 / np.sqrt(h ** 2 + (L / 2) ** 2))
    V_plus1 = A * np.log(d_plus_1 / np.sqrt(h ** 2 + (L / 2) ** 2))
    V = V_plus2 + V_minus1

    ax = plt.subplot()
    ax.plot([0, L / unit.nm], [9, -9], linestyle="--", color="black", label="linear")
    ax.plot(x / unit.nm, V_plus2 / unit.mV, color="red", label="$\mu_+$")
    ax.plot(x / unit.nm, V_minus1 / unit.mV, color="lightblue", label="$\mu_-$")
    ax.plot(x / unit.nm, V / unit.mV, color="green", label="$\mu_+ + \mu_-$")
    # ax.plot(
    #     x / unit.nm,
    #     (V_plus2 - V_minus1) / unit.mV,
    #     color="orange",
    #     label="$\mu_+ + \mu_+$",
    # )
    ax.set_yticks([V[0] / unit.mV, 0, V[-1] / unit.mV])
    ax.set_xticks([0, 18, 36])
    ax.legend()
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("E (meV)")


if __name__ == "__main__":
    analytic_voltage_from_gates()
    # voltage_from_many_gates()
    plt.show()
