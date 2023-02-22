from data import *
import atomic_units as unit


def extrapolate_exchange():
    x = np.array([2, 0, -2])
    y = np.array([9, 15, 21])
    X = np.stack((np.ones(len(x)), x)).T

    theta = np.linalg.inv(X.T @ X) @ (X.T@y)
    print(f"theta = {theta}")
    x0 = theta[0] + theta[1]*(-6)

    print(f"Separation for 1 kHz exchange is {x0} nm")


def hyperfine_modulation_E_field_strength():

    """
    Determine electric field strength required to address single qubits by modulating hyperfine.
    """

    P = 0.999
    t_1q_gate = 0.3 * unit.us

    A = get_A(1, 1)

    B_ac = 2 * np.pi / (gamma_e * t_1q_gate)
    E = np.sqrt(gamma_e * B_ac / (-2 * eta2 * A)) * (1 / P - 1) ** (1 / 4)

    # print(f"B_ac = {B_ac/unit.mT} mT")
    print(f"Electric field requires: {E/unit.MV * unit.m} MV/m")


def rabi_prob(dw, B_ac=1 * unit.mT, c=gamma_e):
    Pr = (c * B_ac / 2) ** 2 / ((c * B_ac) ** 2 + dw ** 2)
    return Pr


if __name__ == "__main__":
    # hyperfine_modulation_E_field_strength()
    # print(f"Pr = {rabi_prob(2*unit.MHz, 1*unit.mT)}")
    extrapolate_exchange()

