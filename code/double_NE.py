from multi_NE import *
from visualisation import *
from pulse_maker import (
    pi_pulse_square,
    block_interpolate_pulse,
    frame_transform_pulse,
    pi_pulse_field_strength,
)
from single_NE import (
    NE_swap_pulse,
    NE_CX_pulse,
    EN_CX_pulse,
    NE_eigensystem,
    NE_couplings,
    get_NE_X,
    NE_transition_pulse,
)
from hamiltonians import (
    get_NE_H0,
    get_NE_Hw,
    multi_NE_H0,
    multi_NE_Hw,
    single_J_coupled_electron_H0,
    get_pulse_hamiltonian,
    get_U0,
    single_electron_H0,
)
from eigentools import get_pi_pulse_tN_from_field_strength
from utils import print_rank2_tensor
from GRAPE import load_grape


B_mag = 1e-3 * unit.T


def NE_CX(tN=None, N=100000, interaction_picture=False):
    A = get_A(1, 1)
    Bz = 2 * unit.T
    Bx, By, T = NE_CX_pulse(tN, N, A, Bz)

    X = get_NE_X(N, Bz, A, Bx, By, T=T, interaction_picture=interaction_picture)
    print_rank2_tensor(
        pt.einsum("ij,ij->ij", (pt.abs(X[-1]) > 1e-5).to(int), pt.angle(X[-1]))
    )
    psi0 = pt.tensor([1, 0, 1, 0], dtype=cplx_dtype) / np.sqrt(2)
    psi = X @ psi0
    plot_psi(psi, T=T)


def EN_CX(tN, N):
    A = get_A(1, 1)
    Bz = 2 * unit.T
    Bx, By, T = EN_CX_pulse(tN, N, A, Bz)
    # Bx, By, T = NE_transition_pulse(1, 3, tN, N, A, Bz)

    X = get_NE_X(N, Bz, A, Bx, By, tN=tN)
    print_rank2_tensor(pt.angle(X[-1]))
    psi0 = pt.tensor([1, 1, 0, 0], dtype=cplx_dtype) / np.sqrt(2)
    psi = X @ psi0
    plot_psi(psi, T=T)


def NE_swap(N_e, N_n):
    N = 2 * N_e + N_n
    A = get_A(1, 1)
    Bz = 2 * unit.T
    Bx, By, T = NE_swap_pulse(N_e, N_n, A, Bz)
    X = get_NE_X(N, Bz, A, Bx, By, T=T)
    print_rank2_tensor(pt.abs(X[-1]))
    psi0 = pt.tensor([1, 1, 0, 0], dtype=cplx_dtype) / np.sqrt(2)
    psi = X @ psi0
    plot_psi(psi, T=T)
    proj_to_qubit_space_1q(X[-1])


def NE_swap_2q(N_e, N_n):

    XY_field = pt.load("fields/electron_flip_XY")
    Bx_e = XY_field[0] * unit.T
    By_e = XY_field[1] * unit.T
    T_e = XY_field[2] * unit.ns

    Bz = 2 * unit.T

    Bx_e, By_e, T_e = block_interpolate_pulse(Bx_e, By_e, T_e, 800000)
    Bx_e, By_e = frame_transform_pulse(Bx_e, By_e, T_e, gamma_e * Bz)

    # N = 2 * N_e + N_n
    A = get_A(1, 1)
    # Bx, By, T = NE_swap_pulse(N_e, N_n, A, Bz)
    Bx = Bx_e
    By = By_e
    T = T_e
    X = get_multi_NE_X(
        len(Bx), Bz, A, get_J_low(1, 2), 2, Bx, By, T=T, deactivate_exchange=True
    )
    psi0 = pt.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=cplx_dtype)
    psi = X @ psi0

    target = gate.kron4(gate.Id, gate.Id, gate.X, gate.X)
    fid = fidelity(pt.abs(X[-1]), real(target))
    print(f"abs fid = {fid}")

    plot_psi(psi, T=T)
    set_trace()
    # project_to_qubit_space(X[-1])
    # print(pt.sum(pt.abs(X[-1]) - gate.NE_swap_2q))


def project_to_qubit_space(T, nuclear_to_electron=True):
    """
    projects 2e, 2n space onto qubit space
    """
    Id4 = pt.kron(gate.Id, gate.Id)
    spin_01 = gate.spin_01.reshape(4, 1)
    n_01 = pt.kron(spin_01, Id4)
    e_01 = pt.kron(Id4, spin_01)
    if nuclear_to_electron:
        phi = e_01.T @ T @ n_01
    else:
        phi = n_01.T @ T @ e_01
    print_rank2_tensor(phi)


def proj_to_qubit_space_1q(T):
    spin_1 = gate.spin_1.reshape(2, 1)
    n_1 = pt.kron(spin_1, gate.Id)
    e_1 = pt.kron(gate.Id, spin_1)
    phi = e_1.T @ T @ n_1
    print_rank2_tensor(phi)


def test_grape_pulse(XY_field=None):
    Bz = 2 * unit.T  # 2*unit.T
    w0 = gamma_e * Bz
    print(f"w0 = {w0/unit.MHz} MHz")
    if XY_field is None:
        XY_field = pt.load("fields/electron_flip_XY")

    Bx = XY_field[0] * unit.T
    By = XY_field[1] * unit.T
    T = XY_field[2] * unit.ns
    print(f"pulse duration = {real(T[-1])/unit.ns} ns")
    Bx, By, T = block_interpolate_pulse(Bx, By, T, 50000)
    Bx, By = frame_transform_pulse(Bx, By, T, w0)
    # plot_fields(Bx, By, T=T)
    # Bx, By = block_interpolate_pulse(Bx, By, 20000)
    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H0_all = single_J_coupled_electron_H0(Bz, -get_A(1, 1), get_J_low(15, 2))
    for H0 in H0_all:
        # H0 = single_electron_H0(0, -get_A(1, 1))
        # U0 = get_U0(H0, T=T)
        # H_IP = pt.einsum("jab,jbc,jcd->jad", dagger(U0), Hw, U0)
        H = sum_H0_Hw(H0, Hw)
        X = get_X_from_H(H, T=T)
        # print_rank2_tensor(X[-1])
        print(f"fidelity = {fidelity(X[-1], gate.X)}")
        print(f"abs fidelity = {fidelity(pt.abs(X[-1]), gate.X.to(real_dtype))}")
        # print_rank2_tensor(X[-1] * pt.abs(X[-1, 0, 0]) / X[-1, 0, 0])
        psi0 = gate.spin_0


def test_grape_with_added_Bz():
    N = 20000
    Bz = 0.1 * unit.T  # 2*unit.T
    w0 = gamma_e * Bz
    print(f"w0 = {w0/unit.MHz} MHz")
    tN = lock_to_frequency(get_A(1, 1), 100 * unit.ns)
    T = linspace(0, tN, N)
    get_rec_min_N(pt.tensor([w0]), tN, verbosity=2)
    w_res = 2 * get_A(1, 1) + 0 * get_J_low(15, 2)[0]
    w_res_frame = -2 * get_A(1, 1) + 2 * get_J_low(15, 2)[0] + w0

    H0 = single_J_coupled_electron_H0(Bz, get_A(1, 1), 0 * get_J_low(15, 2))[0]

    Bx, By = pi_pulse_square(w_res, gamma_e, T=T)
    XY_field = pt.load("fields/single_electron_XY")
    Bx = XY_field[0] * unit.T
    By = XY_field[1] * unit.T
    T = XY_field[2] * unit.ns
    Bx, By, T = block_interpolate_pulse(Bx, By, T, 24000)
    Bx, By = frame_transform_pulse(Bx, By, T, w0)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, T=T)
    print_rank2_tensor(X[-1])

    print(f"abs fidelity = {fidelity(pt.abs(X[-1]), gate.X.to(real_dtype))}")


def test_pulse_frame_transform():
    A = get_A(1, 1)
    J = get_J_low(1, 2)
    N = 500
    Bz = 0 * unit.T  # 2*unit.T
    w0 = gamma_e * Bz
    print(f"w0 = {w0/unit.MHz} MHz")
    tN = lock_to_frequency(get_A(1, 1), 100 * unit.ns)
    T = linspace(0, tN, N)
    get_rec_min_N(pt.tensor([w0]), tN, verbosity=2)
    w_res = 2 * get_A(1, 1) + 0 * get_J_low(15, 2)[0]
    w_res_frame = -2 * get_A(1, 1) + 2 * get_J_low(15, 2)[0] + w0

    H0 = single_electron_H0(Bz, A, J)

    # diagonal also doesn't work when I turn down N and require iterpolation :)
    # B = pi_pulse_field_strength(gamma_e, T[-1])
    # B_mag = pt.cat((linspace(0, 2 * B, N // 2), linspace(2 * B, 0, N // 2)))
    # Bx = pt.einsum("j,j->j", B_mag, pt.cos(w_res * T))
    # By = pt.einsum("j,j->j", B_mag, pt.sin(w_res * T))

    # Even with N=2e4 step grape pulse it's still fucked :(
    XY_field = pt.load("fields/single_electron_XY")
    Bx = XY_field[0] * unit.T
    By = XY_field[1] * unit.T
    T = XY_field[2] * unit.ns

    Bx, By, T = block_interpolate_pulse(Bx, By, T, 1000)
    Bx, By = frame_transform_pulse(Bx, By, T, w0)

    Hw = get_pulse_hamiltonian(Bx, By, gamma_e)
    H = sum_H0_Hw(H0, Hw)
    X = get_X_from_H(H, T=T)
    print(f"len(X) = {len(X)}")
    print_rank2_tensor(X[-1])

    print(f"abs fidelity = {fidelity(pt.abs(X[-1]), gate.X.to(real_dtype))}")

    psi0 = gate.spin_0
    plot_psi(X @ psi0, T=T)


def test_grape_pulse_on_2NE():
    pass


if __name__ == "__main__":
    # NE_CX(N=20000, interaction_picture=True)
    # EN_CX(10000 * unit.ns, 20000)
    # NE_swap(10000, 20000)
    NE_swap_2q(10000, 20000)
    # project_to_qubit_space(gate.NE_swap_2q)
    # test_grape_pulse()
    # test_pulse_frame_transform()
    plt.show()

