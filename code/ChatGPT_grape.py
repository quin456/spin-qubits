import numpy as np
import scipy.optimize

# Define the Hamiltonian of the system
def H(t, omega, phi):
    return omega * np.cos(phi) * np.sigma_x + omega * np.sin(phi) * np.sigma_y


# Define the initial and target states
psi_0 = np.array([1, 0])
psi_targ = np.array([0, 1])

# Define the control pulse
def u(t, omega, phi):
    return omega * np.cos(phi), omega * np.sin(phi)


# Define the objective function to be maximized
def f(params, t):
    omega, phi = params
    U = scipy.linalg.expm(-1j * np.trapz(H(t, omega, phi), t))
    return -np.abs(np.dot(U, psi_0).conj().T @ psi_targ) ** 2


# Define the gradient of the objective function
def grad_f(params, t):
    omega, phi = params
    n = len(t)
    dH_domega = np.cos(phi) * np.sigma_x + np.sin(phi) * np.sigma_y
    dH_dphi = -omega * np.sin(phi) * np.sigma_x + omega * np.cos(phi) * np.sigma_y
    dU_domega = np.zeros((2, 2), dtype=complex)
    dU_dphi = np.zeros((2, 2), dtype=complex)
    for i in range(n - 1):
        U = scipy.linalg.expm(-1j * np.trapz(H(t[: i + 1], omega, phi), t[: i + 1]))
        dU_domega += (U @ dH_domega @ U.conj().T) * (t[i + 1] - t[i])
        dU_dphi += (U @ dH_dphi @ U.conj().T) * (t[i + 1] - t[i])
    grad_omega = 2 * np.real(np.dot(psi_targ.conj().T, dU_domega @ psi_0))
    grad_phi = 2 * np.real(np.dot(psi_targ.conj().T, dU_dphi @ psi_0))
    return -np.array([grad_omega, grad_phi])


# Set the time grid
t = np.linspace(0, 1, 100)

# Set the initial guess for the control pulse
params0 = np.array([1, 0])

# Optimize the control pulse using gradient ascent
result = scipy.optimize.minimize(f, params0, args=(t,), jac=grad_f, method="L-BFGS-B")
omega_opt, phi_opt = result.x

# Calculate the final unitary operator
