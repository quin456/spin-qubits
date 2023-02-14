import torch as pt
import numpy as np
import matplotlib
from scipy import linalg
from enum import Enum

if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
import itertools


from data import default_device, cplx_dtype
import atomic_units as unit


from pdb import set_trace

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def zeros_like_reshape(tensor, shape):
    """
    Returns array of zeros of specified shape with dtype and device of tensor.
    """
    return pt.zeros(*shape, dtype=tensor.dtype, device=tensor.device)


def get_nS_nq_from_A(A):
    """ Returns (number of systems, number of qubits in each system) """
    try:
        if len(A.shape) == 2:
            return len(A), len(A[0])
        elif len(A.shape) == 1:
            return 1, len(A)
    except:
        # A is not an array
        return 1, 1


def normalise(v):
    """ Normalises 1D tensor """
    return v / pt.norm(v)


def dagger(A):
    """  Returns the conjugate transpose of a matrix or batch of matrices.  """
    return pt.conj(pt.transpose(A, -2, -1))


def commutator(A, B):
    """  Returns the commutator [A,B]=AB-BA of matrices A and B.  """
    return pt.matmul(A, B) - pt.matmul(B, A)


def matmul3(A, B, C):
    """  Returns multiple of three matrices A,B,C.  """
    return pt.matmul(A, pt.matmul(B, C))


def batch_trace(T):
    """
    Takes an array of arbitrary shape containing square matrices (in other words a tensor whose innermost dimensions are of
    equal size), and determines the trace of each matrix. Traces are returned in a tensor having the same shape as the
    aforementioned array.
    """
    return (T.diagonal(dim1=-2, dim2=-1)).sum(-1)


def batch_IP(A, B):
    """ 
    Takes two equal sized arrays of square matrices A,B, and returns a bitwise inner product of the matrices of the form
        torch.tensor([<A[0],B[0]>, <A[1],B[1]>, ... , <A[-1],B[-1]>])
    """
    d = A.shape[-1]
    return (1 / d) * batch_trace(pt.matmul(dagger(A), B))


def innerProd(A, B):
    """  Calculates the inner product <A|B>=Phi(A,B) of two matrices A,B.  """
    d = A.shape[-1]
    return pt.trace(pt.matmul(dagger(A), B)).item() / d


def fidelity(A, B):
    """ Calculates fidelity of operators A and B """
    IP = innerProd(A, B)
    return np.real(IP * np.conj(IP))


def batch_fidelity(Ut, Uf):
    IP = batch_IP(Ut, Uf)
    Phi = pt.real(IP * pt.conj(IP))
    return Phi


def wf_fidelity(u, v):
    """ Calculates vector fidelity of u and v """
    return pt.real(pt.dot(u, pt.conj(v)) * pt.dot(pt.conj(u), v))


def get_nq_from_dim(d):
    """ Takes the dimension of the Hilbert space as input, and returns the number of qubits. """
    return int(np.log2(d))


def sigmoid(z):
    return 1 / (1 + pt.exp(-z))


def slow_batch_diag(D):
    """
    More time computing, less time coding
    """
    E = zeros_like_reshape(D, D.shape[:-1])
    if len(D.shape) > 3:
        raise Exception("Not implemented")
    for i in range(D.shape[0]):
        E[i] = pt.diag(D[i])
    return E


#######################################################################################################################
# GRAPE UTILS
#######################################################################################################################
"""
The following functions are used to manipulate and access 'u', which contains the control field amplitudes at 
each timestep, the learning parameters of the GRAPE algorithm. The most natural form for 'u' is an m x N matrix,
but it is converted to and from vector form for input into scipy.minimize.
"""


def uToVector(u):
    """  Takes m x N torch tensor 'u' and converts to 1D tensor in which the columns of u are kept together.  """
    if u is None:
        return u
    if len(u.shape) == 1:
        return u
    return pt.reshape(pt.transpose(u, 0, 1), (u.numel(),))


def uToMatrix(u, m):
    """  
    Inverse of uToVector. Takes m*N length 1D tensor, splits into m N-sized tensors which are stacked together as columns of an
    m x N tensor which is the output of the function.
    """
    N = int(len(u) / m)
    return pt.transpose(pt.reshape(u, (N, m)), 0, 1)


def uIdx(u, m, j, k):
    """  Accesses element (j,k) of vector form u.  """
    if len(u.shape) == 1:
        return u[k * m + j]
    return u[j, k]


def uCol(u, j, m):
    """  Accepts vector form 'u' as input, and returns what would be column 'j' if 'u' were in matrix form  """
    return u[j * m : (j + 1) * m]


#######################################################################################################################
#######################################################################################################################


def psi_from_polar(theta, phi):
    if not pt.is_tensor(theta):
        theta = pt.tensor([theta])
        phi = pt.tensor([phi])
    return pt.stack(
        (pt.cos(theta / 2), pt.einsum("j,j->j", pt.exp(1j * phi), pt.sin(theta / 2)))
    ).T


def get_single_qubit_angles(psi):
    """
    input psi: (N,2) complex array describing single qubit wave function over N timesteps.
    output: (N,2) real array (probably still complex dtype) describing theta, phi over N timesteps.
    """
    reshaped = False
    if len(psi.shape) == 1:
        psi = psi.reshape(1, *psi.shape)
        reshaped = True
    theta = 2 * pt.arctan(pt.abs(psi[:, 1] / psi[:, 0]))
    phi = pt.angle(psi[:, 1]) - pt.angle(psi[:, 0])
    if reshaped:
        return theta[0], phi[0]
    return theta, phi


def psi_to_cartesian(psi):
    """
    psi: (N,2) or (2,) array describing single qubit wave funtion over N timesteps.
    Function returns (N,3) or (3,) array describing cartesian coordinates on the Bloch sphere
    """
    reshaped = False
    if len(psi.shape) == 1:
        psi = psi.reshape(1, *psi.shape)
        reshaped = True

    theta, phi = get_single_qubit_angles(psi)
    x = pt.sin(theta) * pt.cos(phi)
    y = pt.sin(theta) * pt.sin(phi)
    z = pt.cos(theta)
    r = pt.stack((x, y, z)).T

    if reshaped:
        return r[0]
    return r


def forward_prop(U, device=default_device, sys_axis=True):
    """
    Forward propagates U suboperators. U has shape (N,d,d) or (nS,N,d,d)
    """
    if len(U.shape) == 3:
        U = U.reshape(1, *U.shape)
        sys_axis = False

    nS, N, dim, dim = U.shape
    nq = get_nq_from_dim(dim)
    X = pt.zeros((nS, N, dim, dim), dtype=cplx_dtype, device=device)
    X[:, 0, :, :] = U[:, 0]  # forward propagated time evolution operator

    for j in range(1, N):
        X[:, j, :, :] = pt.matmul(U[:, j, :, :], X[:, j - 1, :, :])

    if sys_axis:
        return X
    return X[0]


def fidelity_progress(X, target):
    """
    For each system, determines the fidelity of unitaries in P with the target over the time of the pulse
    """
    multisys = True
    if len(X.shape) == 3:
        X = X.reshape(1, *X.shape)
        multisys = False
    if len(target.shape) == 2:
        target = target.reshape(1, *target.shape)
    nS = len(X)
    N = len(X[0])
    fid = pt.zeros(nS, N)
    for q in range(nS):
        for j in range(N):
            IP = innerProd(target[q], X[q, j])
            # fid[q, j] = np.real(IP * np.conj(IP))
            fid[q, j] = fidelity(target[q], X[q, j])

    if not multisys:
        fid = fid[0]
    return fid


def psi_to_string(psi, pmin=0.01, real_evecs=True):
    """
    Returns triple donor nuclear-electron spin state as a1|000000> + ... + a63|111111>,
    ignoring all a's for which |a|^2 < pmin.
    """
    out = ""
    dim = len(psi)
    nq = get_nq_from_dim(dim)
    add_plus = False
    for j in range(dim):
        if pt.abs(psi[j]) ** 2 > pmin:
            if add_plus:
                out += "+ "
            if real_evecs:
                out += f"{pt.real(psi[j]):0.2f}|{np.binary_repr(j,nq)}> "
            else:
                out += f"({pt.real(psi[j]):0.2f}+{pt.imag(psi[j]):0.2f}i)|{np.binary_repr(j,nq)}> "
            add_plus = True
    return out


def map_psi(A, psi):
    """
    Acts rank 2 tensor on N x dim psi tensor
    """
    return pt.einsum("ab,jb->ja", A, psi)


def print_eigenstates(S):
    for j in range(len(S)):
        print(f"|E{j}> = {psi_to_string(S[:,j])}")


def remove_duplicates(A):
    """
    Removes duplicates from A where equivalence is required to 9 decimal places
    """
    i = 0
    while i < len(A):
        j = i + 1
        while j < len(A):
            if math.isclose(A[i], A[j], rel_tol=1e-9):
                A.pop(j)
                continue
            j += 1
        i += 1
    return A


def clean_vector(v, tol=1e-8):
    for i in range(len(v)):
        if pt.abs(v[i]) < tol:
            v[i] = 0


def print_rank2_tensor(T):
    m, n = T.shape
    for i in range(m):
        if i == 0:
            print("⌈", end="")
        elif i == n - 1:
            print("⌊", end="")
        else:
            print("|", end="")
        for j in range(n):
            print(f"{T[i,j].item():>8.2f}", end="  ")

        if i == 0:
            print("⌉")
        elif i == n - 1:
            print("⌋")
        else:
            print("|")


def label_axis(
    ax, label, x_offset=-0.10, y_offset=-0.10, projection="2D", z_offset=0, fontsize=16
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    x = xlim[0] + x_offset * dx
    y = ylim[0] + y_offset * dy

    # if offset<0:
    #     ax.set_xlim(offset, xlim[1])
    #     ax.set_ylim(offset, ylim[1])
    if projection == "3D":
        zlim = ax.get_ylim()
        dz = zlim[1] - zlim[0]
        z = zlim[0] + z_offset * dz
        ax.text(
            x, y, z, label, fontsize=fontsize, fontweight="bold", va="bottom", ha="left"
        )
    else:
        ax.text(
            x, y, label, fontsize=fontsize, fontweight="bold", va="bottom", ha="left"
        )


def get_rec_min_N(rf, tN, N_period=100, verbosity=0):
    T = 1 / rf
    max_w = pt.max(pt.abs(rf)).item()
    rec_min_N = int(np.ceil(real(N_period * max_w * tN) / (2 * np.pi)))
    if verbosity > 1:
        print(f"resonant freqs = {rf/unit.MHz}")
        print(f"T = {T/unit.ns}")
        print(f"Recommened min N = {rec_min_N}")
    return rec_min_N


def evaluate_timestep_inputs(T, tN, N):
    if T is None:
        if tN is None:
            raise Exception("No time information provided.")
        if N is None:
            raise Exception("No timestep information provided")
        T = linspace(0, tN, N, device=default_device)
    return T


def get_dT(T):
    """
    Gets dt's for each timestep from T tensor containing all time values. 
    """
    dT = pt.zeros_like(T)
    dT[0] = T[0] - 0
    dT[1:] = T[1:] - T[:-1]
    return dT


def linspace(start, end, N, dtype=cplx_dtype, device=default_device):
    # return pt.linspace(start, end, N, dtype=dtype, device=device)
    start = np.float64(start)
    end = np.float64(end)
    return pt.linspace(
        start + (end - start) / np.float64(N), end, N, dtype=dtype, device=device
    )


def maxreal(T):
    return pt.max(pt.real(T))


def real(z):
    try:
        return pt.real(z)
    except:
        return z


def minreal(T):
    return pt.min(pt.real(T))


def rise_ones_fall(N, rise_prop):
    N_rise = int((N * rise_prop) // 1)
    return pt.cat(
        (
            linspace(0, 1, N_rise, device=default_device),
            pt.ones(N - 2 * N_rise, device=default_device),
            linspace(1, 0, N_rise, device=default_device),
        )
    )


def sqrtm(T):
    return pt.tensor(linalg.sqrtm(T), dtype=T.dtype, device=T.device)


class NoiseModels(str, Enum):
    delta_correlated_exchange = "delta-exchange"
    dephasing = "dephasing"


if __name__ == "__main__":
    set_trace()
