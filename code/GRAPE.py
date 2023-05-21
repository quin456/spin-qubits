# coding tools
from abc import abstractmethod, ABC
import time
import warnings


warnings.filterwarnings("ignore")

# library imports
import torch as pt
import numpy as np
import math
import matplotlib

if not pt.cuda.is_available():
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import itertools
from scipy import linalg as la
from scipy.optimize import minimize
from numpy import random as rand
from numpy import sin, cos, sqrt
from datetime import datetime
import ast


# Other python files in project
import atomic_units as unit
import gates as gate
from utils import *
from eigentools import *
from data import *
from hamiltonians import get_H0, single_electron_H0, get_pulse_hamiltonian, get_U0
from visualisation import *
from pulse_maker import get_smooth_E, get_simple_E

time_exp = 0
time_prop = 0
time_grad = 0
time_fid = 0

mergeprop_g = False


jac = True  # keep as True unless testing cost grad function
save_mem = False  # save mem in Hamiltonian arrays.
work_in_eigenbasis = False  # Should be set to False unless testing
""" save_mem incompatible with work_in_eigenbasis and interaction picture """


ngpus = pt.cuda.device_count()

np.set_printoptions(4)
pt.set_printoptions(sci_mode=True)


log_fn = dir + "logs/log.txt"
ID_fn = dir + "logs/ID_tracker.txt"
precision_loss_msg = "Desired error not necessarily achieved due to precision loss."


################################################################################################################
################        HELPER FUNCTIONS        ################################################################
################################################################################################################


def CNOT_targets(nS, nq, device=default_device, native=False):
    if nq == 2:
        if native:
            return pt.einsum(
                "i,ab->iab", pt.ones(nS, device=device), gate.CX_native.to(device)
            )
        return pt.einsum("i,ab->iab", pt.ones(nS, device=device), gate.CX.to(device))
    elif nq == 3:
        return pt.einsum("i,ab->iab", pt.ones(nS, device=device), gate.CX3q.to(device))


def coupler_CX_targets(nS, device=default_device):
    return pt.einsum(
        "s,ab->sab", pt.ones(nS, dtype=cplx_dtype, device=device), gate.coupler_target
    )


def CXr_targets(nS, device=default_device):
    return pt.einsum("i,ab->iab", pt.ones(nS, device=device), gate.CXr.to(device))


def SWAP_targets(n, device=default_device):
    return pt.einsum("i,ab->iab", pt.ones(n, device=device), gate.swap.to(device))


def RSW_targets(n, device=default_device):
    return pt.einsum("i,ab->iab", pt.ones(n, device=device), gate.root_swap.to(device))


def grad(f, x0, dx):
    """Uses difference method to compute grad of function f which takes vector input."""
    df = pt.zeros(len(x0), dtype=real_dtype)
    for j in range(len(x0)):
        x1 = x0.clone()
        x2 = x0.clone()
        x2[j] += dx
        x1[j] -= dx
        df[j] = (f(x2) - f(x1)) / (2 * dx)
    return df


################        BATCH OPERATIONS        ################################################################
def makeBatch(H, T):
    """
    Returns a rank 3 tensor, with the outermost array consisting of each element of 1D tensor T multiplied by matrix H.
    """
    N = len(T)
    return pt.mm(T.unsqueeze(1), H.view(1, H.shape[0] * H.shape[1])).view(N, *H.shape)


def mergeMatmul(A, backwards=False):
    # A has shape Nsys x Nstep x d x d where d = 2^(number of qubits) = dim of H.S.

    if len(A[0]) == 1:
        return A

    h = len(A[0]) // 2
    A[:, :h] = mergeMatmul(A[:, :h], backwards)
    A[:, h:] = mergeMatmul(A[:, h:], backwards)

    if backwards:
        # back propagates target to get P matrices
        A[:, :h] = pt.einsum("sjab,sbc->sjac", A[:, :h], A[:, h])
    else:
        # propagates identity to generate X matrices
        A[:, h:] = pt.einsum("sjab,sbc->sjac", A[:, h:], A[:, h - 1])
    return A


################################################################################################################
################        GRAPE IMPLEMENTATION        ############################################################
################################################################################################################


################       COST FUNCTION        ############################################################


def hamShape(H0, Hw):
    if save_mem:
        nS = len(H0)
        m, N, d = Hw.shape[:-1]
    else:
        nS, m, N, d = Hw.shape[:-1]
    nq = int(np.log2(d))

    return nS, m, N, d, nq


def new_target(Uf):
    # non-zero target elements
    target_nz = pt.tensor(
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ],
        dtype=cplx_dtype,
    )

    nS, d, d = Uf.shape
    # accepted U elements
    U_acc = pt.multiply(
        Uf, pt.einsum("s,ab->sab", pt.ones(nS, dtype=cplx_dtype), target_nz)
    )
    U_sqamp = pt.multiply(U_acc, pt.conj(U_acc))

    # new target will be U_accepted, only normalised so that each row has length 1.
    U_sqlen = pt.sum(U_sqamp, 2)

    target_new = pt.sqrt(
        pt.div(U_sqamp, pt.einsum("sa,b->sab", U_sqlen, pt.ones(d, dtype=cplx_dtype)))
    )
    return target_new


def get_unit_CFs(omega, phase, tN, N, device=default_device):
    omega = omega.to(device)
    phase = phase.to(device)
    T = linspace(0, tN, N, device=device, dtype=pt.float64)
    wt = pt.einsum("k,j->kj", omega, T)
    x_cf_unit = pt.cos(wt + phase)
    y_cf_unit = pt.sin(wt + phase)

    return x_cf_unit.type(cplx_dtype), y_cf_unit.type(cplx_dtype)


def get_control_fields(omega, phase, tN, N, device=default_device):
    """
    Returns x_cf, y_cf, which relate to transverse control field, and have units of joules so that 'u' can be unitless.
    """
    x_cf_unit, y_cf_unit = get_unit_CFs(omega, phase, tN, N, device=device)
    x_cf = x_cf_unit * 0.5 * g_e * mu_B * (1 * unit.T)
    y_cf = y_cf_unit * 0.5 * g_e * mu_B * (1 * unit.T)
    return x_cf, y_cf


class Grape(ABC):
    """
    Runs gradient ascent pulse engineering (GRAPE) optimisation on a quantum
    system.

    Note that all physical quantities are unitless, having already been
    multiplied by the appropriate unit from the atomic_units module.

    Attributes:
        tN (float64):   Duration of pulse.
        N (int):        Number of timesteps.
        u (pt.Tensor):  Control vector of shape (m,N) to be optimised, which
                        modulates amplitudes of m control fields over N timesteps.
        omega (pt.Tensor): (m,) vector containing frequencies of m control fields
        phase (pt.Tensor): (m,) vector containing phases of m control fields.

    """

    def __init__(
        self,
        tN,
        N,
        target=None,
        rf=None,
        u0=None,
        cost_hist=[],
        sp_distance=99999 * 3600,
        verbosity=1,
        filename=None,
        lam=0,
        alpha=0,
        kappa=1,
        operation="Setting up",
        noise_model=None,
        ensemble_size=1,
        cost_momentum=0,
        simulate_spectators=True,
        target_spec=gate.Id,
        X0_spec=gate.Id,
        X0=None,
        simulation_steps=False,
        interaction_picture=False,
        matrix_exp_batches=1,
        stop_fid_avg=0.99999,
        stop_fid_min=0.99999,
        dynamic_opt_plot=False,
    ):
        """
        Contructor for Grape class

        Args:
            tN      (float64):  Duration of pulse
            N       (int):      Number of timesteps
            target  (pt.tensor[complex128] / None):
                                Single unitary or array of unitaries describing
                                desired effect of pulse on system(s).
            rf      (pt.tensor[float64] / None):
                                1D array of frequencies for control fields.
            u0      (pt.tensor[float64] / None):
                                Initial value for control vector, u.
            cost_hist (List[float64]):
                                Cost history prior to starting optimisation.
                                Used for instantiating Grape with date loaded
                                from previous grape optimisation.
            max_time (int):     Time in seconds after which optimisation is
                                terminated of convergence criterion is not
                                satisfied.
            sp_distance (int):  Time between save points (deprecated)
            verbosity (int):    Level of information to print
            operation (str):    Descriptor for grape initialisation process,
                                one of "Setting up", "Loading", "Copying".
            lam     (float):    Amplitude penalisation term
            alpha   (float):    Fluctuation penalisation term
            kappa   (float):    Cost gradient modulation term
            noise_model (<enum: 'NoiseModels'>):
                                Type of noise to apply to system
            ensemble_size
                    (int):      Size of ensemble of systems used to calculate
                                cost in the presence of noise
            cost_momentum
                    (float):    Momentum term for cost calculated in with noise.
            simulate_spectators
                    (bool):     If True, pulse is optimised to perform identity
                                on single qubit systems.
            X0      (pt.tensor[complex128] / None):
                                Initial unitary from which time evolution
                                operators are propagated
            simulation_steps
                    (bool):     If True, additional simulation steps are
                                performed between each of the N timesteps.
                                The N timesteps limit the precision with which
                                control field



        """

        # Store initial class attributes
        self.tN = np.float64(tN)
        self.N = N
        self.dt = np.float64(self.tN / self.N)
        self.target = target if target is not None else self.get_default_targets()
        self.target = self.get_target(target)
        self.u = uToVector(u0)
        self.interaction_picture = interaction_picture
        self.H0 = self.get_H0(Bz=self.Bz)
        self.rf = rf
        self.simulate_spectators = simulate_spectators
        self.target_spec = target_spec
        self.nq_spec = self.get_nq_spec(target_spec)
        self.X0_spec = X0_spec
        self.initialise_spectators()
        self.initialise_control_fields()
        # allow for H0 with no systems axis
        self.reshaped = False
        if len(self.H0.shape) == 2:
            self.H0 = self.H0.reshape(1, *self.H0.shape)
            self.target = self.target.reshape(1, *self.target.shape)
            self.reshaped = True

        self.sp_count = 0
        self.sp_distance = sp_distance

        self.cost_hist = cost_hist if cost_hist is not None else []
        self.Phi_min_hist = []
        self.Phi_avg_hist = []
        self.max_field_hist = []
        self.filename = filename
        self.verbosity = verbosity
        self.operation = operation
        self.iters = 0
        self.time_taken = None
        self.lam = lam
        self.alpha = alpha
        self.kappa = kappa
        self.noise_model = noise_model
        self.ensemble_size = ensemble_size
        self.cost_momentum = cost_momentum
        self.J_prev = None
        self.dJ_prev = None
        self.dJ_hist = []
        self.dJ_spec_hist = []
        self.X0 = X0
        self.matrix_exp_batches = matrix_exp_batches
        self.stop_fid_avg = stop_fid_avg
        self.stop_fid_min = stop_fid_min
        self.time_passed = 0
        self.N_rec = get_rec_min_N(
            rf=self.get_control_frequencies(), tN=self.tN, verbosity=0
        )
        self.simulation_steps = simulation_steps
        if self.simulation_steps:
            self.sim_steps = int(np.ceil(self.N_rec / self.N))
        else:
            self.sim_steps = 1
        self.print_setup_info()
        self.status = "UC"
        self.dynamic_opt_plot = dynamic_opt_plot
        if self.dynamic_opt_plot:
            plt.ion()
            plt.switch_backend("TkAgg")
            self.setup_optimization_tracking_plots()
            # self.setup_Bx_By_tracking_plots()

    def setup_optimization_tracking_plots(self):
        fig, ax = plt.subplots(1, 3)
        fig.suptitle("Optimisation Progress")

        ax[0].set_ylabel("Cost")
        ax[1].set_ylabel("Fidelity")
        ax[2].set_ylabel("Max field (mT)")
        fig.set_size_inches(18, 4)
        fig.tight_layout()
        ax_input = defaultdict(lambda: {})
        ax_input["ylim"] = {0: [0, 1]}
        ax_input["ax"] = {0: ax[0], 1: ax[1], 2: ax[1], 3: ax[2]}
        ax_input["color"] = {0: "black", 1: "blue", 2: "red", 3: "orange"}
        ax_input["legend_label"] = {1: "Avg fidelity", 2: "Min Fidelity"}
        self.dynamic_cost_plot = DynamicOptimizationPlot(n_plots=4, ax_input=ax_input)

    def setup_Bx_By_tracking_plots(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(9, 4)
        fig.suptitle("Optimized fields")
        ax.set_ylabel("B-field (mT)")
        ax.set_xlabel("Time (ns)")
        ax_input = defaultdict(lambda: {})
        ax_input["ax"] = {0: ax, 1: ax}
        ax_input["color"] = {0: color_cycle[0], 1: color_cycle[1]}
        ax_input["legend_label"] = {0: "$B_x$", 2: "$B_y$"}
        self.dynamic_Bx_By_plot = DynamicOptimizationPlot(n_plots=2, ax_input=ax_input)

    def get_default_targets(self):
        pass

    def get_target(self, target):
        target = target if target is not None else self.get_default_targets()
        return target

    def get_H0(self):
        pass

    def spectator_H0(self):
        pass

    def initialise_spectators(self):
        pass

    def get_T(self):
        return linspace(0, self.tN, self.N)

    def get_nq_spec(self, target_spec):
        return get_nq_from_dim(target_spec.shape[-1])

    def initialise_control_fields(self):
        """
        Initialises arrays omega and phi describing the frequency and phase of
        control fields modulated by control vector, 'u'. Also initialises u if
        required.
        """
        if self.rf is None:
            # self.rf = self.get_control_frequencies().sort().values
            self.rf = self.get_control_frequencies()
        else:
            self.rf = real(self.rf).sort().values
        self.omega, self.phi = config_90deg_phase_fields(self.rf)
        self.m = len(self.omega)
        self.x_cf, self.y_cf = get_control_fields(self.omega, self.phi, self.tN, self.N)
        if self.u is None:
            self.u = self.init_u()

    def print_setup_info(self):
        if self.verbosity == -1:
            return
        print("====================================================")
        print(f"{self.operation} GRAPE")
        print("====================================================")
        print(f"(Verbosity level = {self.verbosity})\n")
        print(f"Number of systems: nS = {self.nS}")
        print(f"Pulse duration: tN = {self.tN/unit.ns} ns")
        print(
            f"Number of timesteps N = {self.N}, recommended N is {self.N_rec}", end=""
        )
        print(f", sim steps = {self.sim_steps}") if self.simulation_steps else print("")
        print(f"Noise model: {self.noise_model}")
        if self.noise_model is not None or self.verbosity > 0:
            print(f"Ensemble size for noise simulation = {self.ensemble_size}")
            print(f"Cost momentum for dealing with noise = {self.cost_momentum}")
        print(f"Target Unitary:")
        if len(self.target.shape) >= 2:
            print(self.target[0])
        else:
            print(self.target)
        print(
            f"Found {len(self.rf)} resonant frequencies: rf_max = {pt.max(self.rf)/unit.MHz} MHz, rf_min = {pt.min(self.rf)/unit.MHz} MHz"
        )

        if self.simulate_spectators:
            if self.verbosity >= 1:
                print(f"Number of spectator systems: {len(self.spectator_H0())}")
                print(f"Spectator target(s):")
                if len(self.target_spec.shape) == 3:
                    print_rank2_tensor(self.target_spec[0])
                else:
                    print_rank2_tensor(self.target_spec)

        print(f"Amplitude penalisation: lam = {self.lam:.1e}")
        if self.verbosity >= 1:
            print(f"Variation penalisation: alpha = {self.alpha:.1e}")
        print(f"Cost gradient modulator: kappa = {self.kappa:.1e}")
        if self.verbosity > 2:
            print("{", end=" ")
            for freq in self.rf:
                print(f"{freq/unit.Mrps} Mrad/s,", end=" ")
            print("}")

    def get_control_frequencies(self, device=default_device):
        """
        Gets all unique control frequencies to be used for control fields.
        These are taken as the resonant frequencies corresponding to all two-
        qubit systems, as well as for single qubit spectators if required.

        Arg:
            device (pt.device): The device onto which
        """
        rf = get_multi_system_resonant_frequencies(self.H0, device=device)
        if self.simulate_spectators:
            rf_spectators = get_multi_system_resonant_frequencies(
                self.spectator_H0(), device=device
            )
            rf = pt.cat((rf, rf_spectators))
        return rf

    def u_mat(self, device=default_device):
        return uToMatrix(self.u, self.m).to(device)

    def init_u(self, device="cpu"):
        """
        Generates u0. Less important freuencies are placed in the second half of u0, and can be experimentally initialised to lower values.
        Initialised onto the cpu by default as it will normally be passed to scipy minimize.
        """
        # u0_max = 1 / (gamma_e * self.tN * unit.T)
        # u0_k = pt.cat(
        #     (
        #         pt.linspace(0, u0_max, self.N // 2, dtype=cplx_dtype, device=device),
        #         linspace(
        #             u0_max, 0, self.N - self.N // 2, dtype=cplx_dtype, device=device
        #         ),
        #     )
        # )
        # u0 = pt.einsum(
        #     "k,j->kj", pt.ones(self.m, dtype=cplx_dtype, device=device), u0_k
        # )
        u0 = (
            1
            / (gamma_e * self.tN)
            * pt.ones(self.m, self.N, dtype=cplx_dtype, device=device)
            / unit.T
        )
        return uToVector(u0 / 10)

    def time_evolution(self):
        """
        Calculates the time-evolution operators corresponding to each of the N timesteps,
        and returns as an array, U.

        Relies on self.Hw already being saved
        """
        dim = self.H0.shape[-1]
        nS = len(self.H0)
        N = self.N

        # add axes
        Hw = pt.einsum(
            "s,kjab->skjab",
            pt.ones(nS, dtype=cplx_dtype, device=default_device),
            self.Hw,
        )
        H0 = pt.einsum(
            "j,sab->sjab", pt.ones(N, dtype=cplx_dtype, device=default_device), self.H0
        )
        H_control = pt.einsum("kj,skjab->sjab", self.u_mat(), Hw)
        self.Hc = H_control

        if self.interaction_picture:
            H = H_control
        else:
            H = H_control + H0

        if self.sim_steps == 1:
            U = matrix_exp_array(-1j * H * self.dt)
            return U

        else:
            # scrap this?
            U = pt.matrix_exp(
                pt.zeros(
                    N,
                    2**self.nq,
                    2**self.nq,
                    dtype=cplx_dtype,
                    device=default_device,
                )
            )
            U_s = pt.matrix_exp(-1j * H * ((self.dt / self.sim_steps) / hbar))
            for i in range(self.sim_steps):
                U = pt.matmul(U_s, U)
            return U

    def update_H0(self):
        """Apply changes to H0, such as variations in background noise."""
        if self.noise_model == NoiseModels.dephasing:
            self.apply_delta_dephasing_noise()

    def apply_delta_dephasing_noise(self):
        sigma = 0.0001 * unit.T
        Bz_noise = self.Bz + rand.normal() * sigma
        self.H0 = self.get_H0(Bz=Bz_noise)

    @staticmethod
    def get_X_and_P(U, target, X0=None, device=default_device):
        nS = len(U)
        N = len(U[0])
        dim = U[0].shape[1]  # forward propagated time evolution operator
        nq = get_nq_from_dim(dim)
        if X0 is None:
            X0 = gate.get_Id(nq)
        X = pt.zeros((nS, N, *X0.shape), dtype=cplx_dtype, device=device)
        X[:, 0, :, :] = pt.einsum("sab,bc->sac", U[:, 0], X0)
        P = pt.zeros((nS, N, *X0.shape), dtype=cplx_dtype, device=device)
        P[:, N - 1, :, :] = target  # backwards propagated target

        for j in range(1, N):
            X[:, j, :, :] = pt.matmul(U[:, j, :, :], X[:, j - 1, :, :])
            P[:, -1 - j, :, :] = pt.matmul(dagger(U[:, -j, :, :]), P[:, -j, :, :])

        return X, P

    def propagate(self, device=default_device):
        """
        Determines total time evolution operators to evolve system from t=0 to time t(j) and stores in array P.
        Backward propagates target evolution operators and stores in X.

        This function is sufficiently general to work on all GRAPE implementations.
        """

        self.update_H0()
        self.U = self.time_evolution()
        self.X, self.P = self.get_X_and_P(self.U, self.target, self.X0)
        return self.X, self.P

    def max_field(self):
        return get_max_field(*self.get_Bx_By()).item()

    def average_field(self):
        Bx, By = self.get_Bx_By()
        return pt.mean(pt.sqrt(Bx**2 + By**2)).item()

    def get_opt_state(self):
        return f"nS={self.nS}, tN={self.tN/unit.ns:.0f}ns N={self.N}, λ={self.lam:.1e}, κ={self.kappa:.1e}, time: {self.time_passed:.1f}s, calls: {self.calls_to_cost_fn}, Avg fidelity = {pt.mean(self.Phi)*100:.1f}%, Min fidelity = {minreal(self.Phi)*100:.1f}%, Max field = {self.max_field()/unit.mT:.2f} mT, status={self.status}"

    def callback(self, xk):
        """
        Callback function for optimization of cost function.
        """
        update_period = 5
        self.iters += 1
        self.cost_hist.append(self.J)
        self.Phi_avg_hist.append(pt.mean(self.Phi).item())
        self.Phi_min_hist.append(minreal(self.Phi).item())
        self.max_field_hist.append((self.max_field() / unit.mT).item())
        if self.dynamic_opt_plot:
            if self.iters % update_period == 0:
                self.dynamic_cost_plot.update(
                    [
                        self.cost_hist,
                        self.Phi_avg_hist,
                        self.Phi_min_hist,
                        self.max_field_hist,
                    ]
                )
                # Bx, By = self.get_Bx_By()
                # T = self.get_T()
                # self.dynamic_Bx_By_plot.update(
                #     [self.u_mat()[0, :], self.u_mat()[1, :]], [T / unit.ns, T / unit.ns]
                # )

        self.time_passed = time.time() - self.start_time
        if self.verbosity > -2 and not pt.cuda.is_available():
            # Assume no gpu => running on personal computer. Print progress updates.
            print(self.get_opt_state(), end="\r")
        if self.max_time:
            self.check_time(xk)

        if self.stop_fid_avg < real(pt.mean(self.Phi)) and self.stop_fid_min < minreal(
            self.Phi
        ):
            self.u_opt_terminated = xk
            if self.verbosity > -1:
                print(
                    f"Avg and min stopping fidelities {self.stop_fid_avg*100:.4f}%, {self.stop_fid_min*100:.4f}% have been reached. Terminating optimization."
                )
            raise StopFidelityException

    def check_time(self, xk):
        """
        Checks how long optimisation has been running for on each call to the
        cost function. Terminates optimisation when max time is exceeded, and
        saves current grape object state when a save point is reached.

        Args:
            xk (ndarray[np.float64]): Partially optimised control vector, u.

        Raises:
            TimeoutError:   If the optimisation has run for longer than the
                            max_time set on grape object instantiation.
        """
        if self.time_passed > (self.sp_count + 1) * self.sp_distance:
            self.sp_count += 1
            print(
                f"Save point {self.sp_count} reached, time passed = {self.time_passed}."
            )
            self.save()
        if self.time_passed > self.max_time:
            self.u_opt_terminated = xk
            if self.verbosity > -1:
                print(
                    f"Max time of {self.max_time} has been reached. Optimisation terminated."
                )
            raise TimeoutError

    def spectator_fidelity(self):
        """
        Returns fidelity of an uncoupled qubit with the identity. Qubits which are not coupled should be subject to evolution
        equal to the identity under the influence of the applied pulse.
        """
        raise NotImplementedError

    def fidelity(self):
        """
        Determines the time evolution resulting from control field amplitudes in 'u', and
        calculates the corresponding fidelity Phi=<Uf,Ut> of the final time evolution operator
        Uf with the target operator Ut, and its gradient with respect to 'u'.

        Inputs:

            u: vector containing learning parameters, which are magnetic field amplitudes
                for each control field at each timestep.

            m: number of control fields

            target: target unitary for time evolution operator

            Hw: function of time which returns vector of control Hamiltonian values
                at each time.

            H0: Unperturbed Hamiltonian due to static background z-field

            tN: duration of gate

        """

        # determine time evolution resulting from control fields
        nq = self.nq
        tN = self.tN
        N = self.N
        nS = self.nS
        H0 = self.H0
        Hw = self.Hw
        d = 2**nq

        # add axes
        Hw = pt.einsum(
            "s,kjab->skjab", pt.ones(nS, dtype=cplx_dtype, device=default_device), Hw
        )
        H0 = pt.einsum(
            "j,sab->sjab", pt.ones(N, dtype=cplx_dtype, device=default_device), H0
        )

        X, P = self.propagate()

        Uf = P[:, -1]
        Ut = X[:, -1]
        # fidelity of resulting unitary with target
        Phi = batch_IP(Ut, Uf)
        Phi = pt.real(Phi * pt.conj(Phi))

        # calculate grad of fidelity

        XP_IP = batch_IP(X, P)
        # PHX_IP = batch_IP(P,1j*dt*pt.matmul(Hw,X))
        HwX = pt.einsum("skjab,sjbc->skjac", Hw, X)
        dt = self.dt
        PHX_IP = batch_trace(pt.einsum("sjab,skjbc->skjac", dagger(P), 1j * HwX)) / d

        dPhi = -2 * pt.real(pt.einsum("skj,sj->skj", PHX_IP, XP_IP)) / hbar
        self.Phi = Phi
        return Phi, pt.sum(dPhi, 0) / self.nS

    def fluc_cost(self):
        """
        Cost function term which penalises fluctuations in u between adjacent timesteps.
        """
        u = self.u_mat()
        u_aug = pt.cat((u[:, 0:1], u, u[:, self.N - 1 : self.N]), dim=1)
        J = self.alpha * np.real(pt.sum((u[:, 1:] - u[:, : self.N - 1]) ** 2).item())
        dJ = pt.real(
            uToVector(2 * self.alpha * (2 * u - u_aug[:, 2:] - u_aug[:, 0 : self.N]))
        )
        return J, dJ

    def reg_cost(self):
        """
        Regularisation cost function term which penalises large values of u.
        """
        J = self.lam / (2 * len(self.u)) * np.real(pt.dot(self.u, self.u).item())
        dJ = self.lam / len(self.u) * pt.real(self.u)
        return J, dJ

    def softmax(self):
        """
        Determines softmax function for u summed over k at   each timestep j
        """
        exp_u = real(pt.exp(self.u_mat()))
        J = 0
        dJ = -self.lam * exp_u / pt.sum(exp_u, 0)
        return J, uToVector(dJ)

    def lock_endpoints(self, u_mat, u0_L, uf_L):
        """
        Cost function terms which locks the endpoints of each u_k by applying a large penalty when endpoints
        deviate from specified positions.

        Inputs:
            u_mat: (k,N) tensor - matrix form of u
            u0_L: k element vector representing initial values at which to lock u.
            uf_L: k element vector representing final values at which to lock u.
        """
        L = 1e4  # Any number big enough to enforce lock

        J = (
            0.5 * L * pt.norm(u_mat[:, 0] - u0_L) ** 2
            + pt.norm(u_mat[:, -1] - uf_L) ** 2
        )

        dJ = pt.zeros(*u_mat.shape)
        dJ[:, 0] = L * (u_mat[:, 0] - u0_L)
        dJ[:, -1] = L * (u_mat[:, -1] - uf_L)

        return J, uToVector(dJ)

    def cost(self, u):
        """
        Cost function to be optimised.

        Takes control vector u of shape (m,N) describing weighting to be applied to field k\in[1,...,m] at time step j\in[1,...,N]

        """
        self.calls_to_cost_fn += 1
        J_sum = 0
        dJ_sum = pt.zeros_like(self.u, device=default_device)

        for i in range(self.ensemble_size):
            J_sample, dJ_sample = self.cost_single(u)
            J_sum += J_sample
            dJ_sum += dJ_sample

        J = J_sum / self.ensemble_size
        dJ = dJ_sum / self.ensemble_size

        if self.dJ_prev is not None:
            # J = (1-self.cost_momentum) * J + self.cost_momentum * self.J_prev
            dJ = (1 - self.cost_momentum) * dJ + self.cost_momentum * self.dJ_prev
        self.J_prev = J
        self.dJ_prev = dJ

        self.J = J.item()
        dJ = real(dJ).cpu().detach().numpy()

        return self.J, dJ

    def cost_from_fidelity(self, Phi_avg, dPhi_avg):
        J = 1 - Phi_avg
        dJ = -uToVector(dPhi_avg)

        # hackiness for better convergence I guess
        dJ = dJ * self.tN / self.nS
        return J, dJ / self.kappa

    def cost_single(self, u):
        """
        Cost function to be optimised.

        Takes control vector u of shape (m,N) describing weighting to be applied to field k\in[1,...,m] at time step j\in[1,...,N]

        """
        self.u = pt.tensor(u, dtype=cplx_dtype, device=default_device)

        t0 = time.time()
        Phi, dPhi_avg = self.fidelity()
        Phi_avg = pt.sum(Phi, 0) / self.nS
        global time_fid
        time_fid += time.time() - t0

        J, dJ = self.cost_from_fidelity(Phi_avg, dPhi_avg)
        self.dJ_hist.append(pt.norm(dJ))

        if self.simulate_spectators:
            J_spec, dJ_spec = self.spectator_cost()
            J = (self.nS * J + self.nS_spec * J_spec) / (self.nS + self.nS_spec)
            dJ += dJ_spec
            self.dJ_spec_hist.append(pt.norm(dJ_spec))

        # dJ = dJ*self.tN/self.N
        # dJ = dJ*300*unit.ns/nS

        # print(f"|dJ| = {pt.norm(dJ)}")
        if self.lam:
            J_reg, dJ_reg = self.reg_cost()
            J += J_reg
            dJ += dJ_reg

        if self.alpha:
            J_fluc, dJ_fluc = self.fluc_cost()
            J += J_fluc
            dJ += dJ_fluc

        # print(f"|dJ_spec| = {pt.norm(dJ_spec)}")

        return J, dJ

    def run(self, max_time=None):
        """
        Runs GRAPE optimisaiton. Passes the cost function self.cost and current
        control vector self.u to scipy.minimize. Conjugate gradient is the best
        method for GRAPE optimisations, and jac=True since self.cost calculates
        the gradient. Callback function monitors time spent on optimisation.

        Args:
            max_time (int): Maximum time in seconds to be spent on optimisation
                            before manual termination.
        """
        self.start_time = time.time()
        self.max_time = max_time
        callback = self.callback
        self.calls_to_cost_fn = 0
        if self.verbosity > -1:
            print("\n+++++  RUNNING OPTIMISATION  +++++")
            if self.max_time is None:
                print("No max time set")
            else:
                print(f"max optimisation time = {self.max_time}")
        try:
            opt = minimize(
                self.cost,
                self.u,
                method="CG",
                jac=True,
                callback=callback,
            )

            if self.verbosity > -1:
                print("Optimisation completed")
                print(f"nit = {opt.nfev}, nfev = {opt.nit}")
            self.u = pt.tensor(opt.x, device=default_device)
            self.status = "C"  # complete
        except TimeoutError:
            # max_time exceeded
            self.u = pt.tensor(
                self.u_opt_terminated, dtype=cplx_dtype, device=default_device
            )
            self.status = "TO"  # uncomplete - TimeOut
        except StopFidelityException:
            self.u = pt.tensor(
                self.u_opt_terminated, dtype=cplx_dtype, device=default_device
            )
            self.status = "SF"  # uncomplete - Stopping Fidelity

        self.time_taken = time.time() - self.start_time
        if self.verbosity >= 1:
            print(f"Time taken = {self.time_taken}")

    def save(self, fp=None):
        """
        Saves data to files. Calls functions to save system couplings, fields,
        control vector, and history of cost function over the course of the
        optimisation.
        """
        if fp is None:
            self.ID = self.get_next_ID()
            with open(ID_fn, "a") as f:
                f.write(f"{self.ID}\n")
            fp = f"fields/{self.get_field_filename()}"
        print(f"Saving GRAPE optimisation to: '{fp}'...", end=" ")

        fidelities = self.fidelity()[0]
        avgfid = sum(fidelities) / len(fidelities)
        minfid = min(fidelities).item()
        self.log_result(minfid, avgfid)
        self.save_system_data(fid=fidelities, fp=fp)
        self.save_field(fp)
        self.save_cost_hist(fp)
        print("done.")

    @abstractmethod
    def save_system_data(self, fid=None, fp=None):
        raise Exception(NotImplemented)

    def print_result(self, verbosity=None):
        if verbosity is None:
            verbosity = self.verbosity
        if verbosity == -1:
            return
        fidelities = self.fidelity()[0]
        avgfid = sum(fidelities) / len(fidelities)
        minfid = min(fidelities).item()
        print("\n+++++  PRINTING RESULTS:  +++++")
        if self.nS == 1:
            print(f"Fidelity = {avgfid}")
        else:
            print(f"Average fidelity = {avgfid:.5f}")
            print(f"Min fidelity = {minfid}")
            print(f"Call to cost function: {self.iters}")
        if self.simulate_spectators:
            spectator_fidelities = self.spectator_fidelity()[0]
            spectator_avg_fidelity = sum(spectator_fidelities) / len(
                spectator_fidelities
            )
            print(f"Average spectator fidelity = {spectator_avg_fidelity:.5f}")
        if verbosity >= 2:
            print(f"All fidelities = {fidelities}")
            if self.simulate_spectators:
                print(f"All spectator fidelities = {spectator_fidelities}")
        X_field, Y_field = self.sum_XY_fields()
        max_field = get_max_field(X_field, Y_field)

        print(f"Max field = {max_field*1e3:.2f} mT")
        if verbosity >= 3:
            print(f"GPUs requested = {ngpus}")
            global time_grad, time_exp, time_prop, time_fid
            print(
                f"texp={time_exp}, tprop={time_prop}, tgrad = {time_grad}, tfid={time_fid}"
            )

    def sum_XY_fields(self, device="cpu"):
        """
        Sum contributions of all control fields along x and y axes.

        X_field and Y_field are in units of Tesla, since control vector u is in
        units of Tesla, and control fields have magnitude 1. X_field and Y_field
        should be multiplied by unit.T to get Bx, By, which are unitless in atomic
        units.
        """
        x_cf_unit, y_cf_unit = get_unit_CFs(
            self.omega, self.phi, self.tN, self.N, device=device
        )
        X_field2 = pt.einsum("kj,kj->j", self.u_mat(device=device), x_cf_unit)
        Y_field2 = pt.einsum("kj,kj->j", self.u_mat(device=device), y_cf_unit)
        return X_field2, Y_field2

    def get_Bx_By(self, device="cpu"):
        X_field, Y_field = self.sum_XY_fields(device=device)
        return X_field * unit.T, Y_field * unit.T

    def get_fields(self):
        X_field, Y_field = self.sum_XY_fields()
        Bx = X_field * unit.T
        By = Y_field * unit.T
        return real(Bx), real(By)

    ################################################################################################################
    ################        VISUALISATION        ###################################################################
    ################################################################################################################

    def plot_fidelity(self, ax=None, all_fids=True, legend=True):
        if ax is None:
            ax = plt.subplot()

        if all_fids:
            plot_fidelity(
                fidelity_progress(self.X, self.target), ax=ax, tN=self.tN, legend=legend
            )
        else:
            plot_avg_min_fids(ax, self.X, self.target, self.tN)

    def plot_field_and_evolution(
        self, fp=None, psi0=pt.tensor([1, 0, 1, 0], dtype=cplx_dtype) / np.sqrt(2)
    ):
        """
        Nicely presented plots for thesis
        """
        fig, ax = plt.subplots(2, 2)
        self.plot_XY_fields(ax[0, 0])
        self.plot_psi_with_phase(ax[:, 1], psi0=psi0)
        self.plot_fidelity([1, 0])
        return fig, ax

    def plot_psi_with_phase(
        self,
        ax,
        psi0=pt.tensor([2, 1, 2, 1], dtype=cplx_dtype) / np.sqrt(10),
        amp_legend_loc="best",
        phase_legend_loc="best",
        legend_cols=2,
    ):
        psi = self.X @ psi0
        plot_psi(
            psi[0],
            tN=self.tN,
            ax=ax[0],
            label_getter=alpha_sq_label_getter,
            legend_loc=phase_legend_loc,
            legend_ncols=legend_cols,
        )
        plot_phases(psi[0], tN=self.tN, ax=ax[1], legend_loc=phase_legend_loc)

    def plot_field_and_fidelity(self, fp=None, ax=None, fid_legend=True, all_fids=True):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        else:
            fig = ax[0].get_figure()
        self.plot_XY_fields(ax[0])
        self.plot_fidelity(ax[1], all_fids=all_fids, legend=fid_legend)
        fig.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig.tight_layout()
        if fp is not None:
            fig.savefig(fp)

    def plot_control_fields(self, ax):
        T = np.linspace(0, self.tN / unit.ns, self.N)
        x_cf, y_cf = get_unit_CFs(self.omega, self.phi, self.tN, self.N)
        x_cf = x_cf.cpu().numpy()
        y_cf = y_cf.cpu().numpy()

        axt = ax.twinx()

        k = 1
        axt.plot(
            T,
            x_cf[k],
            linestyle="--",
            color=color_cycle[0],
            label=f"$B_{{{str(k)},x}}$ (mT)",
        )
        axt.plot(
            T,
            y_cf[k],
            linestyle="--",
            color=color_cycle[1],
            linewidth=2,
            label=f"$B_{{{str(k)},y}}$ (mT)",
        )

        k = 0
        ax.plot(T, x_cf[k], color=color_cycle[0], label=f"$B_{{{str(k)},x}}$ (mT)")
        ax.plot(
            T,
            y_cf[k],
            color=color_cycle[1],
            linewidth=2,
            label=f"$B_{{{str(k)},y}}$ (mT)",
        )

        ax.set_xlabel("Time (ns)")
        ax.set_yticks([-1, 0, 1])
        axt.set_yticks([-1, 0, 1])
        ax.set_ylim([-4, 1.1])
        axt.set_ylim([-1.1, 4])
        axt.set_yticklabels(["–1", " 0", " 1"])
        ax.legend(loc="upper right")
        axt.legend(loc="lower right")

    def plot_dJ(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        ax.plot(self.dJ_hist, label="CNOT cost")
        ax.plot(self.dJ_spec_hist, label="spectator")

    def plot_result(self, show_plot=False, save_plot=False):
        u = self.u
        X = self.X
        omegas = self.omega
        m = self.m
        tN = self.tN
        rf = omegas[: len(omegas) // 2]
        N = int(len(u) / m)
        u_m = uToMatrix(u, m).cpu()
        T = np.linspace(0, tN, N)
        fig, ax = plt.subplots(2, 2)
        self.plot_cost_hist(ax[1, 1])

        self.plot_u(ax[0, 0])

        X_field, Y_field = self.sum_XY_fields()
        self.plot_XY_fields(ax[0, 1], X_field, Y_field)
        fids = fidelity_progress(X, self.target)
        plot_fidelity(fids=fids, ax=ax[1, 0], tN=tN, legend=True)
        ax[0, 0].set_xlabel("Time (ns)")

        if save_plot:
            fig.savefig(f"{dir}plots/{self.filename}")
        if show_plot:
            plt.show()

    def plot_u(self, ax, legend_loc=False, legend_ncol=4):
        u_mat = self.u_mat().cpu().numpy()
        t_axis = np.linspace(0, self.tN / unit.ns, self.N)
        w_np = self.omega.cpu().detach().numpy()
        for k in range(self.m):
            if k < self.m / 2:
                linestyle = "-"
                color = color_cycle[k % len(color_cycle)]
            else:
                linestyle = "--"
                color = color_cycle[(k - self.m // 2) % len(color_cycle)]

            ax.plot(
                t_axis,
                u_mat[k] * 1e3,
                label=f"$u_{str(k)}$",
                color=color,
                linestyle=linestyle,
            )
            if y_axis_labels:
                ax.set_ylabel("Field strength (mT)")

        ax.set_xlabel("Time (ns)")
        if legend_loc:
            ax.legend(loc=legend_loc, ncol=legend_ncol)

    def plot_XY_fields(
        self,
        ax=None,
        X_field=None,
        Y_field=None,
        legend_loc="best",
        twinx=None,
        xcol=color_cycle[0],
        ycol=color_cycle[1],
    ):
        if ax is None:
            ax = plt.subplot()
        if X_field is None:
            X_field, Y_field = self.sum_XY_fields()
        X_field = X_field.cpu().numpy()
        Y_field = Y_field.cpu().numpy()
        t_axis = np.linspace(0, self.tN / unit.ns, self.N)
        ax.plot(t_axis, X_field * 1e3, label="$B_x$ (mT)", color=xcol)
        if twinx is None:
            ax.plot(t_axis, Y_field * 1e3, label="$B_y$ (mT)", color=ycol)
        else:
            twinx.plot(t_axis, Y_field * 1e3, label="$B_y$ (mT)", color=ycol)
        ax.set_xlabel(time_axis_label)
        if y_axis_labels:
            ax.set_ylabel("Total applied field")
        if legend_loc:
            ax.legend(loc=legend_loc)

        ax.set_ylabel("$B_x$ (mT)", color=xcol)
        if twinx is not None:
            twinx.set_ylabel("$B_y$ (mT)", color=ycol)

    def plot_cost_hist(self, ax, ax_label=None, yscale="log"):
        ax.plot(self.cost_hist, label="cost")
        ax.set_xlabel("Iterations")
        if y_axis_labels:
            ax.set_ylabel("Cost")
        else:
            ax.legend()
        ax.axhline(0, color="orange", linestyle="--")
        if ax_label is not None:
            ax.set_title(ax_label, loc="left", fontdict={"fontsize": 20})
        ax.set_ylim([0, min(1.2, ax.get_ylim()[1])])

        iters = len(self.cost_hist)

        if yscale == "log":
            x = min(self.cost_hist)
            min_tick = 1
            while not int(x):
                x *= 10
                min_tick /= 10
            ax.set_yscale("log")
            ax.axis([0, iters, min_tick, 1])
        return ax

    def save_field(self, fp=None):
        """
        Saves total control field. Assumes each omega has x field and y field part, so may need to send only half of
        omega tensor to this function while pi/2 offset is being handled manually.
        """
        if fp is None:
            fp = f"{dir}fields/{self.filename}"
        pt.save(self.u, fp)
        X_field, Y_field = self.sum_XY_fields()
        T = linspace(0, self.tN, self.N, dtype=real_dtype, device="cpu") / unit.ns
        fields = pt.stack((X_field, Y_field, T))
        pt.save(fields, f"{fp}_XY")

    def save_cost_hist(self, fp=None):
        pt.save(np.array(self.cost_hist), f"{fp}_cost_hist")

    # LOGGING

    def log_result(self, minfid, avgfid):
        """
        Logs key details of result:
            field filename, minfid, J (MHz), A (MHz), avgfid, fidelities, time date, alpha, kappa, nS, nq, tN (ns), N, ID
        """

        field_filename = self.get_field_filename()
        # fids_formatted = [round(elem,4) for elem in fidelities.tolist()]
        # A_formatted = [round(elem,2) for elem in (pt.real(A)[0]/Mhz).tolist()]
        # J_formatted = [round(elem,2) for elem in (pt.real(J).flatten()/Mhz).tolist()]
        now = datetime.now().strftime("%H:%M:%S %d-%m-%y")
        with open(log_fn, "a") as f:
            f.write(
                "{},{:.4f},{:.4f},{},{},{}\n".format(
                    field_filename, avgfid, minfid, now, self.status, self.time_taken
                )
            )

    ################################################################################################################
    ################        Save Data        #######################################################################
    ################################################################################################################
    def save_system_data(self, fid=None, fp=None):
        """
        Saves system data to a file.

        status carries information about how the optimisation exited, and can take 3 possible values:
            C = complete
            UC = uncomplete - ie reached maximum time and terminated minimize
            T = job terminated by gadi, which means files are not saved (except savepoints)

        """
        J = real(self.J) / unit.MHz if self.J is not None else None
        A = real(self.A) / unit.MHz
        tN = self.tN / unit.ns
        if type(J) == pt.Tensor:
            J = J.tolist()
        if fp is None:
            fp = f"{dir}fields/{self.filename}"
        with open(f"{fp}.txt", "w") as f:
            f.write("J = " + str(J) + "\n")
            f.write("A = " + str((A).tolist()) + "\n")
            f.write("tN = " + str(tN) + "\n")
            f.write("N = " + str(self.N) + "\n")
            f.write("target = " + str(self.target.tolist()) + "\n")
            f.write("Completion status = " + self.status + "\n")
            if fid is not None:
                f.write("fidelity = " + str(fid.tolist()) + "\n")

    @staticmethod
    def get_next_ID():
        with open(ID_fn, "r") as f:
            prev_ID = f.readlines()[-1].split(",")[-1]
        if prev_ID[-1] == "\n":
            prev_ID = prev_ID[:-1]
        ID = prev_ID[0] + str(int(prev_ID[1:]) + 1)
        return ID

    def get_field_filename(self):
        filename = "{}_{}S_{}q_{}ns_{}step".format(
            self.ID, self.nS, self.nq, int(round(self.tN / unit.ns, 0)), self.N
        )
        return filename


class ESR_system(object):
    def __init__(self, J, A, Bz):
        self.J = J
        self.A = A
        self.Bz = Bz


class GrapeESR(Grape):
    """
    Class for running GRAPE optimisations on electron spin systems (ESR: Electron
    Spin Resonance).
    """

    def __init__(
        self, tN, N, J, A, Bz=np.float64(0), J_modulated=False, A_spec=None, **kwargs
    ):
        # save data before running super() initialisation function
        self.J = J
        self.A = A
        self.nS, self.nq = self.get_nS_nq()
        self.Bz = Bz
        self.J_modulated = J_modulated
        self.n_J = self.count_exchange_values()
        self.A_spec = A_spec

        super().__init__(tN, N, **kwargs)

        # perform calculations last
        self.rf = self.get_control_frequencies() if self.rf is None else self.rf

    def get_default_targets(self):
        return CNOT_targets(self.nS, self.nq)

    def initialise_spectators(self):
        if self.A_spec == None:
            self.A_spec = self.get_spectator_A()
        if self.simulate_spectators:
            self.nS_spec = len(self.A_spec)

    def count_exchange_values(self):
        if len(self.J.shape) == 0:
            return 1
        else:
            return self.J.shape[-1]

    def copy(self):
        grape_copy = self.__class__(
            self.J,
            self.A,
            self.tN,
            self.N,
            self.Bz,
            self.target,
            self.rf,
            self.u,
            self.cost_hist,
            self.max_time,
            self.alpha,
            self.lam,
            operation="Copying",
        )
        grape_copy.time_taken = self.time_taken
        grape_copy.propagate()
        return grape_copy

    def get_nS_nq(self):
        return get_nS_nq_from_A(self.A)

    def print_setup_info(self):
        if self.verbosity == -1:
            return
        super().print_setup_info()
        if self.nq == 2:
            system_type = "Electron spin qubits coupled via direct exchange"
        elif self.nq == 3:
            system_type = "Electron spin qubits coupled via intermediate coupler qubit"

        print(f"System type: {system_type}")
        print(f"Bz = {self.Bz/unit.T} T")
        print(
            f"{len(self.get_spectator_A())} hyperfine values, with A_min = {minreal(self.A)/unit.MHz:.2f} MHz, A_max = {maxreal(self.A)/unit.MHz:.2f} MHz"
        )
        print(f"{self.n_J} exchange value", end="")
        if self.n_J == 1:
            print(f": J = {(real(self.J/unit.MHz)).item():.2f} MHz")
        else:
            print(
                f"s, with J_min = {minreal(self.J)/unit.MHz:.2f} MHz, J_max = {maxreal(self.J)/unit.MHz:.2f} MHz"
            )

        if self.verbosity >= 2:
            print(f"Hyperfine: A (MHz) = {real(self.A/unit.MHz)}")
            print(f"Exchange: J (MHz) = {real(self.J/unit.MHz)}")

    def get_H0(self, Bz=0, J=None, device=default_device):
        """
        Free hamiltonian of each system.

        self.A: (nS,nq), self.J: (nS,) for 2 qubit or (nS,2) for 3 qubits
        """
        if J is None:
            J = self.J
        H0 = get_H0(self.A, J, Bz, device=device)
        if self.nS == 1:
            return H0.reshape(1, *H0.shape)
        return H0

    def modulate_J(self):
        # define proportion of exchange values to start and end on
        device = self.J.device
        J0_prop = 0

        # define proportion of time to spend on of rise and fall
        rise_prop = 0.1
        fall_prop = rise_prop
        modulator = rise_ones_fall(J0_prop, self.N, rise_prop, fall_prop)
        if self.nS == 1:
            J_modulated = pt.einsum("j,...->j...", modulator, self.J)
        else:
            J_modulated = pt.einsum("j,s...->sj...", modulator, self.J)

        return J_modulated

    def get_control_frequencies(self, device=default_device):
        if self.J_modulated:
            return get_multi_system_resonant_frequencies(
                self.H0[:, self.N // 2], device=device
            )
        else:
            return super().get_control_frequencies(device=device)

    def get_H0_J_modulated(self, Bz=0, device=default_device):
        """
        Free hamiltonian of each system.

        self.A: (nS,nq, N) and self.J: (nS, N), describe hyperfine and exchange couplings, which are allowed to vary over the N timesteps.
        """
        # if self.nS==1:
        #     A = self.modulate_A().reshape(1,*self.A.shape)
        #     J = self.J.reshape(1,*self.J.shape)
        # else:
        J = self.modulate_J()

        nS, nq = self.get_nS_nq()
        dim = 2**nq

        # Zeeman splitting term is generally rotated out, which is encoded by setting Bz=0
        HZ = 0.5 * gamma_e * Bz * gate.get_Zn(nq)

        # this line only supports nq=2
        if nS == 1:
            H_A = pt.einsum(
                "j,q,qab->jab",
                pt.ones(self.N, dtype=self.A.dtype, device=device),
                self.A.to(device),
                gate.get_PZ_vec(nq).to(device),
            )
            if self.nq == 2:
                H0 = (
                    H_A
                    + pt.einsum(
                        "j,ab->jab",
                        J.to(device),
                        gate.get_coupling_matrices(nq).to(device),
                    )
                    + pt.einsum(
                        "j,ab->jab", pt.ones(self.N, device=device), HZ
                    ).reshape(1, self.N, dim, dim)
                )

            else:
                H0 = (
                    H_A
                    + pt.einsum(
                        "jp,pab->jab",
                        J.to(device),
                        gate.get_coupling_matrices(nq).to(device),
                    )
                    + pt.einsum(
                        "j,ab->jab", pt.ones(self.N, device=device), HZ
                    ).reshape(1, self.N, dim, dim)
                )

        else:
            HZ = pt.einsum("sj,ab->sjab", pt.ones(nS, self.N, device=device), HZ)
            HA = pt.einsum(
                "j,sq,qab->sjab",
                pt.ones(self.N, dtype=self.A.dtype, device=device),
                self.A.to(device),
                gate.get_PZ_vec(nq).to(device),
            )
            if self.nq == 2:
                HJ = pt.einsum(
                    "sj,sab->jab",
                    J.to(device),
                    gate.get_coupling_matrices(nq).to(device),
                )
            else:
                HJ = pt.einsum(
                    "sjp,pab->sjab",
                    J.to(device),
                    gate.get_coupling_matrices(nq).to(device),
                )
            H0 = HA + HJ + HZ

        return H0.to(device)

    def get_Hw(self):
        """
        Gets Hw. Not used for actual optimisation, but sometimes handy for testing and stuff.
        """
        ox = gate.get_Xn(self.nq)
        oy = gate.get_Yn(self.nq)
        Hw = pt.einsum("kj,ab->kjab", self.x_cf, ox) + pt.einsum(
            "kj,ab->kjab", self.y_cf, oy
        )
        return Hw

    def time_evolution_1q(self, device=default_device):
        dim = 2
        m, N = self.x_cf.shape
        dt = self.dt
        u_mat = self.u_mat()
        x_sum = pt.einsum("kj,kj->j", u_mat, self.x_cf)
        y_sum = pt.einsum("kj,kj->j", u_mat, self.y_cf)

        H0 = self.spectator_H0()
        Hw = get_pulse_hamiltonian()

    def get_U(
        self, u_mat, x_cf, y_cf, H0, nq, dt, matrix_exp_batches=1, device=default_device
    ):
        nS = len(H0)
        m, N = u_mat.shape
        dim = 2**nq
        sig_xn, sig_yn = self.get_control_field_operators(nq, device=device)
        u_mat = u_mat
        x_sum = pt.einsum("kj,kj->j", u_mat, x_cf)
        y_sum = pt.einsum("kj,kj->j", u_mat, y_cf)
        if len(H0.shape) == 4:
            H0_t_ax = H0
        else:
            H0_t_ax = pt.einsum("j,sab->sjab", pt.ones(N, device=device), H0.to(device))
        H = H0_t_ax + pt.einsum(
            "s,jab->sjab",
            pt.ones(nS, device=device),
            pt.einsum("j,ab->jab", x_sum, sig_xn)
            + pt.einsum("j,ab->jab", y_sum, sig_yn),
        )
        if matrix_exp_batches == 2:
            H = pt.reshape(H, (nS * N, dim, dim))
            half = len(H) // 2
            U = pt.cat(
                (pt.matrix_exp(-1j * H[:half] * dt), pt.matrix_exp(-1j * H[half:] * dt))
            )
        else:
            # H = pt.reshape(H, (nS * N, dim, dim))
            U = matrix_exp_array(-1j * H * dt)
        del H
        return U

    def time_evolution(self, device=default_device):
        """
        Determines U. Calculates Hw on the fly.
        """
        return self.get_U(
            self.u_mat(),
            self.x_cf,
            self.y_cf,
            self.H0,
            self.nq,
            self.dt,
            device=device,
        )

    def update_H0(self):
        if self.noise_model == NoiseModels.delta_correlated_exchange:
            self.apply_delta_noise_to_J()

    def apply_delta_noise_to_J(self):
        sigma = 0.1 * self.J
        J_noise = self.J + rand.normal() * sigma
        self.H0 = self.get_H0(J=J_noise)

    def get_spectator_A(self):
        return pt.unique(real(pt.flatten(self.A)))

    def spectator_H0(self):
        H0 = pt.einsum("s,ab->sab", 0.5 * gamma_e * self.Bz + self.A_spec, gate.Z)
        return H0

    def get_spec_propagators(self, device=default_device):
        H0_spec = self.spectator_H0()
        U_spec = self.get_U(
            self.u_mat(),
            self.x_cf,
            self.y_cf,
            H0_spec,
            self.nq_spec,
            self.dt,
            device=device,
        )
        self.X_spec, self.P_spec = self.get_X_and_P(
            U_spec, self.target_spec, self.X0_spec
        )
        return self.X_spec, self.P_spec

    def spectator_fidelity(self, device=default_device):
        """
        Returns fidelity of an uncoupled qubit with the identity. Qubits which are not coupled should be subject to evolution
        equal to the identity under the influence of the applied pulse.
        """
        self.get_spec_propagators()
        sig_xn, sig_yn = self.get_control_field_operators(self.nq_spec)
        Phi_spec, dPhi_spec = self.fidelity_from_X_and_P(
            self.X_spec,
            self.P_spec,
            self.x_cf,
            self.y_cf,
            sig_xn,
            sig_yn,
            device=device,
        )
        self.Phi_spec = Phi_spec
        return Phi_spec, dPhi_spec

    def spectator_cost(self):
        Phi_spec, dPhi_spec_avg = self.spectator_fidelity()
        Phi_spec_avg = pt.sum(Phi_spec, 0) / len(self.A)
        J_spec, dJ_spec = self.cost_from_fidelity(Phi_spec_avg, dPhi_spec_avg)
        return J_spec, dJ_spec / 1e2 / self.kappa

    def get_control_field_operators(self, nq, device=default_device):
        sig_xn = gate.get_Xn(nq, device)
        sig_yn = gate.get_Yn(nq, device)
        return sig_xn, sig_yn

    @staticmethod
    def fidelity_from_X_and_P(X, P, x_cf, y_cf, sig_xn, sig_yn, device=default_device):
        nS = len(X)
        dim = X.shape[-2]
        nq = get_nq_from_dim(dim)

        t0 = time.time()
        global time_exp
        time_exp += time.time() - t0

        t0 = time.time()
        global time_prop
        time_prop += time.time() - t0
        Ut = P[:, -1]
        Uf = X[:, -1]
        # fidelity of resulting unitary with target
        IP = batch_IP(Ut, Uf)
        Phi = pt.real(IP * pt.conj(IP))

        t0 = time.time()
        # calculate grad of fidelity
        XP_IP = batch_IP(X, P)

        ox_X = pt.einsum("...ab,...bc->...ac", sig_xn, X)
        oy_X = pt.einsum("...ab,...bc->...ac", sig_yn, X)
        PoxX_IP = batch_trace(pt.einsum("sjab,sjbc->sjac", dagger(P), 1j * ox_X)) / dim
        PoyX_IP = batch_trace(pt.einsum("sjab,sjbc->sjac", dagger(P), 1j * oy_X)) / dim
        del ox_X, oy_X

        Re_IP_x = -2 * pt.real(pt.einsum("sj,sj->sj", PoxX_IP, XP_IP))
        Re_IP_y = -2 * pt.real(pt.einsum("sj,sj->sj", PoyX_IP, XP_IP))

        # average over systems axis
        sum_IP_x = pt.sum(Re_IP_x, 0) / np.float64(nS)
        sum_IP_y = pt.sum(Re_IP_y, 0) / np.float64(nS)
        del Re_IP_x, Re_IP_y

        dPhi_x = pt.einsum("kj,j->kj", pt.real(x_cf), sum_IP_x)
        dPhi_y = pt.einsum("kj,j->kj", pt.real(y_cf), sum_IP_y)
        del sum_IP_x, sum_IP_y

        dPhi = dPhi_x + dPhi_y
        del dPhi_x, dPhi_y
        global time_grad
        time_grad += time.time() - t0

        return Phi, dPhi

    def fidelity(self, device=default_device):
        """
        Adapted grape fidelity function designed specifically for multiple systems with transverse field control Hamiltonians.
        """
        self.propagate(device=device)
        sig_xn, sig_yn = self.get_control_field_operators(self.nq)
        self.Phi, dPhi = self.fidelity_from_X_and_P(
            self.X, self.P, self.x_cf, self.y_cf, sig_xn, sig_yn
        )
        return self.Phi, dPhi

    # class FieldOptimiser(object):
    """
    Optimisation object which passes fidelity function to scipy optimiser, and handles termination of 
    optimisation once predefined time limit is reached.
    """


def process_u_file(filename, SP=None, save_SP=False):
    """
    Gets fidelities and plots from files. save_SP param can get set to True if a save point needs to be logged.
    """
    grape = load_system_data(filename)
    u, cost_hist = load_u(filename, SP=SP)
    grape.u = u
    grape.hist = cost_hist
    grape.plot_result()


def load_system_data(fp, Grape=GrapeESR, **kwargs):
    """
    Retrieves system data (exchange, hyperfine, pulse length, target, fidelity) from file.
    """
    print(f"Loading GRAPE from {fp}")
    with open(f"{fp}.txt", "r") as f:
        lines = f.readlines()
        J = (
            pt.tensor(
                ast.literal_eval(lines[0][4:-1]),
                dtype=cplx_dtype,
                device=default_device,
            )
            * unit.MHz
        )
        A = (
            pt.tensor(
                ast.literal_eval(lines[1][4:-1]),
                dtype=cplx_dtype,
                device=default_device,
            )
            * unit.MHz
        )
        tN = float(lines[2][4:-1]) * unit.ns
        N = int(lines[3][3:-1])
        target = pt.tensor(
            ast.literal_eval(lines[4][9:-1]), dtype=cplx_dtype, device=default_device
        )
        try:
            fid = ast.literal_eval(lines[6][11:-1])
        except:
            fid = None
    u0, cost_hist = load_u(fp)
    grape = Grape(
        tN=tN,
        N=N,
        J=J,
        A=A,
        Bz=0,
        u0=u0,
        cost_hist=cost_hist,
        target=target,
        operation="Loading ",
        **kwargs,
    )
    grape.time_taken = None
    return grape


################################################################################################################
################        Exceptions        ####################################################################
################################################################################################################
class StopFidelityException(Exception):
    pass


################################################################################################################
################        Hamiltonians        ####################################################################
################################################################################################################


def transform_H0(H, S):
    """
    Applies transformation S.T @ H0 @ S for each system, to transform H0 into frame whose basis vectors are the columns of S.
    H0: (nS,N,d,d)
    S: (nS,d,d)
    """
    HS = pt.einsum("sjab,sbc->sjac", H, S)
    H_trans = pt.einsum("sab,sjbc->sjac", dagger(S), HS)
    return H_trans


def transform_Hw(Hw, S):
    """
    Applies transformation S.T @ Hw @ S for each system, to transform Hw into frame whose basis vectors are the columns of S.
    Hw: (m,N,d,d)
    S: (nS,d,d)
    """
    if len(Hw.shape) == 4:
        HS = pt.einsum("kjab,sbc->skjac", Hw, S)
    else:
        HS = pt.einsum("skjab,sbc->skjac", Hw, S)
    H_trans = pt.einsum("sab,skjbc->skjac", dagger(S), HS)
    return H_trans


def evolve_Hw(Hw, U):
    """
    Evolves Hw by applying U @ H0 @ U_dagger to every control field at each timestep for every system.
    Hw: Tensor of shape (m,N,d,d), describing m control Hamiltonians having dimension d for nS systems at N timesteps.
    U: Tensor of shape (nS,N,d,d), ""
    """
    UH = pt.einsum("sjab,kjbc->skjac", U, Hw)
    return pt.einsum("skjab,sjbc->skjac", UH, dagger(U))


def make_Hw(omegas, nq, tN, N, phase=pt.zeros(1), coupled_XY=True):
    """
    Takes input array omegas containing m/2 frequencies.
    Returns an array containing m oscillating fields transverse fields,
    with one in the x direction and one in the y direction for each omega
    """
    mw = len(omegas)
    Xn = gate.get_Xn(nq)
    Yn = gate.get_Yn(nq)
    Zn = gate.get_Zn(nq)
    T = linspace(0, tN, N, device=default_device)

    wt_tensor = pt.kron(omegas, T).reshape((mw, N))
    X_field_ham = pt.einsum("ij,kl->ijkl", pt.cos(wt_tensor - phase), Xn)
    Y_field_ham = pt.einsum("ij,kl->ijkl", pt.sin(wt_tensor - phase), Yn)
    if coupled_XY:
        Hw = 0.5 * g_e * mu_B * 1 * unit.T * (X_field_ham + Y_field_ham)
    else:
        Hw = 0.5 * g_e * mu_B * 1 * unit.T * pt.cat((X_field_ham, Y_field_ham), 0)
    return Hw


def ignore_tensor(trans, d):
    nS = len(trans)
    ignore = pt.ones(len(trans), d, d)
    for s in range(nS):
        for i in range(len(trans[s])):
            ignore[s, int(trans[s][i][0]) - 1, int(trans[s][i][1]) - 1] = 0
            ignore[s, int(trans[s][i][1]) - 1, int(trans[s][i][0]) - 1] = 0
    return pt.cat((ignore, ignore))


def get_HA(A, device=default_device):
    nS, nq = get_nS_nq_from_A(A)
    d = 2**nq
    if nq == 3:
        HA = pt.einsum("sq,qab->sab", A.to(device), gate.get_PZ_vec(nq).to(device))
    elif nq == 2:
        HA = pt.einsum("sq,qab->sab", A.to(device), gate.get_PZ_vec(nq).to(device))
    return HA.to(device)


def get_HJ(J, nq, device=default_device):
    nS = len(J)
    d = 2**nq
    if nq == 3:
        HJ = pt.einsum(
            "sc,cab->sab", J.to(device), gate.get_coupling_matrices(nq).to(device)
        )
    elif nq == 2:
        HJ = pt.einsum(
            "s,ab->sab", J.to(device), gate.get_coupling_matrices(nq).to(device)
        )
    return HJ.to(device)


def get_S_matrix(J, A, device=default_device):
    nS, nq = get_nS_nq_from_A(A)
    d = 2**nq
    if nq != 2:
        raise Exception("Not implemented")
    S = pt.zeros(nS, d, d, dtype=cplx_dtype, device=device)
    for s in range(nS):
        dA = (A[s][0] - A[s][1]).item()
        d = pt.sqrt(
            8 * J[s] ** 2 + 2 * dA**2 - 2 * dA * pt.sqrt(dA**2 + 4 * J[s] ** 2)
        )
        alpha = 2 * J[s] / d
        beta = (-dA + pt.sqrt(dA**2 + 4 * J[s] ** 2)) / d
        S[s] = pt.tensor(
            [[1, 0, 0, 0], [0, alpha, -beta, 0], [0, beta, alpha, 0], [0, 0, 0, 1]]
        )
    return S


################################################################################################################
################        Resonant Frequencies        ############################################################
################################################################################################################


def getFreqs_broke(H0, Hw_shape, device=default_device):
    """
    Determines frequencies which should be used to excite transitions for system with free Hamiltonian H0.
    Useful for >2qubit systems where analytically determining frequencies becomes difficult. Probably won't
    end up being used as 3 qubit CNOTs will be performed as sequences of 2 qubit CNOTs.
    """
    eig = pt.linalg.eig(H0)
    evals = eig.eigenvalues
    S = eig.eigenvectors
    S = S.to(device)
    # S = pt.transpose(pt.stack((S[:,2],S[:,1],S[:,0],S[:,3])),0,1)
    # evals = pt.stack((evals[2],evals[1],evals[0],evals[3]))
    S_T = pt.transpose(S, 0, 1)
    d = len(evals)
    pairs = list(itertools.combinations(pt.linspace(0, d - 1, d, dtype=int), 2))
    # transform shape of control Hamiltonian to basis of energy eigenstates

    Hw_trans = matmul3(S_T, Hw_shape, S)
    Hw_nz = (pt.abs(Hw_trans) > 1e-9).to(int)
    freqs = []

    for i in range(len(H0)):
        for j in range(i + 1, len(H0)):
            if Hw_nz[i, j]:
                freqs.append((pt.real(evals[i] - evals[j])).item())

    # for i in range(len(pairs)):
    #     # The difference between energy levels i,j will be a resonant frequency if the control field Hamiltonian
    #     # has a non-zero (i,j) element.
    #     pair = pairs[i]
    #     idx1=pair[0].item()
    #     idx2 = pair[1].item()
    #     #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
    #     if Hw_nz[idx1,idx2]:
    #         freqs.append((pt.real(evals[pair[1]]-evals[pair[0]])).item())
    #     #if pt.real(Hw_trans[idx1][idx2]) >=1e-9:
    #     # if Hw_nz[idx2,idx1]:
    #     #     freqs.append((pt.real(evals[pair[0]]-evals[pair[1]])).item())
    freqs = pt.tensor(remove_duplicates(freqs), dtype=real_dtype, device=device)
    return freqs


def config_90deg_phase_fields(rf, device=default_device):
    """
    Takes array of control/resonant frequencies (rf). For each frequency, sets up a pair of circular fields rotating pi/2 out of phase.
    First half of omegas are assumed to be more important to desired transition.
    For the love of g*d make sure each omega has a 0 and a pi/2 phase
    """
    zero_phase = pt.zeros(len(rf), dtype=cplx_dtype, device=device)
    piontwo_phase = (
        pt.ones(len(rf), dtype=cplx_dtype, device=device)
        * np.float32(np.pi)
        / np.float32(2)
    )
    phase = pt.cat((zero_phase, piontwo_phase))
    omega = pt.cat((rf, rf))
    return omega, phase.reshape(len(phase), 1)


def get_2q_freqs(J, A, all_freqs=True, device=default_device):
    """
    Analytically determines resonant frequencies for a collection of systems.
    """
    dA = A[0][0] - A[0][1]

    # if True:
    #     return pt.tensor([-2*J[0]-pt.sqrt(dA**2+4*J[0]**2)])

    if all_freqs:
        w = pt.zeros(4 * len(J), device=device)
        for i in range(len(J)):
            w[4 * i + 2] = -2 * J[i] - pt.sqrt(dA**2 + 4 * J[i] ** 2)
            w[4 * i + 3] = -2 * J[i] + pt.sqrt(dA**2 + 4 * J[i] ** 2)
            w[4 * i] = 2 * J[i] - pt.sqrt(dA**2 + 4 * J[i] ** 2)
            w[4 * i + 1] = 2 * J[i] + pt.sqrt(dA**2 + 4 * J[i] ** 2)
    else:
        w = pt.zeros(2 * len(J), device=device)
        for i in range(len(J)):
            w[2 * i + 0] = -2 * J[i] - pt.sqrt(dA**2 + 4 * J[i] ** 2)
            w[2 * i + 1] = -2 * J[i] + pt.sqrt(dA**2 + 4 * J[i] ** 2)

    return w


class GrapeESR_IP(GrapeESR):
    def __init__(self, tN, N, J, A, **kwargs):
        super().__init__(tN, N, J, A, **kwargs)

    def initialise_IP_operators(self, device=default_device):
        sig_xn = gate.get_Xn(self.nq, device)
        sig_yn = gate.get_Yn(self.nq, device)
        H0T = pt.einsum("sab,j->sjab", self.get_H0(), self.get_T())
        U0 = matrix_exp_array(-1j * H0T)

        self.sig_xn_IP = pt.einsum("sjab,bc->sjac", dagger(U0), sig_xn)
        self.sig_yn_IP = pt.einsum("sjab,bc->sjac", dagger(U0), sig_yn)
        self.target = pt.einsum("sab,sbc->sac", dagger(U0[:, -1]), self.target)

    def get_control_field_operators(self, nq, device=default_device):
        if not hasattr(self, "sig_xn_IP"):
            self.initialise_IP_operators()
        return self.sig_xn_IP, self.sig_yn_IP

    def get_U(self, u_mat, x_cf, y_cf, H0, nq, dt, device=default_device):
        nS = len(H0)
        m, N = u_mat.shape
        dim = 2**nq
        sig_xn, sig_yn = self.get_control_field_operators(nq, device=device)
        u_mat = u_mat
        x_sum = pt.einsum("kj,kj->j", u_mat, x_cf)
        y_sum = pt.einsum("kj,kj->j", u_mat, y_cf)
        H = pt.einsum("j,sjab->sjab", x_sum, sig_xn) + pt.einsum(
            "j,sjab->sjab", y_sum, sig_yn
        )
        U = matrix_exp_array(-1j * H * dt)
        del H
        return U


class GrapeESR_AJ_Modulation(GrapeESR):
    def __init__(self, tN, N, J, A, Bz=0, **kwargs):
        self.rise_time = 10 * unit.ns
        self.E = get_smooth_E(tN, N).to(default_device)
        # self.E = get_simple_E(tN, N).to(default_device)
        super().__init__(J, A, tN, N, Bz=Bz, **kwargs)

    def print_setup_info(self):
        Grape.print_setup_info(self)
        system_type = "Electron spin qubits on 2P-1P donors coupled via direct exchange with linear activation and deactivation at start and end of pulse"

        print(f"System type: {system_type}")
        print(f"Bz = {self.Bz/unit.T} T")
        if self.nS == 1:
            print(f"Hyperfine: A = {(self.A/unit.MHz).tolist()} MHz")
            print(f"Exchange: J = {(self.J/unit.MHz).tolist()} MHz")
        else:
            if self.verbosity >= 2:
                print("Hyperfines:")
                for q in range(self.nS):
                    print("".ljust(10) + f"{(self.A[q]/unit.MHz).tolist()}")
                print("Exchange:")
                for q in range(self.nS):
                    print("".ljust(10) + f"{(self.J[q]/unit.MHz).tolist()}")
        print(
            f"Number of timesteps N = {self.N}, recommended N is {get_rec_min_N(rf=self.get_control_frequencies(), tN=self.tN, verbosity = self.verbosity)}"
        )

    def get_H0(self, Bz=0, device=default_device):
        """
        Free hamiltonian of each system.

        self.A: (nS,nq, N) and self.J: (nS, N), describe hyperfine and exchange couplings, which are allowed to vary over the N timesteps.
        """
        # if self.nS==1:
        #     A = self.modulate_A().reshape(1,*self.A.shape)
        #     J = self.J.reshape(1,*self.J.shape)
        # else:
        A = self.modulate_A_old()
        J = self.modulate_J_old()

        nS, nq = self.get_nS_nq()
        dim = 2**nq

        # Zeeman splitting term is generally rotated out, which is encoded by setting Bz=0
        HZ = 0.5 * gamma_e * Bz * gate.get_Zn(nq)

        # this line only supports nq=2
        if nS == 1:
            H0 = (
                pt.einsum("jq,qab->jab", A.to(device), gate.get_PZ_vec(nq).to(device))
                + pt.einsum(
                    "j,ab->jab", J.to(device), gate.get_coupling_matrices(nq).to(device)
                )
                + pt.einsum("j,ab->jab", pt.ones(self.N, device=device), HZ).reshape(
                    1, self.N, dim, dim
                )
            )
        else:
            H0 = (
                pt.einsum("sjq,qab->sjab", A.to(device), gate.get_PZ_vec(nq).to(device))
                + pt.einsum(
                    "sj,ab->sjab",
                    J.to(device),
                    gate.get_coupling_matrices(nq).to(device),
                )
                + pt.einsum("sj,ab->sjab", pt.ones(nS, self.N, device=device), HZ)
            )

        return H0.to(device)

    def time_evolution(self, device=default_device):
        """
        Determines time evolution sub-operators from system Hamiltonian H0 and control Hamiltonians.
        Since A and J vary, H0 already has a time axis in this class, unlike in GrapeESR.
        """
        m, N = self.x_cf.shape
        sig_xn = gate.get_Xn(self.nq, device)
        sig_yn = gate.get_Yn(self.nq, device)
        dim = 2**self.nq
        dt = self.dt
        u_mat = self.u_mat()
        x_sum = pt.einsum("kj,kj->j", u_mat, self.x_cf).to(device)
        y_sum = pt.einsum("kj,kj->j", u_mat, self.y_cf).to(device)
        H = self.H0 + pt.einsum(
            "s,jab->sjab",
            pt.ones(self.nS, device=device),
            pt.einsum("j,ab->jab", x_sum, sig_xn)
            + pt.einsum("j,ab->jab", y_sum, sig_yn),
        )
        H = pt.reshape(H, (self.nS * N, dim, dim))
        U = pt.matrix_exp(-1j * H * dt)
        del H
        U = pt.reshape(U, (self.nS, N, dim, dim))
        return U

    def get_control_frequencies(self, device=default_device):
        analytic = False
        # if analytic:
        #     return get_2E_multi_system_rf_analytic(self.J, self.A)
        return get_multi_system_resonant_frequencies(self.H0[:, self.N // 2])

    def modulate_A(self):
        eta2 = -3e-3 * (unit.um / unit.V) ** 2
        modulator = eta2 * self.E**2
        if self.nS == 1:
            dA = pt.einsum("j,q->jq", modulator, self.A)
            return (
                pt.einsum("q,j->jq", self.A, pt.ones(self.N, device=default_device))
                + dA
            )
        else:
            dA = pt.einsum("j,sq->sjq", modulator, self.A)
            return (
                pt.einsum("sq,j->sjq", self.A, pt.ones(self.N, device=default_device))
                + dA
            )

    def modulate_A_old(self):
        eta2 = -3e-3 * (unit.um / unit.V) ** 2
        modulator = eta2 * self.E**2
        if self.nS == 1:
            dA = pt.einsum("j,q->jq", modulator, self.A)
            return (
                pt.einsum("q,j->jq", self.A, pt.ones(self.N, device=default_device))
                + dA
            )
        else:
            dA = pt.einsum("j,sq->sjq", modulator, self.A)
            return (
                pt.einsum("sq,j->sjq", self.A, pt.ones(self.N, device=default_device))
                + dA
            )

    def modulate_J(self):
        E_max = pt.max(pt.real(self.E))
        modulator = rise_ones_fall(self.N, self.rise_time / self.tN)
        if self.nS == 1:
            return modulator * self.J
        J_mod = pt.einsum("j,s->sj", modulator, self.J)
        return J_mod

    def modulate_J_old(self):
        rise_prop = 100
        modulator = pt.cat(
            (
                pt.linspace(0.01, 1, self.N // rise_prop, device=default_device),
                pt.ones(self.N - 2 * (self.N // rise_prop), device=default_device),
                pt.linspace(1, 0.01, self.N // rise_prop, device=default_device),
            )
        )
        if self.nS == 1:
            return modulator * self.J
        J_mod = pt.einsum("j,s->sj", modulator, self.J)
        return J_mod

    def plot_E_A_J(self):
        fig, ax = plt.subplots(3)
        T = linspace(0, self.tN, self.N)
        A = self.modulate_A()
        J = self.modulate_J()
        plot_E_field(T, self.E, ax[0])
        if self.nS > 1:
            print("Plotting A and J for system 1")
            A = A[0]
            J = J[0]
        plot_A(T, A, ax[1])
        plot_J(T, J, ax[2])


class Grape_ee_Flip(GrapeESR):
    """
    Grape class for optimisations of pulses to flip exchange coupled 1P-2P
    electrons conditional on the 1P nuclear spin, keeping phase intact.
    """

    Ix1 = 0.5 * gate.ZII
    Iy1 = 0.5 * gate.ZII
    Iz1 = 0.5 * gate.ZII
    Sx1 = 0.5 * gate.IXI
    Sy1 = 0.5 * gate.IYI
    Sz1 = 0.5 * gate.IZI
    Sx2 = 0.5 * gate.IIX
    Sy2 = 0.5 * gate.IIY
    Sz2 = 0.5 * gate.IIZ
    S1S2 = 0.25 * pt.kron(gate.Id, gate.sigDotSig)
    Iz1Sz1 = 0.25 * gate.kron3(gate.Z, gate.Z, gate.Id)

    def __init__(self, tN, N, J, A, step, **kwargs):
        # X0 = pt.tensor(
        #     [
        #         [1, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 1],
        #     ],
        #     dtype=cplx_dtype,
        #     device=default_device,
        # )
        # target = pt.tensor(
        #     [
        #         [1, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 0, 1],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 0],
        #         [0, 0, 1, 0],
        #     ],
        #     dtype=cplx_dtype,
        #     device=default_device,
        # )

        if step == 1:
            X0 = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )
            target = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, -1j],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )
        elif step == 2:
            X0 = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )
            target = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, -1j],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )

        elif step == 3:
            # e_ctrl - e_coup CNOT while entangled to nuclear spins
            X0 = pt.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )
            target = pt.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )

        elif step == 4:
            X0 = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 0],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )
            target = pt.tensor(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                ],
                dtype=cplx_dtype,
                device=default_device,
            )

        else:
            raise Exception("step must be either 1 or 2.")

        kwargs["A_spec"] = get_A_spec_single()
        kwargs["X0_spec"] = gate.II
        if step in [1, 2]:
            kwargs["target_spec"] = gate.CX_native
        else:
            kwargs["target_spec"] = gate.II

        kwargs["target"] = target
        kwargs["X0"] = X0
        super().__init__(tN, N, J, A, **kwargs)
        pass

    def get_nS_nq(self):
        nS = 1 if len(self.A.shape) == 1 else len(self.A)
        nq = 3
        return nS, nq

    def get_H0(self, Bz=np.float64(0)):
        return self.get_H0_1n2e(self.J, self.A, Bz=Bz)

    @staticmethod
    def get_H0_1n2e(J, A, Bz=np.float64(0)):
        nS, nq = get_nS_nq_from_A(A)

        if nS == 1:
            H0 = (
                4 * A[1] * Grape_ee_Flip.Iz1Sz1
                + 2 * A[0] * Grape_ee_Flip.Sz2
                + 4 * J * Grape_ee_Flip.S1S2
            )
            return H0.reshape(1, *H0.shape)

        H0 = (
            pt.einsum(
                "s,ab->sab",
                4 * A[:, 1] * pt.ones(nS, device=default_device),
                Grape_ee_Flip.Iz1Sz1,
            )
            + pt.einsum("s,ab->sab", 2 * A[:, 0], Grape_ee_Flip.Sz2)
            + pt.einsum("s,ab->sab", 4 * J, Grape_ee_Flip.S1S2)
        )
        return H0

    def get_control_field_operators(self, nq, device=default_device):
        if nq == 3:
            # non-spectator
            sig_xn = gate.IXI + gate.IIX
            sig_yn = gate.IYI + gate.IIY
        elif nq == 2:
            # spectator
            sig_xn = gate.IX
            sig_yn = gate.IY
        else:
            raise Exception(
                "Invalid qubit count when determining control field operators."
            )
        return sig_xn, sig_yn

    def spectator_H0(self):
        H0_spec = self.A_spec * pt.kron(gate.Z, gate.Z)
        return H0_spec.reshape(1, *H0_spec.shape)


class GrapeRWA(GrapeESR):
    """
    GRAPE class using rotating wave approximation (RWA) to reduce simulation load.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get eigensystem
        self.S, self.D = get_multi_ordered_eigensystems(
            self.H0, H0_phys=self.get_H0(Bz=2 * unit.T)
        )
        self.S_T = pt.transpose(self.S, -1, -2)
        # get eigentarget
        self.target = pt.einsum(
            "sab,sbd->sad", self.S_T, pt.einsum("sbc,scd->sbd", self.target, self.S)
        )

        self.rf_mat = get_multisys_rf_tensor(self.S, self.D)

    def map_matrix_to_eigenbasis(self, A):
        return pt.einsum("sab,sbd->sad", self.S_T, pt.einsum("bc,scd->sbd", A, self.S))

    def get_default_targets(self):
        target = CNOT_targets(self.nS, self.nq)
        U0 = get_U0(self.H0, N=self.N, tN=self.tN)
        return pt.einsum("sab,sbc->sac", dagger(U0[-1]), target)

    def fidelity(self, device=default_device):
        """
        Adapted grape fidelity function designed specifically for multiple systems with transverse field control Hamiltonians.
        """

        x_cf = self.x_cf
        y_cf = self.y_cf
        dim = 2 * self.nq

        sig_xn = self.map_matrix_to_eigenbasis(gate.get_Xn(self.nq, device))
        sig_yn = self.map_matrix_to_eigenbasis(gate.get_Yn(self.nq, device))

        t0 = time.time()
        global time_exp
        time_exp += time.time() - t0

        t0 = time.time()
        self.X, self.P = self.propagate(device=device)
        global time_prop
        time_prop += time.time() - t0
        Ut = self.P[:, -1]
        Uf = self.X[:, -1]
        # fidelity of resulting unitary with target
        IP = batch_IP(Ut, Uf)
        Phi = pt.real(IP * pt.conj(IP))

        t0 = time.time()
        # calculate grad of fidelity
        XP_IP = batch_IP(self.X, self.P)

        ox_X = pt.einsum("ab,sjbc->sjac", sig_xn, self.X)
        oy_X = pt.einsum("ab,sjbc->sjac", sig_yn, self.X)
        PoxX_IP = (
            batch_trace(pt.einsum("sjab,sjbc->sjac", dagger(self.P), 1j * ox_X)) / dim
        )
        PoyX_IP = (
            batch_trace(pt.einsum("sjab,sjbc->sjac", dagger(self.P), 1j * oy_X)) / dim
        )
        del ox_X, oy_X

        Re_IP_x = -2 * pt.real(pt.einsum("sj,sj->sj", PoxX_IP, XP_IP))
        Re_IP_y = -2 * pt.real(pt.einsum("sj,sj->sj", PoyX_IP, XP_IP))

        # sum over systems axis
        sum_IP_x = pt.sum(Re_IP_x, 0)
        sum_IP_y = pt.sum(Re_IP_y, 0)
        del Re_IP_x, Re_IP_y

        dPhi_x = pt.einsum("kj,j->kj", pt.real(x_cf), sum_IP_x)
        dPhi_y = pt.einsum("kj,j->kj", pt.real(y_cf), sum_IP_y)
        del sum_IP_x, sum_IP_y

        dPhi = dPhi_x + dPhi_y
        del dPhi_x, dPhi_y
        global time_grad
        time_grad += time.time() - t0

        return Phi, dPhi


class CouplerGrape(GrapeESR):
    """
    GRAPE class dedicated to paralel optimisation of triple-donor systems.
    """

    def __init__(self, tN, N, J, A, Bz=np.float64(0), **kwargs):
        self.nS, self.nq = get_nS_nq_from_A(A)
        X0 = gate.coupler_Id
        super().__init__(tN, N, J, A, Bz=Bz, X0=X0, **kwargs)

    def get_default_targets(self, device=default_device):
        return coupler_CX_targets(self.nS, device=device)


################################################################################################################
################        Standalone functions        ############################################################
################################################################################################################


def interpolate_u(u, m, N_new):
    """
    Takes an m x N matrix, u_mat, and produces an m x N_new matrix whose rows are found by connecting the dots between the points
    give in the rows of u_mat. This allows us to produce a promising initial guess for u with a large number of timesteps using
    a previously optimised 'u' with a lower number of timesteps.
    """
    u_mat = uToMatrix(u, m)
    m, N = u_mat.shape
    u_new = pt.zeros(m, N_new, dtype=cplx_dtype)
    for j in range(N_new):
        u_new[:, j] = u_mat[:, j * N // N_new]

    return uToVector(u_new)


def refine_u(filename, N_new):
    """
    Uses a previous optimization performed with N timesteps to generate an initial guess u0 with
    N_new > N timesteps, which is then optimised.
    """
    J, A, tN, N, target, _fid = load_system_data(filename)
    u, cost_hist = load_u(filename)
    grape = GrapeESR(J, A, tN, N_new, target, cost_hist=cost_hist)
    m = 2 * len(grape.get_control_frequencies())
    grape.u = interpolate_u(u, m, N_new)
    grape.run()
    grape.result()


def load_u(fp=None, SP=None):
    """
    Loads saved 'u' tensor and cost history from files
    """
    u = pt.load(fp, map_location=pt.device("cpu")).type(cplx_dtype)
    cost_hist = list(pt.load(f"{fp}_cost_hist"))
    return u, cost_hist


def load_grape(fp, Grape=GrapeESR, **kwargs):
    grape = load_system_data(fp, Grape=Grape, **kwargs)
    grape.propagate()
    if grape.simulate_spectators:
        grape.get_spec_propagators()
    return grape


def load_total_field(fp):
    field = pt.load(fp, map_location=pt.device("cpu"))
    Bx = field[0] * unit.T
    By = field[1] * unit.T
    T = field[2] * unit.ns
    return Bx, By, T
