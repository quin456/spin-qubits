import torch as pt
import matplotlib as mpl
import numpy as np

if not pt.cuda.is_available():
    mpl.use("Qt5Agg")
from matplotlib import pyplot as plt


import atomic_units as unit
import gates as gate
from utils import *
from eigentools import *
from hamiltonians import get_H0, multi_NE_H0, get_NE_H0
from data import get_A, get_J, gamma_n, gamma_e, B0, cplx_dtype, default_device


color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class FigureColours:
    sites_colour = "#1DA4BF"


blue = "dodgerblue"
darkblue = "blue"
grey = "silver"
yellow = "orange"
orange = "red"
red = "darkred"

fig_width_double_long = 8
fig_height_single_long = 2.4
fig_height_double_long = 4.5
fig_width_single = 3.2 * 1.2
fig_width_double = fig_width_single * 1.7  # untested
fig_height_single = 2.4 * 1.2

annotate = False
y_axis_labels = False

uparrow = "\u2191"
downarrow = "\u2193"
Uparrow = "⇑"
Downarrow = "⇓"

time_axis_label = "Time (ns)"


def spin_state_label_getter(i, nq, states_to_label=None):
    if states_to_label is not None:
        if i in states_to_label:
            return np.binary_repr(i, nq)
    else:
        return np.binary_repr(i, nq)


def spin_state_ket_label_getter(i, nq=2, states_to_label=None):
    return f"$<{spin_state_label_getter(i, nq, states_to_label=states_to_label)}|\psi>$"


def spin_state_ket_sq_label_getter(i, nq=2, states_to_label=None):
    return f"|{spin_state_ket_label_getter(i, nq=nq, states_to_label=states_to_label)}$|^2$"


def alpha_sq_label_getter(i, nq=2):
    return f"$|α_{{{np.binary_repr(i,nq)}}}|^2$"


def eigenstate_label_getter(i, states_to_label=None):
    if states_to_label is not None:
        if i in states_to_label:
            return f"E{i}"
    else:
        return f"E{i}"


def NE_label_getter(j):

    b = np.binary_repr(j, 2)
    label = "|"
    if b[0] == "0":
        label += Uparrow
    else:
        label += Downarrow
    if b[1] == "0":
        label += uparrow
    else:
        label += downarrow
    label += ">"
    return label


def multi_NE_label_getter(j, label_states=None):
    """ Returns state label corresponding to integer j\in[0,dim] """
    if label_states is not None:
        if j not in label_states:
            return ""
    uparrow = "\u2191"
    downarrow = "\u2193"
    b = np.binary_repr(j, 4)
    if b[2] == "0":
        L2 = uparrow
    else:
        L2 = downarrow
    if b[3] == "0":
        L3 = uparrow
    else:
        L3 = downarrow

    return b[0] + b[1] + L2 + L3


def plot_psi(
    psi,
    tN=None,
    T=None,
    ax=None,
    label_getter=None,
    squared=True,
    fp=None,
    legend_loc="upper center",
    legend_ncols=1,
):
    """
    Plots the evolution of each component of psi.

    Inputs
        psi: (N,d) tensor where d is Hilbert space dimension and N is number of timesteps.
        tN: duration spanned by N timesteps
        ax: axis on which to plot
    """
    psi = psi.cpu()
    if ax is None:
        ax = plt.subplot()
    if label_getter is None:
        label_getter = lambda i: spin_state_label_getter(i, nq)
    N, dim = psi.shape
    nq = get_nq_from_dim(dim)
    if T is None:
        if tN is None:
            print("No time axis information provided to plot_psi")
            T = pt.linspace(0, N - 1, N)
        else:
            T = pt.linspace(0, tN, N)
    for i in range(dim):
        if squared:
            y = pt.abs(psi[:, i]) ** 2
        else:
            y = pt.abs(psi[:, i])
        label = label_getter(i)
        # if squared and label is not None:
        #     label = f"Pr({label})"
        ax.plot(real(T) / unit.ns, y, label=label)
    ax.legend(loc=legend_loc, ncol=legend_ncols)
    ax.set_xlabel("time (ns)")
    if y_axis_labels:
        ax.set_ylabel("$|\psi|$")
    print(squared)
    if fp is not None:
        plt.savefig(fp)

    return ax


def plot_phases(
    psi, tN=None, T=None, ax=None, legend_loc="upper center", relative_to_0=False
):

    if ax is None:
        ax = plt.subplot()
    N, dim = psi.shape
    nq = get_nq_from_dim(dim)
    if T is None:
        T = linspace(0, tN, N)
    phase = pt.zeros(psi.shape, dtype=real_dtype)
    for i in range(dim):
        phase[:, i] = pt.angle(psi)[:, i]  # -pt.angle(psi)[:,0]
        ax.plot(real(T) / unit.ns, phase[:, i], label=f"$\phi_{i}$")
    ax.legend(loc=legend_loc)
    ax.set_xlabel("time (ns)")
    print(f"Final phase = {'π, '.join([str(p) for p in (phase[-1]/np.pi).tolist()])}")
    return ax


def plot_psi_with_phase(psi, T, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 2)

    plot_psi(psi, T=T, ax=ax[0])
    plot_phases(psi, T=T, ax=ax[1])


def plot_fields(Bx, By, tN=None, T=None, ax=None):
    """
    Inputs
        Bx: Magnetic field in x direction over N timesteps (atomic units) 
        By: Magnetic field in y direction over N timesteps (atomic units)
        tN: Duration of pulse
    """
    N = len(Bx)
    if T is None:
        T = linspace(0, tN, N)
    if ax is None:
        ax = plt.subplot()
    ax.plot(T / unit.ns, Bx * 1e3 / unit.T, label="X field (mT)")
    ax.plot(T / unit.ns, By * 1e3 / unit.T, label="Y field (mT)")
    ax.set_xlabel("time (ns)")
    if y_axis_labels:
        ax.set_ylabel("$B_\omega(t)$ (mT)")
    ax.legend()
    return ax


def plot_psi_and_fields(psi, Bx, By, tN):
    fig, ax = plt.subplots(1, 2)
    plot_psi(psi, tN, ax[0])
    plot_fields(Bx, By, tN, ax[1])
    return ax


def visualise_Hw(Hw, tN, eigs=None):
    """
    Generates an array of plots, one for each matrix element of the Hamiltonian Hw, which
    shows the evolution of Hw through time.

    Inputs
        Hw: (N,d,d) tensor describing d x d dimensional Hamiltonian over N timesteps
        tN: duration spanned by Hw.
    """
    N, dim, dim = Hw.shape
    T = pt.linspace(0, tN / unit.ns, N)
    if eigs is not None:
        D = pt.diag(eigs.eigenvalues)
        U0_e = pt.matrix_exp(-1j * pt.einsum("ab,j->jab", D, T))
        S = eigs.eigenvectors
        Hw = dagger(U0_e) @ dagger(S) @ Hw @ S @ U0_e
    fig, ax = plt.subplots(dim, dim)
    for i in range(dim):
        for j in range(dim):
            y = Hw[:, i, j] / unit.MHz
            ax[i, j].plot(T, pt.real(y))
            ax[i, j].plot(T, pt.imag(y))


def plot_fidelity(fids, ax=None, T=None, tN=None, legend=True, printfid=False):
    if ax is None:
        ax=plt.subplot()
    if len(fids.shape) == 1:
        fids = fids.reshape(1, *fids.shape)
    nS = len(fids)
    N = len(fids[0])
    if T is None:
        T = pt.linspace(0, tN, N)
    if nS == 1:
        ax.plot(T / unit.ns, fids[0], label=f"Fidelity")
    else:
        for q in range(nS):
            ax.plot(T / unit.ns, fids[q], label=f"System {q+1} fidelity")
    if legend:
        ax.legend()
    ax.set_xlabel("Time (ns)")
    if y_axis_labels:
        ax.set_ylabel("Fidelity")
    if annotate:
        ax.annotate("Fidelity progress", (0, 0.95))
    if printfid:
        print(f"Achieved fidelity = {fids[:,-1]:.4f}")
    return ax


def plot_avg_min_fids(ax, X, target, tN):
    nS, N = X.shape[:2]
    fid_progress = fidelity_progress(X, target)
    avg_fid = pt.sum(fid_progress, 0) / nS
    min_fid = pt.min(fid_progress, 0).values
    T = linspace(0, tN / unit.ns, N)
    ax.plot(T, avg_fid, color="blue", label="Average fidelity")
    ax.plot(T, min_fid, color="red", label="Minimum fidelity")
    ax.set_xlabel("Time (ns)")
    ax.legend()


def plot_multi_sys_energy_spectrum(E, ax=None):
    dim = E.shape[-1]
    if ax is None:
        ax = plt.subplot()
    for sys in E:
        for i in range(len(sys)):
            ax.axhline(
                pt.real(sys[i] / unit.MHz), label=f"E{dim-1-i}", color=color_cycle[i]
            )
    ax.legend()


def plot_energy_spectrum(E, ax=None):
    if ax is None:
        ax = plt.subplot()
    dim = len(E)
    for i in range(dim):
        ax.axhline(
            pt.real(E[i] / unit.MHz),
            label=f"E{dim-1-i}",
            color=color_cycle[i % len(color_cycle)],
        )


def plot_energy_spectrum_from_H0(H0):
    rf = get_resonant_frequencies(H0)
    S, D = get_ordered_eigensystem(H0)
    plot_energy_spectrum(pt.diagonal(D))


def plot_energy_level_variation(H0, x_axis, x_label, x_unit=unit.MHz, ax=None):
    """
    Accepts array of H0 matrices corresponding to H0 evolution
    """

    N, dim, dim = H0.shape
    S = pt.zeros(N, dim, dim, dtype=cplx_dtype, device=default_device)
    D = pt.zeros_like(S)
    E = pt.zeros(N, dim, dtype=cplx_dtype, device=default_device)
    for j in range(N):
        S[j], D[j] = get_ordered_eigensystem(H0[j])
        E[j] = pt.diag(D[j])

    if ax is None:
        ax = plt.subplot()
    for a in range(dim):
        ax.plot(x_axis / x_unit, E[:, a] / unit.MHz, label=a, color="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (MHz)")

    print("Initial eigenstates:")
    print_rank2_tensor(S[0])
    print("\nFinal eigenstates:")
    print_rank2_tensor(S[-1])

    return E


def plot_exchange_energy_diagram(J=pt.linspace(0, 100, 100) * unit.MHz, A=None, Bz=B0):
    N = len(J)
    if A is None:
        A = get_A(N, 2)
    H0 = get_H0(A, J, Bz=Bz)
    plot_energy_level_variation(H0, J, "Exchange (MHz)")


def plot_NE_energy_diagram(
    Bz=pt.linspace(0, 0.2, 100) * unit.T, N=1000, A=get_A(1, 1), ax=None, fp=None
):

    N = len(Bz)
    dim = 4
    H0 = pt.zeros(N, dim, dim)

    for j in range(N):
        H0[j] = get_NE_H0(A, Bz[j])

    if ax is None:
        fig, ax = plt.subplots(1)
    E = plot_energy_level_variation(H0, Bz, "$B_0$ (mT)", unit.mT, ax=ax)
    ax.annotate("$T_0$", (-0.4, E[0, 1] / unit.MHz + 14))
    ax.annotate("$T^+$", (-0.4, E[0, 1] / unit.MHz - 5))
    ax.annotate("$T^-$", (-0.4, E[0, 1] / unit.MHz - 24))
    ax.annotate("$S_0$", (-0.4, E[0, 3] / unit.MHz))
    ax.annotate(f"{Downarrow}{uparrow}", (5, E[-1, 0] / unit.MHz - 2.5))
    ax.annotate(f"{Uparrow}{uparrow}", (5, E[-1, 1] / unit.MHz - 2.5))
    ax.annotate(f"{Downarrow}{downarrow}", (5, E[-1, 2] / unit.MHz - 5))
    ax.annotate(f"{Uparrow}{downarrow}", (5, E[-1, 3] / unit.MHz - 5))
    ax.set_xlim((-0.5, 5.5))
    if fp is not None:
        fig.savefig(fp)


def plot_NE_alpha_beta(
    Bz=pt.linspace(0, 0.2, 100) * unit.T, N=1000, A=get_A(1, 1), ax=None
):

    n = len(Bz)
    dim = 4
    H0 = pt.zeros(n, dim, dim)
    H0_phys = pt.zeros_like(H0)
    for j in range(n):
        H0[j] = get_NE_H0(A, Bz[j])
        H0_phys[j] = get_NE_H0(A, 1 * unit.T, gamma_n=gamma_e, gamma_e=gamma_n)
    S, D = get_multi_ordered_eigensystems(H0, H0_phys)

    if ax is None:
        ax = plt.subplot()
    plot_alpha_beta(S, D, Bz / unit.mT, ax=ax)
    ax.set_xlabel("$B_0$ (mT)")


def plot_alpha_beta(S, D, x_axis, ax=None):

    alpha = pt.real(S[:, 2, 1])
    beta = pt.real(S[:, 1, 1])
    # bad

    if ax is None:
        ax = plt.subplot()
    ax.plot(x_axis, alpha ** 2, label="$α^2$")
    ax.plot(x_axis, beta ** 2, label="$ß^2$")
    ax.legend()


def show_fidelity(X, T=None, tN=None, target=gate.CX, ax=None):
    print(f"Final unitary:")
    print_rank2_tensor(X[-1] / (X[-1, 0, 0] / pt.abs(X[-1, 0, 0])))
    fids = fidelity_progress(X, target)
    print(f"Final fidelity = {fids[-1]}")

    if ax is None:
        ax = plt.subplot()
    plot_fidelity(ax, fids, tN=tN, T=T)
    ax.set_yticks([0, 1])
    ax.axhline(1, linestyle="--", color="black", linewidth=1)
    ax.set_ylim([0, 1.1])
    return fids


def plot_E_field(T, E, ax=None):
    if ax is None:
        ax = plt.subplot()
    ax.plot(T.cpu() / unit.ns, E.cpu() * unit.m / unit.MV)
    ax.set_ylabel("Electric field (MV/m)")
    ax.set_xlabel("Time (ns)")


def plot_A(T, A, ax=None):
    if ax is None:
        ax = plt.subplot()
    ax.plot(T.cpu() / unit.ns, A.cpu() / unit.MHz)
    ax.set_ylabel("Hyperfine coupling (MHz)")
    ax.set_xlabel("Time (ns)")


def plot_J(T, J, ax=None):
    if ax is None:
        ax = plt.subplot()
    ax.plot(T.cpu() / unit.ns, J.cpu() / unit.MHz)
    ax.set_ylabel("Exchange (MHz)")
    ax.set_xlabel("Time (ns)")


def fidelity_bar_plot(
    fids,
    ax=None,
    f=[0.9999, 0.99, 0.98],
    colours=["green", "orange", "red", "darkred"],
    legend_loc="best",
):
    """
    Accepts nS length array of final fidelities for each system.
    """
    if ax is None:
        ax = plt.subplot()

    nbins = len(colours)

    def get_fid_color(fid):
        for i in range(nbins - 1):
            if fid > f[i]:
                return colours[i]
        else:
            return colours[nbins - 1]

    fids_binned = [[] for _ in range(nbins)]
    sys_binned = [[] for _ in range(nbins)]
    for j in range(len(fids)):
        for i in range(nbins - 1):
            if fids[j] > f[i]:
                i_bin = i
                break
        else:
            i_bin = nbins - 1
        fids_binned[i_bin].append(fids[j])
        sys_binned[i_bin].append(j)

    labels = [f">{fj*100}%" for fj in f] + [f"<{f[-1]*100}%"]
    for i in range(nbins):
        ax.bar(sys_binned[i], fids_binned[i], color=colours[i], label=labels[i])
    color = [get_fid_color(fid) for fid in fids]
    if ax is None:
        ax = plt.subplot()
    nS = len(fids)
    # ax.bar(np.linspace(0,nS-1,nS), fids, np.ones(nS)*0.3, color = color)
    ax.legend(loc=legend_loc, ncol=4)
    ax.set_xlabel("Systems")
    ax.set_ylabel("Fidelity")


def nuclear_spin_tag(
    ax,
    NucSpin,
    r_prop=0.1,
    down_color="black",
    up_color="black",
    loc="lower left",
    dx=0,
    dy=0,
    text=None,
    mult=1,
    dx_text=0,
    dy_text=0,
):
    bbox = ax.get_window_extent().transformed(
        ax.get_figure().dpi_scale_trans.inverted()
    )
    width, height = bbox.width, bbox.height
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_mult = mult * (xlim[1] - xlim[0]) / width
    y_mult = mult * (ylim[1] - ylim[0]) / height
    radius = r_prop * mult

    locy, locx = loc.split()
    if locx == "left":
        x0 = xlim[0]
    elif locx == "center":
        x0 = (xlim[1] + xlim[0]) / 2 - (3 * len(NucSpin) - 1.5) * (radius * x_mult) / 2
    elif locx == "right":
        x0 = xlim[1] - (3 * len(NucSpin) - 1.5) * (radius * x_mult)
    else:
        raise Exception("Invalid location")
    if locy == "upper":
        y0 = ylim[1] - radius * y_mult * 4
    elif locy == "center":
        y0 = (ylim[1] + ylim[0]) / 2 - radius * y_mult * 2
    elif locy == "lower":
        y0 = ylim[0]
    else:
        raise Exception("Invalid location")
    pos = np.array([x0 + (2 * radius) * x_mult + dx, y0 + (2 * radius) * y_mult + dy])
    color = [down_color if nspin else up_color for nspin in NucSpin]
    for j, nspin in enumerate(NucSpin):
        posj = pos + j * np.array([2 * radius * x_mult, 0])
        circ = mpl.patches.Ellipse(
            posj,
            radius * x_mult,
            radius * y_mult,
            facecolor="darkred",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        width = 0.1 * x_mult / 1.71875
        head_width = 0.2 * x_mult / 1.71875
        head_length = 0.05 * y_mult / 0.5949633626483772
        if nspin:
            farrow = mpl.patches.FancyArrow(
                *posj,
                0,
                -1 * radius * y_mult,
                width=width,
                head_length=head_length,
                head_width=head_width,
                color=color[j],
                zorder=3,
            )
        else:
            farrow = mpl.patches.FancyArrow(
                *posj,
                0,
                1 * radius * y_mult,
                width=width,
                head_length=head_length,
                head_width=head_width,
                color=color[j],
                zorder=3,
            )
        ax.add_patch(farrow)
        ax.add_patch(circ)
    if text is not None:
        ax.annotate(
            text,
            (
                x0 + (1.5 * radius) * x_mult + dx + dx_text,
                y0 - radius * 2.5 * y_mult + (2 * radius) * y_mult + dy + dy_text,
            ),
        )


def box_ax(ax, xlim=None, ylim=None, color="red", padding=1):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    xy = [xlim[0] + padding, ylim[0] + padding]
    w = xlim[1] - xlim[0] - 2 * padding
    h = ylim[1] - ylim[0] - 2 * padding
    draw_box(ax, xy, w, h, color=color)


def draw_box(ax, xy, w, h, color="red", linewidth=2):
    box = plt.Rectangle(
        xy, w, h, color=color, fill=False, linewidth=linewidth, alpha=1, zorder=3
    )
    ax.add_patch(box)


def color_bar(
    ax,
    colors,
    padding=0,
    tick_labels=["V3", "V2", "V1", "0 V"],
    orientation="horizontal",
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    Dx = xlim[1] - xlim[0]
    Dy = ylim[1] - ylim[0]
    n = len(colors)
    width = Dx * (1 - 2 * padding)
    height = Dy * (1 - 2 * padding)

    xy0 = np.array([xlim[0] + padding * Dx, ylim[0] + padding * Dy])
    ticks = []
    for i in range(n):
        if orientation == "horizontal":
            rectangle = mpl.patches.Rectangle(
                xy0 + i * np.array([0, height / n]), width, height, color=colors[i]
            )
            ticks.append(xy0[1] + (i + 1 / 2) * height / n)
        else:
            rectangle = mpl.patches.Rectangle(
                xy0 + i * np.array([width / n, 0]), width, height, color=colors[i]
            )
            ticks.append(xy0[0] + (i + 1 / 2) * width / n)

        ax.add_patch(rectangle)
    if orientation == "horizontal":
        ax.set_xticks([])
        ax.set_yticks(ticks=ticks, labels=tick_labels)
    else:
        ax.set_yticks([])
        ax.set_xticks(ticks=ticks, labels=tick_labels)


if __name__ == "__main__":

    # plot_exchange_energy_diagram(J=pt.linspace(-100,100,100)*unit.MHz, A=get_A(100,2), Bz=0.02*unit.T)
    # plot_NE_alpha_beta(Bz = pt.linspace(0,5, 500)*unit.T)
    # plot_NE_energy_diagram(Bz = pt.linspace(0,5, 500)*unit.T)
    # fidelity_bar_plot(np.array([0.99, 0.95, 0.92, 0.99, 0.9999, 0.978]))
    from GRAPE import load_grape

    prev_grape_fp = "fields/c878_1S_2q_100ns_500step"
    grape = load_grape(prev_grape_fp)
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    # ax.plot(x, np.sin(x))
    psi = grape.X[0] @ gate.spin_10
    plot_psi(psi, tN=grape.tN, ax=ax)
    # ax.set_xlim([-1000,10000])
    nuclear_spin_tag(ax, [0, 1, 1], loc="center right", text="15x 1P-1P", mult=1.5)
    # plt.show()

    # ax = plt.subplot()
    # color_bar(ax, colors=['#0000cc', '#0080ff', '#99ccff','lightgray', 'orange', 'red'], tick_labels=['1','2','3','4','5','6'], orientation='vertical')
    plt.show()
