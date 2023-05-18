import torch as pt
import matplotlib as mpl
import numpy as np
from collections import defaultdict


if not pt.cuda.is_available():
    mpl.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from PyQt5.QtWidgets import QApplication

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
# uparrow = "↑"
downarrow = "\u2193"
Uparrow = "⇑"
Downarrow = "⇓"
rangle = "⟩"

time_axis_label = "Time (ns)"


def spin_state_label_getter(i, nq, states_to_label=None):
    if states_to_label is not None:
        if i in states_to_label:
            return np.binary_repr(i, nq)
    else:
        return np.binary_repr(i, nq)


def electron_arrow_spin_state_label_getter(i, nq):
    basis = [downarrow, uparrow]
    label = ""
    comp_state = np.binary_repr(i, nq)
    for q in range(nq):
        label += basis[int(comp_state[q])]
    return label


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
    """Returns state label corresponding to integer j\in[0,dim]"""
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
    legend_loc="best",
    legend_ncols=1,
    colors=None,
    linestyles=None,
    ylabel=None,
    **kwargs,
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
        if colors is not None:
            color = colors[i]
        else:
            color = None
        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = "-"

        ax.plot(
            real(T) / unit.ns,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            **kwargs,
        )
    ax.legend(loc=legend_loc, ncol=legend_ncols)
    ax.set_xlabel("time (ns)")
    if y_axis_labels:
        ax.set_ylabel("$|\psi|$")
    if fp is not None:
        plt.savefig(fp)
    ax.set_ylabel(ylabel)
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


def plot_fields(
    Bx, By, tN=None, T=None, ax=None, ylabel=False, legend_loc="best", **kwargs
):
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
    if ylabel:
        ax.set_ylabel("Magnetic field (mT)")
        legend_unit = " (mT)"
    else:
        legend_unit = ""
    ax.plot(T / unit.ns, Bx * 1e3 / unit.T, label="$B_x$" + legend_unit, **kwargs)
    ax.plot(T / unit.ns, By * 1e3 / unit.T, label="$B_y$" + legend_unit, **kwargs)
    ax.set_xlabel("time (ns)")
    if y_axis_labels:
        ax.set_ylabel("$B_\omega(t)$ (mT)")
    ax.legend(loc=legend_loc)
    return ax


def plot_fields_twinx(
    Bx,
    By,
    T,
    ax=None,
    near_lim=None,
    far_lim=None,
    tick_lim=1,
    ylabels=True,
    prop_zoom_start=0.4,
    prop_zoom_end=0.41,
):
    if ax is None:
        ax = plt.subplot()

    axt = ax.twinx()
    xcol = color_cycle[0]
    ycol = color_cycle[1]

    plot_with_zoom(
        T / unit.ns,
        Bx / unit.mT,
        ax,
        prop_zoom_start=prop_zoom_start,
        prop_zoom_end=prop_zoom_end,
        color=xcol,
    )
    plot_with_zoom(
        T / unit.ns,
        By / unit.mT,
        axt,
        prop_zoom_start=prop_zoom_start,
        prop_zoom_end=prop_zoom_end,
        color=ycol,
    )

    max_field_val = maxreal(pt.cat((Bx, By)))
    if far_lim is None:
        far_lim = 3.5 * max_field_val / unit.mT
    if near_lim is None:
        near_lim = 1.3 * max_field_val / unit.mT
    ax.set_ylim([-far_lim, near_lim])
    axt.set_ylim([-near_lim, far_lim])
    ax.set_yticks([-tick_lim, tick_lim], [-tick_lim, tick_lim], color=xcol)
    axt.set_yticks([-tick_lim, tick_lim], [-tick_lim, tick_lim], color=ycol)
    label_axis(ax, "Zoomed in", x_offset=0.22, y_offset=0.05, fontsize=12)
    if ylabels:
        ax.set_ylabel("Bx (mT)", color=xcol)
        axt.set_ylabel("By (mT)", color=ycol)
    ax.set_xlabel("time (ns)")
    return axt


def plot_with_zoom(
    X, Y, ax, prop_zoom_start, prop_zoom_end, zoom_proportion=0.4, color=color_cycle[0]
):
    x_zoom_start = prop_zoom_start * X[-1]
    x_zoom_end = prop_zoom_end * X[-1]

    i1 = np.argmax(X > x_zoom_start)
    i2 = np.argmax(X > x_zoom_end)

    Y1 = Y[: i1 + 1]
    Y2 = Y[i1 : i2 + 1]
    Y3 = Y[i2:]

    x1 = (1 - zoom_proportion) * len(Y1) / (len(Y1) + len(Y3))

    x2 = x1 + zoom_proportion
    x3 = 1

    X1 = np.linspace(0, x1, len(Y1))
    X2 = np.linspace(x1, x2, len(Y2))
    X3 = np.linspace(x2, x3, len(Y3))

    ax.plot(X1, Y1, color=color)
    ax.plot(X2, Y2, color=color)
    ax.plot(X3, Y3, color=color)

    div_linestyle = "--"
    div_colour = "red"
    ax.axvline(x1, linestyle=div_linestyle, color=div_colour)
    ax.axvline(x2, linestyle=div_linestyle, color=div_colour)

    ax.set_xticks([0, x1, x2, x3], [0, f"{X[i1]:.0f}", f"{X[i2]:.0f}", f"{X[-1]:.0f}"])


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
        ax = plt.subplot()
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
    ax.plot(x_axis, alpha**2, label="$α^2$")
    ax.plot(x_axis, beta**2, label="$ß^2$")
    ax.legend()


def show_fidelity(X, T=None, tN=None, target=gate.CX, ax=None):
    print(f"Final unitary:")
    print_rank2_tensor(X[-1] / (X[-1, 0, 0] / pt.abs(X[-1, 0, 0])))
    fids = fidelity_progress(X, target)
    print(f"Final fidelity = {fids[-1]}")

    if ax is None:
        ax = plt.subplot()
    plot_fidelity(fids, ax, tN=tN, T=T)
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
    systems_ax=None,
    ax=None,
    f=[0.9999, 0.999, 0.99],
    colours=["green", "orange", "red"],
    labels=None,
    legend_loc="upper left",
    ylim=[0.999, 1.0005],
    put_xlabel=True,
    **kwargs,
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

    if systems_ax is None:
        systems_ax = np.linspace(0, len(fids) - 1, len(fids))
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
        sys_binned[i_bin].append(systems_ax[j])
    i = 2
    if labels == None:
        labels = [f">{fj*100:.2f}%" for fj in f] + [f"<{f[-1]*100:.2f}%"]
    for i in range(nbins):
        ax.bar(
            sys_binned[i],
            fids_binned[i],
            color=colours[i],
            label=labels[i],
            **kwargs,
        )
    color = [get_fid_color(fid) for fid in fids]
    if ax is None:
        ax = plt.subplot()
    nS = len(fids)
    # ax.bar(np.linspace(0,nS-1,nS), fids, np.ones(nS)*0.3, color = color)
    ax.legend(loc=legend_loc, ncol=4)
    if put_xlabel:
        ax.set_xlabel("Systems")
    ax.set_ylabel("Fidelity (%)")
    ax.set_ylim(ylim)
    ax.set_yticks([0.999, 1], ["99.9%", "100%"])
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70])


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


def plot_spheres(sites, color="blue", radius=0.4, ax=None, alpha=1, zorder=0):
    for site in sites:
        circle = plt.Circle(
            site, radius=radius, color=color, alpha=alpha, zorder=zorder
        )
        ax.add_patch(circle)


def get_all_sites(x_range, y_range, padding):
    X = np.linspace(
        x_range[0] - padding,
        x_range[1] + padding,
        int(x_range[1] - x_range[0] + 2 * padding + 1),
    )
    Y = np.linspace(
        y_range[0] - padding,
        np.ceil(y_range[1]) + padding,
        int(np.ceil(y_range[1] - y_range[0] + 2 * padding + 1)),
    )
    all_sites = []
    for x in X:
        for y in Y:
            all_sites.append(np.array([x, y]))
    return all_sites


def plot_6_sites(
    x_target,
    y_target,
    ax=None,
    round_func=None,
    color="lightblue",
    alpha=0.5,
    orientation="flat",
):
    if round_func is not None:
        x_target = round_func(x_target)
        y_target = round_func(y_target)
    long = np.linspace(0, 1, 2)
    short = np.linspace(0, 2, 3)
    if orientation == "flat":
        sites_x = short
        sites_y = long
    else:
        sites_x = long
        sites_y = short

    sites = []

    for x in sites_x:
        for y in sites_y:
            sites.append((x + x_target, y + y_target))

    if ax is None:
        ax = plt.subplot()
    plot_spheres(sites, ax=ax, alpha=alpha, color=color)
    # plot_spheres([(x_target, y_target)], ax=ax, color='black', alpha=1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_9_sites(
    x_target, y_target, ax=None, round_func=None, neighbour_color="lightblue", alpha=0.5
):
    if round_func is not None:
        x_target = round_func(x_target)
        y_target = round_func(y_target)
    sites_x = np.linspace(-1, 1, 3)
    sites_y = np.linspace(-1, 1, 3)
    sites = []

    for x in sites_x:
        for y in sites_y:
            sites.append((x + x_target, y + y_target))

    if ax is None:
        ax = plt.subplot()
    plot_spheres(sites, ax=ax, alpha=alpha, color=neighbour_color)
    # plot_spheres([(x_target, y_target)], ax=ax, color='black', alpha=1)
    ax.set_aspect("equal")
    ax.axis("off")


def add_colorbar(cmap, ax=None):
    # Add the color bar
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(-0.5, 0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.5, cax=ax)
    custom_ticks = np.array([-1, -0.5, 0, 0.5, 1]) / 2  # Example tick values
    # custom_tick_labels = ["0", "π/2", "π", "3π/2", "2π"]  # Example tick labels
    custom_tick_labels = ["-π", "-π/2", "0", "π/2", "π"]
    # custom_tick_labels = custom_ticks

    cbar.set_ticks(custom_ticks)  # Normalize the tick values
    cbar.set_ticklabels(custom_tick_labels)


def plot_unitary(
    unitary_matrix,
    ax=None,
    label_getter=None,
    colorbar=False,
    colorbar_ax=None,
    cmap=None,
):
    m, n = unitary_matrix.shape
    if label_getter is None:
        label_getter = lambda i: np.binary_repr(int(i), int(np.log2(m)))

    y, x = np.meshgrid(np.arange(m), np.arange(n))
    z = np.abs(unitary_matrix)
    print_rank2_tensor(pt.angle(unitary_matrix))
    if cmap is None:
        cmap = cm.viridis
    hue = np.angle(unitary_matrix) / (2 * np.pi) + 0.5
    # hue += hue < 0

    colors = cmap(hue)

    print_rank2_tensor(pt.angle(unitary_matrix) / np.pi)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.bar3d(
        x.ravel(),
        y.ravel(),
        np.zeros_like(z).ravel(),
        0.5,
        0.5,
        z.ravel(),
        shade=True,
        color=colors.reshape(-1, 4),
    )
    ax.grid(False)
    # ax.set_xlabel('Real')
    # ax.set_ylabel('Imaginary')
    # ax.set_zlabel('Magnitude')
    x = np.linspace(1, m, m)
    ax.set_xticks(x - 1, [label_getter(int(i - 1)) for i in x])
    ax.set_yticks(x, [label_getter(int(i - 1)) for i in x])

    if colorbar:
        add_colorbar(cmap, colorbar_ax)


class DynamicOptimizationPlot:
    def __init__(self, n_plots=1, ax_input=defaultdict(lambda: {})):
        # ax=None, colors="blue", linestyle="-", ylim=None
        self.n_plots = n_plots
        self.initialize_ax_data(ax_input)

    def initialize_ax_data(self, ax_input):
        default_color = "blue"
        default_linestyle = "-"
        default_ylim = None
        for k in range(self.n_plots):
            if k not in ax_input["ax"]:
                ax_input["ax"][k] = plt.subplot()
            if k not in ax_input["color"]:
                ax_input["color"][k] = default_color
            if k not in ax_input["linestyle"]:
                ax_input["linestyle"][k] = default_linestyle
            if k not in ax_input["ylim"]:
                ax_input["ylim"][k] = default_ylim
            if k not in ax_input["legend_label"]:
                ax_input["legend_label"][k] = None

        self.ax = ax_input["ax"]
        self.color = ax_input["color"]
        self.linestyle = ax_input["linestyle"]
        self.ylim = ax_input["ylim"]
        self.legend_label = ax_input["legend_label"]
        self.line = [
            self.ax[k].plot(
                [],
                [],
                color=self.color[k],
                linestyle=self.linestyle[k],
                label=self.legend_label[k],
            )[0]
            for k in range(self.n_plots)
        ]
        for k in range(self.n_plots):
            if self.ylim[k] is not None:
                self.ax[k].set_ylim(self.ylim[k])
            if self.legend_label[k] is not None:
                self.ax[k].legend()

    def update(self, data):
        for k in range(self.n_plots):
            self.line[k].set_data(range(len(data[k])), data[k])
            self.ax[k].relim()
            if self.ylim[k] is None:
                self.ax[k].autoscale_view()
            else:
                self.ax[k].set_xlim([0, len(data[k])])
            self.ax[k].get_figure().canvas.flush_events()


if __name__ == "__main__":
    # Plot quantum state tomography
    plot_unitary(pt.tensor([[1, 0], [0, 1j]]))

    plt.show()
