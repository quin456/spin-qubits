from GRAPE import GrapeESR, load_grape
from architecture_2P_coupler import HS_side_view, make_HS_array
from visualisation import *
from utils import *
from run_couplers import coupler_fidelity_bars
from run_grape import get_fids_and_field_from_fp
from cnot_1P_2P_1P import *
from unique_configs import get_donor_sites_1P_2P
from run_grape import run_CNOTs


folder = "paper-plots/"


def parallel_1P_1P_CNOTs():
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(5, 3.7)
    ax0_twinx = ax[0].twinx()
    ax[0].set_ylim([-0.7, 1.5])
    ax0_twinx.set_ylim([-1.5, 0.7])

    grape_fp = "fields/c965_15S_2q_2000ns_2500step"
    grape = load_grape(fp=grape_fp)

    grape.print_result()

    fids, grad = grape.fidelity()
    grape.plot_XY_fields(ax[0], twinx=ax0_twinx, legend_loc=False)
    fidelity_bar_plot(
        fids,
        ax=ax[1],
        f=[0.999, 0.998],
        colours=["green", "orange"],
        legend_loc="upper center",
    )
    ax[1].set_ylim([0.998, 1.0007])
    ax[1].set_yticks([0.998, 0.999, 1], ["99.8", "99.9", "100"])
    ax[1].axhline(1, linestyle="--", linewidth=1, color="black")
    ax[1].set_xticks([0, 4, 9, 14], ["1", "5", "10", "15"])

    fig.tight_layout()
    fig.savefig(f"{folder}parallel-1P-1P.pdf")


def draw_HS_architecture(fp=None):
    fig, ax = plt.subplots(1, 1)
    make_HS_array(ax, 5)
    if fp is not None:
        fig.savefig(fp)


def draw_HS_unit_and_side_view(fp=None):
    fig1, ax = plt.subplots(1, 2)
    make_stab_unit_cell(ax[0], fontsize=15)
    HS_side_view(ax[1])
    # ax[1].set_aspect('equal')
    # fig.set_size_inches(fig_width_double, fig_height_single)
    fig1.set_size_inches(fig_width_double, fig_height_single)
    fig1.tight_layout()
    # fig2.set_size_inches(fig_width_single, fig_width_single*1.1)
    if fp is not None:
        fig1.savefig(fp)
    # if fp2 is not None:
    #      fig2.savefig(fp2)


def CNOTs_coupler(fn="coupler-fid-bars.pdf"):
    fp_100 = "fields/g350_100S_3q_4000ns_8000step"
    fp_81 = "fields/g344_81S_3q_4000ns_8000step"
    fp_nospec = "fields/g337_100S_3q_4000ns_8000step"

    ax = plt.subplot()
    coupler_fidelity_bars(fp_nospec, ax, simulate_spectators=False)

    fig = ax.get_figure()
    fig.set_size_inches(5.5, 2.5)
    fig.tight_layout()
    fig.savefig(folder + fn)


def CNOTs_1P_2P(fn="fidbars-1P-2P.pdf"):
    fp = "fields/g354_69S_2q_2500ns_8000step"
    fids = get_fids_from_fp(
        fp,
        simulate_spectators=True,
        Grape=GrapeESR,
        A_spec=pt.tensor([get_A(1, 1)], device=default_device),
    )
    ax = plt.subplot()
    fidelity_bar_plot(fids, ax, f=[0.999999, 0.99999, 0.9999])
    delta = max(fids) - min(fids)
    ax.set_ylim(min(fids) - 0.5 * delta, 1 + 0.2 * delta)
    # ax.get_figure().set_size_inches


def plot_1P_1P_placement(fn="placement-1P-1P.pdf", ax=None):
    if ax is None:
        ax = plt.subplot()

    padding = 0
    x_range = [0, 40]
    y_range = [0, 9]
    all_sites = get_all_sites(x_range, y_range, 0)
    plot_spheres(all_sites, ax=ax, alpha=0.1)
    plot_6_sites(2, 4, ax=ax, color="orange")
    # plot_6_sites(37,4,ax=ax, color = 'orange')
    # plot_6_sites(42,4,ax=ax, color = 'orange')
    plot_6_sites(36, 2, ax=ax, color="orange")
    plot_6_sites(36, 6, ax=ax, color="orange")

    incorp_sites = [[2, 4], [38, 3], [37, 7]]
    plot_spheres(incorp_sites, color="red", ax=ax)

    ax.annotate("1P", [2, 1], fontsize=20)
    ax.annotate("2P", [32, 1], fontsize=20)

    ax_length = 5
    ax_origin = np.array([15, 2])
    ax_font_size = 12
    ax.arrow(
        *ax_origin,
        0,
        ax_length,
        shape="full",
        lw=1,
        length_includes_head=True,
        head_width=0.2,
    )
    ax.arrow(
        *ax_origin,
        ax_length,
        0,
        shape="full",
        lw=1,
        length_includes_head=True,
        head_width=0.2,
    )
    ax.annotate("[1,-1,0]", ax_origin + np.array([-0, -1.3]), fontsize=ax_font_size)
    ax.annotate(
        "[1,1,0]", ax_origin + np.array([-1.6, 0.7]), rotation=90, fontsize=ax_font_size
    )

    ax.set_xlim(x_range[0] - padding, x_range[1] + padding)
    ax.set_ylim(y_range[0] - padding, y_range[1] + padding)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.set_size_inches(7, 2)
    fig.savefig(folder + fn)


def multi_sys_2e_flip(
    fp=None,
    grape_fp1="fields/g378_69S_3q_6974ns_8000step",
    grape_fp3="fields/g379_69S_3q_5983ns_8000step",
):
    kwargs_2P_EE = get_2P_EE_swap_kwargs()

    fids1, fields = get_fids_and_field_from_fp(
        grape_fp1, get_from_grape=False, **kwargs_2P_EE
    )

    fig, ax = plt.subplots(2, 1)
    # fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3,  2]})

    n_1P_2P = 69
    sys_1P_2P = np.linspace(1, n_1P_2P, n_1P_2P)
    sys_spec = np.array([1])
    div = 4
    systems_ax = np.concatenate(
        (
            sys_1P_2P,
            sys_1P_2P + n_1P_2P + div,
            sys_spec + 2 * n_1P_2P + 2 * div,
            sys_spec + 2 * n_1P_2P + 2 * div + 1,
        )
    )

    fidelity_bar_plot(
        fids,
        systems_ax=systems_ax,
        f=[0.9999, 0.999, 0.99],
        ax=ax[1],
        colours=["green", "orange", "red"],
    )
    ax[1].set_ylim(0.99, 1.006)
    ax[1].set_yticks([0.99, 0.995, 1.00], ["99.0", "99.5", "100"])
    ax[1].set_ylabel("Fidelity (%)")
    ax[1].set_xticks([1, n_1P_2P, 2 * n_1P_2P + div], [1, n_1P_2P, 2 * n_1P_2P])
    axt = ax[0].twinx()

    Bx, By, T = fields
    xcol = color_cycle[0]
    ycol = color_cycle[1]
    ax[0].plot(T / unit.ns, 1e3 * Bx / unit.mT, color=xcol)
    axt.plot(T / unit.ns, 1e3 * By / unit.mT, color=ycol)
    yticks = [-0.3, 0.3]
    axt.set_yticks(yticks, yticks, color=ycol)
    ax[0].set_yticks(yticks, yticks, color=xcol)
    ax[0].set_ylim([-0.8, 0.3])
    axt.set_ylim([-0.3, 0.8])
    ax[0].set_ylabel("Bx (mT)", color=xcol)
    ax[0].set_xlabel("time (ns)")

    y_offset = 0.61
    label_axis(
        ax[1],
        f"$T_{Downarrow}^{{2e}}$",
        x_offset=0.2,
        y_offset=y_offset,
    )
    label_axis(
        ax[1],
        f"$T_{Uparrow}^{{2e}}$",
        x_offset=0.65,
        y_offset=y_offset,
    )
    label_axis(
        ax[1],
        f"$T_{{{Downarrow}, {Uparrow}}}^{{1e}}$",
        x_offset=0.9,
        y_offset=y_offset,
    )
    axt.set_ylabel("By (mT)", color=ycol)
    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp)


def get_2e_flip_fig():
    show_2e_flip(fp=f"{folder}{'e-spin-flip.pdf'}")


def get_2e_entangle_fig():
    show_2P_1P_CX_pulse(fp=f"{folder}{'coupler-target-CX-pulse.pdf'}")


def generate_system_table(
    A_2P=-2 * get_A_1P_2P(70)[:, 1], J=4 * get_J_1P_2P(70), ncols=7
):
    """
    Generates latex code for table of A and J data.
    """
    n = len(A_2P)

    nrows = n // ncols

    for i in range(nrows):
        for j in range(ncols):
            k = j * nrows + i
            print(
                f"{k+1} &{real(A_2P[k]/unit.MHz):.1f} &{real(J[k]/unit.MHz):.1f}",
                end="",
            )
            if j != ncols - 1:
                print(" &", end="")
            else:
                print("\\\\")


def plot_lattice_sites(
    fp=None,
    dist=51,
    separation_2P=6,
    orientation=0,
    y0=-1,
    y1=6,
    ax_ticks=False,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    x0 = 0
    x1 = dist

    padding = 1
    all_sites = get_all_sites([-1, dist], [y0 - 1, y1], padding=padding)
    plot_spheres(all_sites, ax=ax, alpha=0.1)
    ax.set_xlim([x0 - padding - 0.5, dist + padding + 0.5])
    ax.set_ylim([y0 - padding - 0.5, y1 + padding + 0.5])
    ax.set_aspect("equal")
    sites_color = "#1DA4BF"
    alpha = 0.5
    # plot_9_sites(0, 0, ax=ax, neighbour_color=sites_color)

    linewidth = 1.5
    fontsize = 13

    d = 1
    delta_x = dist - 4
    delta_y = 0

    alpha = 0.5

    sites_colour = FigureColours.sites_colour
    sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites_1P_2P(
        delta_x, delta_y, d=d, separation_2P=separation_2P, orientation=orientation
    )
    for sites in [sites_2P_lower, sites_2P_upper, sites_1P]:
        for j, site in enumerate(sites):
            sites[j] = (site[0], site[1] - 1)
    plot_spheres(sites_2P_lower, color=sites_colour, alpha=alpha, ax=ax)
    plot_spheres(sites_2P_upper, color=sites_colour, alpha=alpha, ax=ax)
    plot_spheres(sites_1P, color=sites_colour, alpha=alpha, ax=ax)

    s2L = sites_2P_lower[5]
    s2U = sites_2P_upper[3]
    s1 = sites_1P[0]

    # ax.plot([s2L[0], s2U[0]], [s2L[1], s2U[1]], color='black', linestyle='dotted', label='Hyperfine separation')
    # ax.plot([(s2L[0]+s2U[0])/2, s1[0]], [(s2L[1]+s2U[1])/2, s1[1]], color='black', linestyle='dashed', label='Exchange separation')

    plot_spheres([s2U], color="red", ax=ax, zorder=3)
    plot_spheres([s2L], color="red", ax=ax, zorder=3)
    plot_spheres([s1], color="red", ax=ax, zorder=3)

    # ax.plot([10,10], [5,6])
    # ax_length=5
    # ax_origin = np.array([11,1])
    # ax.arrow(*ax_origin, 0, ax_length, shape='full', lw=1.2, length_includes_head=True, head_width=0.4)
    # ax.arrow(*ax_origin, ax_length, 0, shape='full', lw=1.2, length_includes_head=True, head_width=0.4)
    # ax.annotate('[1,-1,0]', ax_origin + np.array([-0,-1.5]))
    # ax.annotate('[1,1,0]', ax_origin + np.array([-1.8,0.3]), rotation=90)

    if orientation == 0:
        ax.set_xlabel("[1,1,0]")
        ax.set_ylabel("[-1,1,0]")
    else:
        ax.set_xlabel("[1,-1,0]")
        ax.set_ylabel("[1,1,0]")
    if ax_ticks:
        ax.set_yticks([-2, 2, 6])
        ax.set_yticklabels([0, 4, 8])
        ax.set_xticks(np.linspace(1, dist, 5).astype(int) + 1)
        ax.set_xticklabels(np.linspace(1, dist, 5).astype(int))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    # ax.legend(loc='upper center')
    # ax.axis('off')


def show_configs(fp=None):
    fig, ax = plt.subplots(2, 1)
    dist = 21
    separation_2P = 2
    plot_lattice_sites(dist=dist, separation_2P=separation_2P, ax=ax[0], y1=3)
    plot_lattice_sites(
        dist=dist, separation_2P=separation_2P, ax=ax[1], y1=3, y0=-2, orientation=1
    )

    fontsize = 12

    x_1P = 0.68
    y_1P = 0.33

    x_2P_1 = 0.16
    y_2P_1 = 0.0
    x_2P_2 = 0.2
    y_2P_2 = 0.38
    label_axis(ax[0], "2P", x_offset=x_2P_1, y_offset=y_2P_1, fontsize=fontsize)
    label_axis(ax[0], "1P", x_offset=x_1P, y_offset=y_1P, fontsize=fontsize)
    label_axis(ax[1], "2P", x_offset=x_2P_2, y_offset=y_2P_2, fontsize=fontsize)
    label_axis(ax[1], "1P", x_offset=x_1P, y_offset=y_1P, fontsize=fontsize)

    fig.set_size_inches(8 / 2.54, 7 / 2.54)
    if fp is not None:
        fig.savefig(fp)


def single_CX_for_many_exchanges():
    J_1_9 = pt.linspace(1, 9, 9)
    J = pt.cat((J_1_9, 10 * J_1_9, 100 * J_1_9))

    J = 2.5 * unit.MHz
    lam = 0
    kappa = 1e2

    print(run_CNOTs(
        1500 * unit.ns,
        2000,
        J=pt.tensor(J, dtype=cplx_dtype),
        A=get_A_1P_2P(1),
        save_data=False,
        lam=lam,
        kappa=kappa,
        verbosity=-1,
    ).get_opt_state())


if __name__ == "__main__":
    # parallel_1P_1P_CNOTs()
    # draw_HS_architecture(fp=f'{folder}HS-distance-5.pdf')
    # draw_HS_unit_and_side_view(fp=f'{folder}architecture-unit-cell-and-side.pdf')
    # HS_array_configs(fp=f'{folder}voltage-configs.pdf')
    # coupler_CNOTs()
    # CNOTs_1P_2P()
    # plot_1P_1P_placement()
    # multi_sys_2e_flip(fp=f"{folder}multi-sys-2e-flip.pdf")
    # HS_side_view()
    # get_2e_flip_fig()
    # get_2e_entangle_fig()
    # multi_2P_1P_CX(f"{folder}multi-sys-2P-1P.pdf")
    # all_multi_system_pulses(
    #     fp=f"{folder}/all-multi-sys-pulses.pdf",
    #     grape_fp1="fields/g399_70S_3q_4991ns_8000step",
    #     grape_fp2="fields/g392_70S_2q_3000ns_8000step",
    #     grape_fp3="fields/g409_70S_3q_3966ns_8000step",
    # )
    # small_MW_1_3("fields/c1326_1S_3q_479ns_1000step", fp=f"{folder}MW1-single.pdf")
    # small_MW_1_3("fields/c1350_1S_3q_479ns_2500step", fp=f"{folder}MW3-single.pdf")
    # single_systems(
    #     fp1 = 'fields/c1359_1S_3q_479ns_1000step',
    #     fp2 = 'fields/c1366_1S_2q_500ns_1000step',
    #     fp3="fields/c1354_1S_3q_479ns_2500step", fp=f"{folder}single-system-pulses.pdf"
    # )
    # generate_system_table()
    # show_configs(f"{folder}1P-2P-configs.pdf")

    single_CX_for_many_exchanges()
    plt.show()
