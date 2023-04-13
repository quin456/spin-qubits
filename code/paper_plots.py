from GRAPE import GrapeESR, load_grape
from architecture_2P_coupler import HS_side_view, make_HS_array
from visualisation import *
from utils import *
from run_couplers import coupler_fidelity_bars
from run_grape import get_fids_and_field_from_fp
from run_n_entangle import get_2P_EE_swap_kwargs


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


def control_coupler_E_flip(fp="fields/g366_138S_2q_6000ns_8000step"):

    kwargs_2P_EE = get_2P_EE_swap_kwargs()

    fids, fields = get_fids_and_field_from_fp(fp, **kwargs_2P_EE)

    fig, ax = plt.subplots(2, 1)
    # fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3,  2]})
    fidelity_bar_plot(fids, f=[0.9999, 0.999, 0.99], ax=ax[1])
    ax[1].set_ylim(0.99, 1.002)
    ax[1].set_yticks([0.99, 0.995, 1.00])
    ax[1].set_xticks([0, 137], [1, 138])
    axt = ax[0].twinx()

    Bx, By, T = fields
    xcol = color_cycle[0]
    ycol = color_cycle[1]
    ax[0].plot(T / unit.ns, 1e3 * Bx / unit.mT, color=xcol)
    axt.plot(T / unit.ns, 1e3 * By / unit.mT, color=ycol)
    ax[0].set_yticks([-0.3, 0.8], color=xcol)
    ax[0].set_ylabel('Bx (mT)', color=xcol)
    ax[0].set_xlabel("time (ns)")
    axt.set_ylabel('By (mT)', color=ycol)
    axt.set_yticks([-0.8, 0.3], color=ycol)
    fig.tight_layout()


if __name__ == "__main__":
    # parallel_1P_1P_CNOTs()
    # draw_HS_architecture(fp=f'{folder}HS-distance-5.pdf')
    # draw_HS_unit_and_side_view(fp=f'{folder}architecture-unit-cell-and-side.pdf')
    # HS_array_configs(fp=f'{folder}voltage-configs.pdf')
    # coupler_CNOTs()
    # CNOTs_1P_2P()
    # plot_1P_1P_placement()
    #control_coupler_E_flip()
    HS_side_view()
    plt.show()

