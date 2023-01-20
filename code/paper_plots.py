


from GRAPE import GrapeESR, load_grape
from architecture_HS_1P_1P import HS_side_view, make_HS_array, HS_array_configs
from visualisation import *
from utils import *


folder = 'paper-plots/'


def parallel_1P_1P_CNOTs():

    fig,ax = plt.subplots(2,1)
    fig.set_size_inches(5,3.7)
    ax0_twinx = ax[0].twinx()
    ax[0].set_ylim([-0.7,1.5])
    ax0_twinx.set_ylim([-1.5,0.7])

    grape_fp = 'fields/c965_15S_2q_2000ns_2500step'
    grape = load_grape(fp = grape_fp, minus_phase=True)
    
    grape.print_result()

    fids, grad = grape.fidelity()
    grape.plot_XY_fields(ax[0], twinx=ax0_twinx, legend_loc=False)
    fidelity_bar_plot(fids, ax=ax[1], f=[0.999, 0.998], colours=['green', 'orange'], legend_loc='upper center')
    ax[1].set_ylim([0.998, 1.0007])
    ax[1].set_yticks([0.998, 0.999, 1], ['99.8', '99.9', '100'])
    ax[1].axhline(1, linestyle='--', linewidth=1, color='black')
    ax[1].set_xticks([0, 4, 9, 14], ['1','5','10','15'])

    fig.tight_layout()
    fig.savefig(f'{folder}parallel-1P-1P.pdf')


def draw_HS_architecture(fp=None):

    fig,ax = plt.subplots(1,1)
    make_HS_array(ax, 5)
    if fp is not None: fig.savefig(fp)


def draw_HS_unit_and_side_view(fp=None):
    fig1,ax = plt.subplots(1,2)
    make_stab_unit_cell(ax[0], fontsize=15)
    HS_side_view(ax[1])
    #ax[1].set_aspect('equal')
    #fig.set_size_inches(fig_width_double, fig_height_single)
    fig1.set_size_inches(fig_width_double, fig_height_single)
    fig1.tight_layout()
    #fig2.set_size_inches(fig_width_single, fig_width_single*1.1)
    if fp is not None:
            fig1.savefig(fp)
    # if fp2 is not None:
    #      fig2.savefig(fp2)





if __name__ == '__main__':
    parallel_1P_1P_CNOTs()
    #draw_HS_architecture(fp=f'{folder}HS-distance-5.pdf')
    #draw_HS_unit_and_side_view(fp=f'{folder}architecture-unit-cell-and-side.pdf')
    #HS_array_configs(fp=f'{folder}voltage-configs.pdf')
    plt.show()






