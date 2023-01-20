

import torch as pt
import matplotlib

if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure 
import numpy as np 

from GRAPE import GrapeESR, GrapeESR_AJ_Modulation, load_grape
import gates as gate
from utils import psi_from_polar, normalise, label_axis
from eigentools import get_resonant_frequencies, lock_to_frequency
from hamiltonians import get_pulse_hamiltonian, get_U0
from visualisation import plot_psi
from visualisation import *
from single_spin import show_single_spin_evolution
from data import dir, cplx_dtype, gamma_e, gamma_n, J_100_14nm, J_100_18nm, get_A, get_J
import atomic_units as unit
from architecture_design import *
from electrons import plot_free_electron_evolution, get_free_electron_evolution
from transition_visualisation import visualise_E_transitions
from single_NE import *
from multi_NE import *
from misc_calculations import *
from voltage_plot import plot_CNOTs
from electrons import investigate_3E_resfreqs
from single_spin import *


import qiskit
from qiskit.visualization import plot_bloch_vector

plots_folder = f"{dir}thesis-plots/"

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'
rangle = '⟩'

max_time = 10

xoff_half_long = -0.08 
yoff_half_long = -0.15


################################################################################################################
################        ALL PLOTS TO BE USED IN THESIS WILL BE GENERATED HERE        ###########################
################################################################################################################




################################################################################################################
################        CHAPTER 1 INTRODUCTION        ##########################################################
################################################################################################################

def plot_rotation(psi_cartesian, angle0, d_angle, ax, zorder=3):
    rho = 0.2
    r=psi_cartesian.copy(); r[0]=psi_cartesian[1]; r[1] = -psi_cartesian[0]
    x,y,z = r
    
    rmid = r/2 
    v1 = np.array([-rmid[1], rmid[0], 0]); v1=v1/np.sqrt(np.dot(v1,v1))
    v2 = np.array([0, rmid[2], -rmid[1]])
    
    # Gram Schmidt
    v2 = v2 - np.dot(v2,v1)*v1 
    v2 = v2 / np.sqrt(np.dot(v2,v2))

    p1 = v1+rmid
    p2 = v2+rmid
    print(f"rmid={rmid}, v1={v1}, t1 = {p1}")
    #ax.plot([p1[0],rmid[0]], [p1[1],rmid[1]], [p1[2],rmid[2]])
    #ax.plot([p2[0],rmid[0]], [p2[1],rmid[1]], [p2[2],rmid[2]])

    n = 50
    rho = 0.2
    theta = np.linspace(angle0,angle0+d_angle,n)
    
    arc = np.einsum('i,a->ia',np.ones(n),rmid) + np.einsum('i,a->ia',rho*np.cos(theta),v1) + np.einsum('i,a->ia',rho*np.sin(theta),v2)


    ax.plot(arc[:,0], arc[:,1], arc[:,2], zorder=zorder, linewidth=3)



def bloch_sphere_XZH(psi, fp=None):

    psi = pt.tensor([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1.001, 0.001],
        [np.cos(np.pi/8), np.sin(np.pi/8)]
    ])
    psi_cartesian = psi_to_cartesian(psi).numpy()

    fig = plt.figure()
    ax = []
    ax.append(fig.add_subplot(1, 3, 1, projection='3d'))
    ax.append(fig.add_subplot(1, 3, 2, projection='3d'))
    ax.append(fig.add_subplot(1, 3, 3, projection='3d'))
    
    plot_bloch_vector(psi_cartesian[0], ax=ax[0])
    plot_bloch_vector(psi_cartesian[1], ax=ax[1])
    plot_bloch_vector(psi_cartesian[2], ax=ax[2])
    plot_rotation(psi_cartesian[0], 0, np.pi, ax[0])
    plot_rotation(psi_cartesian[1], -np.pi, np.pi, ax[1])
    plot_rotation(psi_cartesian[2], -np.pi/2, np.pi, ax[2])

    z_offset = -0.5
    x_offset = 0.15
    label_axis(ax[0], 'X', projection='3D', x_offset=x_offset, z_offset = z_offset)
    label_axis(ax[1], 'Z', projection='3D', x_offset=x_offset, z_offset = z_offset)
    label_axis(ax[2], 'H', projection='3D', x_offset=x_offset, z_offset = z_offset)


    fig.set_size_inches(1.2*fig_width_double, 1.2*fig_height_single)
    if fp is not None: plt.savefig(fp)


def energy_level_picture(H0, state_labels=None, energy_labels=None, colors=color_cycle, ax=None, fp=None, ax_label=None):

    S,D = get_ordered_eigensystem(H0)
    energies = pt.real(pt.diagonal(D))

    x0=0.05
    y0=(max(energies)-min(energies))/3 *0.05

    if ax is None: 
        fig,ax = plt.subplots(1)
        fig.set_size_inches(2.5, 4.5)

    for j,energy in enumerate(energies):
        ax.axhline(energy, 0, 0.84, color=colors[j%len(colors)])
        if state_labels is not None:
            ax.annotate(state_labels[j], (0.85, energy-0.08))
        if energy_labels is not None:
            ax.annotate(energy_labels[j], (x0,energy+y0))

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')

    ax.set_ylim(min(energies)-3*y0, max(energies)+6*y0)
    ax.set_xlim(0, 1)
    
    if ax_label is not None:
        ax.annotate(ax_label, [-0,min(energies)])
    if fp is not None:
        plt.savefig(fp)
    

def two_electron_energy_level_picture(J=get_J(1,2), Bz=0.05*unit.T, ax=None, fp = None, detuning=True):

    if detuning:
        #labels = [f"$\omega_z+J$", f"$2A-J$",  f"$-2A-J$", f"$-\omega_z+J$"]
        labels = ["$|T_+⟩$", "$|T_0⟩$", "$|S_0⟩$", "$|T_-⟩$"]
        A=get_A(1,2)
    else:
        #labels = [f"$\omega_z+2A+J$", f"$J$",  f"-J", f"$-\omega_z-2A+J$"]
        labels = ["$|T_+⟩$", "$|T_0⟩$", "$|S_0⟩$", "$|T_-⟩$"]
        A=get_A(1,2, NucSpin=[1,1])
    H0 = get_H0(A*5,J,Bz=Bz)

    energy_level_picture(H0, state_labels=None, energy_labels=labels, ax=ax, fp=fp, colors = ['blue', 'green', 'green', 'red'])




def NE_energy_level_picture(ax=None, fp=None):
    

    energies = [10, 0, 9, 3]
    energy_labels = [f"$A + (\omega_e-\omega_n)/2$", f"$-A - (\omega_e-\omega_n)/2$", f"$-A + (\omega_e+\omega_n)/2$" , f"$A - (\omega_e+\omega_n)/2$"]
    spin_labels = [f"{Uparrow}{uparrow}", f"{Uparrow}{downarrow}",f"{Downarrow}{uparrow}",f"{Downarrow}{downarrow}"]
    energy_level_picture(energies, spin_labels, energy_labels, ax=ax)

    plt.arrow(0.7,energies[2], 0, energies[3]-energies[2], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[3], 0, energies[2]-energies[3], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[1], 0, energies[3]-energies[1], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')
    plt.arrow(0.7,energies[3], 0, energies[1]-energies[3], head_width=0.02, head_length=0.2, length_includes_head=True, color='red')

    plt.annotate("$\omega_e$",(0.71,energies[2] + 0.5*(energies[3]-energies[2])))
    plt.annotate("$\omega_n$",(0.71,energies[1] + 0.5*(energies[3]-energies[1])))

    if fp is not None:
        plt.savefig(fp)

def triple_E_energy_level_picture(ax,A, J, ax_label=None):
    #energies = [10, 7.3,7,6.7,3.3,3,2.7,0]
    H0=get_H0(A,J,Bz=0.04*unit.T)
    colors = ['blue', 'green', 'green', 'green', 'orange', 'orange', 'orange', 'red']
    spin_labels = [f"$|000{rangle}$", f"$a|001{rangle}+b|010{rangle}+c|100{rangle}$", "","", f"$a|011{rangle}+b|101{rangle}+c|110{rangle}$", "", "", f"|111{rangle}"]
    energy_level_picture(H0, energy_labels=spin_labels, colors=colors, ax=ax, ax_label=ax_label)

#f"$\alpha|001{rangle}+\beta|010{rangle}+\gamma|100{rangle}$"
def free_2E_evolution(fp=None):
    A = get_A(1,2); J=get_J(1,2); tN=10*unit.ns
    H0 = get_H0(A,J)
    U0 = get_U0(H0, tN, 1000)
    psi0 = pt.tensor([0.1, 0.3, 0.7, 0.2], dtype=cplx_dtype); normalise(psi0)
    psi0 = normalise(pt.tensor([0.5, 0, 0.5, 0], dtype=cplx_dtype))
    psi = U0@psi0

    plot_psi(psi, tN, squared=True, fp=fp)

def exchange_label_getter(i):
    if i in [1,2,4]:
        return f"Pr({np.binary_repr(i,3)})"

def compare_free_3E_evols(fp=None):
    '''
    Compares 3E evolution for 101 vs 001 nuclear spin configurations
    '''
    tN = 10*unit.ns 
    N = 500
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(fig_width_double_long, fig_height_single_long)


    plot_free_electron_evolution(tN, N, get_A(1,3,[0,1,0]), get_J(50,3, J1=J_100_14nm, J2=J_100_18nm)[15], ax=ax[0], label_getter=exchange_label_getter)
    plot_free_electron_evolution(tN, N, get_A(1,3,[0,0,1]), get_J(1,3), ax=ax[1], label_getter=exchange_label_getter)
    plt.tight_layout()
    if fp is not None:
        fig.savefig(fp)

def show_2E_Hw(J,A, tN, N, fp=None):
    H0 = get_H0(A,J)
    rf = get_resonant_frequencies(H0, gate.get_Xn(2)+gate.get_Yn(2))
    T = pt.linspace(0,tN,N)
    w = rf[1]
    Bx=pt.cos(w*T); By=pt.sin(w*T)
    Hw =  get_pulse_hamiltonian(Bx, By, gamma_e, gate.get_Xn(2), gate.get_Yn(2))
    S,D = get_ordered_eigensystem(get_H0(A,J))
    U0 = get_U0(H0, tN, N)
    visualise_Hw(S.T@dagger(U0)@Hw@U0@S, tN)

def intro_and_review(chapter='intro-'):


    #bloch_sphere_XZH(psi_from_polar(np.pi/4,0), fp = f'{plots_folder}Ch1-bloch-sphere.pdf')
    #show_single_spin_evolution(tN=100*unit.ns, fp = f"{plots_folder}Ch1-analytic-example.pdf")
    #two_electron_energy_level_picture(fp=f"{plots_folder}Ch1-2E-energy-levels.pdf")
    run_single_electron_grape(f"{plots_folder}{chapter}GRAPE-example.pdf")

def HHex(chapter='H-Hex-'):
    def compare_symmetry(fp=None):
        fig,ax = plt.subplots(1)
        dist=18
        sq_centre = np.array([dist,0])
        diag_centre = np.ceil(np.array([dist/np.sqrt(2), dist/np.sqrt(2)]))
        hex_centre = np.ceil(np.array([dist*np.sqrt(3)/2, dist*1/2]))
        all_sites = get_all_sites([0,dist], [2,dist/np.sqrt(2)-2], padding=dist//5)
        plot_spheres(all_sites, ax=ax, alpha=0.1)
        sites_color='#1DA4BF'
        alpha = 0.5
        plot_9_sites(0, 0, ax=ax, neighbour_color=sites_color)
        plot_9_sites(*sq_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
        plot_9_sites(*diag_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
        plot_9_sites(*hex_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)

        linecolor_15='black'
        linecolor_25 = 'darkred'
        linewidth = 1.5
        fontsize = 13
        ax.plot([1.5, sq_centre[0]-1.5], [0,sq_centre[1]], color=linecolor_15, linewidth=linewidth)
        ax.plot([np.sqrt(2), diag_centre[0]-np.sqrt(2)], [np.sqrt(2),diag_centre[1]-np.sqrt(2)], color=linecolor_15, linewidth=linewidth)
        ax.plot([1.5, hex_centre[0]-1.5], [np.sqrt(3)/2,hex_centre[1]-np.sqrt(3)/2], color=linecolor_25, linewidth=linewidth)

        n=100
        ang_adjust = np.arctan(hex_centre[1]/hex_centre[0])
        theta1 = np.linspace(0,ang_adjust, n)
        theta2 = np.linspace(0,np.pi/4, n)
        r1 = 12
        r2 = 7
        ax.plot(r1*np.cos(theta1), r1*np.sin(theta1), color=linecolor_25)
        ax.plot(r2*np.cos(theta2), r2*np.sin(theta2), color=linecolor_15)

        ax_length=3
        ax_origin = np.array([-1,7])
        ax.arrow(*ax_origin, 0, ax_length, shape='full', lw=1, length_includes_head=True, head_width=0.2)
        ax.arrow(*ax_origin, ax_length, 0, shape='full', lw=1, length_includes_head=True, head_width=0.2)
        ax.annotate('[1,-1,0]', ax_origin + np.array([-0,-1.3]))
        ax.annotate('[1,1,0]', ax_origin + np.array([-1.6,-0.5]), rotation=90)

        ax.annotate('0°', sq_centre + [-3.5,0.45], fontsize=fontsize)
        ax.annotate('45°', r2*diag_centre/dist + [-1.05,0.55], fontsize=fontsize)
        ax.annotate('30°', r1*hex_centre/dist + [-0.95,0.6], fontsize=fontsize, color=linecolor_25)
        fig.set_size_inches(1.4*fig_width_single, 1.2*fig_height_single)
        if fp is not None:
            plt.savefig(fp)

    def save_single_cell(fp):
        fig,ax = plt.subplots(1,1)
        plot_single_cell(ax)
        ax.set_xlabel('[100] (nm)')
        ax.set_ylabel('[010] (nm)')
        fig.set_size_inches(1.2*fig_width_single, 1.2*fig_height_single)
        fig.tight_layout()
        fig.savefig(fp)
        
    def allowed_transitions(A=get_A(1,3), J=get_J(1,3),fp=None, labels=None):
        fig,ax = plt.subplots(1,2)
        visualise_E_transitions(A=A, J=J, ax=ax[0])
        visualise_E_transitions(A=pt.zeros_like(A), J=J, ax=ax[1])
        fig.set_size_inches(8, 4)
        labels = ['$A_1=-A_2=A_3$', '$A_1=A_2=A_3$']
        if labels is not None:
            label_axis(ax[0],labels[0], 0.3, -0.1)
            label_axis(ax[1],labels[1], 0.4,-0.1)
        if fp is not None:
            fig.savefig(fp)

    def coupler_225_attempt(fp = None, grape_fp='fields/g243_225S_3q_5000ns_5000step'):
        grape = load_grape(grape_fp, minus_phase=False)
        fig, ax = plt.subplot_mosaic([['upper left', 'upper right'], ['lower', 'lower']], gridspec_kw={'height_ratios':[1,0.6]})
        div = len(grape.cost_hist)//5
        cost_24hr_idx = np.argmax(grape.cost_hist[div:])+div
        grape.cost_hist = grape.cost_hist[:cost_24hr_idx-180] + grape.cost_hist[cost_24hr_idx+180:]
        ax['upper right'].axvline(cost_24hr_idx, linestyle='--', color='red', label='24 hr mark')
        grape.plot_cost_hist(ax['upper right'])

        grape.plot_fidelity(ax['upper left'], all_fids=False)
        fidelity_bar_plot(grape.fidelity()[0], ax['lower'], colours=['green', 'orange', 'red'], f=[0.999, 0.99], legend_loc='upper left')
        fig.set_size_inches(fig_width_double*1.08, fig_height_single*1.3)
        ax['lower'].set_yticks([0.9,1])
        ax['lower'].set_xticks([0,19,39,59,80])
        ax['lower'].set_xticklabels([1,20,40,60,81])
        fig.set_size_inches(fig_width_double, fig_height_double_long)
        ax['lower'].set_ylim([0.90,1+0.4*0.02])
        fig.tight_layout()
        nuclear_spin_tag(ax['upper left'], [1,0,1], text = '225x', dx=100, dy=0.3, dx_text=180)
        grape.print_result()
        if fp is not None: fig.savefig(fp)

    def log_cost(grape_fp):

        cost_hist = list(pt.load(f"{grape_fp}_cost_hist"))
        grape = GrapeESR(get_J(1,2), get_A(1,2),10,10, cost_hist=cost_hist)
        ax = plt.subplot()
        grape.plot_cost_hist(ax)


    #plot_cell_array(4,4, filename=f"{plots_folder}{chapter}cell_array.pdf")
    #generate_CNOTs(fp = f"{plots_folder}{chapter}CNOT-voltage-cells.pdf")
    #generate_2_coupler_conditions(fp = f"{plots_folder}{chapter}coupler_V_configs.pdf")
    #plot_CNOTs(fp=f"{plots_folder}{chapter}CNOT-voltage-schedule.pdf")
    #plot_annotated_cell(filename=f"{plots_folder}{chapter}single_cell.pdf")
    #save_single_cell(f"{plots_folder}{chapter}single_cell.pdf")
    #numbered_qubits_cell()
    #compare_symmetry(fp = f"{plots_folder}{chapter}symmetry-comparison.pdf")
    #allowed_transitions(fp=f'{plots_folder}{chapter}allowed_triple_E_transitions.pdf')
    fp_225 = 'fields/g248_225S_3q_5000ns_5000step'
    fp_225 = 'fields/g251_225S_3q_5000ns_5000step'
    fp_225 = 'fields/g264_225S_3q_5000ns_5000step'
    fp_225 = 'fields/g256_81S_3q_5000ns_5000step'
    fp_15 = 'fields/g235_15S_2q_2000ns_5000step'

    coupler_225_attempt(fp = f"{plots_folder}{chapter}coupler_225.pdf",grape_fp=fp_225)
    #log_cost(grape_fp='fields/g248_225S_3q_5000ns_5000step')
    #visualise_frequency_overlap(f'{plots_folder}{chapter}frequency-overlap.pdf')
    plt.show()



def pulse_design(chapter="Ch3-"):

    def grape_1s2q(fp=None, fp1=None, fp2=None, prev_grape_fp = None):
        if prev_grape_fp is None:
            grape = GrapeESR(J=get_J(1,2), A=get_A(1,2),tN=20*unit.ns, N=500, max_time=None, verbosity=2)
            grape.run()
            grape.save()
        else:
            grape = load_grape(prev_grape_fp)
        fig,ax = plt.subplots(1,2, gridspec_kw={'width_ratios':[1,0.35]})
        fig1, ax1 = plt.subplots(1,2)
        fig2, ax2 = plt.subplots(1,2)
        grape.plot_u(ax[0], legend_loc = 'lower center')
        ax[0].set_ylim(-0.7,0.5)
        grape.plot_cost_hist(ax[1])
        grape.plot_field_and_fidelity(fp1, ax=ax1)
        grape.plot_psi_with_phase(ax2, phase_legend_loc='upper center', amp_legend_loc='upper right', legend_cols=2)
        ax2[0].set_ylim([0,1.3])
        ax2[0].set_yticks([0,1])
        ax2[1].set_yticks([-np.pi,0,np.pi])
        ax2[1].set_yticklabels(["$-\pi$",0,"$\pi$"])
        # label_axis(ax[0], "(a)", x_offset=-0.08, y_offset=-0.15)
        # label_axis(ax[1], "(b)", x_offset=-0.08, y_offset=-0.15)
        # label_axis(ax1[0], "(a)", x_offset=-0.08, y_offset=-0.15)
        # label_axis(ax1[1], "(b)", x_offset=-0.08, y_offset=-0.15)
        # label_axis(ax2[0], "(a)", x_offset=-0.08, y_offset=-0.15)
        # label_axis(ax2[1], "(b)", x_offset=-0.08, y_offset=-0.15)
        fig.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig1.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig2.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig.tight_layout()
        fig1.tight_layout()
        fig2.tight_layout()
        nuclear_spin_tag(ax1[0], [0,1], mult=1, dx=10)
        nuclear_spin_tag(ax[0], [0,1], mult=1, loc='center center', dx=-5, dy=-0.03)
        nuclear_spin_tag(ax2[0], [0,1], mult=1, loc='upper left', dy=-0.05)

        if fp is not None: fig.savefig(fp)
        if fp1 is not None: fig1.savefig(fp1)
        if fp2 is not None: fig2.savefig(fp2)

    def grape_no_rf():
        rf = pt.tensor([0], dtype=real_dtype, device=default_device)
        grape = GrapeESR(J=get_J(15,2), A=get_A(15,2), tN=1000*unit.ns, N=2500, rf=rf, max_time=2745)
        grape.run()
        grape.plot_result()

    def grape_1s3q(fp=None, grape_fp=None, lam=1e8, max_time=None):
        A = get_A(1, 3)
        J = get_J(1,3, J1=J_100_18nm, J2=J_100_18nm/3)
        
        if grape_fp is None:
            grape = GrapeESR(J, A, tN=400*unit.ns, N=1000, max_time=max_time, save_data=False, lam=0)
            grape.run()
            grape.save()
        else:
            grape = load_grape(grape_fp, GrapeESR)
        #grape.plot_result()
        fig,ax = plt.subplots(1,2)
        grape.plot_field_and_fidelity(ax=ax)
        grape.print_result()
        nuclear_spin_tag(ax[0], [1,0,1])
        if fp is not None: fig.savefig(fp)



    def NE_EN_CX(fp=f"{plots_folder}{chapter}NE_EN_CX.pdf"):
        fig,ax = plt.subplots(2,3, gridspec_kw={'width_ratios': [1.5, 0.5,  1]})
        show_NE_CX(get_A(1,1),2*unit.T, 1000, ax=ax[0], legend_loc = 'center right')
        show_EN_CX(get_A(1,1),2*unit.T, 400 , ax=ax[1], legend_loc = 'center right')
        fig.set_size_inches(fig_width_double_long, fig_height_double_long)
        fig.tight_layout()
        y_offset=-0.3
        x_offset_1 = -0.3
        label_axis(ax[0,0], '(a)', y_offset=y_offset)
        #label_axis(ax[0,1], '(b)', y_offset=y_offset, x_offset=x_offset_1)
        label_axis(ax[0,2], '(b)', y_offset=y_offset)
        label_axis(ax[1,0], '(c)', y_offset=y_offset)
        #label_axis(ax[1,1], '(e)', y_offset=y_offset, x_offset=x_offset_1)
        label_axis(ax[1,2], '(d)', y_offset=y_offset)
        ax[0,2].set_yticks([0,0.5])
        ax[1,2].set_yticks([0,0.5])
        fig.savefig(fp)

    def plot_swap_schedule(fp = f"{plots_folder}{chapter}swap_schedule.pdf"):
        fig,ax = plt.subplots(1)
        ax2 = ax.twinx()
        color='black'
        color2='red'


        B_e = 0.52
        B_n = 1
        t0=0

        div=1e3
        tn_CX = 10760.332 /div
        tn_wait = 171.237 /div
        te_CX = 34.188 /div
        te_wait = 59.790 /div

        ndigits=2
        ndigits1=2

        t0_tick = '0'
        t1_tick=str(round(te_CX,ndigits))
        t2_tick = str(round(te_CX+te_wait, ndigits))
        t3_tick = str(round(te_CX+te_wait+tn_CX, ndigits1))
        t4_tick = str(round(te_CX+te_wait+tn_CX+tn_wait, ndigits1))
        t5_tick = str(round(te_CX+te_wait+tn_CX+tn_wait+te_CX, ndigits1))
        t6_tick = str(round(te_CX+te_wait+tn_CX+tn_wait+te_CX+te_wait, ndigits1))
        ticks = [t0_tick,t1_tick,t2_tick,t3_tick,t4_tick,t5_tick,t6_tick]


        xn_CX = tn_CX/70
        xn_wait = tn_wait/1.9

        x0 = 0
        x1 = te_CX
        x2 = te_CX+te_wait
        x3 = te_CX+te_wait+xn_CX
        x4 = te_CX+te_wait+xn_CX+xn_wait
        x5 = te_CX+te_wait+xn_CX+xn_wait+te_CX
        x6 = te_CX+te_wait+xn_CX+xn_wait+te_CX+te_wait

        x = np.array([x0, x1, x2, x3, x4, x5, x6])
        B = [0,B_e,B_e,0, 0, B_n,B_n,0, 0, B_e, B_e, 0, 0]
        T = [x[0], x[0], x[1], x[1],x[2],x[2], x[3], x[3],x[4],x[4],  x[5],  x[5],x[6]]
        ax.plot(T,B, color=color)
        ax.set_xticks(x, ticks)
        ax.set_xlabel("time (µs)")
        ax.set_ylabel("$B_{ac}$ (mT)", color=color)
        ax.set_yticks([0,B_e,B_n], [0,B_e, B_n])


        w_n = -93.03
        w_e = 55990
        y1=0.2
        y2=0.8
        ax2.set_ylabel('$\omega/2\pi$ (MHz)', color=color2)
        ax2.set_yticks([y1,y2], [w_n, w_e])
        ax.set_ylim([-0.3,1.3])
        ax2.plot([x[0],x[1]], [y2,y2], color=color2)
        ax2.plot([x[2],x[3]], [y1,y1], color=color2)
        ax2.plot([x[4],x[5]], [y2,y2], color=color2)


        fig.set_size_inches(fig_width_double_long, fig_height_single_long)
        fig.tight_layout()

        fig.savefig(fp)

    def plot_NE_energy_diagram_and_alpha_beta(fp=None):
        fig,ax = plt.subplots(1,2)
        plot_NE_energy_diagram(Bz = pt.linspace(0,5, 100)*unit.mT, N=1000, ax=ax[0])
        plot_NE_alpha_beta(Bz = pt.linspace(0,5, 100)*unit.mT, N=1000, ax=ax[1])

        y_offset=-0.125
        label_axis(ax[0], '(a)', y_offset=y_offset)
        label_axis(ax[1], '(b)', y_offset=y_offset)
        fig.set_size_inches(1.15*fig_width_double,fig_height_single)
        fig.tight_layout()



        if fp is not None: fig.savefig(fp)

    def all_15_2q_CNOTs(fp1 = None,tN=2000*unit.ns, N=2500, max_time=1800, lam=1e7, prev_grape_fp=None):

        old_big_field_prev_grape_fp = 'fields/c877_15S_2q_200ns_500step'
        nS = 15; nq = 2
        A = get_A(nS, nq)
        J = get_J(nS, nq)
        if prev_grape_fp is None:
            grape = GrapeESR(J, A, tN, N, max_time=max_time, lam=0)
            grape.run()
            grape.save()
        else:
            grape = load_grape(prev_grape_fp, max_time=max_time, lam=lam)
            grape.print_result()

        #grape.plot_result()

        fig1, ax1 = plt.subplots(1,2)
        grape.plot_field_and_fidelity(None, ax1, fid_legend=False, all_fids=False)
        nuclear_spin_tag(ax1[1], [0,1], text = '15x 1P-1P', loc='center left', dx=100, dx_text=-80, dy_text=-0.05)
        fig1.savefig(fp1)


    #NE_energy_level_picture(fp=f"{plots_folder}Ch3-NE-energy-levels.pdf")
    #plot_NE_energy_diagram_and_alpha_beta(fp=f"{plots_folder}{chapter}NE-energy-diagram-alpha-beta.pdf")

    #free_2E_evolution(fp = f"{plots_folder}Ch3-2E-free-evolution.pdf")


    #grape_1s2q(fp = f"{plots_folder}Ch3-2E-u-and-cost.pdf", fp1 = f"{plots_folder}Ch3-2E-field-and-fidelity.pdf", fp2 = f"{plots_folder}Ch3-psi-evolution.pdf", prev_grape_fp='fields/c878_1S_2q_100ns_500step')
    #grape_no_rf()
    grape_1s3q(max_time=None, grape_fp='fields/c905_1S_3q_400ns_1000step', fp=f'{plots_folder}{chapter}3E-field-and-fidelity.pdf')
    #grape = GrapeESR(get_J(1,3), get_A(1,3), tN=100*unit.ns, N=500, max_time=max_time); grape.run(); grape.plot_field_and_fidelity(f"{plots_folder}Ch3-3E-field-and-evolution.pdf")

    #show_2E_Hw(get_J(1,2),get_A(1,2),30*unit.ns,500, "Ch3-2E-Hw.pdf")

    #compare_free_3E_evols(fp = f"{plots_folder}Ch3-3E-free-evol-comparison.pdf")

    #investigate_3E_resfreqs(fp = f"{plots_folder}{chapter}all-allowed-3E-transitions.pdf")
    #investigate_3E_resfreqs(N=2000)

    #NE_EN_CX()
    #show_NE_swap(get_A(1,1),2*unit.T, 100000, 40000, fp=f"{plots_folder}{chapter}NE_swap.pdf")
    #plot_swap_schedule()

    #all_15_2q_CNOTs(fp1=f"{plots_folder}Ch3-15-2E-field-and-fidelity.pdf", prev_grape_fp='fields/g235_15S_2q_2000ns_5000step', max_time=None)

def no_coupler():
    def plot_exchange_switch(A=get_A(1,3), J=get_J(1,3), fp=None):

                
        def label_getter(i):
            if i in [1,2,4]:
                return f"Pr({np.binary_repr(i,3)[0]}{np.binary_repr(i,3)[-1]})"
        def analytic_time_limit(J=get_J(1,2), A=get_A(1,1), fidelity=0.99):
            return pt.real(np.arccos(np.sqrt(fidelity)) / np.sqrt(2*A**2 + 4*J**2)).item()



        N=5000
        tN=20*unit.ns
        t0 = 5*unit.ns
        psi0 = pt.tensor([0,0,1,0,0,0,0,0], dtype=cplx_dtype)
        psi1 = pt.einsum('j,a->ja', pt.ones(N//2), psi0)
        psi2 = get_free_electron_evolution(J=J, A=A, tN=tN/2, N=N//2, psi0=psi0)

        j=0
        while pt.abs(psi2[j,2]) > 0.99:
            j+=1 
        print(f"From graph: Pr(100) < 0.99 after t-t0 = {j/N*tN/unit.ns} ns")
        print(f"Analytic: Pr(~100) < 0.99 after t-t0 = {analytic_time_limit(J=pt.max(pt.real(J)))/unit.ns:.3f} ns")
        psi = pt.cat((psi1, psi2))
        fig,ax = plt.subplots(1,1)
        plot_psi(psi, tN, legend_loc = 'center left', ax=ax, label_getter=exchange_label_getter)
        fig.set_size_inches(fig_width_double_long, fig_height_double_long)
        ax.set_xlabel("time (ns)")
        ax.axvline(10, color='black', linestyle='--', linewidth=1)
        ax.annotate("$t=t_0$",[9,0.5])
        fig.tight_layout()
        if fp is not None:
            fig.savefig(fp)

    def triple_E_transitions(A=get_A(1,3), J=get_J(1,3),fp=None, labels=None):
        fig,ax = plt.subplots(1,2)
        visualise_E_transitions(A=A, J=J, ax=ax[0])
        triple_E_energy_level_picture(ax=ax[1], A=A, J=J)
        fig.set_size_inches(10, 5)
        if labels is not None:
            label_axis(ax[0],labels[0], 0.2, 0)
            label_axis(ax[1],labels[1], -0.1,0)
        if fp is not None:
            fig.savefig(fp)
    
    def two_E_transitions(A=get_A(1,2), J=get_J(1,2), fp=None, detuning=True, ax_labels=None):
        fig,ax = plt.subplots(1,2)
        NucSpin=[0,1] if detuning else [1,1]
        visualise_E_transitions(A=get_A(1,2, NucSpin=NucSpin), J=get_J(1,2), ax=ax[0])
        two_electron_energy_level_picture(ax=ax[1], detuning=detuning)
        fig.tight_layout()
        fig.set_size_inches(5, 3)
        ax[0].set_xlim(-4,4)
        if ax_labels is not None:
            label_axis(ax[0],ax_labels[0], x_offset=0.2)
            label_axis(ax[1],ax_labels[1])
        if fp is not None:
            fig.savefig(fp)

    def compare_2E_transitions_detuning(fp=None):
        fig,ax = plt.subplots(2,2)
        visualise_E_transitions(A=get_A(1,2, NucSpin=[0,1]), J=get_J(1,2), ax=ax[0,0])
        two_electron_energy_level_picture(ax=ax[0,1], detuning=True)
        visualise_E_transitions(A=get_A(1,2, NucSpin=[1,1]), J=get_J(1,2), ax=ax[1,0])
        two_electron_energy_level_picture(ax=ax[1,1], detuning=False)
        fig.tight_layout()
        fig.set_size_inches(13, 10)
        ax[0,0].annotate("(a)", [0,0])
        ax[0,1].annotate("(b)", [0,0])
        ax[1,0].annotate("(c)", [0,0])
        ax[1,1].annotate("(d)", [0,0])
        if fp is not None:
            fig.savefig(fp)

    def triple_E_grape_failure(fp=None):
        fig,ax = plt.subplots(1,2)
        grape = GrapeESR(J=get_J(1,3), A=get_A(1,3, NucSpin=[1,1,1]), tN=100*unit.ns, N=200, max_time=2)
        grape.run()
        grape.plot_cost_hist(ax[1])
        grape = GrapeESR(J=get_J(1,3), A=get_A(1,3, NucSpin=[1,0,1]), tN=100*unit.ns, N=200, max_time=2)
        grape.run()
        grape.plot_cost_hist(ax[0])
        label_axis(ax[0], '(a)', -0.15,-0.1)
        label_axis(ax[1], '(b)', -0.15, -0.1)
        fig.set_size_inches(fig_width_double, fig_height_single)
        if fp is not None: fig.savefig(fp)

    def NE_swap_spin_states(tN, N, A, Bz, ax, psi0=gate.spin_11):
        Bx,By = NE_swap_pulse(tN,N,A,Bz, ax[0])
        H0 = get_NE_H0(A, Bz)
        X = get_NE_X(Bx, By, Bz, A, tN, N)
        X = get_IP_X(X,H0,tN,N)
        plot_psi(X@psi0, tN, ax)
    
    def NE_swap_exchange_comparison(fp=None):
        fig,ax = plt.subplots(2,1)
        Bz = 2*unit.T

        N=500000
        double_NE_swap_with_exchange(J=get_J(1,2)/71, N=N, deactivate_exchange=True, ax=ax[0], label_states=[2,8,10])
        double_NE_swap_with_exchange(J=get_J(1,2)/71, N=N, deactivate_exchange=False, ax=ax[1], label_states=[2,8,10,1,4,5])
        label_axis(ax[0], '(a)', x_offset=-0.13, y_offset=-0.1)
        label_axis(ax[1], '(b)', x_offset=-0.13, y_offset=-0.1)

        if fp is not None: fig.savefig(fp)


        #NE_swap_spin_states(tN, N, A, Bz, ax[0])


    #plot_exchange_switch(A=get_A(1,3), J=get_J(1,3)/10, fp = f"{plots_folder}NC-exchange-switch.pdf")
    triple_E_transitions(fp = f"{plots_folder}NC-triple-E-transitions.pdf", labels=['a','b'])
    #triple_E_transitions(A=get_A(1,3,NucSpin=[1,1,1]), J=get_J(1,3), fp=f"{plots_folder}NC-triple-E-transitions-no-detuning.pdf", labels=['c','d'])
    #two_E_transitions(fp = f"{plots_folder}NC-allowed-2E-transitions.pdf", ax_labels=['a','b'])
    #two_E_transitions(fp = f"{plots_folder}NC-allowed-2E-transitions-no-detuning.pdf", detuning=False, ax_labels=['c', 'd'])
    #compare_2E_transitions_detuning(fp=f"{plots_folder}NC-transitions-2E-comparison-detuning.pdf")
    #triple_E_grape_failure(fp=f"{plots_folder}NC-transitions-3E-cost_convergence_comparison.pdf")
    #show_NE_swap(get_A(1,1),2*unit.T,500*unit.ns,100000)
    #NE_swap_exchange_comparison(fp=f"{plots_folder}NC-swap-with-exchange-comparison")
    #plot_load_time_vs_J(fid_min=0.99, Jmin=0.5*unit.MHz, Jmax=20*unit.MHz, tN_max=100*unit.ns, A=get_A(1,3), n=100, fp=f"{plots_folder}NC-J-vs-max-load-time-99.pdf")


def HSquare(chapter = 'HS-'):

    def plot_lattice_sites(fp=None):
        fig,ax = plt.subplots(1,1)
        dist=51
        x0=0; x1=dist
        y0 = -1; y1=6
        padding=1
        all_sites = get_all_sites([-1,dist], [-1,6], padding=padding)
        plot_spheres(all_sites, ax=ax, alpha=0.1)
        ax.set_xlim([x0-padding-0.5,dist+padding+0.5])
        ax.set_ylim([y0-padding-0.5, y1+padding+0.5])
        ax.set_aspect('equal')
        sites_color='#1DA4BF'
        alpha = 0.5
        #plot_9_sites(0, 0, ax=ax, neighbour_color=sites_color)
        
        linewidth = 1.5
        fontsize = 13

        d=1 
        delta_x=dist-4
        delta_y=0
        separation_2P=6
        alpha = 0.5

        sites_colour = FigureColours.sites_colour
        sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites_1P_2P(delta_x, delta_y, d=d, separation_2P=separation_2P)
        for sites in [sites_2P_lower, sites_2P_upper, sites_1P]:
            for j,site in enumerate(sites):
                sites[j]=(site[0],site[1]-1)
        plot_spheres(sites_2P_lower, color=sites_colour, alpha=alpha, ax=ax)
        plot_spheres(sites_2P_upper, color=sites_colour, alpha=alpha, ax=ax)
        plot_spheres(sites_1P, color=sites_colour, alpha=alpha, ax=ax)

        s2L = sites_2P_lower[5]
        s2U = sites_2P_upper[3]
        s1 = sites_1P[0]

        #ax.plot([s2L[0], s2U[0]], [s2L[1], s2U[1]], color='black', linestyle='dotted', label='Hyperfine separation')
        #ax.plot([(s2L[0]+s2U[0])/2, s1[0]], [(s2L[1]+s2U[1])/2, s1[1]], color='black', linestyle='dashed', label='Exchange separation')

        plot_spheres([s2U], color='red', ax=ax, zorder=3)
        plot_spheres([s2L], color='red', ax=ax, zorder=3)
        plot_spheres([s1], color='red', ax=ax, zorder=3)


        #ax.plot([10,10], [5,6])
        # ax_length=5
        # ax_origin = np.array([11,1])
        # ax.arrow(*ax_origin, 0, ax_length, shape='full', lw=1.2, length_includes_head=True, head_width=0.4)
        # ax.arrow(*ax_origin, ax_length, 0, shape='full', lw=1.2, length_includes_head=True, head_width=0.4)
        # ax.annotate('[1,-1,0]', ax_origin + np.array([-0,-1.5]))
        # ax.annotate('[1,1,0]', ax_origin + np.array([-1.8,0.3]), rotation=90)
        
        ax.set_xlabel('[1,-1,0]')
        ax.set_ylabel('[1,1,0]')
        ax.set_yticks([-2,2,6])
        ax.set_yticklabels([0,4,8])
        ax.set_xticks([2,12,22,32,42,49])
        ax.set_xticklabels([0,10,20,30,40,47])
        #ax.legend(loc='upper center')
        #ax.axis('off')
        fig.set_size_inches(fig_width_double, 0.6*fig_height_single)
        fig.tight_layout()
        if fp is not None:
            plt.savefig(fp)

    def voltage_scale(fp=f'{plots_folder}{chapter}voltage-scale.pdf'):
        fig,ax=plt.subplots(1,1)
        padding=0.04
        color_bar(ax, [blue, grey, yellow, orange, red], tick_labels=[], orientation='vertical', padding=padding)
        ax.axis('off')
        xlim = ax.get_xlim(); ylim=ax.get_ylim()
        ax.annotate('+V', [xlim[1], ylim[0]], fontsize=30)
        ax.annotate('–V', [xlim[0], ylim[0]], fontsize=30)
        fig.set_size_inches(20,0.4)
        fig.savefig(f'{plots_folder}{chapter}voltage-scale.pdf')


    def compare_symmetry(fp=None):
        fig,ax = plt.subplots(1)
        dist=18
        sq_centre = np.array([dist,0])
        diag_centre = np.ceil(np.array([dist/np.sqrt(2), dist/np.sqrt(2)]))
        hex_centre = np.ceil(np.array([dist*np.sqrt(3)/2, dist*1/2]))
        all_sites = get_all_sites([0,dist], [0,dist/np.sqrt(2)], padding=dist//5)
        plot_spheres(all_sites, ax=ax, alpha=0.1)
        sites_color='#1DA4BF'
        alpha = 0.5
        plot_9_sites(0, 0, ax=ax, neighbour_color=sites_color)
        plot_9_sites(*sq_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
        plot_9_sites(*diag_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
        plot_9_sites(*hex_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)

        linecolor_15='black'
        linecolor_25 = 'darkred'
        linewidth = 1.5
        fontsize = 13
        n=100
        ang_adjust = np.arctan(hex_centre[1]/hex_centre[0])
        theta1 = np.linspace(0,ang_adjust, n)
        theta2 = np.linspace(0,np.pi/4, n)
        r1 = 12
        r2 = 7
        if fp is not None:
            plt.savefig(fp)

    def all_48_1P_2P_CNOTs(grape_fp='fields-gadi/fields/g228_48S_2q_5000ns_10000step', fp=None):
        grape_fp = 'fields/g259_48S_2q_5000ns_10000step'
        grape_fp='fields/g265_48S_2q_5000ns_10000step'
        grape_fp='fields/g269_48S_2q_5000ns_10000step'
        grape=load_grape(grape_fp, Grape=GrapeESR_AJ_Modulation)
        print("EXCHANGE - HYPERFINE")
        print("================================================")
        J=pt.real(grape.J/unit.MHz)
        A = pt.real(A_2P_1P_fab[:,0])/unit.MHz
        for n in range(12):
            n2=n+12
            n3=n+24
            n4=n+36
            print(f'{n+1} &{J[n]:.2f} &{A[n]:.1f} &{n2+1} &{J[n2]:.2f} &{A[n2]:.1f} &{n3+1} &{J[n3]:.2f} &{A[n3]:.1f} &{n4+1} &{J[n4]:.2f} &{A[n4]:.1f}\\\\')
        #grape.plot_result()    
        xcol=color_cycle[0]
        ycol = color_cycle[1]
        fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3,  2]})
        ax2=ax[0].twinx()
        ax[0].set_yticks([-0.5,0,0.5], color=xcol)
        ax2.set_yticks([-0.5,0,0.5])
        grape.plot_XY_fields(ax[0], legend_loc = False, twinx=ax2, xcol=xcol, ycol=ycol)
        grape.print_result()
        fidelity_bar_plot(grape.fidelity()[0], ax=ax[1], f=[0.9999,0.999,0.99], colours=['green','lightgreen','orange','red'], legend_loc='upper center')
        ax[0].set_ylim([-0.5,1.5])
        ax2.set_ylim([-1.5,0.5])
        ax[1].set_yticks([0.98,1])
        ax[1].set_xticks([0,9,19,29,39,47])
        ax[1].set_xticklabels([1,10,20,30,40,48])
        fig.set_size_inches(fig_width_double, fig_height_double_long)
        ax[1].set_ylim([0.98,1+0.4*0.02])
        fig.tight_layout()
        if fp is not None: fig.savefig(fp)
        set_trace()

    def draw_HS_architecture(fp=None, fp2=None, orientation='hori'):
        if orientation=='hori':
            fig1,ax = plt.subplots(1,2)
        else:
            #fig1,ax = plt.subplot_mosaic([['left','upper right'], ['left', 'lower right']], figsize=(7.5,3.5),layout="constrained", gridspec_kw={'width_ratios': [3,  1.2]})
            
            fig1,ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [0.95,  1.2]})
        make_stab_unit_cell(ax[0], fontsize=15)
        HS_side_view(ax[1])
        #make_HS_array(ax['left'], 5)
        #ax[1].set_aspect('equal')
        #fig.set_size_inches(fig_width_double, fig_height_single)
        if orientation=='hori':
            fig1.set_size_inches(1.3*fig_width_double, 1.3*fig_height_single)
        else:
            fig1.set_size_inches(1.2*fig_width_single, 2.4*fig_height_single)

        fig1.tight_layout()
        #fig2.set_size_inches(fig_width_single, fig_width_single*1.1)
        if fp is not None:
             fig1.savefig(fp)
        # if fp2 is not None:
        #      fig2.savefig(fp2)
    def save_HS_array(fp=None):
        fig,ax=plt.subplots(1,1)
        make_HS_array(ax, 5)
        fig.set_size_inches(7,7)
        if fp is not None: fig.savefig(fp)

    #plot_lattice_sites(fp = f'{plots_folder}{chapter}1P-2P-lattice-placement.pdf')
    #all_48_1P_2P_CNOTs(fp = f'{plots_folder}{chapter}all-48-1P-2P-pulse-and-fidelity-bars.pdf')
    compare_symmetry()
    #draw_HS_architecture(fp = f'{plots_folder}{chapter}architecture_layout.pdf', fp2=f'{plots_folder}{chapter}architecture-layout-sideon.pdf', orientation='stoopy')
    #illustrative_configs(3, fp=f'{plots_folder}{chapter}illustrative-configs.pdf')
    #voltage_scale()
    #save_HS_array(fp=f'{plots_folder}{chapter}array.pdf')
    #stabilizer_activations(fp=f'{plots_folder}{chapter}stabilizer-activations.pdf')
    #readout(fp=f'{plots_folder}{chapter}readout-configs.pdf')



if __name__=='__main__':
    #intro_and_review()
    #HHex()
    pulse_design()
    HSquare()

    #no_coupler()


    plt.show()