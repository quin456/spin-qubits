

import torch as pt
import matplotlib
if not pt.cuda.is_available():
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure 
import numpy as np 

from GRAPE import GrapeESR
import gates as gate
from utils import psi_from_polar, get_A, get_J, normalise, get_resonant_frequencies, label_axis
from hamiltonians import get_pulse_hamiltonian, get_U0
from visualisation import plot_psi
from visualisation import *
from single_spin import show_single_spin_evolution
from data import dir, cplx_dtype, gamma_e, gamma_n, J_100_14nm, J_100_18nm
from atomic_units import *
from architecture_design import plot_cell_array, plot_annotated_cell, generate_CNOTs, numbered_qubits_cell, plot_single_cell
from electrons import plot_free_electron_evolution, get_free_electron_evolution
from transition_visualisation import visualise_E_transitions
from single_NE import show_NE_swap, NE_swap_pulse, get_NE_H0, get_NE_X, get_IP_X
from gates import spin_11
from multi_NE import double_NE_swap_with_exchange
from misc_calculations import plot_load_time_vs_J
from voltage_plot import plot_CNOTs
from electrons import investigate_3E_resfreqs


from qiskit.visualization import plot_bloch_vector

plots_folder = f"{dir}thesis-plots/"

uparrow = u'\u2191'
downarrow = u'\u2193'
Uparrow = '⇑'
Downarrow = '⇓'
rangle = '⟩'

max_time = 10


################################################################################################################
################        ALL PLOTS TO BE USED IN THESIS WILL BE GENERATED HERE        ###########################
################################################################################################################




################################################################################################################
################        CHAPTER 1 INTRODUCTION        ##########################################################
################################################################################################################


def bloch_sphere(psi, fp=None):
    blochs = psi_to_cartesian(psi).numpy()
    plot_bloch_vector(blochs)
    if fp is not None: plt.savefig(fp)


def energy_level_picture(H0, state_labels=None, energy_labels=None, colors=colors, ax=None, fp=None, ax_label=None):

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
    
    set_trace()

def two_electron_energy_level_picture(J=get_J(1,2), Bz=0.05*tesla, ax=None, fp = None, detuning=True):

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
    H0=get_H0(A,J,Bz=0.04*tesla)
    colors = ['blue', 'green', 'green', 'green', 'orange', 'orange', 'orange', 'red']
    spin_labels = [f"$|000{rangle}$", f"$a|001{rangle}+b|010{rangle}+c|100{rangle}$", "","", f"$a|011{rangle}+b|101{rangle}+c|110{rangle}$", "", "", f"|111{rangle}"]
    energy_level_picture(H0, energy_labels=spin_labels, colors=colors, ax=ax, ax_label=ax_label)

#f"$\alpha|001{rangle}+\beta|010{rangle}+\gamma|100{rangle}$"
def free_2E_evolution(fp=None):
    A = get_A(1,2); J=get_J(1,2); tN=10*nanosecond
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
    tN = 10*nanosecond 
    N = 500
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(double_long_width, single_long_height)


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
    S,D = get_ordered_2E_eigensystem(A, J)
    U0 = get_U0(H0, tN, N)
    visualise_Hw(S.T@dagger(U0)@Hw@U0@S, tN)

def chapter_1():
    #bloch_sphere(psi_from_polar(np.pi/2,np.pi/4), fp = f'{plots_folder}Ch1-bloch-sphere.pdf')
    #show_single_spin_evolution(tN=100*nanosecond, fp = f"{plots_folder}Ch1-analytic-example.pdf")
    two_electron_energy_level_picture(fp=f"{plots_folder}Ch1-2E-energy-levels.pdf")


def chapter_2(chapter='Ch2-'):
    #plot_cell_array(4,4, filename=f"{plots_folder}Ch2-cell_array.pdf")
    generate_CNOTs(fp = f"{plots_folder}Ch2-CNOT-voltage-cells.pdf")
    #plot_CNOTs(fp=f"{plots_folder}{chapter}CNOT-voltage-schedule.pdf")
    #plot_annotated_cell(filename=f"{plots_folder}Ch2-single_cell.pdf")
    #numbered_qubits_cell()
    #plot_single_cell()
    plt.show()



def chapter_3(chapter="Ch3-"):

    def grape_1s2q(fp=None, fp1=None):
        grape = GrapeESR(J=get_J(1,2),A=get_A(1,2),tN=20*nanosecond,N=500, max_time=5); grape.run()
        fig,ax = plt.subplots(1,2)
        grape.plot_u(ax[0])
        grape.plot_cost_hist(ax[1])
        fig.set_size_inches(double_long_width, single_long_height)
        fig1,ax1 = grape.plot_field_and_evolution()
        fig1.set_size_inches(double_long_width, double_long_height)
        plt.tight_layout()
        if fp is not None: fig.savefig(fp)
        if fp1 is not None: fig1.savefig(fp1)


    #NE_energy_level_picture(fp=f"{plots_folder}Ch3-NE-energy-levels.pdf")

    #free_2E_evolution(fp = f"{plots_folder}Ch3-2E-free-evolution.pdf")


    #grape_1s2q(fp = f"{plots_folder}Ch3-2E-u-and-cost.pdf", fp1 = f"{plots_folder}Ch3-2E-field-and-evolution.pdf")

    #grape = GrapeESR(get_J(1,3), get_A(1,3), tN=100*nanosecond, N=500, max_time=max_time); grape.run(); grape.plot_field_and_fidelity(f"{plots_folder}Ch3-3E-field-and-evolution.pdf")

    #show_2E_Hw(get_J(1,2),get_A(1,2),30*nanosecond,500, "Ch3-2E-Hw.pdf")

    #compare_free_3E_evols(fp = f"{plots_folder}Ch3-3E-free-evol-comparison.pdf")

    investigate_3E_resfreqs(fp = f"{plots_folder}{chapter}all-allowed-3E-transitions.pdf")

def no_coupler():
    def plot_exchange_switch(A=get_A(1,3), J=get_J(1,3), fp=None):

                
        def label_getter(i):
            if i in [1,2,4]:
                return f"Pr({np.binary_repr(i,3)[0]}{np.binary_repr(i,3)[-1]})"
        def analytic_time_limit(J=get_J(1,2), A=get_A(1,1), fidelity=0.99):
            return pt.real(np.arccos(np.sqrt(fidelity)) / np.sqrt(2*A**2 + 4*J**2)).item()



        N=5000
        tN=20*nanosecond
        t0 = 5*nanosecond
        psi0 = pt.tensor([0,0,1,0,0,0,0,0], dtype=cplx_dtype)
        psi1 = pt.einsum('j,a->ja', pt.ones(N//2), psi0)
        psi2 = get_free_electron_evolution(J=J, A=A, tN=tN/2, N=N//2, psi0=psi0)

        j=0
        while pt.abs(psi2[j,2]) > 0.99:
            j+=1 
        print(f"From graph: Pr(100) < 0.99 after t-t0 = {j/N*tN/nanosecond} ns")
        print(f"Analytic: Pr(~100) < 0.99 after t-t0 = {analytic_time_limit(J=pt.max(pt.real(J)))/nanosecond:.3f} ns")
        psi = pt.cat((psi1, psi2))
        fig,ax = plt.subplots(1,1)
        plot_psi(psi, tN, legend_loc = 'center left', ax=ax, label_getter=exchange_label_getter)
        fig.set_size_inches(double_long_width, double_long_height)
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
        grape = GrapeESR(J=get_J(1,3), A=get_A(1,3, NucSpin=[1,1,1]), tN=100*nanosecond, N=200, max_time=2)
        grape.run()
        grape.plot_cost_hist(ax[1])
        grape = GrapeESR(J=get_J(1,3), A=get_A(1,3, NucSpin=[1,0,1]), tN=100*nanosecond, N=200, max_time=2)
        grape.run()
        grape.plot_cost_hist(ax[0])
        label_axis(ax[0], '(a)', -0.15,-0.1)
        label_axis(ax[1], '(b)', -0.15, -0.1)
        fig.set_size_inches(2*square_size, square_size)
        if fp is not None: fig.savefig(fp)

    def NE_swap_spin_states(tN, N, A, Bz, ax, psi0=spin_11):
        Bx,By = NE_swap_pulse(tN,N,A,Bz, ax[0])
        H0 = get_NE_H0(A, Bz)
        X = get_NE_X(Bx, By, Bz, A, tN, N)
        X = get_IP_X(X,H0,tN,N)
        plot_psi(X@psi0, tN, ax)
    
    def NE_swap_exchange_comparison(fp=None):
        fig,ax = plt.subplots(2,1)
        Bz = 2*tesla

        N=500000
        double_NE_swap_with_exchange(J=get_J(1,2)/71, N=N, deactivate_exchange=True, ax=ax[0], label_states=[2,8,10])
        double_NE_swap_with_exchange(J=get_J(1,2)/71, N=N, deactivate_exchange=False, ax=ax[1], label_states=[2,8,10,1,4,5])
        label_axis(ax[0], '(a)', x_offset=-0.13, y_offset=-0.1)
        label_axis(ax[1], '(b)', x_offset=-0.13, y_offset=-0.1)

        if fp is not None: fig.savefig(fp)


        #NE_swap_spin_states(tN, N, A, Bz, ax[0])


    #plot_exchange_switch(A=get_A(1,3), J=get_J(1,3)/10, fp = f"{plots_folder}NC-exchange-switch.pdf")
    #triple_E_transitions(fp = f"{plots_folder}NC-triple-E-transitions.pdf", labels=['a','b'])
    #triple_E_transitions(A=get_A(1,3,NucSpin=[1,1,1]), J=get_J(1,3), fp=f"{plots_folder}NC-triple-E-transitions-no-detuning.pdf", labels=['c','d'])
    #two_E_transitions(fp = f"{plots_folder}NC-allowed-2E-transitions.pdf", ax_labels=['a','b'])
    #two_E_transitions(fp = f"{plots_folder}NC-allowed-2E-transitions-no-detuning.pdf", detuning=False, ax_labels=['c', 'd'])
    #compare_2E_transitions_detuning(fp=f"{plots_folder}NC-transitions-2E-comparison-detuning.pdf")
    #triple_E_grape_failure(fp=f"{plots_folder}NC-transitions-3E-cost_convergence_comparison.pdf")
    #show_NE_swap(get_A(1,1),2*tesla,500*nanosecond,100000)
    #NE_swap_exchange_comparison(fp=f"{plots_folder}NC-swap-with-exchange-comparison")
    #plot_load_time_vs_J(fid_min=0.99, Jmin=0.5*Mhz, Jmax=20*Mhz, tN_max=100*nanosecond, A=get_A(1,3), n=100, fp=f"{plots_folder}NC-J-vs-max-load-time-99.pdf")


if __name__=='__main__':
    #chapter_2()
    chapter_3()

    #no_coupler()

    plt.show()