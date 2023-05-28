import numpy as np
import torch as pt
import pandas as pd

from visualisation import *
from GRAPE import GrapeESR, get_max_field
from utils import *
from run_grape import run_CNOTs, get_rec_min_N

nS_2D = [1, 2, 3, 4, 5, 6, 7, 8, 9]
tN_2D = [100, 170, 500, 500, 700, 700, 700, 700, 900]
Bmax_2D = [1.26, 1.48, 1.03, 1.5, 1.14, 1.31, 1.56, 1.79, 2.14]
N_2D = [None, None, None, 1500, 1500, 1500, 1800, 1500, 1800]

log_fp = "fid-vs-tN-vs-nS/results_log.csv"

"""
Run GRAPE optimisations with on different numbers of systems with different 
pulse durations to observe relationship between number of systems, required 
pulse duration, fidelity, and pulse size.
"""

fp = "fid-vs-tN-vs-nS/tN-500-2000_nS-1-15.csv"


def test_pulse_times(T=linspace(0, 2000, 20) * unit.ns, fids=None):
    nS = 1
    n = len(T)
    J = get_J(nS, 2)
    A = get_A(nS, 2)
    rf = get_resonant_frequencies(get_H0(A, J))
    if fids is None:
        fids = [None] * n
    for k in range(n):
        N = get_rec_min_N(rf, T[k], N_period=5)
        grape = run_CNOTs(
            tN=T[k],
            N=N,
            nS=1,
            max_time=10,
            verbosity=-1,
            save_data=False,
            stop_fid_avg=0.99,
            stop_fid_min=0.99,
            lam=1e7,
            kappa=1,
        )
        fids[k] = pt.mean(grape.Phi).item() * 100
        minfid = minreal(grape.Phi).item() * 100
        print(
            f"tN = {grape.tN/unit.ns} ns, N = {grape.N}, avg fid = {fids[k]:.1f}%, min fid = {minfid:.1f}%, max_field = {get_max_field(*grape.get_Bx_By())/unit.mT:.1f} mT, status={grape.status}"
        )

    return fids


def initialise_T_row(T, fp=fp):
    with open(fp, "w") as f:
        f.write("," + ", ".join([f"{real(t).item() / unit.ns:.1f}" for t in T]) + "\n")


def add_fids_line(fids, nS, fp):
    with open(fp, "a") as f:
        f.write(str(nS) + ", " + ", ".join([f"{fid:.1f}" for fid in fids]) + "\n")


def redo_fids_line():
    pass


def get_fids_line_and_save(nS, fp=fp):
    fids = test_pulse_times()
    add_fids_line(fids, nS, fp=fp)


def plot_tN_vs_nS_2D():
    ax = plt.subplot(projection="3d")
    ax.scatter(nS_2D, Bmax_2D, tN_2D)


"""
Best approach might be to just generate a fuck tonne of data and log it all. 
There will be holes that will need to be filled in where convergence has failed...
I can then plot what I want to plot. 
Max field will be too much of a pain as a variable.
Fidelity vs tN vs nS.
Even just tN vs nS would be interesting.
Hopefully fidelity stabilizes a bit for more systems.
"""


def log(grape, fp=log_fp):
    Bmax = get_max_field(*grape.get_Bx_By()) / unit.mT
    tN = grape.tN / unit.ns
    Phi_avg = pt.mean(grape.Phi) * 100
    Phi_min = minreal(grape.Phi) * 100
    calls = grape.calls_to_cost_fn
    opt_time = grape.time_taken
    with open(fp, "a") as f:
        f.write(
            f"\n{tN:.0f},{grape.N},{grape.nS},{Phi_avg:.1f},{Phi_min:.1f},{Bmax:.2f},{grape.lam:.0e},{grape.kappa:.0e},{opt_time:.0f},{calls}, {grape.status}"
        )


def run_and_log(tN, N, J, A, **kwargs):
    grape = run_CNOTs(
        tN,
        N,
        J=J,
        A=A,
        verbosity=-1,
        save_data=False,
        stop_fid_avg=0.99,
        stop_fid_min=0.99,
        **kwargs,
    )
    print(grape.get_opt_state())
    log(grape)


def run_some_grapes():
    T = pt.arange(550, 1500, 100).to(real_dtype) * unit.ns
    kappa = 1e2
    lam = 1e7

    for nS in range(22, 40, 4):
        for tN in T:
            A = get_A_1P_2P(nS)
            J = get_J_1P_2P(nS)
            rf = get_multi_system_resonant_frequencies(get_H0(A, J))
            N = get_rec_min_N(rf, tN, 5)
            run_and_log(
                tN,
                N,
                J,
                A,
                kappa=kappa,
                lam=lam,
                max_time=1800,
                dynamic_opt_plot=True,
                simulate_spectators=False,
                dynamic_params=True,
            )


def select_smallest_two(df):
    return df.nsmallest(2, "tN")


def get_data_from_log(fp=log_fp, fp_fig=None):
    Bmax = 2
    df = pd.read_csv(fp)
    df = df[df.index > 400]
    # df = df[df["nS"] < 50]
    # df['Bmax'] = pd.to_numeric(df['Bmax'])
    df_SF = df[df["status"] == " SF"]
    df_failed = df[df["status"] != " SF"]

    df_SF_2mT = df_SF[df_SF["Bmax"] < Bmax]
    df_SF_g_2mT = df_SF[df_SF["Bmax"] > Bmax]
    df_shortest = df_SF_2mT.loc[df_SF_2mT.groupby("nS")["tN"].idxmin()]

    df_new = df_SF_2mT.groupby("nS").apply(select_smallest_two).reset_index(drop=True)

    cmap = cm.viridis

    ax = plt.subplot()
    fig = ax.get_figure()
    fig.set_size_inches(11 / 2.54, 8 / 2.54)
    # df_failed.plot(x="nS", y="tN", kind="scatter", color='gray', ax=ax, marker='x')
    df_SF_2mT.plot(x="nS", y="tN", kind="scatter", c="Bmax", colormap=cmap, ax=ax)
    print(df_shortest)
    fig.tight_layout()
    ax.set_xlabel("Distinct 1P-2P qubit pairs")
    ax.set_ylabel("Pulse Duration (ns)")
    colorbar = ax.collections[-1].colorbar
    colorbar.set_label("Max Field Strength (mT)")
    ax.set_yticks([100, 500, 1000, 1500, 2000])
    colorbar.set_ticks([0.6, 1.2, 1.8])

    if fp_fig is not None:
        fig.savefig(fp_fig)


if __name__ == "__main__":
    T = linspace(0, 2000, 20) * unit.ns
    # initialise_T_row(T)
    # add_fids_line([np.random.rand() for _ in range(20)], nS=2)
    # test_pulse_times()
    # get_fids_line_and_save(1)
    # plot_tN_vs_nS_2D()
    run_some_grapes()
    get_data_from_log()
    plt.show()
