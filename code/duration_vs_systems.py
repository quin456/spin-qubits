import numpy as np
import torch as pt


from visualisation import *
from GRAPE import GrapeESR, get_max_field
from utils import *
from run_grape import run_CNOTs, get_rec_min_N


"""
Run GRAPE optimisations with on different numbers of systems with different 
pulse durations to observe relationship between number of systems, required 
pulse duration, fidelity, and pulse size.
"""


def test_pulse_times(T=linspace(0, 2000, 20) * unit.ns):
    J = get_J_1P_2P(1)
    A = get_A_1P_2P(1)
    rf = get_resonant_frequencies(get_H0(A, J))
    for tN in T:
        N = get_rec_min_N(rf, tN, N_period=5)
        grape = run_CNOTs(
            tN=tN,
            N=N,
            nS=1,
            max_time=N / 50,
            verbosity=-1,
            save_data=False,
            stop_fid_avg=0.99,
            stop_fid_min=0.99,
            lam=1e5,
        )
        print(
            f"tN = {grape.tN/unit.ns} ns, N = {grape.N}, fidelity = {grape.fidelity()[0][0].item()*100:.1f}%, max_field = {get_max_field(*grape.get_Bx_By())/unit.mT:.1f} mT, status={grape.status}"
        )


"""
Best approach might be to just generate a fuck tonne of data and log it all. 
There will be holes that will need to be filled in where convergence has failed...
I can then plot what I want to plot. 
Max field will be too much of a pain as a variable.
Fidelity vs tN vs nS.
Even just tN vs nS would be interesting.
Hopefully fidelity stabilizes a bit for more systems.
"""


if __name__ == "__main__":
    test_pulse_times()
