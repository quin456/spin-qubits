import torch as pt
import numpy as np

import atomic_units as unit
from utils import *
from eigentools import *
from data import *
from GRAPE import Grape_ee_Flip, load_grape


def run_n_entangled_CX(
    tN=1000 * unit.ns,
    N=2000,
    nS=4,
    max_time=70,
    verbosity=2,
    prev_grape_fp=None,
    **kwargs
):
    step = 1
    fp = "J-50-100"
    print(fp)
    print(step)
    tN = lock_to_frequency(get_A(1, 1), tN)
    A = get_A_1P_2P(nS, NucSpin=[1, -1])
    J = get_J_1P_2P(nS, fp=fp)

    if prev_grape_fp is not None:
        grape = load_grape(
            prev_grape_fp, Grape=Grape_ee_Flip, step=step, verbosity=verbosity, **kwargs
        )
    else:
        grape = Grape_ee_Flip(tN, N, J, A, step=step, verbosity=verbosity, **kwargs)

    grape.run(max_time=max_time)
    grape.print_result(verbosity=verbosity)
    grape.plot_result()
    grape.save()


if __name__ == "__main__":
    # run_n_entangled_CX(tN=300 * unit.ns, N=1000, nS=1, max_time=60, lam=1e3)
    run_n_entangled_CX(
        tN=4000 * unit.ns,
        N=8000,
        nS=70,
        max_time=23.5 * 3600,
        lam=1e7,
        # prev_grape_fp="fields/c1356_1S_3q_479ns_2500step",
    )
