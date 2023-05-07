import torch as pt
import numpy as np

import atomic_units as unit
from utils import *
from eigentools import *
from data import *
from GRAPE import Grape_ee_Flip, load_grape


def run_ee_flip_grape(
    tN=1000 * unit.ns,
    N=2000,
    nS=4,
    max_time=70,
    verbosity=2,
    prev_grape_fp=None,
    **kwargs
):
    tN = lock_to_frequency(get_A(1, 1), tN)
    A = get_A_1P_2P(nS, NucSpin=[1, -1])
    J = get_J_1P_2P(nS)

    if prev_grape_fp is not None:
        grape = load_grape(
            prev_grape_fp, Grape=Grape_ee_Flip, step=1, verbosity=verbosity, **kwargs
        )
    else:
        grape = Grape_ee_Flip(tN, N, J, A, step=1, verbosity=verbosity, **kwargs)
    grape.run(max_time=max_time)
    grape.print_result(verbosity=verbosity)
    grape.plot_result()
    grape.save()


if __name__ == "__main__":
    # run_ee_flip_grape(
    #     tN=5000 * unit.ns, N=8000, nS=70, max_time=23.5 * 3600, lam=1e7,
    # )
    run_ee_flip_grape(tN=200 * unit.ns, N=1000, nS=1, max_time=60, lam=1e3)

