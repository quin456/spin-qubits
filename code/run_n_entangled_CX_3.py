import torch as pt
import numpy as np

import atomic_units as unit
from utils import *
from eigentools import *
from data import *
from GRAPE import Grape_ee_Flip, load_grape
from run_n_entangled_CX import run_n_entangled_CX


if __name__ == "__main__":
    run_n_entangled_CX(
        tN=400 * unit.ns, N=800, nS=2, max_time=5, lam=1e3, step=3
    )  # , J=J_single, A=A_1P_2P_single)
    # run_n_entangled_CX(
    #     tN=4000 * unit.ns,
    #     N=8000,
    #     nS=70,
    #     max_time=23.5 * 3600,
    #     lam=1e7,
    #     step=3
    #     # prev_grape_fp="fields/c1356_1S_3q_479ns_2500step",
    # )
