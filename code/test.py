import torch as pt
import gates as gate
import numpy as np
import time

from typing import List

depth = 0


# def propagate(U):

#     Uf = U[0]
#     for Uj in U[1:]:
#         Uf = Uj @ Uf
#     return Uf


# def merge_prop(U):
#     N = len(U)
#     if len(U) == 1:
#         return U
#     return merge_prop(U[N // 2 :]) @ merge_prop(U[: N // 2])


def my_func(x: List[int])->float:
    return float(x)

if __name__ == "__main__":
    from collections import defaultdict 
    def f(): return 0
    di = defaultdict(f)
    di[1]=2
    breakpoint()


