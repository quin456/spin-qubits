import torch as pt
import gates as gate
import numpy as np
import time

from typing import List


A = [1,2,3,4,5,6,7,8,9,0]
for k in range(len(A)):
    A.pop(0)
    print(len(A),k)


