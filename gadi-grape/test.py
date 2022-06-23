
import torch as pt 
import scipy
import numpy as np
import matplotlib.pyplot as plt
import time




#print(x)
#print(y)

#plt.plot(x,y)
#plt.savefig("tensorplot")

ngpus = pt.cuda.device_count()
print(f"ngpus={ngpus}")

T1 = pt.zeros(1000,1000,600, device='cuda',dtype=pt.complex128)
T2=pt.matrix_exp(T1)

