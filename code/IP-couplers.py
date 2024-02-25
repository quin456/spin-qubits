
from multi_NE import multi_NE_H0, multi_NE_Hw
from eigentools import *
from utils import *
from data import *
from visualisation import *
import gates as gate


from GRAPE import *





def analyse_3NE_transition():

    J = get_J(225, 3, J1=J_100_14nm, J2=J_100_18nm)
    A = get_A(1,1)
    Bz = 0*unit.mT
    H0 = multi_NE_H0(Bz, A, J[0])
    dim = H0.shape[-1]
    H0_phys = multi_NE_H0(2*unit.mT, A, J[0])
    S, D = get_ordered_eigensystem(H0, H0_phys)
    Hw_shape = gate.get_Sx_sum(3) + gate.get_Sy_sum(3) + gate.get_Ix_sum(3) + gate.get_Iy_sum(3)
    
    trans = get_allowed_transitions(S=S, D=D, Hw_shape=Hw_shape)
    rf = get_resonant_frequencies(S=S, D=D, Hw_shape=Hw_shape)


    for k in range(dim):
        print(f"|{k}> = {psi_to_string(S[:,k])}")

    # for t in trans:
    #     print(f"{psi_to_string(S[:,t[0]])} <--> {psi_to_string(S[:,t[0]])}")
    




if __name__ == '__main__':
    analyse_3NE_transition()