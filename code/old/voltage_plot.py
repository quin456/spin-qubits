import numpy as np 
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt 
import pdb

n_wires=5

N = 1000  #plot points

t_SW = 10

#t_CX = 10 #Oscar
t_CX = 1000 #Usman

t_tunnel = 100

t_CNOT = t_CX #2*t_SW + t_CX + 2*t_tunnel
t_readout = t_tunnel


# X stabilizer steps
X_steps = 6

Z_steps = 2

# measurement steps
M_steps = 2
I_steps = 2


offset=75
foffset=30

V0 = 0     # zero as in Supp.
Vn1=-16       # load code qubits
Vn2=-32     # load couplers
Vn3=-48     # change electrostatic potential

Vp = 18     # change electrostatic potential
Vw = -18
Vb = -36
dv = 4


colours = ['green', 'purple', 'red', 'blue', 'gold']



CNOT_57=np.array([Vn3,Vn3,Vn2,Vn1,V0])
CNOT_67=np.array([Vn3,Vn3,Vn1,Vn2,V0])
CNOT_45_16_35_26 = np.array([Vn2,Vn2,Vn3,Vn3,V0])

CNOT_45=np.array([Vn1,0,-30,0,0])
CNOT_16=np.array([Vn1,0,0,-30,0])
CNOT_35=np.array([0,Vn1,-30,0,0])
CNOT_26=np.array([0,Vn1,-10,Vn3,0])

#dumber
CNOT_57 = [Vn2,Vn2,Vn2,Vn1,Vn1]
CNOT_67 = [Vn2,Vn2,Vn1,Vn2,Vn1]
CNOT_45_16 = [Vn2,Vn3,Vn2,Vn2,V0]
CNOT_35_26 = [Vn3,Vn2,Vn2,Vn2,V0]

#dumb
CNOT_57 = [Vn3,Vn3,Vn2,Vn1,V0]
CNOT_67 = [Vn3,Vn3,Vn1,Vn2,V0]
CNOT_45_16 = [Vn3,Vn1,Vn2,Vn2,V0]
CNOT_35_26 = [Vn1,Vn3,Vn2,Vn2,V0]


unload_black = np.array([Vb+dv,Vb+dv,Vb-dv,Vb-dv,Vp])
unload_white = np.array([Vw+dv,Vw+dv,Vw-dv,Vw-dv,Vp])

unload_black_wait = np.array([Vb-dv,Vb-dv,Vb-dv,Vb-dv,Vp])
unload_white_wait = np.array([Vw-dv,Vw-dv,Vw-dv,Vw-dv,Vp])

init_white = np.array([Vw-dv,Vw-dv,Vw+dv,Vw+dv,Vp])
init_black = np.array([Vb-dv,Vb-dv,Vb+dv,Vb+dv,Vp])


def step_voltage(V_config, V, step):

    for i in range(n_wires):
        V[i,2*step] += V_config[i]
        V[i,2*step+1] += V_config[i]

    return V



def get_T(steps,t_step):

    t=0
    T = np.zeros(2*steps)

    for n in range(steps):
        T[2*n]=t 
        t=t+t_step
        T[2*n+1]=t
    return T

def CNOTs(V):


    
    V=step_voltage(CNOT_57, V, step=0)
    V=step_voltage(CNOT_67, V, step=1)
    V=step_voltage(CNOT_45_16, V, step=2)
    V=step_voltage(CNOT_35_26,V,step=3)
    V=step_voltage(CNOT_57, V, step=4)
    V=step_voltage(CNOT_67, V, step=5)
    return V


def Zstab(V):
    V = np.array([2*Z_steps*[offset*i] for i in range(n_wires)]) - foffset

    t=0
    T = get_T(Z_steps,t_CNOT)

    V=step_voltage(CNOT_45_16_35_26, V, step=1)

    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'red')
    plt.plot(T,V[3], color = 'blue')
    plt.plot(T,V[4], color = 'gold')

    plot_grid(T)
    for i in range(X_steps-1):
        plt.vlines((i+1)*t_CNOT,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')

    plt.annotate("(a)", (t_CNOT*(0.4), offset*(n_wires-0.7)))
    plt.annotate("(b)", (t_CNOT*(1.4), offset*(n_wires-0.7)))
    plt.annotate("(c)", (t_CNOT*(2.4), offset*(n_wires-0.7)))
    plt.annotate("(d)", (t_CNOT*(3.4), offset*(n_wires-0.7)))
    plt.annotate("(e)", (t_CNOT*(4.4), offset*(n_wires-0.7)))

    plt.savefig(filename)
    plt.show()

def plot_grid(T):
    
    for i in range(n_wires):
        plt.hlines(offset*i-foffset,0,T[-1], linewidth=0.95, linestyles='dashed',color=colours[i])
        for j in [Vn3,Vn2,Vn1]:
            plt.hlines(offset*i-foffset+j,0,T[-1], linewidth=0.45, linestyles='dashed',color='black')

    yticks = [0]*20
    for w in range(n_wires):
        yticks[(n_wires-1)*w] = Vn3 + offset*w - foffset
        yticks[(n_wires-1)*w+1] = Vn2 + offset*w - foffset
        yticks[(n_wires-1)*w+2] = Vn1 + offset*w - foffset
        yticks[(n_wires-1)*w+3] = offset*w - foffset


    plt.yticks(yticks, 5 * ['$V_{3}$', '$V_{2}$', '$V_{1}$', '$0$'])

    #plt.yticks([offset*i-foffset for i in range(n_wires)], n_wires*[0])

def plot_CNOTs(fp=None):
    V = np.array([2*X_steps*[offset*i] for i in range(n_wires)]) - foffset

    t=0
    T = get_T(X_steps,t_CNOT)
    V = CNOTs(V)

    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'red')
    plt.plot(T,V[3], color = 'blue')
    plt.plot(T,V[4], color = 'gold')

    plot_grid(T)
    for i in range(X_steps-1):
        plt.vlines((i+1)*t_CNOT,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')

    plt.annotate("(a)", (t_CNOT*(0.4), offset*(n_wires-0.7)))
    plt.annotate("(b)", (t_CNOT*(1.4), offset*(n_wires-0.7)))
    plt.annotate("(c)", (t_CNOT*(2.4), offset*(n_wires-0.7)))
    plt.annotate("(d)", (t_CNOT*(3.4), offset*(n_wires-0.7)))
    plt.annotate("(e)", (t_CNOT*(4.4), offset*(n_wires-0.7)))
    plt.annotate("(f)", (t_CNOT*(5.4), offset*(n_wires-0.7)))

    if fp is not None: plt.savefig(fp)


def measure(V):

    V = step_voltage(unload_black, V, step=0)
    V = step_voltage(unload_white, V, step=1)



    return V

def init_single(V):

    V = step_voltage(init_white, V, step=0)
    V = step_voltage(init_black, V, step=1)

    return V

'''
def plot_measurements(filename, d=1):
    
    V = np.array([2*d*(M_steps+I_steps)*[offset*i] for i in range(n_wires)])-foffset

    T = get_T(d*(M_steps+I_steps), t_readout)
    
    V[:,:2*M_steps]=measure(V[:,:2*M_steps])
    for i in range(d-1):
        V[:,2*M_steps+i*(M_steps+I_steps)] = V[:,2*M_steps-1]
    V[:,2*(d-1)*(M_steps+I_steps)+2*M_steps:]=init_single(V[:,2*(d-1)*(M_steps+I_steps)+2*M_steps:])
    

    
    plot_grid(T)

    plt.vlines(t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(2*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')


    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'red')
    plt.plot(T,V[3], color = 'blue')
    plt.plot(T,V[4], color = 'gold')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')

    plt.savefig(filename)
    plt.show()
'''


def plot_more_measurements(filename):
    # distance is 5 and its hardcoded :)
    d=5

    rows=(1+d//2)
    steps = rows*M_steps+I_steps
    tmin = 0
    tmax = steps * t_tunnel
    T= np.linspace(tmin, tmax, N)
    dt = T[1]-T[0]
    V = np.array([N*[offset*i] for i in range(n_wires+4)])-foffset
    V[-1]-=foffset


    i=0
    while T[i] < t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        for w in range(2):
            V[2*w,i]+=unload_black_wait[w]
            V[2*w+1,i]+=unload_black_wait[w]
        i+=1


    while T[i] < 2*t_tunnel:
        for w in range(2,n_wires):
            V[w+4,i] += unload_black[w]
        
        V[0,i]+=unload_black_wait[0]
        V[1,i]+=unload_black_wait[1]
        V[2,i]+=unload_black[0]
        V[3,i]+=unload_black[1]
        V[4,i]+=unload_black_wait[0]
        V[5,i]+=unload_black_wait[1]
        i+=1

    
    while T[i] < 3*t_tunnel:
        for w in range(2,n_wires):
            V[w+4,i] += unload_black[w]
        
        V[0,i]+=unload_black[0]
        V[1,i]+=unload_black[1]
        V[2,i]+=unload_black_wait[0]
        V[3,i]+=unload_black_wait[1]
        V[4,i]+=unload_black_wait[0]
        V[5,i]+=unload_black_wait[1]
        i+=1

    while T[i] < rows*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w+4,i] += unload_black[w]
            i+=1
            t+=dt
    
    while T[i] < (rows+1)*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        for w in range(2):
            V[2*w,i]+=unload_white_wait[w]
            V[2*w+1,i]+=unload_white_wait[w]
        i+=1

    while T[i] < (rows+2)*t_tunnel:
        for w in range(2,n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white_wait[0]
        V[1,i]+=unload_white_wait[1]
        V[2,i]+=unload_white[0]
        V[3,i]+=unload_white[1]
        V[4,i]+=unload_white_wait[0]
        V[5,i]+=unload_white_wait[1]
        i+=1

    while T[i] < (rows+3)*t_tunnel:
        for w in range(2,n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white[0]
        V[1,i]+=unload_white[1]
        V[2,i]+=unload_white_wait[0]
        V[3,i]+=unload_white_wait[1]
        V[4,i]+=unload_white_wait[0]
        V[5,i]+=unload_white_wait[1]
        i+=1

    while T[i] < (2*rows)*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w+4,i] += unload_white[w]
            i+=1
            t+=dt

    while T[i] < tmax - t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white[0]
        V[1,i]+=unload_white[1]
        V[2,i]+=unload_white[0]
        V[3,i]+=unload_white[1]
        i+=1

    while i < len(T):
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        V[0,i]+=unload_black[0]
        V[1,i]+=unload_black[1]
        V[2,i]+=unload_black[0]
        V[3,i]+=unload_black[1]
        i+=1
    

    #plot_grid(T)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.vlines(2*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(3*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+2)*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+1)*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-2*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')

    for i in range(n_wires-1):
        plt.hlines(offset*(i+4)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i+4)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+4)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')

    for i in range(2):
        plt.hlines(offset*(i)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i+2)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+2)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+2)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')


    plt.hlines(offset*8-2*foffset,0,T[-1], linewidth=0.95, linestyles='dashed',color=colours[4])

    yticks = [0]*(3*(n_wires+4-1)+2)
    for i in range(n_wires+4-1):
        yticks[3*i] = i*offset - foffset + Vb
        yticks[3*i+1] = i*offset - foffset + Vw
        yticks[3*i+2] = i*offset - foffset
    yticks[-2]=8*offset-2*foffset
    yticks[-1]=8*offset-2*foffset+Vp

    plt.yticks(yticks, (n_wires+4-1)*['Vb', 'Vw', 0] + [0, 'Vt'])


    #plt.yticks([offset*i-foffset for i in range(n_wires)], n_wires*[0])


    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'green')
    plt.plot(T,V[3], color = 'purple')
    plt.plot(T,V[4], color = 'green')
    plt.plot(T,V[5], color = 'purple')
    plt.plot(T,V[6], color = 'red')
    plt.plot(T,V[7], color = 'blue')
    plt.plot(T,V[8], color = 'gold')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')


    plt.annotate("(1)", (t_tunnel*(0.25), offset*(n_wires+4-1.1)))
    plt.annotate("(2)", (t_tunnel*(1.25), offset*(n_wires+4-1.1)))
    plt.annotate("(3)", (t_tunnel*(2.25), offset*(n_wires+4-1.1)))
    plt.annotate("(4)", (t_tunnel*(3.25), offset*(n_wires+4-1.1)))
    plt.annotate("(5)", (t_tunnel*(4.25), offset*(n_wires+4-1.1)))
    plt.annotate("(6)", (t_tunnel*(5.25), offset*(n_wires+4-1.1)))
    plt.annotate("(7)", (t_tunnel*(6.25), offset*(n_wires+4-1.1)))
    plt.annotate("(8)", (t_tunnel*(7.25), offset*(n_wires+4-1.1)))

    #ax.set_aspect(1/)
    plt.savefig(filename)
    plt.show()


def plot_Zstab_measurements(filename):
    # distance is 5 and its hardcoded :)
    d=5

    rows=(1+d//2)
    steps = rows*M_steps+I_steps
    tmin = 0
    tmax = steps * t_tunnel
    T= np.linspace(tmin, tmax, N)
    dt = T[1]-T[0]
    V = np.array([N*[offset*i] for i in range(n_wires+4)])-foffset
    V[-1]-=foffset


    i=0
    while T[i] < t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        for w in range(2):
            V[2*w,i]+=unload_black_wait[w]
            V[2*w+1,i]+=unload_black_wait[w]
        i+=1


    while T[i] < 2*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        
        V[0,i]+=unload_black_wait[0]
        V[1,i]+=unload_black_wait[1]
        V[2,i]+=unload_black[0]
        V[3,i]+=unload_black[1]
        i+=1

    
    while T[i] < 3*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        
        V[0,i]+=unload_black[0]
        V[1,i]+=unload_black[1]
        V[2,i]+=unload_black[0]
        V[3,i]+=unload_black[1]
        i+=1

    while T[i] < rows*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w+4,i] += unload_black[w]
            i+=1
            t+=dt
    
    while T[i] < (rows+1)*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        for w in range(2):
            V[2*w,i]+=unload_white_wait[w]
            V[2*w+1,i]+=unload_white_wait[w]
        i+=1

    while T[i] < (rows+2)*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white_wait[0]
        V[1,i]+=unload_white_wait[1]
        V[2,i]+=unload_white[0]
        V[3,i]+=unload_white[1]
        i+=1

    while T[i] < (rows+3)*t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white[0]
        V[1,i]+=unload_white[1]
        V[2,i]+=unload_white[0]
        V[3,i]+=unload_white[1]
        i+=1

    while T[i] < (2*rows)*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w+4,i] += unload_white[w]
            i+=1
            t+=dt

    while T[i] < tmax - t_tunnel:
        for w in range(n_wires):
            V[w+4,i] += unload_white[w]
        V[0,i]+=unload_white[0]
        V[1,i]+=unload_white[1]
        V[2,i]+=unload_white[0]
        V[3,i]+=unload_white[1]
        i+=1

    while i < len(T):
        for w in range(n_wires):
            V[w+4,i] += unload_black[w]
        V[0,i]+=unload_black[0]
        V[1,i]+=unload_black[1]
        V[2,i]+=unload_black[0]
        V[3,i]+=unload_black[1]
        i+=1
    

    #plot_grid(T)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.vlines(2*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(3*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+2)*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+1)*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-2*t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-t_tunnel,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax,-1*offset,offset*(n_wires+4-0.5), linewidth=0.6,linestyles='dashed',color='black')

    for i in range(n_wires-1):
        plt.hlines(offset*(i+4)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i+4)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+4)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')

    for i in range(2):
        plt.hlines(offset*(i)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i+2)-foffset,0,T[-1], linewidth=0.75, linestyles='dashed',color=colours[i])
        plt.hlines(offset*(i)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+2)-foffset+Vb,0,T[-1], linewidth=0.75, linestyles='dashed')
        plt.hlines(offset*(i+2)-foffset+Vw,0,T[-1], linewidth=0.75, linestyles='dashed')


    plt.hlines(offset*8-2*foffset,0,T[-1], linewidth=0.95, linestyles='dashed',color=colours[4])

    yticks = [0]*(3*(n_wires+4-1)+2)
    for i in range(n_wires+4-1):
        yticks[3*i] = i*offset - foffset + Vb
        yticks[3*i+1] = i*offset - foffset + Vw
        yticks[3*i+2] = i*offset - foffset
    yticks[-2]=8*offset-2*foffset
    yticks[-1]=8*offset-2*foffset+Vp

    plt.yticks(yticks, (n_wires+4-1)*['Vb', 'Vw', 0] + [0, 'Vt'])


    #plt.yticks([offset*i-foffset for i in range(n_wires)], n_wires*[0])


    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'green')
    plt.plot(T,V[3], color = 'purple')
    plt.plot(T,V[4], color = 'green')
    plt.plot(T,V[5], color = 'purple')
    plt.plot(T,V[6], color = 'red')
    plt.plot(T,V[7], color = 'blue')
    plt.plot(T,V[8], color = 'gold')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')


    plt.annotate("(1)", (t_tunnel*(0.25), offset*(n_wires+4-1.1)))
    plt.annotate("(2)", (t_tunnel*(1.25), offset*(n_wires+4-1.1)))
    plt.annotate("(3)", (t_tunnel*(2.25), offset*(n_wires+4-1.1)))
    plt.annotate("(4)", (t_tunnel*(3.25), offset*(n_wires+4-1.1)))
    plt.annotate("(5)", (t_tunnel*(4.25), offset*(n_wires+4-1.1)))
    plt.annotate("(6)", (t_tunnel*(5.25), offset*(n_wires+4-1.1)))
    plt.annotate("(7)", (t_tunnel*(6.25), offset*(n_wires+4-1.1)))
    plt.annotate("(8)", (t_tunnel*(7.25), offset*(n_wires+4-1.1)))

    #ax.set_aspect(1/)
    plt.savefig(filename)
    plt.show()


def plot_measurements(filename, d=5):
    rows=(1+d//2)
    steps = rows*M_steps+I_steps
    tmin = 0
    tmax = steps * t_tunnel
    T= np.linspace(tmin, tmax, N)
    dt = T[1]-T[0]
    V = np.array([N*[offset*i] for i in range(n_wires)])-foffset
    V[-1]-=foffset


    i=0
    while T[i] < t_tunnel:
        for w in range(n_wires):
            V[w,i] += unload_black_wait[w]
        i+=1


    while T[i] < rows*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w,i] += unload_black[w]
            i+=1
            t+=dt
    
    while T[i] < (rows+1)*t_tunnel:
        for w in range(n_wires):
            V[w,i] += unload_white_wait[w]
        i+=1

    while T[i] < (2*rows)*t_tunnel:
        t=0
        while t < t_tunnel:
            for w in range(n_wires):
                V[w,i] += unload_white[w]
            i+=1
            t+=dt

    while T[i] < tmax - t_tunnel:
        for w in range(n_wires):
            V[w,i] += unload_white[w]
        i+=1

    while i < len(T):
        for w in range(n_wires):
            V[w,i] += unload_black[w]
        i+=1
    

    #plot_grid(T)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.vlines(2*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(3*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+2)*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines((rows+1)*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-2*t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax-t_tunnel,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')
    plt.vlines(tmax,-1*offset,offset*(n_wires-0.5), linewidth=0.6,linestyles='dashed',color='black')

    for i in range(n_wires-1):
        plt.hlines(offset*i-foffset,0,T[-1], linewidth=0.95, linestyles='dashed',color=colours[i])
        plt.hlines(offset*i-foffset+Vb,0,T[-1], linewidth=0.95, linestyles='dashed')
        plt.hlines(offset*i-foffset+Vw,0,T[-1], linewidth=0.95, linestyles='dashed')
    plt.hlines(offset*4-2*foffset,0,T[-1], linewidth=0.95, linestyles='dashed',color=colours[4])

    yticks = [0]*(3*(n_wires-1)+2)
    for i in range(n_wires-1):
        yticks[3*i] = i*offset - foffset + Vb
        yticks[3*i+1] = i*offset - foffset + Vw
        yticks[3*i+2] = i*offset - foffset
    yticks[-2]=4*offset-2*foffset
    yticks[-1]=4*offset-2*foffset+Vp

    plt.yticks(yticks, (n_wires-1)*['Vb', 'Vw', 0] + [0, 'Vt'])


    #plt.yticks([offset*i-foffset for i in range(n_wires)], n_wires*[0])


    plt.plot(T,V[0], color = 'green')
    plt.plot(T,V[1], color = 'purple')
    plt.plot(T,V[2], color = 'red')
    plt.plot(T,V[3], color = 'blue')
    plt.plot(T,V[4], color = 'gold')


    plt.ylabel('Gate voltages')
    plt.xlabel('time (ns)')


    plt.annotate("(1)", (t_tunnel*(0.25), offset*(n_wires-1.1)))
    plt.annotate("(2)", (t_tunnel*(1.25), offset*(n_wires-1.1)))
    plt.annotate("(3)", (t_tunnel*(2.25), offset*(n_wires-1.1)))
    plt.annotate("(a)", (t_tunnel*(1.25), offset*(n_wires-0.7)))
    plt.annotate("(b)", (t_tunnel*(rows+1.25), offset*(n_wires-0.7)))
    plt.annotate("(c)", (t_tunnel*(2*rows+0.25), offset*(n_wires-0.7)))
    plt.annotate("(e)", (t_tunnel*(2*rows+1.25), offset*(n_wires-0.7)))

    #ax.set_aspect(1/)
    plt.savefig(filename)
    plt.show()



def total_measurement(d=5):
    V = np.array([2*(M_steps+I_steps)*d*[offset*i] for i in range(n_wires*d)])-foffset
    print(np.shape(V))

    for i in range(d):
        #V[d:(d+1)*n_wires,i*n_wires:(i+1)*n_wires] = step_voltage(unload_black, V[d:(d+1)*n_wires,i*n_wires:(i+1)*n_wires], step=i*d+1)
        V[d:d+n_wires,i*n_wires:(i+1)*n_wires] = step_voltage(unload_white, V[d:d+n_wires,i*n_wires:(i+1)*n_wires], step=i*d+2)
        #V[d:d+n_wires,i*n_wires:(i+1)*n_wires] = step_voltage(init_white, V[d:d+n_wires,i*n_wires:(i+1)*n_wires], step=i*d+3)
        #V[d:d+n_wires,i*n_wires:(i+1)*n_wires] = step_voltage(init_white, V[d:d+n_wires,i*n_wires:(i+1)*n_wires], step=i*d+4)
    
    t=0
    T = np.zeros(2*(M_steps+I_steps)*d)
    for n in range((M_steps+I_steps)*d):
        T[2*n]=t 
        t=t+t_CNOT 
        T[2*n+1]=t


    return T,V

def plot_total_measurement(filename=None,d=3):
    T,V = total_measurement(d)
    for i in range(n_wires):
        plt.hlines(offset*i,0,T[-1],linestyles='dashed',color='black')

    for i in range(d):
        plt.plot(T,V[i*d+0], color = 'green')
        plt.plot(T,V[i*d+1], color = 'purple')
        plt.plot(T,V[i*d+2], color = 'red')
        plt.plot(T,V[i*d+3], color = 'blue')
        plt.plot(T,V[i*d+4], color = 'gold')

    plt.show()






def plot_all(filename = None,d=3):


    steps = X_steps + d * (M_steps+I_steps)
    T = get_T(steps)

    V = np.array([2*steps*[offset*i] for i in range(d*n_wires)])

    V_m = np.zeros((n_wires,2*M_steps))

    V_m = measure(V_m)
    print(V_m)

    for i in range(d):
        V[i*n_wires:(i+1)*n_wires, 0:2*X_steps] = CNOTs(V[i*n_wires:(i+1)*n_wires, 0:2*X_steps])

        
        V[i*n_wires:(i+1)*n_wires, 2*X_steps+2*i*M_steps:2*X_steps+2*(i+1)*M_steps] = measure(V[i*n_wires:(i+1)*n_wires, 2*X_steps+2*i*M_steps:2*X_steps+2*(i+1)*M_steps])

        #V[i*n_wires:(i+1)*n_wires, 2*X_steps+2*i*M_steps:2*X_steps+2*(i+1)*M_steps] = measure(V[i*n_wires:(i+1)*n_wires, 2*X_steps+2*i*M_steps:2*X_steps+2*(i+1)*M_steps])
        
        for j in range(d):
            for s in range(2*M_steps):
                V[i*n_wires, 2*X_steps+2*j*M_steps+s] += V_m[0,s]
                #V[i*n_wires+1, 2*X_steps+2*j*M_steps:2*X_steps+2*(j+1)*M_steps]


    for i in range(d):
        plt.plot(T,V[i*n_wires+0], color = 'green')
        plt.plot(T,V[i*n_wires+1], color = 'purple')
        plt.plot(T,V[i*n_wires+2], color = 'red')
        plt.plot(T,V[i*n_wires+3], color = 'blue')
        plt.plot(T,V[i*n_wires+4], color = 'gold')
    if filename:
        plt.savefig(filename)
    plt.show()




if __name__ == '__main__':
    #plot_all()



    #plot_more_measurements("all_measure_voltages.pdf")
    #plot_total_measurement()
    plot_CNOTs()
    #plot_measurements("measure_voltages.pdf")

    #plot_total_measurement()
    plt.show()


