



def linear_Bz_decrease(tN=50*nanosecond, N=10000, psi0=pt.kron(spin_up,spin_down), A=get_A(1,1)*Mhz, Bz=2*tesla):
    H0 = H_hyperfine(A) + H_zeeman(Bz)

    t_sw = np.pi/(4*A)
    N_swap = int(t_sw*N/tN) 
    N_dropoff = N-N_swap
    
    u = pt.cat((pt.linspace(0,-2,N_dropoff), -2*pt.ones(N_swap)))
    H_control = 1*tesla * (gamma_e*Sz - gamma_P*Iz)

    Hw = pt.einsum('j,ab->jab', u, H_control)

    H = sum_H0_Hw(H0,Hw)
    U = get_subops(H,tN/N)
    X = forward_prop(U)
    psi = pt.matmul(X,psi0)

    plot_spin_states(psi,tN)


def NE_grape_X(u, H0, H_control, tN, N, target = gate.swap):
    
    pulse = gaussian(u, tN, N) # (tesla - as in, if I print it it will be around 2 teslas. NOT in atomic units yet.)
    pulse = sigmoid_pulse(u, tN, N)
    Hw = pt.einsum('j,ab->jab',pulse,H_control)

    H = sum_H0_Hw(H0,Hw)
    U = get_subops(H,tN/N)
    X = forward_prop(U)
    return X

def NE_grape_cost(u, H0, H_control, tN, N, target = gate.swap):
    X = NE_grape_X(u, H0, H_control, tN, N, target = target)
    return 1 - fidelity(X[-1], target)


def gaussian(u, tN, N):
    T = pt.linspace(0,tN,N)
    t_center = tN/2
    return u[0]*pt.exp(-((T-t_center)/(u[1]*nanosecond))**2)    

def sigmoid(x,alpha,A):
    return A / (1 + pt.exp(-alpha*x))


def sigmoid_pulse(u, tN, N):
    alpha,A,w,t0 = u
    T = pt.linspace(0,tN,N) / nanosecond
    return sigmoid(T-t0,alpha,A) - sigmoid(T-w-t0,alpha,A)

def optimise_NE_swap_Bz():

    tN=100*nanosecond
    N=500
    A = get_A(1,1) * Mhz
    Bz = 2*tesla
    sigma0 = 5 # spread of pulse
    mag0 = -2 # maximum magnitude of pulse

    H0 = NE_H0(A,Bz)
    H_control = 1*tesla * (gamma_e*Sz - gamma_P*Iz)
    cost = lambda u: NE_grape_cost(u,H0,H_control,tN,N)

    w0= 1.6*np.pi/(4*A) / nanosecond
    alpha0 = 5
    amp0=-2
    t0_0 = 10
    u0 = pt.tensor([alpha0,amp0,w0, t0_0])


    optimisation = minimize(cost,u0, method = 'Nelder-Mead')
    u = optimisation.x

    X = NE_grape_X(u, H0, H_control, tN, N)
    psi0 = pt.kron(spin_up,spin_down)
    psi = pt.matmul(X,psi0)

    fig,ax=plt.subplots(1,2)
    plot_spin_states(psi,tN,ax[0])



    ax[1].plot(pt.linspace(0,tN/nanosecond,N),sigmoid_pulse(u, tN, N))
    print(optimisation)


def analyse_3NE_eigensystem(S):
    dim = 64

    print("============================================================================================================================================")

    for j in range(dim):
        evec = S[:,j]
        print(f"Eigenstatestate {j}:")
        for i in range(dim):
            if pt.real(S[i,j]) > 0.01:
                print(f"---> {np.binary_repr(i,6)}, Pr={pt.abs(S[i,j])**2}")
