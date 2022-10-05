
def placement_symmetries(delta_x = 50, delta_y = 0):

    sites_2P_xs = np.linspace(0,4,5)
    sites_2P_ys = np.linspace(0,1,2)
    sites_1P_xs = np.linspace(1,3,3)
    sites_1P_ys = np.linspace(0,1,2)

    sites_2P = []
    sites_1P = []

    for x in sites_2P_xs:
        for y in sites_2P_ys:
            sites_2P.append((x,y))
    for x in sites_1P_xs:
        for y in sites_1P_ys:
            sites_1P.append((x+delta_x,y+delta_y))

    unique_2P = [] #indices of unique 2P sites 

    donor_pairs = list(combinations(sites_2P, 2))
    
    distances = []
    for pair in donor_pairs:
        d = distance(*pair)
        if d not in distances:
            distances.append(d)

    print(f"dx = {delta_x}, dy = {delta_y}")

    print(f"Unique 2P hyperfines = {len(distances)}")

    distance_tups = []
    distance_2A = []
    distance_tups_unique = []
    for pair in donor_pairs:
        distance_2A.append(distance(*pair))
        for site in sites_1P:
            dtup = (distance(*pair), distance(pair[0],site), distance(pair[1],site))
            distance_tups.append(dtup)
            if (dtup not in distance_tups_unique) and ((dtup[0], dtup[2], dtup[1]) not in distance_tups_unique):
                distance_tups_unique.append(dtup)

    print(f"Unique J-couplings = {len(distance_tups_unique)}")

    distance_tups_unique = []
    distance_2A_checks = []
    for i in range(len(distance_2A)):
        for j in range(len(distance_tups)):
            dtup = distance_tups[j]
            if (dtup not in distance_tups_unique) and ((dtup[0], dtup[2], dtup[1]) not in distance_tups_unique):
                #if distance(2A)
                distance_tups_unique.append(distance_tups[j])

    print(f"Unique J-couplings = {len(distance_tups_unique)}")






def save_grapes(J=get_J_1P_2P(48), A=get_A_1P_2P(48), tN=5000*unit.ns, N=5000, idxs = np.linspace(0,47,48).astype(int), max_time=60, lam=1e8):

    for i in idxs:
        grape = GrapeESR_AJ_Modulation(J[i], A[i], tN, N, Bz=0, target=CNOT_targets(1,2), max_time=max_time, lam=lam)
        grape.run()
        grape.save(f"grape_bunch/grape{i}")


def test_sum(tN = 5000*unit.ns, N=5000, nS=2, max_time=15, div=1, lam=0, save_grapes=False):
    save_data=False
    nq=2
    target_single = CNOT_targets(1, nq)
    J = get_J_1P_2P(48)
    A = get_A_1P_2P(48)

    grapes = []

    if save_grapes:
        grape = GrapeESR_AJ_Modulation(J[0], A[0], tN, N, Bz=0, target=target_single, max_time=max_time, save_data=save_data, lam=lam)
        grape.run()
        grape.save("grape_bunch/grape0")
        grape = GrapeESR_AJ_Modulation(J[1], A[1], tN, N, Bz=0, target=target_single, max_time=max_time, save_data=save_data, lam=lam)
        grape.run()
        grape.save("grape_bunch/grape1")
    
    grape0 = load_grape("grape_bunch/grape0", GrapeESR_AJ_Modulation)
    grape0.print_result()
    grape1 = load_grape("grape_bunch/grape1", GrapeESR_AJ_Modulation)
    grapes=[grape0, grape1]

    grape = sum_grapes(grapes)
    grape.plot_result()

    for i in range(48):
        grape.J = J[i]/div
        grape.A = A[i]
        grape.H0=grape.get_H0()
        #X_free = get_electron_X(grape.tN, grape.N, 0, grape.A, grape.J)
        X_free = get_X_from_H(grape.H0, tN, N)
        grape.propagate()


        print(f"CX, free fids: {fidelity(grape.X[0,-1], gate.CX):.3f} {fidelity(grape.X[0,-1], X_free[0,-1]):.3f}")

