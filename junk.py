
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


