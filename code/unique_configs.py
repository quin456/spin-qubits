import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
import atomic_units as unit
from data import *
from utils import maxreal, minreal

folder = "exchange_data_fab/"


def get_unique_configs(fp):
    pos_1P_og = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])
    pos_1P = np.array([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    pos_2P_1_offset_1 = np.array([5, 100])
    pos_2P_2_offset_1 = np.array([-5, 100])
    pos_2P_1_offset_2 = np.array([40, 0])
    pos_2P_2_offset_2 = np.array([50, 0])

    # 3 and 4 are dumb
    pos_2P_1_offset_3 = np.array([100, 5])
    pos_2P_2_offset_3 = np.array([100, -5])
    pos_2P_1_offset_4 = np.array([0, 60])
    pos_2P_2_offset_4 = np.array([0, 50])
    pos_2P_1 = pos_1P + np.einsum("i,j->ji", np.ones(6), pos_2P_1_offset_2)
    pos_2P_2 = pos_1P + np.einsum("i,j->ji", np.ones(6), pos_2P_2_offset_2)

    # pos_2P_1 = np.array([[10, 11, 12, 10, 11, 12], [5, 5, 5, 6, 6, 6]])
    # pos_2P_2 = np.array([[10, 11, 12, 10, 11, 12], [-5, -5, -5, -4, -4, -4]])

    unique_configs = {}

    unique_hyperfines = {}
    A_dict = {}
    i_A = 0

    # J_1P_2P = pt.cat((J_100_110_18nm, J_rand))
    A_1P_2P = np.array(
        [36.6, 36.3, 35.7, 35.2, 35.1, 34.9, 34.8, 34.5, 33.0, 35.8, 33.9]
    )

    # A_1P_2P = pt.load(f"{folder}{fp}_A_gen")
    J_1P_2P = pt.load(f"{folder}{fp}_J_gen")

    def distance(r1, r2):
        return np.sqrt(np.dot(r1 - r2, r1 - r2))

    for i1 in range(6):
        for i2 in range(6):
            for i3 in range(6):
                r1 = pos_1P[:, i1]
                r2 = pos_2P_1[:, i2]
                r3 = pos_2P_2[:, i3]
                d1 = distance(r1, r2)
                d2 = distance(r1, r3)
                d3 = distance(r2, r3)

                if d1 > d2:
                    dtemp = d1
                    d1 = d2
                    d2 = dtemp

                d_trip = f"{d1:.2f},{d2:.2f},{d3:.2f}"

                if d3 not in A_dict.keys():
                    A_dict[d3] = A_1P_2P[i_A]
                    i_A += 1
                unique_configs[d_trip] = unique_configs.get(d_trip, []) + [[d1, d2, d3]]
                unique_hyperfines[f"{d3:.2f}"] = unique_hyperfines.get(
                    f"{d3:.2f}", []
                ) + [[i2, i3]]

    n_sys = len(unique_configs.keys())

    A_1P_2P = [[] for i in range(n_sys)]
    keys = list(unique_configs.keys())

    for i in range(n_sys):
        A_key = unique_configs[keys[i]][0][2]
        A_1P_2P[i] = [A_P / unit.MHz, A_dict[A_key]]

    J_100_110_18nm = pt.cat((J_100_18nm, J_110_18nm))
    J_rand = pt.rand(39) * (
        maxreal(J_100_110_18nm) - minreal(J_100_110_18nm)
    ) + minreal(J_100_110_18nm)

    print(f"{len(unique_configs.keys())} unique configurations")

    # for key in unique_configs.keys():
    #     print(key)
    #     print(unique_configs[key])
    return A_1P_2P, J_1P_2P


def save_unique_configs(fp):
    A, J = get_unique_configs(fp)
    pt.save(pt.tensor(A), f"{exch_data_folder}{fp}_A")
    pt.save(pt.tensor(J), f"{exch_data_folder}{fp}_J")


def generate_A_and_J_from_uniform(
    nA=100, nJ=710, Amin=32, Amax=40, Jmin=10, Jmax=100, fp=None
):

    A_gen = np.random.uniform(Amin, Amax, nA)
    J_gen = np.random.uniform(Jmin, Jmax, nJ)

    pt.save(A_gen, f"{folder}{fp}_A_gen")
    pt.save(J_gen, f"{folder}{fp}_J_gen")


def inspect_A_J(fp="U_1P_2P"):
    A = pt.load(f"{exch_data_folder}{fp}_A")
    J = pt.load(f"{exch_data_folder}{fp}_J")
    print(A)
    print(J)
    print(f"nA = {len(A)}")
    print(f"nJ = {len(J)}")


def add_one_J():
    pass


#############################################################################
### EVERYTHING BELOW HERE TAKEN FROM MISC CALCS
#############################################################################


def plot_sites(sites, ax, color="red"):
    for site in sites:
        circle = plt.Circle(site, radius=0.1, color=color)
        ax.add_patch(circle)


def plot_donor_placement(
    sites_2P_upper, sites_2P_lower, sites_1P, delta_x, delta_y, d=1, i=3, j=1, k=4
):

    fig, ax = plt.subplots(1)
    ax.set_aspect(1)
    plot_squares(sites_2P_upper, ax=ax, L=d)
    plot_squares(sites_2P_lower, ax=ax, L=d)
    plot_squares(sites_1P, ax=ax, L=d)
    plot_sites((sites_2P_lower[i], sites_2P_upper[j], sites_1P[k]), ax)

    if delta_y == 0:
        head_length = 0.3
        head_width = 0.15
        ax.arrow(delta_x - 3, 0, 0, 1, head_length=head_length, head_width=head_width)
        ax.arrow(delta_x - 3, 0, 1, 0, head_length=head_length, head_width=head_width)
        ax.text(delta_x - 1.5, 0, "[1,-1,0]", rotation=0)
        ax.text(delta_x - 3.6, 1.5, "[1,1,0]", rotation=0)
        # ax.set_xlabel('[1,1,0]')
        # ax.set_ylabel('[-1,1,0]')
        ax.axis("off")
    elif delta_x == 0:
        ax.set_xlabel("[1,-1,0]")
        ax.set_ylabel("[1,1,0]")


def plot_select_donor_placement(delta_x=10, delta_y=0):

    sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites_1P_2P(delta_x, delta_y)
    plot_donor_placement(sites_2P_upper, sites_2P_lower, sites_1P, delta_x, delta_y)


def classify_2P_pairs(donor_2P_pairs):
    distances = []
    for pair in donor_2P_pairs:
        d = distance(*pair)
        if d not in distances:
            distances.append(d)
    print(f"Unique 2P hyperfines = {len(distances)}")


def midpoint(r1, r2):
    return ((r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2)


def get_unique_distance_triples(sites_2P_upper, sites_2P_lower, sites_1P):
    """
    Gets list of distances of the form(d(2P_1, 2P_2), d(2P_1, 1P), d(2P_2, 1P))
    """
    donor_2P_pairs = []
    donor_triples = []
    for r1 in sites_2P_upper:
        for r2 in sites_2P_lower:
            donor_2P_pairs.append((r1, r2))
            for r3 in sites_1P:
                donor_triples.append((r1, r2, r3))

    classify_2P_pairs(donor_2P_pairs)
    # classify_triplets()

    # def classify_triplets(triplets)
    distance_tups = []
    distance_tups_unique = []

    # define distances between 2Ps and from 2P midpoint to 1P
    d_pairs = []
    d_pairs_unique = []
    for pair in donor_2P_pairs:
        for site in sites_1P:
            dtup = (distance(*pair), distance(pair[0], site), distance(pair[1], site))
            distance_tups.append(dtup)
            d_pair = (distance(*pair), distance(midpoint(*pair), site))
            d_pairs.append(d_pair)
            if (dtup not in distance_tups_unique) and (
                (dtup[0], dtup[2], dtup[1]) not in distance_tups_unique
            ):
                distance_tups_unique.append(dtup)
            if d_pair not in d_pairs_unique:
                d_pairs_unique.append(d_pair)

    print(f"{len(distance_tups_unique)} unique systems using d_trips")
    print(f"{len(d_pairs_unique)} unique systems using d_pairs")
    return d_pairs_unique


def get_AJ_distances(d_pairs):
    d_A = []
    d_J = []
    for d_pair in d_pairs:
        if d_pair[0] not in d_A:
            d_A.append(d_pair[0])
        if d_pair[1] not in d_J:
            d_J.append(d_pair[1])
    return d_A, d_J


def get_donor_sites_1P_2P(delta_x, delta_y, d=1, separation_2P=2, orientation=0):

    delta_x *= d
    delta_y *= d

    sites_2P_xs = np.linspace(1, 3, 3) * d
    sites_2P_ys_lower = np.linspace(-1, 0, 2) * d
    sites_2P_ys_upper = np.linspace(1 + separation_2P, 2 + separation_2P, 2) * d
    sites_1P_xs = np.linspace(1, 3, 3) * d
    sites_1P_ys = np.linspace(0 + separation_2P / 2, 1 + separation_2P / 2, 2) * d
    sites_2P_lower = []
    sites_2P_upper = []
    sites_2P_left = []
    sites_2P_right = []
    sites_1P_1 = []
    sites_1P_0 = []

    for x in sites_2P_xs:
        for y in sites_2P_ys_upper:
            sites_2P_upper.append((x, y))
            sites_2P_right.append((y + 2, x))
        for y in sites_2P_ys_lower:
            sites_2P_lower.append((x, y))
            sites_2P_left.append((y + 2, x))
    for x in sites_1P_xs:
        for y in sites_1P_ys:
            sites_1P_0.append((y + 1 + delta_x, x + delta_y))
            sites_1P_1.append((x + delta_x, y + delta_y))
    if orientation == 0:
        return sites_2P_left, sites_2P_right, sites_1P_0
    else:
        return sites_2P_upper, sites_2P_lower, sites_1P_1


def placement_symmetries_1P_2P(
    delta_x=50, delta_y=0, separation_2P=2, verbose=False, save=False
):

    d = 1
    J_rand = minreal(J_100_18nm) + pt.rand(100) * (
        maxreal(J_100_18nm) - minreal(J_100_18nm)
    )
    J_2P_1P_fab = pt.cat((J_100_18nm, J_110_18nm, J_rand))
    A_2P_fab_old = (
        pt.tensor(
            [87.3, 90.2, 87.0, 88.9, 74.7, 78.5, 69.2, 69.1, 63.3], dtype=cplx_dtype
        )
        * unit.MHz
    )
    A_2P_fab = (
        pt.tensor(
            [36.3, 35.7, 35.2, 34.9, 34.8, 34.5, 33.0, 31.9, 31.7, 30.7],
            dtype=cplx_dtype,
        )
        * unit.MHz
    )

    sites_2P_upper, sites_2P_lower, sites_1P = get_donor_sites_1P_2P(
        delta_x, delta_y, d=d, separation_2P=separation_2P
    )

    print(f"dx = {delta_x}, dy = {delta_y}")

    d_pairs = get_unique_distance_triples(sites_2P_upper, sites_2P_lower, sites_1P)
    nS = len(d_pairs)

    d_A, d_J = get_AJ_distances(d_pairs)
    print(f"{len(d_A)} unique hyperfines, {len(d_J)} unique exchange values.")

    J_map = {}
    A_map = {}
    set_trace()
    for i in range(len(d_J)):
        J_map[d_J[i]] = J_2P_1P_fab[i]
    for i in range(len(d_A)):
        A_map[d_A[i]] = A_2P_fab[i]

    A_2P = pt.zeros(nS, dtype=real_dtype, device=default_device)
    A_2P_1P = pt.zeros(nS, 2, dtype=real_dtype)
    A_1P_2P = pt.zeros(nS, 2, dtype=real_dtype)
    J = pt.zeros(nS, dtype=real_dtype, device=default_device)
    for q in range(nS):
        A_2P[q] = A_map[d_pairs[q][0]]
        A_1P_2P[q, 0] = A_P
        A_1P_2P[q, 1] = A_2P[q]
        A_2P_1P[q, 0] = A_2P[q]
        A_2P_1P[q, 1] = -A_P
        J[q] = J_map[d_pairs[q][1]]

    if verbose:
        print("2P separations:")
        for i in range(len(d_A)):
            print(f"{d_A[i]*atom_spacing:.3f} nm")
        print("Exchange:")
        for i in range(len(d_J)):
            print(f"{d_J[i]*atom_spacing:.3f} nm, {pt.real(J[i])/unit.MHz:.2f} MHz")

    for q in range(nS):
        print(f"A = {A_2P[q]/unit.MHz:.2f} MHz, J = {J[q]/unit.MHz:.2f} MHz")

    if save:
        pt.save(A_2P / unit.MHz, f"{exch_data_folder}A_2P")
        pt.save(A_2P_1P / unit.MHz, f"{exch_data_folder}A_2P_1P")
        pt.save(A_1P_2P / unit.MHz, f"{exch_data_folder}A_1P_2P")
        pt.save(J / unit.MHz, f"{exch_data_folder}J_big")
    return A_2P, J


if __name__ == "__main__":
    fp = "U_1P_2P"
    fp_big = "U_1P_2P_big"
    fp_tight = "U_1P_2P_tight"
    fp_10_20 = "U_1P_2P_10_20"
    fp_50_100 = "U_1P_2P_50_100"
    fp = fp_10_20
    generate_A_and_J_from_uniform(fp=fp, Jmin=10, Jmax=20)
    get_unique_configs(fp)
    save_unique_configs(fp)
    # inspect_A_J(fp)

