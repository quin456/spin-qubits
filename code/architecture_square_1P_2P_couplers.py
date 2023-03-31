import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


from old.architecture_design import place_wire, place_qubit
from visualisation import *


x_hat = np.array([1, 0])
y_hat = np.array([0, 1])


# colours
SET_colour = "gray"
q_data_colour = "orange"
q_anc_colour = "purple"
q_coup_colour = "gray"
coupler_vert_colour = q_coup_colour
coupler_hori_colour = q_coup_colour
q_colours = [coupler_hori_colour, q_data_colour, coupler_vert_colour, q_coup_colour]

q_colours_sq_2P_couplers = [
    q_data_colour,
    q_anc_colour,
    q_coup_colour,
]
d_HS = 18  # 47*d_lattice
q_radius = 0.175 * d_HS
q_coup_radius = 0.115 * d_HS
q_locs_HS = [
    [[-d_HS, 0]],
    [[0, 0]],
    [[0, -d_HS]],
    [[-3 * d_HS / 2, 0], [-d_HS / 2, 0], [0, -d_HS / 2], [0, d_HS / 2]],
]
HS_cell_size = 2 * d_HS
read_color = "red"
q_sep = 2 * d_HS

q_radius = 0.15 * d_HS
q_radius_active = 0.27 * d_HS
q_radius_inactive = 0.13 * d_HS
q_radii = [q_radius_inactive, q_radius_active]

SET_operating_color = "lightblue"
SET_flag_2e_color = "blue"
SET_no_Sb_color = "gold"
SET_size = 13
tile_pad = 0
tile_alpha = 0
Xstab_tile_color = "lightgreen"
Zstab_tile_color = "lightblue"
read_radius = 1.8 * q_radius


def HS_side_view(ax=None, r0=np.array([-2 * d_HS, 0])):
    padding = 3
    if ax is None:
        ax = plt.subplot()
    n_qubits = 5
    q_separation = 18
    q_radius = 2
    wire_radius = 2.5
    SET_length = SET_size
    SET_height = q_radius * 2 / 3
    SET_colour = grey
    SD_height = 20
    gate_colour = "orange"
    gate_height = 46
    q_colours = [q_data_colour, coupler_hori_colour] * (n_qubits // 2)
    q_colours += [q_data_colour]
    locs = [k * q_separation for k in range(n_qubits)]
    xlim = [
        -q_radius - padding - n_qubits // 2 * q_separation,
        q_radius + n_qubits // 2 * q_separation + padding,
    ]

    def place_sideon_SET(loc):
        SET = plt.Rectangle(
            [loc[0] - SET_length / 2, loc[1] - SET_height / 2],
            SET_length,
            SET_height,
            angle=0,
            color=SET_colour,
            zorder=0,
        )
        # SET_outline = plt.Rectangle(loc-width*(x_hat+y_hat)/2, width, height, angle=0, color = 'black', linewidth=1.5, fill=False)
        outline = plt.Rectangle(
            [loc[0] - SET_length / 2, loc[1] - SET_height / 2],
            SET_length,
            SET_height,
            angle=0,
            fill=False,
            color="black",
            linewidth=1,
            zorder=0,
        )
        ax.add_patch(SET)
        ax.add_patch(outline)

    def place_lower_wire(y_pos, colour):
        SET = plt.Rectangle(
            [xlim[0], y_pos - wire_radius],
            xlim[1] - xlim[0],
            2 * wire_radius,
            angle=0,
            color=colour,
            zorder=0,
        )
        border_up = plt.Rectangle(
            [xlim[0], y_pos - wire_radius],
            xlim[1] - xlim[0],
            0.1,
            angle=0,
            color="black",
            zorder=2,
        )
        border_down = plt.Rectangle(
            [xlim[0], y_pos + wire_radius],
            xlim[1] - xlim[0],
            0.1,
            angle=0,
            color="black",
            zorder=2,
        )
        ax.add_patch(SET)
        ax.add_patch(border_up)
        ax.add_patch(border_down)

    for k in range(n_qubits):
        loc = np.array([locs[k], 0])
        place_qubit(ax, r0 + loc, q_colours[k], q_radius)
        if k % 2 == 1:
            place_qubit(ax, [r0[0] + loc[0], SD_height], SET_colour, wire_radius)
            place_sideon_SET(loc + r0)
        else:
            place_qubit(ax, [r0[0] + loc[0], gate_height], gate_colour, wire_radius)

    place_lower_wire(-SD_height, SET_colour)
    place_lower_wire(-gate_height, gate_colour)
    ax.set_xlim(xlim)
    ax.set_ylim([-46 - wire_radius - padding, 46 + wire_radius + padding])
    ax.set_aspect("equal")
    ax.set_yticks([-46, -20, 0, 20, 46])
    ax.set_xticks([-36, -18, 0, 18, 36])
    # ax.axis('off')
    ax.set_xlabel("[1,-1,0] (nm)")
    ax.set_ylabel("[0,0,1] (nm)")
    ax.get_figure().tight_layout()


def plot_HS_grid(ax, r0, left_link=False, down_link=False):
    X = [-d_HS, 0, 0]
    Y = [0, 0, -d_HS]
    if left_link:
        X[0] *= 2
    if down_link:
        Y[2] *= 2
    ax.plot(r0[0] + np.array(X), r0[1] + np.array(Y), color="black", zorder=0)


def make_HS_unit_cell(
    ax,
    r0=np.array([0, 0]),
    left_link=False,
    down_link=False,
    place_SET=True,
    q_colours=q_colours_sq_2P_couplers,
    i=0,
    j=0,
    q_radius=q_radius,
    cell_activations=[1, 1, 1],
):

    plot_HS_grid(ax, r0, left_link=left_link, down_link=down_link)
    n_qubit_types_HS = 3

    # place qubits
    for k in range(n_qubit_types_HS):
        for q_loc in q_locs_HS[k]:
            color = q_colours[k]
            if color == q_colours[0]:
                if (i + j) % 2 != 0:
                    if j % 2 == 0:
                        color = q_colours[0]
                    else:
                        color = q_colours[1]
                else:
                    if j % 2 == 0:
                        color = q_colours[2]
            place_qubit(
                ax, q_loc + r0, color, q_radii[cell_activations[k]], couplers=True
            )


def place_HS_cell_SET(ax, r0, color="silver"):
    # width=d_HS/3
    # loc=r0-d_HS*np.array([1,1])/2
    width = SET_size
    height = width
    loc = r0 - d_HS * np.array([1, 1])

    SET = plt.Rectangle(
        loc - width * (x_hat + y_hat) / 2, width, height, color=color, angle=0
    )
    SET_outline = plt.Rectangle(
        loc - width * (x_hat + y_hat) / 2,
        width,
        height,
        angle=0,
        color="black",
        linewidth=1.5,
        fill=False,
    )
    ax.add_patch(SET)
    ax.add_patch(SET_outline)


def HS_11_single_square(r0=np.array([0, 0]), fontsize=17, ax=None):
    if ax is None:
        ax = plt.subplot()
    make_HS_unit_cell(ax, r0, left_link=True, down_link=False, q_radius=q_radius)
    place_qubit(ax, r0, q_colours[1], q_radius, textcolor="white", fontsize=fontsize)
    place_qubit(
        ax,
        r0 - d_HS * x_hat,
        q_colours[0],
        q_radius,
        textcolor="white",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - d_HS * x_hat - 2 * d_HS * y_hat,
        q_colours[0],
        q_radius,
        textcolor="white",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - d_HS * y_hat,
        q_colours[2],
        q_radius,
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat,
        q_colours[1],
        q_radius,
        textcolor="white",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat - d_HS * y_hat,
        q_colours[2],
        q_radius,
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat - 2 * d_HS * y_hat,
        q_colours[1],
        q_radius,
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - 2 * d_HS * y_hat,
        q_colours[1],
        q_radius,
        textcolor="purple",
        fontsize=fontsize,
    )
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    place_HS_cell_SET(ax, r0)
    pad = d_HS / 4
    # grid lines
    # ax.plot([r0[0]-2*d_HS, r0[0]-2*d_HS], [r0[1]+d_HS, r0[1]-d_HS], zorder=0, color='black')
    # ax.plot([r0[0], r0[0]], [r0[1]+d_HS, r0[1]-d_HS], zorder=0, color='black')

    ax.set_aspect("equal")
    ax.set_xticks([-d_HS, -SET_size / 2, 0, SET_size / 2, d_HS])
    ax.set_yticks([-d_HS, -2.5, 0, 2.5, d_HS])
    alpha = 0.4
    SD_col = grey
    G_col = "orange"
    G_alpha = 0.4
    place_wire(ax, -d_HS, grey, "hori", xlim, alpha, 5)
    place_wire(ax, 0, G_col, "hori", xlim, G_alpha, 5)
    place_wire(ax, -2 * d_HS, G_col, "vert", xlim, G_alpha, 5)
    place_wire(ax, -d_HS, grey, "vert", xlim, G_alpha, 5)
    place_wire(ax, -2 * d_HS, G_col, "hori", xlim, alpha, 5)
    place_wire(ax, 0, G_col, "vert", xlim, alpha, 5)
    ax.set_xlabel("[1,-1,0] (nm)")
    ax.set_ylabel("[1,1,0] (nm)")
    ax.get_figure().tight_layout()


def get_distance_xy(distance):
    if isinstance(distance, np.ndarray) or isinstance(distance, list):
        if len(distance) == 2:
            distance_x, distance_y = distance
        else:
            raise Exception("Distance array must have length of 2.")
    elif isinstance(distance, int):
        distance_x = distance
        distance_y = distance
    else:
        raise Exception("Distance must be an int, list, or numpy.ndarray.")
    return distance_x, distance_y


def draw_grid(distance, ax):
    distance_x, distance_y = get_distance_xy(distance)
    for i in range(2 * distance_x - 1):
        ax.plot(
            [i * q_sep, i * q_sep], [0, 2 * (distance_y - 1) * q_sep], color="black"
        )
    for j in range(2 * distance_y - 1):
        ax.plot(
            [0, 2 * (distance_x - 1) * q_sep], [j * q_sep, j * q_sep], color="black"
        )


def generate_2Pcoup_sq_qubit_array(distance, ax=None):
    """
    distance (int): distance of code, needs to be odd.
    """
    distance_x, distance_y = get_distance_xy(distance)

    if ax is None:
        ax = plt.subplot()
    for i in range(2 * distance_x - 1):
        for j in range(2 * distance_y - 1):
            q_loc = q_sep * np.array([i, j])
            is_data = (i + j) % 2 == 0
            if is_data:
                q_colour = q_data_colour
            else:
                q_colour = q_anc_colour
            place_qubit(ax, q_loc, q_colour)
            if i < 2 * (distance_x - 1):
                place_qubit(
                    ax, q_loc + 0.5 * q_sep * x_hat, q_coup_colour, q_coup_radius
                )
            if j < 2 * (distance_y - 1):
                place_qubit(
                    ax, q_loc + 0.5 * q_sep * y_hat, q_coup_colour, q_coup_radius
                )

    draw_grid(distance, ax)
    ax.set_xlim(-q_sep, q_sep * (2 * distance_x - 1))
    ax.set_ylim(-q_sep, q_sep * (2 * distance_y - 1))
    ax.set_aspect("equal")


if __name__ == "__main__":
    generate_2Pcoup_sq_qubit_array(3)
    plt.show()

