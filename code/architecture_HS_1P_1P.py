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
q_meas_colour = "black"
q_flag_colour = "purple"
q_coup_colour = "grey"
q_data_colour = "orange"
q_colours = [q_meas_colour, q_flag_colour, q_data_colour, q_coup_colour]

q_colours_HS = [
    q_meas_colour,
    q_flag_colour,
    q_data_colour,
    q_meas_colour,
    q_meas_colour,
    q_meas_colour,
]
d_HS = 18  # 47*d_lattice
q_radius = 0.2 * d_HS
# q_coup_radius = 0.1*d_HS]
q_locs_HS = [
    [[-d_HS, 0]],
    [[0, 0]],
    [[0, -d_HS]],
    [[-3 * d_HS / 2, 0], [-d_HS / 2, 0], [0, -d_HS / 2], [0, d_HS / 2]],
]
HS_cell_size = 2 * d_HS
read_color = "red"

q_radius = 0.15 * d_HS
q_radius_active = 0.27 * d_HS
q_radius_inactive = 0.13 * d_HS
q_radii = [q_radius_inactive, q_radius_active]

SET_operating_color = "lightblue"
SET_flag_2e_color = "blue"
SET_no_Sb_color = "gold"
SET_size = 13
tile_pad = 0
tile_alpha = 0.5
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
    q_colours = [q_flag_colour, q_meas_colour] * (n_qubits // 2)
    q_colours += [q_flag_colour]
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
    q_colours=q_colours_HS,
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
                        color = q_colours[3]
                    else:
                        color = q_colours[4]
                else:
                    if j % 2 == 0:
                        color = q_colours[5]
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


def HS_left_boundary_cell(
    ax,
    j,
    distance,
    q_colours=q_colours,
    q_activations=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    place_stabilizer_tiles=True,
):
    cell_activations = get_cell_activations(q_activations, -1, j)
    flag_loc = HS_cell_size * np.array([-1, j])
    flag_loc_right = np.array([2 * (distance - 1) * d_HS, (2 * j) * d_HS])
    # flag_loc = HS_cell_size*np.array([i,j])
    place_qubit(ax, flag_loc - d_HS * y_hat, q_colours[2], q_radii[cell_activations[2]])
    place_qubit(ax, flag_loc, q_colours[1], q_radii[cell_activations[1]])
    # place_HS_cell_SET(ax, np.array([-HS_cell_size,(2*i)*d_HS]))
    if place_stabilizer_tiles:
        if j % 2 == 0:
            side_tile = mpl.patches.Polygon(
                flag_loc
                + np.array(
                    [
                        [(distance - 1) * HS_cell_size, -d_HS],
                        [(distance - 1) * HS_cell_size, d_HS],
                        [(2 * distance - 1) * d_HS, 0],
                    ]
                ),
                zorder=0,
                color=Xstab_tile_color,
                alpha=tile_alpha,
            )
        else:
            side_tile = mpl.patches.Polygon(
                flag_loc + np.array([[0, -d_HS], [-d_HS, 0], [0, d_HS]]),
                color=Xstab_tile_color,
                alpha=tile_alpha,
                zorder=0,
            )
        ax.add_patch(side_tile)


def place_HS_SETs(ax, distance, colors=["silver", "orange"]):
    for i in range(distance - 1):
        flag_loc_left = np.array([-HS_cell_size, (2 * i) * d_HS])
        flag_loc_right = np.array([2 * (distance - 1) * d_HS, (2 * i) * d_HS])
        if i % 2 != 0:
            place_HS_cell_SET(ax, flag_loc_left, color=colors[i % 2])
        else:
            place_HS_cell_SET(ax, flag_loc_right, color=colors[i % 2])
        if i % 2 == 0:
            place_HS_cell_SET(
                ax,
                np.array([i * HS_cell_size, (2 * distance - 2) * d_HS]),
                color=colors[0],
            )
        else:
            place_HS_cell_SET(
                ax, np.array([i * HS_cell_size, (2 * distance) * d_HS]), color=colors[1]
            )
        for j in range(distance - 1):
            if (i + j) % 2 == 0:
                flag_loc = HS_cell_size * np.array([i, j])
                place_HS_cell_SET(ax, flag_loc, color=colors[(i) % 2])
    # place_HS_cell_SET(ax, np.array([(distance-1)*HS_cell_size, (2*distance-2)*d_HS]), color=colors[0])


def place_HS_wires(ax, distance, colors):
    xlim = d_HS * np.array([-3.5, (distance - 1)])
    ylim = d_HS * np.array([-2.5, (distance + 2.5)])
    for i in range(-2, distance):
        flag_loc_left = np.array([-HS_cell_size,])
        flag_loc_right = np.array([2 * (distance - 1) * d_HS, (2 * i) * d_HS])
        y_SET_wire = (2 * i + 1) * d_HS
        x_SET_wire = (2 * i + 1) * d_HS
        color_SD = colors[1] if i % 2 == 0 else colors[0]
        color_gate = colors[3] if i % 2 == 0 else colors[2]
        color_gate_vert = colors[5] if i % 2 == 0 else colors[4]
        x_gate_wire = (2 * i) * d_HS
        y_gate_wire = (2 * i) * d_HS
        place_wire(ax, x_SET_wire, color_SD, "vert", ylim, 1, thickness=3)
        if i > -2 and i < distance - 1:
            place_wire(ax, x_gate_wire, color_gate_vert, "vert", ylim, 1, thickness=3)

        # place_wire(ax, x_gate_wire, color_gate,'vert', xlim, 1, thickness=2)

        if i == -2:
            continue
        place_wire(ax, y_SET_wire, color_SD, "hori", xlim, 1, thickness=3)
        place_wire(ax, y_gate_wire, color_gate, "hori", xlim, 1, thickness=3)
        continue
        break


def HS_upper_boundary_cell(
    ax,
    i,
    distance,
    q_colours=q_colours,
    q_activations=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    place_stabilizer_tiles=True,
):
    upper_data_loc = np.array([i * HS_cell_size, (2 * distance - 3) * d_HS])
    lower_data_loc = np.array([i * HS_cell_size, -d_HS])
    place_qubit(ax, upper_data_loc, q_colours[2], q_radius)
    place_qubit(ax, upper_data_loc + d_HS * y_hat, q_colours[1], q_radius)
    place_qubit(
        ax, lower_data_loc - d_HS * y_hat - HS_cell_size * x_hat, q_colours[1], q_radius
    )
    # place_HS_cell_SET(ax, np.array([(i-1)*HS_cell_size, -2*d_HS]))
    if i % 2 == 0:
        place_qubit(ax, upper_data_loc + d_HS * (x_hat + y_hat), q_colours[3], q_radius)
        ax.plot(
            [
                upper_data_loc[0],
                upper_data_loc[0],
                upper_data_loc[0] + HS_cell_size,
                upper_data_loc[0] + HS_cell_size,
            ],
            [
                upper_data_loc[1] - d_HS,
                upper_data_loc[1] + d_HS,
                upper_data_loc[1] + d_HS,
                upper_data_loc[1] - d_HS,
            ],
            color="black",
            zorder=0,
        )
        if place_stabilizer_tiles:
            stab_tile = plt.Rectangle(
                lower_data_loc
                + 1 * d_HS * y_hat
                - HS_cell_size * np.array([1 - tile_pad, 1 - tile_pad]),
                HS_cell_size * (1 - 2 * tile_pad),
                HS_cell_size * (0.5 - 2 * tile_pad),
                color=Zstab_tile_color,
                zorder=0,
                alpha=tile_alpha,
            )
            ax.add_patch(stab_tile)
    else:
        place_qubit(
            ax,
            np.array([i * HS_cell_size - 3 * d_HS, -HS_cell_size]),
            q_colours[0],
            q_radius,
        )
        ax.plot(
            [
                lower_data_loc[0] - HS_cell_size,
                lower_data_loc[0] - HS_cell_size,
                lower_data_loc[0] - 2 * HS_cell_size,
                lower_data_loc[0] - 2 * HS_cell_size,
            ],
            [
                lower_data_loc[1],
                lower_data_loc[1] - d_HS,
                lower_data_loc[1] - d_HS,
                lower_data_loc[1],
            ],
            color="black",
            zorder=0,
        )
        if place_stabilizer_tiles:
            stab_tile = plt.Rectangle(
                upper_data_loc
                + 2 * d_HS * y_hat
                - HS_cell_size * np.array([1 - tile_pad, 1 - tile_pad]),
                HS_cell_size * (1 - 2 * tile_pad),
                HS_cell_size * (0.5 - 2 * tile_pad),
                color=Zstab_tile_color,
                zorder=0,
                alpha=tile_alpha,
            )
            ax.add_patch(stab_tile)


def get_cell_activations(q_activations, i, j):
    cell_activations = [0, 0, 0]
    if j % 2 == 0:
        cell_activations[2] = q_activations[0]
        if i % 2 == 0:
            cell_activations[1] = q_activations[2]
            cell_activations[0] = q_activations[6]
        else:
            cell_activations[1] = q_activations[3]
            cell_activations[0] = q_activations[7]
    else:
        cell_activations[2] = q_activations[1]
        if i % 2 == 0:
            cell_activations[1] = q_activations[4]
            cell_activations[0] = q_activations[8]
        else:
            cell_activations[1] = q_activations[5]
            cell_activations[0] = q_activations[9]
    return cell_activations


def make_HS_qubit_array(
    ax,
    distance,
    SET_colors=[grey, grey],
    wire_colors=[grey, grey, yellow, yellow, yellow, yellow],
    q_colours=[
        q_meas_colour,
        q_flag_colour,
        q_data_colour,
        q_meas_colour,
        q_meas_colour,
        q_meas_colour,
    ],
    q_activations=[1, 1, 1, 1, 1, 1, 1, 1],
    place_stabilizer_tiles=True,
):
    """
    wire_colours =? [SET1, SET2, hgate1, hgate2, vgate1, vgate2] 

    """
    # plot distance-1 cells, then handle edges
    q_radii = [q_radius_inactive, q_radius_active]
    for i in range(distance - 1):
        HS_upper_boundary_cell(
            ax,
            i,
            distance,
            q_colours=q_colours,
            q_activations=q_activations,
            place_stabilizer_tiles=place_stabilizer_tiles,
        )
        HS_left_boundary_cell(
            ax,
            i,
            distance,
            q_colours=q_colours,
            q_activations=q_activations,
            place_stabilizer_tiles=place_stabilizer_tiles,
        )
        for j in range(distance - 1):
            flag_loc = HS_cell_size * np.array([i, j])
            # cell_activations: [measure, flag, data]

            cell_activations = get_cell_activations(q_activations, i, j)

            make_HS_unit_cell(
                ax,
                flag_loc,
                left_link=True,
                down_link=j > 0,
                q_colours=q_colours,
                place_SET=(i + j) % 2 == 0,
                i=i,
                j=j,
                cell_activations=cell_activations,
            )  # or j==0 or i==0 or i==distance-2)
            if place_stabilizer_tiles:
                tile_color = Xstab_tile_color if (i + j) % 2 == 0 else Zstab_tile_color
                stab_tile = plt.Rectangle(
                    flag_loc - HS_cell_size * np.array([1 - tile_pad, 0.5 - tile_pad]),
                    HS_cell_size * (1 - 2 * tile_pad),
                    HS_cell_size * (1 - 2 * tile_pad),
                    color=tile_color,
                    zorder=0,
                    alpha=tile_alpha,
                )
                ax.add_patch(stab_tile)
    # place_qubit(ax, np.array([-HS_cell_size,-d_HS]), q_colours[2], q_radius)
    ax.plot(
        [-HS_cell_size, -HS_cell_size],
        [-HS_cell_size, (2 * distance - 3) * d_HS],
        color="black",
        zorder=0,
    )

    pad = 10
    ax.set_xlim([-2 * HS_cell_size - pad, (distance - 1.5) * HS_cell_size + pad])
    ax.set_ylim([-HS_cell_size - pad, (distance - 0.5) * HS_cell_size + pad])
    ax.set_aspect("equal")
    ax.axis("off")


def make_HS_array(
    ax=None,
    distance=5,
    SET_colors=[grey, grey],
    wire_colors=[grey, grey, yellow, yellow, yellow, yellow],
    q_activations=10 * [0],
    zoomed=True,
    place_wires=True,
    place_stabilizer_tiles=True,
):
    """
    wire_colours: [SET1 S/D, SET2 S/D, G_h1, G_h2, G_v1, G_v2]
    SET_colours: [SET1, SET2]
    qubit_activations: [data1, data2, flag11, flag12, flag21, flag22, meas11, meas12, meas21, meas22  ]
    """
    if ax is None:
        ax = plt.subplot()
    make_HS_qubit_array(
        ax,
        distance,
        SET_colors=SET_colors,
        wire_colors=wire_colors,
        q_activations=q_activations,
        place_stabilizer_tiles=place_stabilizer_tiles,
    )
    if place_wires:
        place_HS_SETs(ax, distance, colors=SET_colors)
        place_HS_wires(ax, distance, colors=wire_colors)

    if zoomed:
        padding = 0.3 * d_HS
        ax.set_xlim(-2 * d_HS - padding, 2 * (distance - 2) * d_HS + padding)
        ax.set_ylim(-2 * d_HS - padding, 2 * (distance - 2) * d_HS + padding)


SD_a = "lightblue"
SD_i = grey
G_attract = orange
G_repel = darkblue
inactive = grey
SD_a = "#99CCFF"
SD_a = "#66B2FF"


def HS_array_configs(fp=None):
    fig, ax = plt.subplots(2, 2)
    make_HS_array(
        ax=ax[0, 0],
        wire_colors=[SD_a, SD_i, G_repel, inactive, G_attract, G_attract],
        SET_colors=[SD_a, SD_i],
        q_activations=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        zoomed=True,
    )
    make_HS_array(
        ax=ax[0, 1],
        wire_colors=[SD_i, SD_a, G_repel, inactive, G_attract, G_attract],
        SET_colors=[SD_i, SD_a],
        q_activations=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        zoomed=True,
    )
    make_HS_array(
        ax=ax[1, 0],
        wire_colors=[SD_a, SD_i, inactive, G_attract, G_repel, inactive],
        SET_colors=[SD_a, SD_i],
        q_activations=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        zoomed=True,
    )
    make_HS_array(
        ax=ax[1, 1],
        wire_colors=[SD_a, SD_i, inactive, G_attract, inactive, G_repel],
        SET_colors=[SD_a, SD_i],
        q_activations=[0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        zoomed=True,
    )

    fig.set_size_inches(10, 10)
    if fp is not None:
        fig.savefig(fp)


if __name__ == "__main__":
    # fig,ax = plt.subplots(1,1)
    # make_HS_array(ax=ax, wire_colors = [SD_a, SD_i, G_repel, inactive, G_attract, G_attract], SET_colors=[SD_a, SD_i], q_activations=[1,0,1,1,0,0,1,1,0,0])
    # make_HS_array(ax=ax, wire_colors = [SD_a, SD_i, inactive, G_attract, G_repel, inactive], SET_colors=[SD_a, SD_i], q_activations=[0,0,1,0,0,0,1,0,0,0], zoomed=True)
    # HS_array_configs()
    make_HS_array(
        distance=7, place_wires=False, place_stabilizer_tiles=True, zoomed=False
    )
    plt.show()
