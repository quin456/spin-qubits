import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from pdb import set_trace

from utils import label_axis
from visualisation import *

distorted = False

# all distance values are in nm

n_qubit_types_HH = 4
n_qubit_types_HS = 3
couplers = True

# distance between lattice sites
a_lattice = 0.543
d_lattice = 0.384

# number of lattice sites in x and y direction between adjacent qubits in horizontal and vertical directions
xh = 32 * a_lattice
yh = 0 * a_lattice
xd = 16 * a_lattice
yd = 28 * a_lattice

x_hat = np.array([1, 0])
y_hat = np.array([0, 1])


# colours
SET_colour = "gray"
q_coup_colour = "gray"
q_data_colour = "purple"
q_meas_colour = q_coup_colour
q_flag_colour = q_data_colour
q_data_colour = "orange"
q_anc_colour = "purple"
q_colours = [q_data_colour, q_coup_colour, q_anc_colour]
V_colours = ["lightgray", "orange", "red", "black", "blue", "lightblue"]
V_colours = ["lightgray", "orange", "red", "#0000cc", "#0080ff", "#99ccff"]
# V_colours = [np.array([224,224,224,256])/256, 'orange', 'red', [0,0,204/256,1], [0,128/256,255/256,1], [153/256,204/256,255/256,1]]
d = 18
q_locs = [[[0, 0]], [[0, -d], [d, 0]]]

# locations within unit cell
SET_x = 46 * a_lattice
SET_y = 42 * a_lattice
R_SET = 6.5  # radius
nhat = np.array([np.cos(23.4 * np.pi / 180), np.sin(23.4 * np.pi / 180)])

# SET locations


SET_loc = np.array(
    [
        [SET_x, SET_y],
        [-SET_x - 2 * R_SET, SET_y],
        [-SET_x - 2 * R_SET, -SET_y],
        [SET_x, -SET_y],
    ]
)
WCs = np.array([[xh, yd], [-xh, yd], [-xh, -yd], [xh, -yd]])


SET_angle = np.array([0, 0, 0, 0, 45])
SET_length = np.array([0.6, 1, 1, 0.6]) * 4 * R_SET
SET_width = np.array([1, 1, 1, 1]) * R_SET
# -23.4*np.pi/180
if not distorted:
    cell_length = 4 * xh + 4 * xd  # 276
    cell_height = 4 * yd  # 160

    q_meas_loc = np.array([[0, 0]])
    q_meas_num = [7]
    flag_x = 2 * xh
    flag_y = 0
    data_x = 2 * (xh + xd)
    data_y = 2 * yd
    measure_x = 0
    measure_y = 0
    q_flag_loc = np.array([[-2 * xh, 0], [2 * xh, 0]])
    q_coup_loc = np.array(
        [
            [-xh, 0],
            [xh, 0],
            [2 * xh + xd, yd],
            [-2 * xh - xd, yd],
            [-2 * xh - xd, -yd],
            [2 * xh + xd, -yd],
        ]
    )
    q_data_loc = np.array(
        [[-data_x, data_y], [-data_x, -data_y], [data_x, data_y], [data_x, -data_y]]
    )
    q_flag_num = [5, 6]
    q_data_num = [4, 3, 1, 2]

    def plot_HH_grid(ax, r0):
        grid_color = "gray"
        grid_zorder = 0.25
        grid_opacity = 1
        grid_xh = flag_x * np.array([-1, 1])
        grid_yh = np.zeros(2)
        grid_xUL = np.array([-2 * xh - 2 * xd, -flag_x])
        grid_yUL = np.array([2 * yd, 0])
        grid_xLL = np.array([-2 * xh - 2 * xd, -flag_x])
        grid_yLL = np.array([-2 * yd, 0])
        grid_xLR = np.array([2 * xh + 2 * xd, flag_x])
        grid_yLR = np.array([-2 * yd, 0])
        grid_xUR = np.array([2 * xh + 2 * xd, flag_x])
        grid_yUR = np.array([2 * yd, 0])

        x0, y0 = r0
        ax.plot(
            grid_xh + x0,
            grid_yh + y0,
            color=grid_color,
            zorder=grid_zorder,
            alpha=grid_opacity,
        )
        ax.plot(
            grid_xUL + x0,
            grid_yUL + y0,
            color=grid_color,
            zorder=grid_zorder,
            alpha=grid_opacity,
        )
        ax.plot(
            grid_xLL + x0,
            grid_yLL + y0,
            color=grid_color,
            zorder=grid_zorder,
            alpha=grid_opacity,
        )
        ax.plot(
            grid_xLR + x0,
            grid_yLR + y0,
            color=grid_color,
            zorder=grid_zorder,
            alpha=grid_opacity,
        )
        ax.plot(
            grid_xUR + x0,
            grid_yUR + y0,
            color=grid_color,
            zorder=grid_zorder,
            alpha=grid_opacity,
        )

    nwire_vert = 4
    wire_vert_x = [-2 * xh - 2 * xd, -xh, xh]
    wire_vert_colours = ["orange", "red", "blue"]

    nwire_hori = 2
    wire_hori_y = [yd, -yd]
    wire_hori_colours = ["green", "purple"]

else:
    cell_length = 6 * xh + 2 * xd
    cell_height = 2 * yd

    q_meas_loc = np.array([[0, 0]])
    q_flag_loc = np.array([[-2 * xh, 0], [2 * xh, 0]])
    q_coup_loc = np.array(
        [
            [-2 * xh - xd, 40],
            [-2.5 * xh, -yd],
            [-2 * xh - xd, -yd],
            [2 * xh + xd, yd],
            [-xh, 0],
            [xh, 0],
        ]
    )
    q_data_loc = np.array(
        [[-3 * xh - xd, yd], [-3 * xh - xd, -yd], [3 * xh + xd, -yd], [3 * xh + xd, yd]]
    )

    nwire_vert = 4
    wire_vert_x = [-2 * xh - xd, -xh, xh, 2 * xh + xd]
    wire_vert_colours = ["orange", "red", "blue", "purple"]
    nwire_hori = 1
    wire_hori_y = [2 * yh]
    wire_hori_colours = ["green"]


q_locs_HH = [q_meas_loc, q_flag_loc, q_data_loc, q_coup_loc]
q_radius = 8.5 * a_lattice
q_coup_radius = 5.5 * a_lattice


# wires

wire_width = 5
default_wire_opacity = 0.3


def double_arrow(ax, x, y, dx, dy, width=0.05, color="darkblue", linewidth=0.001):
    ax.arrow(
        x + dx / 2.01,
        y + dy / 2.01,
        dx / 1.99,
        dy / 1.99,
        width=width,
        length_includes_head=True,
        color=color,
        linewidth=linewidth,
    )
    ax.arrow(
        x + dx / 1.99,
        y + dy / 1.99,
        -dx / 2.01,
        -dy / 2.01,
        width=width,
        length_includes_head=True,
        color=color,
        linewidth=linewidth,
    )


def stadium(ax, start, end, radius, color=SET_colour):

    circ1 = plt.Circle(start, radius, color=color)
    circ2 = plt.Circle(end, radius, color=color)
    length = np.linalg.norm(start - end)
    unit_vec = (end - start) / length
    normal = np.array([-unit_vec[1], unit_vec[0]])
    angle = np.arctan(unit_vec[1] / unit_vec[0]) * 180 / np.pi
    print(angle)
    mid_section = plt.Rectangle(
        start - radius * normal, length, 2 * radius, angle=angle, color=color
    )

    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.add_patch(mid_section)


def unit_vector(theta):
    theta = theta * np.pi / 180
    nhat = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-nhat[1], nhat[0]])
    return nhat, normal


def rect_SET(ax, start, length, width, angle, color=SET_colour):
    nhat, normal = unit_vector(angle)
    end = start + length * nhat
    # length = np.linalg.norm(start-end)
    # unit_vec = (end-start)/length
    # angle = np.arctan(nhat[1]/nhat[0])*180/np.pi
    mid_section = plt.Rectangle(
        start - 0.5 * width * normal, length, width, angle=angle, color=color
    )
    ax.add_patch(mid_section)


def place_SET(ax, location, slant=0):

    # slant_vec = np.array([np.cos(slant),np.sin(slant)])
    # slant_vec_normal = np.array([np.sin(slant),-np.cos(slant)])
    # r1 = location - R_SET*slant_vec
    # r2 = location + R_SET*slant_vec
    # circ1 = plt.Circle(r1, R_SET, color=SET_colour)
    # circ2 = plt.Circle(r2, R_SET, color=SET_colour)
    # square = plt.Rectangle(location-R_SET*(slant_vec-slant_vec_normal), 2*R_SET, 2*R_SET, angle=slant*180/np.pi, color=SET_colour)
    # ax.add_patch(circ1)
    # ax.add_patch(circ2)
    # ax.add_patch(square)

    start = location
    end = location + 2 * R_SET * np.array([np.cos(slant), np.sin(slant)])
    stadium(ax, start, end, R_SET)


def place_wire(ax, loc, colour, orientation, lim, wire_opacity, thickness=5):
    length = lim[1] - lim[0]
    outline = 0.07
    if orientation == "vert":
        wire = plt.Rectangle(
            [loc - thickness / 2, lim[0]],
            thickness,
            length - lim[0],
            color=colour,
            alpha=wire_opacity,
            zorder=0,
        )
        top = plt.Rectangle(
            [loc - thickness / 2 - outline, lim[0]],
            outline,
            length - lim[0],
            color="black",
            alpha=wire_opacity,
            zorder=0,
        )
        bottom = plt.Rectangle(
            [loc + thickness / 2, lim[0]],
            outline,
            length - lim[0],
            color="black",
            alpha=wire_opacity,
            zorder=0,
        )
    elif orientation == "hori":
        wire = plt.Rectangle(
            [lim[0], loc - thickness / 2],
            length - lim[0],
            thickness,
            color=colour,
            alpha=wire_opacity,
            zorder=0,
        )
        top = plt.Rectangle(
            [lim[0], loc - thickness / 2 - outline],
            length - lim[0],
            outline,
            color="black",
            alpha=wire_opacity,
            zorder=0,
        )
        bottom = plt.Rectangle(
            [lim[0], loc + thickness / 2],
            length - lim[0],
            outline,
            color="black",
            alpha=wire_opacity,
            zorder=0,
        )
    else:
        raise Exception("Invalid wire orientation")

    ax.add_patch(wire)
    if False:
        ax.add_patch(top)
        ax.add_patch(bottom)


def UR_SET(ax, r0):
    length = 18
    width = 9
    loc = r0 + WCs[0] + np.array([-6.5, 0])
    rect_SET(ax, loc, length, width, 0)


def UL_SET(ax, r0):
    length = 18
    width = 9
    loc = r0 + WCs[1] + np.array([6.5 - length, 0])
    rect_SET(ax, loc, length, width, 0)


def LR_SET(ax, r0):
    width = 12
    length = 25
    loc = r0 + WCs[3] - np.array([length / 2, 0])
    rect_SET(ax, loc, length, width, 0)

    length2 = 2 * 18.7 / np.sqrt(3)
    loc2 = (
        loc
        + np.array([length, width / 2])
        + width / 2 * np.array([-np.sqrt(3) / 2, -1 / 2])
    )
    rect_SET(ax, loc2, 2 * 18.7 / np.sqrt(3), width, -60)


def LL_SET(ax, r0):
    width = 12
    loc = r0 + WCs[2] - np.array([11, 0])
    rect_SET(ax, loc, 13.5, width, 0)

    length2 = 2 * 18.7 / np.sqrt(3)
    loc2 = (
        loc + np.array([0, width / 2]) + width / 2 * np.array([np.sqrt(3) / 2, -1 / 2])
    )
    rect_SET(ax, loc2, length2, width, 240)

    # ax.scatter([loc[0]],[loc[1]])
    # ax.scatter([loc2[0]],[loc2[1]])


def place_HH_cell_SETs(ax, r0):
    LL_SET(ax, r0)
    LR_SET(ax, r0)
    UR_SET(ax, r0)
    UL_SET(ax, r0)
    # if not distorted:
    #     for i in range(len(SET_loc)):
    #         rect_SET(ax,SET_loc[i],SET_length[i],SET_width[i],SET_angle[i])
    #         #place_SET(ax,np.array(r0+SET_loc[i]), SET_angle[i])


def place_qubit(
    ax,
    loc,
    color,
    q_radius=q_radius,
    text=None,
    textcolor="black",
    fontsize=17,
    couplers=couplers,
    q_coup_radius=q_coup_radius,
):
    if color == q_coup_colour and not couplers:

        set_trace()
        return
    if color == q_coup_colour:
        q_radius = q_coup_radius

    ax.add_patch(plt.Circle(loc, q_radius, color=color, zorder=3))
    ax.add_patch(
        plt.Circle(loc, q_radius, color="black", linewidth=1, fill=False, zorder=3)
    )
    if ax is not None:
        ax.annotate(
            text,
            loc - q_radius * (x_hat + y_hat) * 0.5,
            fontsize=fontsize,
            color=textcolor,
        )


def make_HH_unit_cell(ax, r0=np.array([0, 0]), xlim=[-150, 150], ylim=[-90, 90]):

    # ax.vlines(wire_vert_x+r0[0], ylim[0], ylim[1], colors=wire_vert_colours, linewidth=wire_width, alpha=wire_opacity,zorder=1)

    # for i in range(nwire_vert):
    #    place_wire(ax,wire_vert_x[i]+r0[0],wire_vert_colours[i],'vert', ylim)

    # place gridlines
    plot_HH_grid(ax, r0)

    place_HH_cell_SETs(ax, r0)

    # place qubits
    for i in range(n_qubit_types_HH):
        for q_loc in q_locs_HH[i]:
            place_qubit(ax, q_loc + r0, q_colours[i], couplers=True)

    # place wires
    # ax.hlines(wire_hori_y+r0[1], xlim[0], xlim[1], colors=wire_hori_colours, linewidth=wire_width, alpha=wire_opacity)
    # for j in range(nwire_hori):
    #    place_wire(ax,wire_hori_y[j]+r0[1],wire_hori_colours[j], 'hori', xlim)


def place_all_wires(ax, m, n, wire_colours=None, wire_opacity=default_wire_opacity):
    n_hori_wires = nwire_hori * m
    n_vert_wires = nwire_vert * n + 1
    if wire_colours == None:
        wire_colours = (
            m * wire_hori_colours + n * wire_vert_colours + wire_vert_colours[0:1]
        )
        print(wire_colours)
    wire_orientations = ["hori"] * n_hori_wires + ["vert"] * n_vert_wires
    lims = n_hori_wires * [ax.get_xlim()] + n_vert_wires * [ax.get_ylim()]
    wire_positions = []
    for i in range(m):
        y0 = i * cell_height
        wire_positions += [y0 + y for y in wire_hori_y]
    for j in range(n):
        x0 = j * cell_length
        wire_positions += [x0 + x for x in wire_vert_x]
    wire_positions += [n * cell_length + wire_vert_x[0]]

    nwires = len(wire_positions)
    for n in range(nwires):
        place_wire(
            ax,
            wire_positions[n],
            wire_colours[n],
            wire_orientations[n],
            lims[n],
            wire_opacity,
        )

    place_wire(
        ax, wire_hori_y[1] + 4 * cell_height, "purple", "hori", lims[0], wire_opacity
    )
    place_wire(ax, wire_hori_y[0] - cell_height, "green", "hori", lims[0], wire_opacity)
    place_wire(
        ax, wire_vert_x[0] + 4 * cell_length, "red", "vert", lims[-1], wire_opacity
    )


def plot_single_cell(ax=None, wire_colours=None, wire_opacity=default_wire_opacity):
    r0 = np.array([0, 0])
    if ax == None:
        ax = plt.subplot()
    ax.set_aspect("equal")
    xlim = [-110 * a_lattice, 110 * a_lattice]
    ylim = [-70 * a_lattice, 70 * a_lattice]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    place_all_wires(ax, 1, 1, wire_colours=wire_colours, wire_opacity=wire_opacity)
    make_HH_unit_cell(ax, xlim=xlim, ylim=ylim)


def annotate_cell(ax):
    arrow_color = "blue"
    anno_fontsize = 8
    arrow_width = 0.5
    double_arrow(
        ax,
        -flag_x + q_radius,
        flag_y,
        xh - 2 * q_radius,
        0,
        arrow_width,
        color=arrow_color,
    )
    ax.annotate(
        "17.7 nm", xy=(-1.80 * xh, 0.1 * yd), rotation=0, fontsize=anno_fontsize
    )
    double_arrow(
        ax, q_radius - xh, flag_y, xh - 2 * q_radius, 0, arrow_width, color=arrow_color
    )
    ax.annotate(
        "17.7 nm", xy=(-0.80 * xh, 0.1 * yd), rotation=0, fontsize=anno_fontsize
    )

    nx = 0.5
    ny = np.sqrt(3) / 2
    double_arrow(
        ax,
        -data_x + nx * q_radius,
        data_y - ny * q_radius,
        xd - 2 * nx * q_radius,
        -yd + 2 * ny * q_radius,
        arrow_width,
        color=arrow_color,
    )
    ax.annotate(
        "17.7 nm",
        xy=(-data_x - 0.13 * xd, data_y - 0.88 * yd),
        rotation=-60,
        fontsize=anno_fontsize,
    )
    double_arrow(
        ax,
        -flag_x - xd + nx * q_radius,
        yd - ny * q_radius,
        xd - 2 * nx * q_radius,
        -yd + 2 * ny * q_radius,
        arrow_width,
        color=arrow_color,
    )
    ax.annotate(
        "17.7 nm",
        xy=(-2 * xh - xd - 0.13 * xd, yd - 0.88 * yd),
        rotation=-60,
        fontsize=anno_fontsize,
    )

    ax.annotate("(0,0)", xy=(0, -q_radius * 2))
    ax.annotate("(32,0)", xy=(xh, -q_radius * 2))
    ax.annotate("(64,0)", xy=(2 * xh + q_radius, 0))
    ax.annotate("(80,28)", xy=(2.2 * xh + xd - 6.5 * q_radius, 1.25 * yd))
    ax.annotate("(96,56)", xy=(2 * xh + 2 * xd - 6.5 * q_radius, 2 * yd))
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")


def plot_annotated_cell(filename=None):
    fig, ax = plt.subplots(1, 1)
    plot_single_cell(ax)
    annotate_cell(ax)
    if filename is not None:
        fig.savefig(filename)


def number_qubits(ax):
    for i in range(len(q_data_loc)):
        loc = q_data_loc[i] - np.ones(2) * q_radius / 2
        num = q_data_num[i]
        ax.annotate(num, xy=loc)
    for i in range(len(q_flag_loc)):
        loc = q_flag_loc[i] - np.ones(2) * q_radius / 2
        num = q_flag_num[i]
        ax.annotate(num, xy=loc)
    for i in range(len(q_meas_loc)):
        loc = q_meas_loc[i] - np.ones(2) * q_radius / 2
        num = q_meas_num[i]
        ax.annotate(num, xy=loc, color="white")


def numbered_qubits_cell(filename=None):
    fig, ax = plt.subplots(1, 1)
    plot_single_cell(ax)
    number_qubits(ax)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if filename is not None:
        fig.savefig(filename)


def is_populated(i, j):
    return (i + j) % 2 == 0


def place_hori_qubits(ax, r0):
    place_qubit(ax, q_meas_loc[0] + r0, q_colours[0])
    place_qubit(ax, q_flag_loc[0] + r0, q_colours[1])
    place_qubit(ax, q_flag_loc[1] + r0, q_colours[1])
    place_qubit(ax, q_coup_loc[0] + r0, q_colours[3])
    place_qubit(ax, q_coup_loc[1] + r0, q_colours[3])
    grid_color = "gray"
    grid_zorder = 0.25
    grid_opacity = 1
    grid_xh = flag_x * np.array([-1, 1])
    grid_yh = np.zeros(2)

    x0, y0 = r0
    ax.plot(
        grid_xh + x0,
        grid_yh + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )


def place_top_qubits(ax, r0):

    place_qubit(ax, q_coup_loc[4] + r0, q_colours[3])
    place_qubit(ax, q_coup_loc[5] + r0, q_colours[3])
    place_qubit(ax, q_data_loc[1] + r0, q_colours[2])
    if couplers:
        place_qubit(ax, q_data_loc[3] + r0, q_colours[2])
    place_hori_qubits(ax, r0)

    x0, y0 = r0
    grid_color = "gray"
    grid_zorder = 0.25
    grid_opacity = 1
    grid_xLL = np.array([-2 * xh - 2 * xd, -flag_x])
    grid_yLL = np.array([-2 * yd, 0])
    grid_xLR = np.array([2 * xh + 2 * xd, flag_x])
    grid_yLR = np.array([-2 * yd, 0])

    LL_SET(ax, r0)
    LR_SET(ax, r0)

    ax.plot(
        grid_xLL + x0,
        grid_yLL + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )
    ax.plot(
        grid_xLR + x0,
        grid_yLR + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )


def place_bottom_qubits(ax, r0):
    place_qubit(ax, q_coup_loc[3] + r0, q_colours[3])
    place_qubit(ax, q_coup_loc[2] + r0, q_colours[3])
    place_qubit(ax, q_data_loc[0] + r0, q_colours[2])
    place_qubit(ax, q_data_loc[2] + r0, q_colours[2])
    place_hori_qubits(ax, r0)
    x0, y0 = r0
    grid_color = "gray"
    grid_zorder = 0.25
    grid_opacity = 1
    grid_xUL = np.array([-2 * xh - 2 * xd, -flag_x])
    grid_yUL = np.array([2 * yd, 0])
    grid_xUR = np.array([2 * xh + 2 * xd, flag_x])
    grid_yUR = np.array([2 * yd, 0])

    UL_SET(ax, r0)
    UR_SET(ax, r0)

    ax.plot(
        grid_xUL + x0,
        grid_yUL + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )
    ax.plot(
        grid_xUR + x0,
        grid_yUR + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )


def place_left_qubits(ax, r0):
    place_qubit(ax, q_flag_loc[1] + r0, q_colours[1])
    place_qubit(ax, q_coup_loc[2] + r0, q_colours[3])
    place_qubit(ax, q_coup_loc[5] + r0, q_colours[3])

    x0, y0 = r0
    grid_color = "gray"
    grid_zorder = 0.25
    grid_opacity = 1
    grid_xUR = np.array([2 * xh + 2 * xd, flag_x])
    grid_yUR = np.array([2 * yd, 0])
    grid_xLR = np.array([2 * xh + 2 * xd, flag_x])
    grid_yLR = np.array([-2 * yd, 0])

    LR_SET(ax, r0)
    UR_SET(ax, r0)

    ax.plot(
        grid_xLR + x0,
        grid_yLR + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )
    ax.plot(
        grid_xUR + x0,
        grid_yUR + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )


def place_right_qubits(ax, r0):
    place_qubit(ax, q_flag_loc[0] + r0, q_colours[1])
    place_qubit(ax, q_coup_loc[3] + r0, q_colours[3])
    place_qubit(ax, q_coup_loc[4] + r0, q_colours[3])

    x0, y0 = r0
    grid_color = "gray"
    grid_zorder = 0.25
    grid_opacity = 1
    grid_xUL = np.array([-2 * xh - 2 * xd, -flag_x])
    grid_yUL = np.array([2 * yd, 0])
    grid_xLL = np.array([-2 * xh - 2 * xd, -flag_x])
    grid_yLL = np.array([-2 * yd, 0])

    LL_SET(ax, r0)
    UL_SET(ax, r0)

    ax.plot(
        grid_xUL + x0,
        grid_yUL + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )
    ax.plot(
        grid_xLL + x0,
        grid_yLL + y0,
        color=grid_color,
        zorder=grid_zorder,
        alpha=grid_opacity,
    )


def plot_cell_array(m, n, filename=None):

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    xlim = [-150 * a_lattice, (n - 1 / 2) * cell_length + 10 * a_lattice + 3 * xd]
    ylim = [
        -1.1 * yd - 90 * a_lattice,
        (m - 1 / 2) * cell_height + 10 * a_lattice + 2 * yd,
    ]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for i in range(n):
        for j in range(m):
            if is_populated(i, j):
                r0 = np.array([i * cell_length, j * cell_height])
                make_HH_unit_cell(ax, r0, xlim, ylim)

    for i in range(n):
        if is_populated(i, m):
            r0 = np.array([i * cell_length, m * cell_height])
            place_top_qubits(ax, r0)
        if is_populated(i, -1):
            r0 = np.array([i * cell_length, -1 * cell_height])
            place_bottom_qubits(ax, r0)
    LL_SET(ax, np.array([m * cell_length, n * cell_height]))

    for j in range(m):
        if is_populated(-1, j):
            r0 = np.array([-1 * cell_length, j * cell_height])
            place_left_qubits(ax, r0)
        if is_populated(n, j):
            r0 = np.array([n * cell_length, j * cell_height])
            place_right_qubits(ax, r0)

    place_all_wires(ax, m, n)

    ax.set_xlabel("[100] (nm)")
    ax.set_ylabel("[010] (nm)")
    ax.set_aspect("equal")
    # ax.axis('off')
    fig.set_size_inches(1.2 * fig_width_single, 1.2 * fig_height_single)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)


# green, purple, yellow, red, blue
Phi_L = [-3, -3, 0, -2, -1, 0]
L_loaded = [q_coup_loc[0]]
Phi_R = [-3, -3, 0, -1, -2, 0]
R_loaded = [q_coup_loc[1]]
Phi_U = [-2, -1, 0, -3, -3, 0]
U_loaded = [q_coup_loc[2], q_coup_loc[3]]
Phi_D = [-1, -2, 0, -3, -3, 0]
D_loaded = [q_coup_loc[4], q_coup_loc[5]]
Phi_UL = [-2, -1, 0, -3, -1, 0]
UL_loaded = [q_coup_loc[3]]
Phi_DR = [-1, -2, 0, -1, -3, 0]
D_loaded = [q_coup_loc[4], q_coup_loc[5]]


def CNOT(ax, Phi, loaded):
    plot_single_cell(ax, wire_colours=[V_colours[p] for p in Phi], wire_opacity=1)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    for loc in loaded:
        place_qubit(ax, loc, "red")


import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.figure import figaspect


def generate_2_coupler_conditions(fp=None):
    fig, ax = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.1]})
    CNOT(ax[0], Phi_L, L_loaded)
    CNOT(ax[1], Phi_UL, UL_loaded)

    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    Dx = xlim[1] - xlim[0]
    Dy = ylim[1] - ylim[0]
    ax[2].set_xlim(0, Dx / 5)
    ax[2].set_ylim(Dy)

    colors = V_colours[-3:] + V_colours[0:1]
    color_bar(ax[2], colors=colors)

    fig.set_size_inches(fig_width_double, fig_height_single * 0.8)
    fig.tight_layout()
    if fp is not None:
        fig.savefig(fp)


def generate_CNOTs(fp=None):
    fig, ax = plt.subplots(1, 4)
    CNOT(ax[0], Phi_L, L_loaded)
    CNOT(ax[1], Phi_R, R_loaded)
    CNOT(ax[2], Phi_U, U_loaded)
    CNOT(ax[3], Phi_D, D_loaded)
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-3.5, vmax=0.5)
    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])

    label_axis(ax[0], "(a), (e)", y_offset=-0.17)
    label_axis(ax[1], "(b), (f)", y_offset=-0.17)
    label_axis(ax[2], "(c)", y_offset=-0.17)
    label_axis(ax[3], "(d)", y_offset=-0.17)

    ncolors = 4
    cw = 256 // 4  # color width
    colors = np.zeros((256, 4))
    for n in range(ncolors):
        colors[n * cw : (n + 1) * cw, :] = V_colours[n - 3]

    # cmap = ListedColormap(colors)
    # ax[4].set_aspect(1.5)
    # ax[4].set_yticks([0, -1, -2, -3], ['$V_0$', '$V_1$', '$V_2$', '$V_3$'])
    # cb1 = mpl.colorbar.ColorbarBase(ax[4], cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')
    # cb1.set_label('Voltage')

    w, h = figaspect(0.2)
    fig.set_size_inches(w, h)

    if fp is not None:
        fig.savefig(fp)


def plot_HS_grid(ax, r0, left_link=False, down_link=False):
    X = [-d_HS, 0, 0]
    Y = [0, 0, -d_HS]
    if left_link:
        X[0] *= 2
    if down_link:
        Y[2] *= 2
    ax.plot(r0[0] + np.array(X), r0[1] + np.array(Y), color="black", zorder=0)


if __name__ == "__main__":
    """ HEAVY HEXAGON """
    couplers = True
    # plot_cell_array(4,4, filename="cell_array")
    # generate_CNOTs()
    # generate_2_coupler_conditions()
    # plot_annotated_cell(filename="single_cell")
    # numbered_qubits_cell()
    # plot_single_cell()


############################################################################################################
#                HEAVY SQUARE
############################################################################################################
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
# q_coup_radius = 0.1*d_HS
q_radii = [q_radius, q_radius, q_radius, q_coup_radius]
q_locs_HS = [
    [[-d_HS, 0]],
    [[0, 0]],
    [[0, -d_HS]],
    [[-3 * d_HS / 2, 0], [-d_HS / 2, 0], [0, -d_HS / 2], [0, d_HS / 2]],
]
HS_cell_size = 2 * d_HS
read_color = "red"


SET_operating_color = "lightblue"
SET_flag_2e_color = "blue"
SET_no_Sb_color = "gold"
SET_size = 13
tile_pad = 0
tile_alpha = 0
Xstab_tile_color = "lightgreen"
Zstab_tile_color = "lightblue"
read_radius = 1.8 * q_radius


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


def make_HS_unit_cell(
    ax,
    r0=np.array([0, 0]),
    left_link=False,
    down_link=False,
    place_SET=True,
    q_colours=q_colours,
    i=0,
    j=0,
    q_radius=q_radius,
):

    plot_HS_grid(ax, r0, left_link=left_link, down_link=down_link)

    print(q_colours)
    # place qubits
    for k in range(2):
        for q_loc in q_locs[k]:
            if k == 0:
                if (i + j) % 2 == 0:
                    color = q_colours[-1]
                else:
                    color = q_colours[0]
            else:
                color = q_colours[k]
            place_qubit(ax, q_loc + r0, color, q_radius, couplers=True)


def make_stab_unit_cell(ax=None, r0=np.array([d_HS, 0]), fontsize=17):
    if ax is None:
        ax = plt.subplot()
    q_radius = 0.15 * d_HS
    make_HS_unit_cell(ax, r0, left_link=True, down_link=False, q_radius=q_radius)
    place_qubit(
        ax, r0, q_colours[1], q_radius, text="6", textcolor="white", fontsize=fontsize
    )
    place_qubit(
        ax,
        r0 - d_HS * x_hat,
        q_colours[0],
        q_radius,
        text="7",
        textcolor="white",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - d_HS * y_hat,
        q_colours[2],
        q_radius,
        text="4",
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat,
        q_colours[1],
        q_radius,
        text="5",
        textcolor="white",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat - d_HS * y_hat,
        q_colours[2],
        q_radius,
        text="3",
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat + d_HS * y_hat,
        q_colours[2],
        q_radius,
        text="1",
        textcolor="black",
        fontsize=fontsize,
    )
    place_qubit(
        ax,
        r0 + d_HS * y_hat,
        q_colours[2],
        q_radius,
        text="2",
        textcolor="black",
        fontsize=fontsize,
    )
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    place_HS_cell_SET(ax, r0)
    ax.set_ylim(ylim)
    ax.plot(
        [r0[0] - 2 * d_HS, r0[0] - 2 * d_HS],
        [r0[1] + d_HS, r0[1] - d_HS],
        zorder=0,
        color="black",
    )
    ax.plot([r0[0], r0[0]], [r0[1] + d_HS, r0[1] - d_HS], zorder=0, color="black")
    ax.set_aspect("equal")
    ax.set_xticks([-d_HS, -SET_size / 2, 0, SET_size / 2, d_HS])
    ax.set_yticks([-d_HS, -2.5, 0, 2.5, d_HS])
    alpha = 0.4
    SD_col = grey
    G_col = "orange"
    G_alpha = 0.4
    place_wire(ax, -d_HS, grey, "hori", xlim, alpha, 5)
    place_wire(ax, 0, G_col, "hori", xlim, G_alpha, 5)
    place_wire(ax, d_HS, G_col, "vert", xlim, G_alpha, 5)
    place_wire(ax, -d_HS, G_col, "vert", xlim, G_alpha, 5)
    place_wire(ax, d_HS, grey, "hori", xlim, alpha, 5)
    place_wire(ax, 0, grey, "vert", xlim, alpha, 5)
    ax.set_xlabel("[1,-1,0] (nm)")
    ax.set_ylabel("[1,1,0] (nm)")
    ax.get_figure().tight_layout()


def make_atomic_unit_cell(ax=None, r0=np.array([47, 0]), fontsize1P=15, fontsize2P=15):
    if ax is None:
        ax = plt.subplot()
    d_HS = 47
    HS_cell_size = 2 * d_HS
    q1P_radius = 0.15 * d_HS
    q2P_radius = 0.2 * d_HS
    q1P_col = "black"
    q2P_col = "darkred"
    make_HS_unit_cell(
        ax, r0, left_link=True, down_link=False, q_radius=0.0001 * q_radius
    )
    place_qubit(
        ax, r0, q2P_col, q2P_radius, text="2P", textcolor="white", fontsize=fontsize2P
    )
    place_qubit(
        ax,
        r0 - d_HS * x_hat,
        q1P_col,
        q1P_radius,
        text="1P",
        textcolor="white",
        fontsize=fontsize1P,
    )
    place_qubit(
        ax,
        r0 - d_HS * y_hat,
        q1P_col,
        q1P_radius,
        text="1P",
        textcolor="white",
        fontsize=fontsize1P,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat,
        q2P_col,
        q2P_radius,
        text="2P",
        textcolor="white",
        fontsize=fontsize2P,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat - d_HS * y_hat,
        q1P_col,
        q1P_radius,
        text="1P",
        textcolor="white",
        fontsize=fontsize1P,
    )
    place_qubit(
        ax,
        r0 - HS_cell_size * x_hat + d_HS * y_hat,
        q1P_col,
        q1P_radius,
        text="1P",
        textcolor="white",
        fontsize=fontsize1P,
    )
    place_qubit(
        ax,
        r0 + d_HS * y_hat,
        q1P_col,
        q1P_radius,
        text="1P",
        textcolor="white",
        fontsize=fontsize1P,
    )
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    # place_HS_cell_SET(ax, r0)
    ax.set_ylim(ylim)
    ax.plot(
        [r0[0] - 2 * d_HS, r0[0] - 2 * d_HS],
        [r0[1] + d_HS, r0[1] - d_HS],
        zorder=0,
        color="black",
    )
    ax.plot([r0[0], r0[0]], [r0[1] + d_HS, r0[1] - d_HS], zorder=0, color="black")
    ax.set_aspect("equal")
    ax.set_xticks([-d_HS, 0, d_HS])
    ax.set_yticks([-d_HS, 0, d_HS])
    alpha = 0.4
    SD_col = grey
    G_col = "orange"
    G_alpha = 0.4
    # place_wire(ax, -d_HS, grey, 'hori', xlim, alpha, 5)
    # place_wire(ax, 0, G_col, 'hori', xlim, G_alpha, 5)
    # place_wire(ax, d_HS, G_col, 'vert', xlim, G_alpha, 5)
    # place_wire(ax, -d_HS, G_col, 'vert', xlim, G_alpha, 5)
    # place_wire(ax, d_HS, grey, 'hori', xlim, alpha, 5)
    # place_wire(ax, 0, grey, 'vert', xlim, alpha, 5)
    ax.set_xlabel("[1,-1,0] (lattice sites)")
    ax.set_ylabel("[1,1,0] (lattice sites)")
    ax.get_figure().tight_layout()


def HS_upper_boundary_cell(ax, i, distance, q_colours=q_colours):
    upper_data_loc = np.array([i * HS_cell_size, (2 * distance - 3) * d_HS])
    lower_data_loc = np.array([i * HS_cell_size, -d_HS])
    place_qubit(ax, upper_data_loc, q_colours[2], q_radii[0])
    place_qubit(ax, upper_data_loc + d_HS * y_hat, q_colours[1], q_radii[0])
    place_qubit(
        ax,
        lower_data_loc - d_HS * y_hat - HS_cell_size * x_hat,
        q_colours[1],
        q_radii[0],
    )
    if couplers:
        place_qubit(
            ax,
            lower_data_loc - d_HS / 2 * y_hat - 2 * d_HS * x_hat,
            q_colours[3],
            q_radii[3],
        )
        place_qubit(ax, upper_data_loc + d_HS / 2 * y_hat, q_colours[3], q_radii[3])
    # place_HS_cell_SET(ax, np.array([(i-1)*HS_cell_size, -2*d_HS]))
    if i % 2 == 0:
        place_qubit(
            ax, upper_data_loc + d_HS * (x_hat + y_hat), q_colours[3], q_radii[0]
        )
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
            q_radii[0],
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


couplers = False


def HS_left_boundary_cell(ax, i, distance, q_colours=q_colours):
    flag_loc = np.array([-HS_cell_size, (2 * i) * d_HS])
    flag_loc_right = np.array([2 * (distance - 1) * d_HS, (2 * i) * d_HS])
    place_qubit(
        ax, np.array([-HS_cell_size, (2 * i + 1) * d_HS]), q_colours[2], q_radii[0]
    )
    place_qubit(ax, flag_loc, q_colours[1], q_radii[0])
    if couplers:
        place_qubit(ax, flag_loc + d_HS / 2 * y_hat, q_colours[3], q_radii[3])
        place_qubit(ax, flag_loc - d_HS / 2 * y_hat, q_colours[3], q_radii[3])
    # place_HS_cell_SET(ax, np.array([-HS_cell_size,(2*i)*d_HS]))
    if i % 2 == 0:
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
    place_HS_cell_SET(
        ax,
        np.array([(distance - 1) * HS_cell_size, (2 * distance - 2) * d_HS]),
        color=colors[0],
    )


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


def make_HS_qubit_array(
    ax,
    distance,
    SET_colors=[grey, grey],
    wire_colors=[grey, grey, yellow, yellow, yellow, yellow],
    q_colours=q_colours,
):
    """
    wire_colours =? [SET1, SET2, hgate1, hgate2, vgate1, vgate2] 

    """
    # plot distance-1 cells, then handle edges
    for i in range(distance - 1):
        # HS_upper_boundary_cell(ax, i, distance, q_colours=q_colours)
        # HS_left_boundary_cell(ax, i, distance, q_colours=q_colours)
        for j in range(distance - 1):
            flag_loc = HS_cell_size * np.array([i, j])
            make_HS_unit_cell(
                ax,
                flag_loc,
                left_link=True,
                down_link=j > 0,
                q_colours=q_colours,
                place_SET=(i + j) % 2 == 0,
                i=i,
                j=j,
            )  # or j==0 or i==0 or i==distance-2)
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
    place_qubit(ax, np.array([-HS_cell_size, -d_HS]), q_colours[2], q_radii[0])
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
    ax,
    distance,
    SET_colors=[grey, grey],
    wire_colors=[grey, grey, yellow, yellow, yellow, yellow],
    q_colours=q_colours,
):
    make_HS_qubit_array(
        ax,
        distance,
        SET_colors=SET_colors,
        wire_colors=wire_colors,
        q_colours=q_colours,
    )
    place_HS_SETs(ax, distance, colors=SET_colors)
    place_HS_wires(ax, distance, colors=wire_colors)


def logical_operators():
    pass


def surface_code_defects(defects=[np.array([3, 3])], distance=5, ax=None):
    defect_color = "lightgray"
    defect_length = 1.8
    dq_00 = d_HS * np.array([-2, -1])
    if ax is None:
        ax = plt.subplot()
    make_HS_array(ax, distance)
    for defect in defects:
        loc = dq_00 + HS_cell_size * defect
        # place_qubit(ax, loc, color=defect_color)
        # place_qubit(ax, loc+d_HS*y_hat, color=defect_color)
        # place_qubit(ax, loc-d_HS*y_hat, color=defect_color)
        place_qubit(ax, loc + d_HS * (x_hat + y_hat), color=defect_color)
        place_qubit(ax, loc + d_HS * (x_hat - y_hat), color=defect_color)
        place_qubit(ax, loc - d_HS * (x_hat + y_hat), color=defect_color)
        place_qubit(ax, loc - d_HS * (x_hat - y_hat), color=defect_color)
        place_HS_cell_SET(ax, loc + d_HS * y_hat, color=SET_no_Sb_color)
        place_HS_cell_SET(ax, loc - d_HS * y_hat, color=SET_no_Sb_color)
        place_HS_cell_SET(ax, loc + d_HS * (2 * x_hat + y_hat), color=SET_no_Sb_color)
        place_HS_cell_SET(ax, loc + d_HS * (2 * x_hat - y_hat), color=SET_no_Sb_color)

        X = [
            loc[0] - defect_length * d_HS,
            loc[0] + defect_length * d_HS,
            loc[0] + defect_length * d_HS,
            loc[0] - defect_length * d_HS,
            loc[0] - defect_length * d_HS,
        ]
        Y = [
            loc[1] - defect_length * d_HS,
            loc[1] - defect_length * d_HS,
            loc[1] + defect_length * d_HS,
            loc[1] + defect_length * d_HS,
            loc[1] - defect_length * d_HS,
        ]
        ax.plot(X, Y, linestyle="--", color="blue")


def flag_readout_1(ax, distance=5):
    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, blue, grey, orange, red],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    for j in range(distance // 2 + 1):
        for i in range(distance // 2):
            place_qubit(
                ax, np.array([4 * i, 4 * j]) * d_HS, read_color, q_radius=read_radius
            )


def flag_readout_2(ax, distance=5):
    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, grey, grey, red, red],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    for j in range(distance // 2 + 1):
        for i in range(distance // 2 + 1):
            place_qubit(ax, np.array([4 * i, 4 * j]) * d_HS, grey)
            if i == 0 and j == distance // 2:
                continue
            place_qubit(
                ax,
                np.array([-2 * d_HS, 0]) + np.array([4 * i, 4 * j]) * d_HS,
                read_color,
                q_radius=read_radius,
            )


def flag_readout_3(ax, distance=5):
    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, grey, blue, red, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    for j in range(distance // 2 + 1):
        for i in range(distance // 2 + 1):
            place_qubit(ax, np.array([4 * i, 4 * j]) * d_HS, grey)
            if j != 0 or i != distance // 2:
                place_qubit(
                    ax,
                    np.array([-2 * d_HS, -2 * d_HS]) + np.array([4 * i, 4 * j]) * d_HS,
                    read_color,
                    q_radius=read_radius,
                )
            if i == 0 and j == distance // 2:
                continue
            place_qubit(
                ax, np.array([-2 * d_HS, 0]) + np.array([4 * i, 4 * j]) * d_HS, grey
            )


def flag_readout_4(ax, distance=5):
    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, grey, grey, grey, red],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    for j in range(distance // 2 + 1):
        for i in range(distance // 2 + 1):
            place_qubit(ax, np.array([4 * i, 4 * j]) * d_HS, grey)
            if i != 0 or j != distance // 2:
                place_qubit(
                    ax,
                    np.array([0 - 2 * d_HS, 0]) + np.array([4 * i, 4 * j]) * d_HS,
                    grey,
                )
            if j != 0 or i != distance // 2:
                place_qubit(
                    ax,
                    np.array([-2 * d_HS, -2 * d_HS]) + np.array([4 * i, 4 * j]) * d_HS,
                    grey,
                )
            place_qubit(
                ax,
                np.array([0, -2 * d_HS]) + np.array([4 * i, 4 * j]) * d_HS,
                read_color,
                q_radius=read_radius,
            )


def measure_readout(ax, distance=5):
    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, grey, red, red, red],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    for j in range(distance // 2 + 1):
        for i in range(distance // 2 + 1):
            place_qubit(ax, np.array([4 * i, 4 * j]) * d_HS, grey)
            place_qubit(
                ax, np.array([0, -2 * d_HS]) + np.array([4 * i, 4 * j]) * d_HS, grey
            )
            if j != 0 or i != distance // 2:
                place_qubit(
                    ax,
                    np.array([-2 * d_HS, -2 * d_HS]) + np.array([4 * i, 4 * j]) * d_HS,
                    grey,
                )
            if i != 0 or j != distance // 2:
                place_qubit(
                    ax, np.array([-2 * d_HS, 0]) + np.array([4 * i, 4 * j]) * d_HS, grey
                )
        place_qubit(ax, np.array([1, 4 * j]) * d_HS, read_color, q_radius=read_radius)
        place_qubit(ax, np.array([5, 4 * j]) * d_HS, read_color, q_radius=read_radius)


def readout(distance=5, fp=None):
    fig, ax = plt.subplots(2, 2)
    make_HS_array(
        ax[0, 0],
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, blue, "silver", orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    flag_readout_1(ax[0, 1], distance)
    flag_readout_2(ax[1, 0], distance)
    # flag_readout_3(ax[1,0], distance)
    # flag_readout_4(ax[1,1], distance)
    measure_readout(ax[1, 1], distance)
    x_offset = 0.08
    y_offset = -0.12
    fontsize = 16
    label_axis(
        ax[0, 0], "CNOTs", x_offset=x_offset, y_offset=y_offset, fontsize=fontsize
    )
    label_axis(
        ax[0, 1], "read flag 1", x_offset=x_offset, y_offset=y_offset, fontsize=fontsize
    )
    label_axis(
        ax[1, 0], "read flag 2", x_offset=x_offset, y_offset=y_offset, fontsize=fontsize
    )
    # label_axis(ax[1,0], 'read flag 3', x_offset=x_offset, y_offset=y_offset, fontsize=fontsize)
    # label_axis(ax[1,1], 'read flag 4', x_offset=x_offset, y_offset=y_offset, fontsize=fontsize)
    label_axis(
        ax[1, 1],
        "read measure",
        x_offset=x_offset,
        y_offset=y_offset,
        fontsize=fontsize,
    )

    fig.set_size_inches(fig_width_double, 1.05 * fig_width_double)
    if fp is not None:
        fig.savefig(fp)


def illustrative_configs(distance=5, fp=None):
    fig, ax = plt.subplots(2, 3)
    make_HS_array(
        ax[0, 0],
        distance=distance,
        SET_colors=[grey, grey],
        wire_colors=[grey] * 6,
        q_colours=[
            q_meas_colour,
            q_flag_colour,
            q_data_colour,
            q_meas_colour,
            q_meas_colour,
            q_meas_colour,
        ],
    )
    # make_HS_array(ax[0,1], distance=distance, SET_colors=[grey, grey], wire_colors=[grey, grey, grey,grey, 'silver', darkblue], q_colours = [grey, q_flag_colour, q_data_colour, grey])
    make_HS_array(
        ax[0, 1],
        distance=distance,
        SET_colors=[grey, grey],
        wire_colors=["silver", "silver", "silver", "silver", red, red],
        q_colours=[
            q_meas_colour,
            grey,
            q_data_colour,
            q_meas_colour,
            q_meas_colour,
            q_meas_colour,
        ],
    )
    make_HS_array(
        ax[0, 2],
        distance=distance,
        SET_colors=[yellow, yellow],
        wire_colors=[yellow, yellow, "silver", grey, "silver", grey],
        q_colours=[grey, q_flag_colour, grey, grey, grey, grey],
    )
    make_HS_array(
        ax[1, 0],
        distance=distance,
        SET_colors=[yellow, yellow],
        wire_colors=[yellow, yellow, "silver", "silver", orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, grey, grey, grey],
    )
    make_HS_array(
        ax[1, 1],
        distance=distance,
        SET_colors=[grey, grey],
        wire_colors=[grey, grey, blue, "silver", grey, grey],
        q_colours=[
            grey,
            q_flag_colour,
            q_data_colour,
            q_meas_colour,
            grey,
            q_meas_colour,
        ],
    )
    make_HS_array(
        ax[1, 2],
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, blue, "silver", orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    # make_HS_array(ax[1,1], distance=distance, SET_colors=[orange,grey], wire_colors=[orange,grey, blue,grey, 'darkblue', grey])
    x_offset = 0
    y_offset = -0.15
    label_axis(ax[0, 0], "a", x_offset=x_offset, y_offset=y_offset)
    label_axis(ax[0, 1], "b", x_offset=x_offset, y_offset=y_offset)
    label_axis(ax[0, 2], "c", x_offset=x_offset, y_offset=y_offset)
    label_axis(ax[1, 0], "d", x_offset=x_offset, y_offset=y_offset)
    label_axis(ax[1, 1], "e", x_offset=x_offset, y_offset=y_offset)
    label_axis(ax[1, 2], "f", x_offset=x_offset, y_offset=y_offset)
    # make_HS_array(ax, distance=distance, SET_colors=['silver', 'cornflowerblue'], wire_colors=['silver', 'cornflowerblue', 'silver', 'red', 'silver', 'orange'])
    fig.set_size_inches(fig_width_double, 0.7 * fig_width_double)

    if fp is not None:
        fig.savefig(fp)


def stabilizer_activations(fp=None, distance=5):
    fig, ax = plt.subplots(2, 2)
    make_HS_array(
        ax[0, 0],
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, blue, "silver", orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, q_meas_colour, grey, grey],
    )
    make_HS_array(
        ax[0, 1],
        distance=distance,
        SET_colors=[grey, yellow],
        wire_colors=[grey, yellow, grey, blue, orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, grey, q_meas_colour, grey],
    )
    make_HS_array(
        ax[1, 0],
        distance=distance,
        SET_colors=[grey, yellow],
        wire_colors=[grey, yellow, blue, grey, orange, orange],
        q_colours=[grey, q_flag_colour, q_data_colour, grey, grey, q_meas_colour],
    )
    make_HS_array(
        ax[1, 1],
        distance=distance,
        SET_colors=[yellow, grey],
        wire_colors=[yellow, grey, grey, blue, orange, orange],
        q_colours=[q_meas_colour, q_flag_colour, q_data_colour, grey, grey, grey],
    )
    fig.set_size_inches(fig_width_double, 1.05 * fig_width_double)
    if fp is not None:
        fig.savefig(fp)


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


def make_conventional_square_array(distance=4):
    global d_HS
    fucku = 1 * d_HS
    # set_trace()
    del d_HS
    d_HS = 1.5 * fucku
    fig, ax = plt.subplots(1, 1)
    for i in range(distance):

        loc_d1 = np.array([2 * i + 1, -1]) * d_HS
        # loc_a1 = loc_d1 + np.array([1,0])*d_HS
        loc_d2 = np.array([-1, 2 * i + 1]) * d_HS
        # loc_a2 = loc_d1 + np.array([0,1])*d_HS
        place_qubit(ax, loc_d1, "orange")
        # place_qubit(ax, loc_a1, 'black')
        place_qubit(ax, loc_d2, "orange")
        # place_qubit(ax, loc_a2, 'black')
        ax.axhline(2 * i * d_HS, color="black")
        ax.axvline(2 * i * d_HS, color="black")
        ax.axhline((2 * i + 1) * d_HS, color="black")
        ax.axvline((2 * i + 1) * d_HS, color="black")
        for j in range(distance):
            loc_d1 = np.array([i, j]) * 2 * d_HS
            loc_a1 = loc_d1 + np.array([1, 0]) * d_HS
            loc_d2 = loc_d1 + np.array([1, 1]) * d_HS
            loc_a2 = loc_d1 + np.array([0, 1]) * d_HS
            place_qubit(ax, loc_d1, "orange")
            place_qubit(ax, loc_a1, "black")
            place_qubit(ax, loc_d2, "orange")
            place_qubit(ax, loc_a2, "black")

    ax.set_xlim([-2 * d_HS, distance * 2 * d_HS])
    ax.set_ylim([-2 * d_HS, distance * 2 * d_HS])
    ax.set_aspect("equal")

    ############################################################################################################
    ############################################################################################################
    #           1P-1P stuff for paper
    ############################################################################################################
    ############################################################################################################

    pad=1
    ax.set_ylim([-2 * d_HS - pad, pad])
    ax.set_xlim([-2 * d_HS - pad, pad])


def HS_11(distance=5):

    ax = plt.subplot()

    make_HS_array(
        ax,
        distance=distance,
        SET_colors=[grey, yellow],
        wire_colors=[grey, yellow, grey, blue, grey, orange],
        q_colours=[
            q_meas_colour,
            q_flag_colour,
            q_data_colour,
            q_meas_colour,
            q_meas_colour,
            q_meas_colour,
        ],
    )


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
    q_colours = [
        q_data_colour,
        q_coup_colour,
        q_anc_colour,
        q_coup_colour,
        q_data_colour,
    ]
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
        print(q_radius, q_colours[k])
        place_qubit(ax, r0 + loc, q_colours[k], q_radius, q_coup_radius=q_radius)
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


if __name__ == "__main__":

    """ HEAVY SQUARE """
    couplers = False

    # make_HS_array(plt.subplot(), 13)
    # surface_code_defects(np.array([[2,2], ]), distance=5)

    # plt.subplot().add_patch(mpl.patches.Polygon([[0,0],[1,2],[2,3]]))

    # illustrative_configs()
    # readout()
    # HS_side_view()
    # make_stab_unit_cell()
    make_atomic_unit_cell()
    # stabilizer_activations()
    # make_HS_qubit_array(plt.subplot(), 5)
    # make_conventional_square_array()

    # HS_11()

    plt.show()

