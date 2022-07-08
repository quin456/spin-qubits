
import numpy as np
import matplotlib.pyplot as plt 
from pdb import set_trace

distorted = False

# all distance values are in nm

n_qubit_types = 4


# distance between lattice sites
a_lattice = 0.543

# number of lattice sites in x and y direction between adjacent qubits in horizontal and vertical directions
xh = 32*a_lattice; yh=0*a_lattice
xd = 16*a_lattice; yd = 28*a_lattice


# colours 
SET_colour = 'gray'
q_meas_colour = 'black'
q_flag_colour = 'white'
q_coup_colour = 'gray'
q_data_colour = 'orange'
q_colours = [q_meas_colour, q_flag_colour, q_coup_colour, q_data_colour]
V_colours = ['lightgray', 'orange', 'red', 'black', 'blue', 'lightblue']
V_colours = ['lightgray', 'orange', 'red', '#0000cc', '#0080ff', '#99ccff']
V_colours = [np.array([224,224,224,256])/256, 'orange', 'red', [0,0,204/256,1], [0,128/256,255/256,1], [153/256,204/256,255/256,1]]

# locations within unit cell
SET_x = 46*a_lattice; SET_y = 42*a_lattice
R_SET = 6.5 #radius
nhat = np.array([np.cos(23.4*np.pi/180), np.sin(23.4*np.pi/180)])

#SET locations 


SET_loc = np.array([[SET_x,SET_y],[-SET_x-2*R_SET,SET_y], [-SET_x-2*R_SET,-SET_y], [SET_x,-SET_y]])
WCs = np.array([[xh,yd],[-xh,yd],[-xh,-yd],[xh,-yd]])


SET_angle = np.array([0,0,0, 0,45])
SET_length = np.array([0.6,1,1,0.6])*4*R_SET
SET_width = np.array([1,1,1,1])*R_SET
#-23.4*np.pi/180
if not distorted:
    cell_length = 4*xh+4*xd #276
    cell_height = 4*yd #160

    q_meas_loc = np.array([[0,0]])
    q_meas_num = [7]
    flag_x = 2*xh; flag_y=0
    data_x = 2*(xh+xd); data_y = 2*yd
    measure_x=0; measure_y=0
    q_flag_loc = np.array([[-2*xh,0],[2*xh,0]])
    q_coup_loc = np.array([[-xh,0],[xh,0],[2*xh+xd,yd],[-2*xh-xd,yd],[-2*xh-xd,-yd],[2*xh+xd,-yd] ])
    q_data_loc = np.array([[-data_x,data_y],[-data_x,-data_y],[data_x,data_y],[data_x,-data_y]])
    q_flag_num = [5,6]
    q_data_num = [4,3,1,2]

    def plot_grid(ax,r0):
        grid_color='gray'
        grid_zorder=0.25
        grid_opacity=1
        grid_xh = flag_x*np.array([-1,1])
        grid_yh = np.zeros(2)
        grid_xUL=np.array([-2*xh-2*xd,-flag_x])
        grid_yUL=np.array([2*yd,0])
        grid_xLL=np.array([-2*xh-2*xd,-flag_x])
        grid_yLL=np.array([-2*yd,0])
        grid_xLR=np.array([2*xh+2*xd,flag_x])
        grid_yLR=np.array([-2*yd,0])
        grid_xUR=np.array([2*xh+2*xd,flag_x])
        grid_yUR=np.array([2*yd,0])

        x0,y0=r0
        ax.plot(grid_xh+x0,grid_yh+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
        ax.plot(grid_xUL+x0,grid_yUL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
        ax.plot(grid_xLL+x0,grid_yLL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
        ax.plot(grid_xLR+x0,grid_yLR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
        ax.plot(grid_xUR+x0,grid_yUR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)


    nwire_vert = 4
    wire_vert_x = [-2*xh-2*xd,-xh,xh]
    wire_vert_colours = ['orange', 'red', 'blue']

    nwire_hori = 2
    wire_hori_y = [yd,-yd]
    wire_hori_colours = ['green','purple']

else:
    cell_length = 6*xh+2*xd
    cell_height = 2*yd


    q_meas_loc = np.array([[0,0]])
    q_flag_loc = np.array([[-2*xh,0],[2*xh,0]])
    q_coup_loc = np.array([[-2*xh-xd,40],[-2.5*xh,-yd],[-2*xh-xd,-yd],[2*xh+xd,yd], [-xh,0],[xh,0]])
    q_data_loc = np.array([[-3*xh-xd,yd],[-3*xh-xd,-yd],[3*xh+xd,-yd],[3*xh+xd,yd]])

    nwire_vert = 4
    wire_vert_x = [-2*xh-xd,-xh,xh, 2*xh+xd]
    wire_vert_colours = ['orange', 'red', 'blue', 'purple']
    nwire_hori = 1
    wire_hori_y = [2*yh]
    wire_hori_colours = ['green']


q_locs = [q_meas_loc, q_flag_loc, q_coup_loc, q_data_loc]
q_radius = 6.5*a_lattice


# wires 

wire_width = 5
default_wire_opacity=0.3

def double_arrow(ax,x,y,dx,dy,width=0.05, color='darkblue',linewidth=0.001):
    ax.arrow(x+dx/2.01,y+dy/2.01,dx/1.99,dy/1.99,width=width, length_includes_head=True, color=color,linewidth=linewidth)
    ax.arrow(x+dx/1.99,y+dy/1.99,-dx/2.01,-dy/2.01,width=width, length_includes_head=True, color=color, linewidth=linewidth)


def stadium(ax,start,end,radius, color=SET_colour):

    circ1 = plt.Circle(start, radius, color=color)
    circ2 = plt.Circle(end, radius, color=color)
    length = np.linalg.norm(start-end)
    unit_vec = (end-start)/length
    normal = np.array([-unit_vec[1], unit_vec[0]])
    angle = np.arctan(unit_vec[1]/unit_vec[0])*180/np.pi
    print(angle)
    mid_section = plt.Rectangle(start - radius*normal, length, 2*radius, angle=angle, color=color)

    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.add_patch(mid_section)

def unit_vector(theta):
    theta=theta*np.pi/180
    nhat=np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-nhat[1], nhat[0]])
    return nhat,normal


def rect_SET(ax,start,length,width,angle,color=SET_colour):
    nhat,normal = unit_vector(angle)
    end = start + length*nhat
    # length = np.linalg.norm(start-end)
    # unit_vec = (end-start)/length
    #angle = np.arctan(nhat[1]/nhat[0])*180/np.pi
    mid_section = plt.Rectangle(start - 0.5*width*normal, length, width, angle=angle, color=color)
    ax.add_patch(mid_section)

def place_SET(ax, location, slant):
    
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
    end = location + 2*R_SET*np.array([np.cos(slant),np.sin(slant)])
    stadium(ax,start,end,R_SET)

def place_wire(ax,loc,colour,orientation, lim, wire_opacity):
    length = lim[1]-lim[0]
    middle = (lim[0]+lim[1])/2
    if orientation == 'vert':
        wire = plt.Rectangle([loc-wire_width/2, lim[0]], wire_width, length-lim[0], color=colour, alpha = wire_opacity)
    elif orientation == 'hori':
        wire = plt.Rectangle([lim[0], loc-wire_width/2], length-lim[0], wire_width, color=colour, alpha = wire_opacity)
    else: raise Exception("Invalid wire orientation")

    ax.add_patch(wire)

def UR_SET(ax,r0):
    length=18
    width=9
    loc=r0+WCs[0]+np.array([-6.5,0])
    rect_SET(ax,loc,length,width,0)

def UL_SET(ax,r0):
    length=18
    width=9
    loc=r0+WCs[1]+np.array([6.5-length,0])
    rect_SET(ax,loc,length,width,0)

def LR_SET(ax,r0):
    width=12
    length=25
    loc=r0+WCs[3]-np.array([length/2,0])
    rect_SET(ax,loc,length,width,0)
    
    length2 = 2*18.7/np.sqrt(3)
    loc2 = loc+np.array([length,width/2])+width/2 * np.array([-np.sqrt(3)/2,-1/2])
    rect_SET(ax,loc2,2*18.7/np.sqrt(3),width,-60)

def LL_SET(ax,r0):
    width=12
    loc=r0+WCs[2]-np.array([11,0])
    rect_SET(ax,loc,13.5,width,0)


    length2 = 2*18.7/np.sqrt(3)
    loc2 = loc+np.array([0,width/2])+width/2 * np.array([np.sqrt(3)/2,-1/2])
    rect_SET(ax,loc2,length2,width,240)

    #ax.scatter([loc[0]],[loc[1]])
    #ax.scatter([loc2[0]],[loc2[1]])

def place_cell_SETs(ax,r0):
    LL_SET(ax,r0)
    LR_SET(ax,r0)
    UR_SET(ax,r0)
    UL_SET(ax,r0)
    # if not distorted:
    #     for i in range(len(SET_loc)):
    #         rect_SET(ax,SET_loc[i],SET_length[i],SET_width[i],SET_angle[i])
    #         #place_SET(ax,np.array(r0+SET_loc[i]), SET_angle[i])

def place_qubit(ax,loc,color):
    ax.add_patch(plt.Circle(loc, q_radius, color = color))
    ax.add_patch(plt.Circle(loc, q_radius, color = 'black', linewidth=1, fill=False))

def construct_unit_cell(ax,r0=np.array([0,0]),xlim=[-150,150],ylim=[-90,90]):

    #ax.vlines(wire_vert_x+r0[0], ylim[0], ylim[1], colors=wire_vert_colours, linewidth=wire_width, alpha=wire_opacity,zorder=1)

    #for i in range(nwire_vert):
    #    place_wire(ax,wire_vert_x[i]+r0[0],wire_vert_colours[i],'vert', ylim)

    # place gridlines
    plot_grid(ax,r0)


    place_cell_SETs(ax,r0)

    # place qubits
    for i in range(n_qubit_types):
        for q_loc in q_locs[i]:
            place_qubit(ax,q_loc+r0, q_colours[i])


    # place wires 
    #ax.hlines(wire_hori_y+r0[1], xlim[0], xlim[1], colors=wire_hori_colours, linewidth=wire_width, alpha=wire_opacity)
    #for j in range(nwire_hori):
    #    place_wire(ax,wire_hori_y[j]+r0[1],wire_hori_colours[j], 'hori', xlim)


def place_all_wires(ax, m,n,wire_colours = None, wire_opacity=default_wire_opacity):
    n_hori_wires = nwire_hori*m
    n_vert_wires = (nwire_vert*n+1)
    if wire_colours==None:
        wire_colours = m*wire_hori_colours + n*wire_vert_colours + wire_vert_colours[0:1]
        print(wire_colours)
    wire_orientations = ['hori']*n_hori_wires + ['vert']*n_vert_wires
    lims = n_hori_wires * [ax.get_xlim()] + n_vert_wires * [ax.get_ylim()]
    wire_positions = []
    for i in range(m): 
        y0 = i*cell_height
        wire_positions += [y0+y for y in wire_hori_y]
    for j in range(n): 
        x0 = j*cell_length
        wire_positions += [x0+x for x in wire_vert_x]
    wire_positions += [n*cell_length + wire_vert_x[0]]

    nwires = len(wire_positions)
    for n in range(nwires):
        place_wire(ax,wire_positions[n], wire_colours[n], wire_orientations[n], lims[n], wire_opacity)

def plot_single_cell(ax=None,wire_colours=None, wire_opacity = default_wire_opacity):
    r0=np.array([0,0])
    if ax==None: ax = plt.subplot()
    ax.set_aspect('equal')
    xlim=[-110*a_lattice,110*a_lattice]
    ylim=[-70*a_lattice,70*a_lattice]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    place_all_wires(ax,1,1, wire_colours=wire_colours,wire_opacity=wire_opacity)
    construct_unit_cell(ax,xlim=xlim, ylim=ylim)

def annotate_cell(ax):
    arrow_color = 'blue'
    anno_fontsize=8
    arrow_width=0.5
    double_arrow(ax,-flag_x+q_radius,flag_y,xh-2*q_radius,0, arrow_width, color=arrow_color)
    ax.annotate('17.7 nm', xy=(-1.80*xh,0.1*yd), rotation=0,fontsize=anno_fontsize)
    double_arrow(ax,q_radius-xh,flag_y,xh-2*q_radius,0, arrow_width, color=arrow_color)
    ax.annotate('17.7 nm', xy=(-0.80*xh,0.1*yd), rotation=0, fontsize=anno_fontsize)

    nx = 0.5; ny = np.sqrt(3)/2
    double_arrow(ax,-data_x+nx*q_radius,data_y-ny*q_radius,xd-2*nx*q_radius,-yd+2*ny*q_radius,arrow_width, color=arrow_color)
    ax.annotate('17.7 nm', xy=(-data_x-0.13*xd,data_y-0.88*yd), rotation=-60,fontsize=anno_fontsize)
    double_arrow(ax,-flag_x-xd+nx*q_radius,yd-ny*q_radius,xd-2*nx*q_radius,-yd+2*ny*q_radius, arrow_width, color=arrow_color)
    ax.annotate('17.7 nm', xy=(-2*xh-xd-0.13*xd,yd-0.88*yd), rotation=-60,fontsize=anno_fontsize)

    ax.annotate('(0,0)', xy=(0,-q_radius*2))
    ax.annotate('(32,0)', xy=(xh,-q_radius*2))
    ax.annotate('(64,0)', xy=(2*xh+q_radius,0))
    ax.annotate('(80,28)', xy=(2.2*xh+xd-6.5*q_radius,1.25*yd))
    ax.annotate('(96,56)', xy=(2*xh+2*xd-6.5*q_radius,2*yd))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

def plot_annotated_cell(filename=None):
    fig,ax = plt.subplots(1,1)
    plot_single_cell(ax)
    annotate_cell(ax)
    if filename is not None:
        fig.savefig(filename)


def number_qubits(ax):
    for i in range(len(q_data_loc)):
        loc = q_data_loc[i] - np.ones(2) * q_radius/2
        num = q_data_num[i]
        ax.annotate(num, xy=loc)
    for i in range(len(q_flag_loc)):
        loc = q_flag_loc[i] - np.ones(2) * q_radius/2
        num = q_flag_num[i]
        ax.annotate(num, xy=loc)
    for i in range(len(q_meas_loc)):
        loc = q_meas_loc[i] - np.ones(2) * q_radius/2
        num = q_meas_num[i]
        ax.annotate(num, xy=loc,color='white')

def numbered_qubits_cell(filename=None):
    fig,ax = plt.subplots(1,1)
    plot_single_cell(ax)
    number_qubits(ax)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    if filename is not None:
        fig.savefig(filename)


def is_populated(i,j):
    return (i+j)%2 == 0

def place_hori_qubits(ax,r0):
    place_qubit(ax,q_meas_loc[0]+r0, q_colours[0])
    place_qubit(ax,q_flag_loc[0]+r0, q_colours[1])
    place_qubit(ax,q_flag_loc[1]+r0, q_colours[1])
    place_qubit(ax,q_coup_loc[0]+r0, q_colours[2])
    place_qubit(ax,q_coup_loc[1]+r0, q_colours[2])
    grid_color='gray'
    grid_zorder=0.25
    grid_opacity=1
    grid_xh = flag_x*np.array([-1,1])
    grid_yh = np.zeros(2)

    x0,y0=r0
    ax.plot(grid_xh+x0,grid_yh+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
    

def place_top_qubits(ax,r0):
    
    place_qubit(ax,q_coup_loc[4]+r0, q_colours[2])
    place_qubit(ax,q_coup_loc[5]+r0, q_colours[2])
    place_qubit(ax,q_data_loc[1]+r0, q_colours[3])
    place_qubit(ax,q_data_loc[3]+r0, q_colours[3])
    place_hori_qubits(ax,r0)
    
    x0,y0=r0
    grid_color='gray'
    grid_zorder=0.25
    grid_opacity=1
    grid_xLL=np.array([-2*xh-2*xd,-flag_x])
    grid_yLL=np.array([-2*yd,0])
    grid_xLR=np.array([2*xh+2*xd,flag_x])
    grid_yLR=np.array([-2*yd,0])

    LL_SET(ax,r0)
    LR_SET(ax,r0)

    ax.plot(grid_xLL+x0,grid_yLL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
    ax.plot(grid_xLR+x0,grid_yLR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)

def place_bottom_qubits(ax,r0):
    
    place_qubit(ax,q_coup_loc[3]+r0, q_colours[2])
    place_qubit(ax,q_coup_loc[2]+r0, q_colours[2])
    place_qubit(ax,q_data_loc[0]+r0, q_colours[3])
    place_qubit(ax,q_data_loc[2]+r0, q_colours[3])
    place_hori_qubits(ax,r0)
    x0,y0=r0
    grid_color='gray'
    grid_zorder=0.25
    grid_opacity=1
    grid_xUL=np.array([-2*xh-2*xd,-flag_x])
    grid_yUL=np.array([2*yd,0])
    grid_xUR=np.array([2*xh+2*xd,flag_x])
    grid_yUR=np.array([2*yd,0])

    UL_SET(ax,r0)
    UR_SET(ax,r0)

    ax.plot(grid_xUL+x0,grid_yUL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
    ax.plot(grid_xUR+x0,grid_yUR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)

def place_left_qubits(ax,r0):
    place_qubit(ax,q_flag_loc[1]+r0, q_colours[1])
    place_qubit(ax,q_coup_loc[2]+r0, q_colours[2])
    place_qubit(ax,q_coup_loc[5]+r0, q_colours[2])

    x0,y0=r0
    grid_color='gray'
    grid_zorder=0.25
    grid_opacity=1
    grid_xUR=np.array([2*xh+2*xd,flag_x])
    grid_yUR=np.array([2*yd,0])
    grid_xLR=np.array([2*xh+2*xd,flag_x])
    grid_yLR=np.array([-2*yd,0])

    LR_SET(ax,r0)
    UR_SET(ax,r0)

    ax.plot(grid_xLR+x0,grid_yLR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
    ax.plot(grid_xUR+x0,grid_yUR+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)


def place_right_qubits(ax,r0):
    place_qubit(ax,q_flag_loc[0]+r0, q_colours[1])
    place_qubit(ax,q_coup_loc[3]+r0, q_colours[2])
    place_qubit(ax,q_coup_loc[4]+r0, q_colours[2])

    x0,y0=r0
    grid_color='gray'
    grid_zorder=0.25
    grid_opacity=1
    grid_xUL=np.array([-2*xh-2*xd,-flag_x])
    grid_yUL=np.array([2*yd,0])
    grid_xLL=np.array([-2*xh-2*xd,-flag_x])
    grid_yLL=np.array([-2*yd,0])

    LL_SET(ax,r0)
    UL_SET(ax,r0)

    ax.plot(grid_xUL+x0,grid_yUL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)
    ax.plot(grid_xLL+x0,grid_yLL+y0, color=grid_color,zorder=grid_zorder, alpha=grid_opacity)

def plot_cell_array(m,n, filename=None):
    
    fig,ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    xlim=[-150*a_lattice,(n-1/2)*cell_length+10*a_lattice+3*xd]
    ylim=[-1.1*yd-90*a_lattice,(m-1/2)*cell_height+10*a_lattice+2*yd]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for i in range(n):
        for j in range(m):
            if is_populated(i,j):
                r0 = np.array([i*cell_length, j*cell_height])
                construct_unit_cell(ax, r0,xlim,ylim)
    

    for i in range(n):
        if is_populated(i,m):
            r0 = np.array([i*cell_length, m*cell_height])
            place_top_qubits(ax,r0)
        if is_populated(i,-1):
            r0 = np.array([i*cell_length, -1*cell_height])
            place_bottom_qubits(ax,r0)
    LL_SET(ax,np.array([m*cell_length, n*cell_height]))
    
    for j in range(m):
        if is_populated(-1,j):
            r0 = np.array([-1*cell_length, j*cell_height])
            place_left_qubits(ax,r0)
        if is_populated(n,j):
            r0 = np.array([n*cell_length, j*cell_height])
            place_right_qubits(ax,r0)

    place_all_wires(ax,m,n)

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_aspect('equal')
    if filename is not None:
        fig.savefig(filename)
        
# green, purple, yellow, red, blue
Phi_L = [-3,-3,0,-2,-1,0]
L_loaded = [q_coup_loc[0]]
Phi_R = [-3,-3,0,-1,-2,0]
R_loaded = [q_coup_loc[1]]
Phi_U = [-2,-1,0,-3,-3,0]
U_loaded = [q_coup_loc[2],q_coup_loc[3]]
Phi_D = [-1,-2,0,-3,-3,0]
D_loaded = [q_coup_loc[4],q_coup_loc[5]]

def CNOT(ax, Phi, loaded):
    plot_single_cell(ax, wire_colours=[V_colours[p] for p in Phi],wire_opacity=1)
    for loc in loaded:
        place_qubit(ax,loc,'red')

import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.figure import figaspect

def generate_CNOTs():
    fig,ax = plt.subplots(1,5)
    CNOT(ax[0],Phi_L,L_loaded)
    CNOT(ax[1], Phi_R, R_loaded)
    CNOT(ax[2], Phi_U, U_loaded)
    CNOT(ax[3], Phi_D, D_loaded)
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-3.5, vmax=0.5)
    ax[0].set_xlabel('x (nm)')
    ax[1].set_xlabel('x (nm)')
    ax[2].set_xlabel('x (nm)')
    ax[3].set_xlabel('x (nm)')
    ax[0].set_ylabel('y (nm)')


    ncolors = 4
    cw = 256//4 #color width
    colors=np.zeros((256,4))
    for n in range(ncolors):
        colors[n*cw:(n+1)*cw,:] = V_colours[n-3]

    cmap = ListedColormap(colors)

    cb1 = mpl.colorbar.ColorbarBase(ax[4], cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Voltage')

    ax[4].set_aspect(1.5)
    w, h = figaspect(10)
    fig.set_size_inches(w/140, h/40)

    ax[4].set_yticks([0, -1, -2, -3], ['$V_0$', '$V_1$', '$V_2$', '$V_3$'])
    plt.yticks()
if __name__ == '__main__':
    #plot_cell_array(4,4, filename="cell_array")
    #generate_CNOTs()
    #plot_annotated_cell(filename="single_cell")
    #numbered_qubits_cell()
    #plot_single_cell()
    plt.show()

