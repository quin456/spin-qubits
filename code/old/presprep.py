

import torch as pt 
import matplotlib as mpl
import numpy as np
if not pt.cuda.is_available():
    mpl.use('Qt5Agg')
from matplotlib import pyplot as plt 
from visualisation import *
from misc_calculations import *


def compare_symmetry(fp=None):
    fig,ax = plt.subplots(1)
    dist=18
    sq_centre = np.array([dist,0])
    diag_centre = np.ceil(np.array([dist/np.sqrt(2), dist/np.sqrt(2)]))
    hex_centre = np.ceil(np.array([dist*np.sqrt(3)/2, dist*1/2]))
    all_sites = get_all_sites([0,dist], [2,dist/np.sqrt(2)-2], padding=dist//5)
    plot_spheres(all_sites, ax=ax, alpha=0.1)
    sites_color='#1DA4BF'
    alpha = 0.5
    plot_9_sites(0, 0, ax=ax, neighbour_color=sites_color)
    plot_9_sites(*sq_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
    plot_9_sites(*diag_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)
    plot_9_sites(*hex_centre, ax=ax, round_func = np.ceil, neighbour_color=sites_color, alpha=alpha)

    linecolor_15='black'
    linecolor_25 = 'darkred'
    linewidth = 1.5
    fontsize = 13
    ax.plot([1.5, sq_centre[0]-1.5], [0,sq_centre[1]], color=linecolor_15, linewidth=linewidth)
    ax.plot([np.sqrt(2), diag_centre[0]-np.sqrt(2)], [np.sqrt(2),diag_centre[1]-np.sqrt(2)], color=linecolor_15, linewidth=linewidth)
    ax.plot([1.5, hex_centre[0]-1.5], [np.sqrt(3)/2,hex_centre[1]-np.sqrt(3)/2], color=linecolor_25, linewidth=linewidth)

    n=100
    ang_adjust = np.arctan(hex_centre[1]/hex_centre[0])
    theta1 = np.linspace(0,ang_adjust, n)
    theta2 = np.linspace(0,np.pi/4, n)
    r1 = 12
    r2 = 7
    ax.plot(r1*np.cos(theta1), r1*np.sin(theta1), color=linecolor_25)
    ax.plot(r2*np.cos(theta2), r2*np.sin(theta2), color=linecolor_15)

    ax_length=3
    ax_origin = np.array([-1,7])
    ax.arrow(*ax_origin, 0, ax_length, shape='full', lw=1, length_includes_head=True, head_width=0.2)
    ax.arrow(*ax_origin, ax_length, 0, shape='full', lw=1, length_includes_head=True, head_width=0.2)
    ax.annotate('[1,-1,0]', ax_origin + np.array([-0,-1.3]))
    ax.annotate('[1,1,0]', ax_origin + np.array([-1.6,-0.5]), rotation=90)

    ax.annotate('0°', sq_centre + [-3.5,0.45], fontsize=fontsize)
    ax.annotate('45°', r2*diag_centre/dist + [-1.05,0.55], fontsize=fontsize)
    ax.annotate('30°', r1*hex_centre/dist + [-0.95,0.6], fontsize=fontsize, color=linecolor_25)
    fig.set_size_inches(1.4*fig_width_single, 1.2*fig_height_single)
    if fp is not None:
        plt.savefig(fp)


compare_symmetry()
plt.show()