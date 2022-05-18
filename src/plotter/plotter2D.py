"""
This file consist main plotter code for DryVR reachtube output
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np 
from typing import List 

colors = ['red', 'green', 'blue', 'yellow', 'black']

def plot(
    data, 
    x_dim: int = 0, 
    y_dim_list: List[int] = [1], 
    color = 'b', 
    fig = None, 
    x_lim = (float('inf'), -float('inf')), 
    y_lim = (float('inf'), -float('inf'))
):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    for rect in data:
        lb = rect[0]
        ub = rect[1]
        for y_dim in y_dim_list:
            rect_patch = patches.Rectangle((lb[x_dim], lb[y_dim]), ub[x_dim]-lb[x_dim], ub[y_dim]-lb[y_dim], color = color)
            ax.add_patch(rect_patch)
            x_min = min(lb[x_dim], x_min)
            y_min = min(lb[y_dim], y_min)
            x_max = max(ub[x_dim], x_max)
            y_max = max(ub[y_dim], y_max)

    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
    return fig, (x_min, x_max), (y_min, y_max)

def plot_reachtube_tree(root, agent_id, x_dim: int=0, y_dim_list: List[int]=[1], color='b', fig = None):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    queue = [root]
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0,len(trace),2):
            data.append([trace[i], trace[i+1]])
        fig, x_lim, y_lim = plot(data, x_dim, y_dim_list, color, fig, x_lim, y_lim)

        queue += node.child

    return fig

def plot_map(map, color = 'b', fig = None):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                ax.plot([lane_seg.start[0], lane_seg.end[0]],[lane_seg.start[1], lane_seg.end[1]], color) 
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                x = np.cos(phase_array)*lane_seg.radius + lane_seg.center[0]
                y = np.sin(phase_array)*lane_seg.radius + lane_seg.center[1]
                ax.plot(x,y,color)
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    return fig

def plot_simulation_tree(root, agent_id, x_dim: int=0, y_dim_list: List[int]=[1], color='b', fig = None):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    queue = [root]
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        for y_dim in y_dim_list:
            ax.plot(trace[:,x_dim], trace[:,y_dim], color)
            x_min = min(x_min, trace[:,x_dim].min())
            x_max = max(x_max, trace[:,x_dim].max())

            y_min = min(y_min, trace[:,y_dim].min())
            y_max = max(y_max, trace[:,y_dim].max())

        queue += node.child
    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
    
    return fig
