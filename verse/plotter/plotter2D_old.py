"""
This file consist main plotter code for DryVR reachtube output
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np 
from typing import List 
from PIL import Image, ImageDraw
import io

colors = ['red', 'green', 'blue', 'yellow', 'black']

def plot(
    data, 
    x_dim: int = 0, 
    y_dim_list: List[int] = [1], 
    color = 'b', 
    fig = None, 
    x_lim = None, 
    y_lim = None
):
    if fig is None:
        fig = plt.figure()
    
    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()
    
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

def plot_reachtube_tree(root, agent_id, x_dim: int=0, y_dim_list: List[int]=[1], color='b', fig = None, x_lim = None, y_lim = None):
    if fig is None:
        fig = plt.figure()
    
    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()

    queue = [root]
    idx = 0
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0,len(trace),2):
            data.append([trace[i], trace[i+1]])
        if False:
            fig, x_lim, y_lim = plot(data, x_dim, y_dim_list, 'y', fig, x_lim, y_lim)
        else:
            fig, x_lim, y_lim = plot(data, x_dim, y_dim_list, color, fig, x_lim, y_lim)
            
        if node.assert_hits:
            fig, x_lim, y_lim = plot([data[-1]], x_dim, y_dim_list, 'k', fig, x_lim, y_lim)
        queue += node.child
        idx += 1

    return fig

def plot_map(map, color = 'b', fig = None, x_lim = None,y_lim = None):
    if fig is None:
        fig = plt.figure()
    
    ax = fig.gca()
    
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                start1 = lane_seg.start + lane_seg.width/2 * lane_seg.direction_lateral
                end1 = lane_seg.end + lane_seg.width/2 * lane_seg.direction_lateral
                ax.plot([start1[0], end1[0]],[start1[1], end1[1]], color) 
                start2 = lane_seg.start - lane_seg.width/2 * lane_seg.direction_lateral
                end2 = lane_seg.end - lane_seg.width/2 * lane_seg.direction_lateral
                ax.plot([start2[0], end2[0]],[start2[1], end2[1]], color) 
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                r1 = lane_seg.radius - lane_seg.width/2
                x = np.cos(phase_array)*r1 + lane_seg.center[0]
                y = np.sin(phase_array)*r1 + lane_seg.center[1]
                ax.plot(x,y,color)

                r2 = lane_seg.radius + lane_seg.width/2
                x = np.cos(phase_array)*r2 + lane_seg.center[0]
                y = np.sin(phase_array)*r2 + lane_seg.center[1]
                ax.plot(x,y,color)
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    return fig

def plot_simulation_tree(root, agent_id, x_dim: int=0, y_dim_list: List[int]=[1], color='b', fig = None, x_lim = None, y_lim = None):
    if fig is None:
        fig = plt.figure()
    
    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()
    
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    
    queue = [root]
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

def generate_simulation_anime(root, map, fig = None):
    if fig is None:
        fig = plt.figure()
    fig = plot_map(map, 'g', fig)
    timed_point_dict = {}
    stack = [root]
    ax = fig.gca()
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = ax.get_ylim()
    while stack != []:
        node = stack.pop()
        traces = node.trace
        for agent_id in traces:
            trace = traces[agent_id]
            color = 'b'
            if agent_id == 'car2':
                color = 'r'
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][1])
                x_max = max(x_max, trace[i][1])
                y_min = min(y_min, trace[i][2])
                y_max = max(y_max, trace[i][2])
                if round(trace[i][0],5) not in timed_point_dict:
                    timed_point_dict[round(trace[i][0],5)] = [(trace[i][1:],color)]
                else:
                    timed_point_dict[round(trace[i][0],5)].append((trace[i][1:],color))
        stack += node.child

    frames = []
    for time_point in timed_point_dict:
        point_list = timed_point_dict[time_point]
        plt.xlim((x_min-2, x_max+2))
        plt.ylim((y_min-2, y_max+2))
        plot_map(map,color = 'g', fig = fig)
        for data in point_list:
            point = data[0]
            color = data[1]
            ax = plt.gca()
            ax.plot([point[0]], [point[1]], markerfacecolor = color, markeredgecolor = color, marker = '.', markersize = 20)
            x_tail = point[0]
            y_tail = point[1]
            dx = np.cos(point[2])*point[3]
            dy = np.sin(point[2])*point[3]
            ax.arrow(x_tail, y_tail, dx, dy, head_width = 1, head_length = 0.5)
        plt.pause(0.05)
        plt.clf()
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format = 'png')
    #     im = Image.open(img_buf)
    #     frames.append(im)
    #     plt.clf()
    # frame_one = frames[0]
    # frame_one.save(fn, format = "GIF", append_images = frames, save_all = True, duration = 100, loop = 0)