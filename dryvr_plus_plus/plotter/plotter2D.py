"""
This file consist main plotter code for DryVR reachtube output
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import plotly.graph_objects as go
from typing import List
from PIL import Image, ImageDraw
import io

colors = ['red', 'green', 'blue', 'yellow', 'black']


def plot(
    data,
    x_dim: int = 0,
    y_dim_list: List[int] = [1],
    color='b',
    fig=None,
    x_lim=None,
    y_lim=None
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
            rect_patch = patches.Rectangle(
                (lb[x_dim], lb[y_dim]), ub[x_dim]-lb[x_dim], ub[y_dim]-lb[y_dim], color=color)
            ax.add_patch(rect_patch)
            x_min = min(lb[x_dim], x_min)
            y_min = min(lb[y_dim], y_min)
            x_max = max(ub[x_dim], x_max)
            y_max = max(ub[y_dim], y_max)

    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
    return fig, (x_min, x_max), (y_min, y_max)


def plot_reachtube_tree(root, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='b', fig=None, x_lim=None, y_lim=None):
    if fig is None:
        fig = plt.figure()

    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()

    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0, len(trace), 2):
            data.append([trace[i], trace[i+1]])
        fig, x_lim, y_lim = plot(
            data, x_dim, y_dim_list, color, fig, x_lim, y_lim)

        queue += node.child

    return fig

def plot_reachtube_tree_branch(root, agent_id, x_dim: int=0, y_dim_list: List[int]=[1], color='b', fig = None, x_lim = None, y_lim = None):
    if fig is None:
        fig = plt.figure()
    
    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()

    stack = [root]
    while stack != []:
        node = stack.pop()
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0,len(trace),2):
            data.append([trace[i], trace[i+1]])
        fig, x_lim, y_lim = plot(data, x_dim, y_dim_list, color, fig, x_lim, y_lim)

        if node.child:
            stack += [node.child[0]]

    return fig

def plot_map(map, color = 'b', fig = None, x_lim = None,y_lim = None):
    if fig is None:
        fig = plt.figure()

    ax = fig.gca()
    if x_lim is None:
        x_lim = ax.get_xlim()
    if y_lim is None:
        y_lim = ax.get_ylim()

    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                start1 = lane_seg.start + lane_seg.width/2 * lane_seg.direction_lateral
                end1 = lane_seg.end + lane_seg.width/2 * lane_seg.direction_lateral
                ax.plot([start1[0], end1[0]], [start1[1], end1[1]], color)
                start2 = lane_seg.start - lane_seg.width/2 * lane_seg.direction_lateral
                end2 = lane_seg.end - lane_seg.width/2 * lane_seg.direction_lateral
                ax.plot([start2[0], end2[0]], [start2[1], end2[1]], color)
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(
                    start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                r1 = lane_seg.radius - lane_seg.width/2
                x = np.cos(phase_array)*r1 + lane_seg.center[0]
                y = np.sin(phase_array)*r1 + lane_seg.center[1]
                ax.plot(x, y, color)

                r2 = lane_seg.radius + lane_seg.width/2
                x = np.cos(phase_array)*r2 + lane_seg.center[0]
                y = np.sin(phase_array)*r2 + lane_seg.center[1]
                ax.plot(x, y, color)
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    return fig


def plot_simulation_tree(root, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='b', fig=None, x_lim=None, y_lim=None):
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
            ax.plot(trace[:, x_dim], trace[:, y_dim], color)
            x_min = min(x_min, trace[:, x_dim].min())
            x_max = max(x_max, trace[:, x_dim].max())

            y_min = min(y_min, trace[:, y_dim].min())
            y_max = max(y_max, trace[:, y_dim].max())

        queue += node.child
    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])

    return fig


def generate_simulation_anime(root, map, fig=None):
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
                if round(trace[i][0], 5) not in timed_point_dict:
                    timed_point_dict[round(trace[i][0], 5)] = [
                        (trace[i][1:], color)]
                else:
                    timed_point_dict[round(trace[i][0], 5)].append(
                        (trace[i][1:], color))
        stack += node.child

    frames = []
    for time_point in timed_point_dict:
        point_list = timed_point_dict[time_point]
        plt.xlim((x_min-2, x_max+2))
        plt.ylim((y_min-2, y_max+2))
        plot_map(map, color='g', fig=fig)
        for data in point_list:
            point = data[0]
            color = data[1]
            ax = plt.gca()
            ax.plot([point[0]], [point[1]], markerfacecolor=color,
                    markeredgecolor=color, marker='.', markersize=20)
            x_tail = point[0]
            y_tail = point[1]
            dx = np.cos(point[2])*point[3]
            dy = np.sin(point[2])*point[3]
            ax.arrow(x_tail, y_tail, dx, dy, head_width=1, head_length=0.5)
        plt.pause(0.05)
        plt.clf()
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format = 'png')
    #     im = Image.open(img_buf)
    #     frames.append(im)
    #     plt.clf()
    # frame_one = frames[0]
    # frame_one.save(fn, format = "GIF", append_images = frames, save_all = True, duration = 100, loop = 0)


def plotly_map(map, color='b', fig=None, x_lim=None, y_lim=None):
    if fig is None:
        fig = go.figure()

    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                start1 = lane_seg.start + lane_seg.width/2 * lane_seg.direction_lateral
                end1 = lane_seg.end + lane_seg.width/2 * lane_seg.direction_lateral
                # fig.add_trace(go.Scatter(x=[start1[0], end1[0]], y=[start1[1], end1[1]],
                #                          mode='lines',
                #                          line_color='black',
                #                          showlegend=False,
                #                          # text=theta,
                #                          name='lines'))
                start2 = lane_seg.start - lane_seg.width/2 * lane_seg.direction_lateral
                end2 = lane_seg.end - lane_seg.width/2 * lane_seg.direction_lateral
                # fig.add_trace(go.Scatter(x=[start2[0], end2[0]], y=[start2[1], end2[1]],
                #                          mode='lines',
                #                          line_color='black',
                #                          showlegend=False,
                #                          # text=theta,
                #                          name='lines'))
                fig.add_trace(go.Scatter(x=[start1[0], end1[0], end2[0], start2[0]], y=[start1[1], end1[1], end2[1], start2[1]],
                                         mode='lines',
                                         line_color='black',
                                         #  fill='toself',
                                         #  fillcolor='rgba(255,255,255,0)',
                                         #  line_color='rgba(0,0,0,0)',
                                         showlegend=False,
                                         # text=theta,
                                         name='lines'))
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(
                    start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                r1 = lane_seg.radius - lane_seg.width/2
                x = np.cos(phase_array)*r1 + lane_seg.center[0]
                y = np.sin(phase_array)*r1 + lane_seg.center[1]
                fig.add_trace(go.Scatter(x=x, y=y,
                                         mode='lines',
                                         line_color='black',
                                         showlegend=False,
                                         # text=theta,
                                         name='lines'))

                r2 = lane_seg.radius + lane_seg.width/2
                x = np.cos(phase_array)*r2 + lane_seg.center[0]
                y = np.sin(phase_array)*r2 + lane_seg.center[1]
                fig.add_trace(go.Scatter(x=x, y=y,
                                         mode='lines',
                                         line_color='black',
                                         showlegend=False,
                                         # text=theta,
                                         name='lines'))
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    return fig


def plotly_simulation_tree(root, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='b', fig=None, x_lim=None, y_lim=None):
    if fig is None:
        fig = go.Figure()
    i = 0
    fg_color = ['rgb(31,119,180)', 'rgb(255,127,14)', 'rgb(44,160,44)', 'rgb(214,39,40)', 'rgb(148,103,189)',
                'rgb(140,86,75)', 'rgb(227,119,194)', 'rgb(127,127,127)', 'rgb(188,189,34)', 'rgb(23,190,207)']
    bg_color = ['rgba(31,119,180,0.2)', 'rgba(255,127,14,0.2)', 'rgba(44,160,44,0.2)', 'rgba(214,39,40,0.2)', 'rgba(148,103,189,0.2)',
                'rgba(140,86,75,0.2)', 'rgba(227,119,194,0.2)', 'rgba(127,127,127,0.2)', 'rgba(188,189,34,0.2)', 'rgba(23,190,207,0.2)']
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print(node.mode)
        # [[time,x,y,theta,v]...]
        trace = np.array(traces[agent_id])
        # print(trace)
        for y_dim in y_dim_list:
            trace_y = trace[:, y_dim].tolist()
            trace_x = trace[:, x_dim].tolist()
            theta = [i/pi*180 for i in trace[:, x_dim+2]]
            trace_x_rev = trace_x[::-1]
            # print(trace_x)
            trace_upper = [i+1 for i in trace_y]
            trace_lower = [i-1 for i in trace_y]
            trace_lower = trace_lower[::-1]
            # print(trace_upper)
            # print(trace[:, y_dim])
            fig.add_trace(go.Scatter(x=trace_x+trace_x_rev, y=trace_upper+trace_lower,
                                     fill='toself',
                                     fillcolor=bg_color[i % 10],
                                     line_color='rgba(255,255,255,0)',
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
                                     mode='lines',
                                     line_color=fg_color[i % 10],
                                     text=theta,
                                     name='lines'))
            i += 1
        queue += node.child
    fig.update_traces(mode='lines')
    return fig


def plotly_simulation_anime(root, map=None, fig=None):
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    # fig = plot_map(map, 'g', fig)
    timed_point_dict = {}
    stack = [root]
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    while stack != []:
        node = stack.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][1])
                x_max = max(x_max, trace[i][1])
                y_min = min(y_min, trace[i][2])
                y_max = max(y_max, trace[i][2])
                if round(trace[i][0], 2) not in timed_point_dict:
                    timed_point_dict[round(trace[i][0], 2)] = [
                        trace[i][1:].tolist()]
                else:
                    init = False
                    for record in timed_point_dict[round(trace[i][0], 2)]:
                        if record == trace[i][1:].tolist():
                            init = True
                            break
                    if init == False:
                        timed_point_dict[round(trace[i][0], 2)].append(
                            trace[i][1:].tolist())
            time = round(trace[i][0], 2)
        stack += node.child
    # fill in most of layout
    # print(time)
    duration = int(600/time)
    fig_dict["layout"]["xaxis"] = {
        "range": [(x_min-10), (x_max+10)],
        "title": "x position"}
    fig_dict["layout"]["yaxis"] = {
        "range": [(y_min-2), (y_max+2)],
        "title": "y position"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": duration, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": duration,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    # make data
    point_list = timed_point_dict[0]
    # print(point_list)
    data_dict = {
        "x": [data[0] for data in point_list],
        "y": [data[1] for data in point_list],
        "mode": "markers + text",
        "text": [(round(data[3], 2), round(data[2]/pi*180, 2)) for data in point_list],
        "textposition": "bottom center",
        # "marker": {
        #     "sizemode": "area",
        #     "sizeref": 200000,
        #     "size": 2
        # },
        "name": "Current Position"
    }
    fig_dict["data"].append(data_dict)

    # make frames
    for time_point in timed_point_dict:
        frame = {"data": [], "layout": {
            "annotations": []}, "name": str(time_point)}
        # print(timed_point_dict[time_point])
        point_list = timed_point_dict[time_point]
        # point_list = list(OrderedDict.fromkeys(timed_point_dict[time_point]))
        trace_x = [data[0] for data in point_list]
        trace_y = [data[1] for data in point_list]
        trace_theta = [data[2] for data in point_list]
        trace_v = [data[3] for data in point_list]
        data_dict = {
            "x": trace_x,
            "y": trace_y,
            "mode": "markers + text",
            "text": [(round(trace_theta[i]/pi*180, 2), round(trace_v[i], 2)) for i in range(len(trace_theta))],
            "textposition": "bottom center",
            # "marker": {
            #     "sizemode": "area",
            #     "sizeref": 200000,
            #     "size": 2
            # },
            "name": "current position"
        }
        frame["data"].append(data_dict)
        for i in range(len(trace_x)):
            ax = np.cos(trace_theta[i])*trace_v[i]
            ay = np.sin(trace_theta[i])*trace_v[i]
            # print(trace_x[i]+ax, trace_y[i]+ay)
            annotations_dict = {"x": trace_x[i]+ax+0.1, "y": trace_y[i]+ay,
                                # "xshift": ax, "yshift": ay,
                                "ax": trace_x[i], "ay": trace_y[i],
                                "arrowwidth": 2,
                                # "arrowside": 'end',
                                "showarrow": True,
                                # "arrowsize": 1,
                                "xref": 'x', "yref": 'y',
                                "axref": 'x', "ayref": 'y',
                                # "text": "erver",
                                "arrowhead": 1,
                                "arrowcolor": "black"}
            frame["layout"]["annotations"].append(annotations_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [time_point],
            {"frame": {"duration": duration, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": duration}}
        ],
            "label": time_point,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        # print(len(frame["layout"]["annotations"]))

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = plotly_map(map, 'g', fig)
    i = 0
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print(node.mode)
        # [[time,x,y,theta,v]...]
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            # print(trace)
            trace_y = trace[:, 2].tolist()
            trace_x = trace[:, 1].tolist()
            # theta = [i/pi*180 for i in trace[:, 3]]
            color = 'green'
            if agent_id == 'car1':
                color = 'red'
            fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
                                     mode='lines',
                                     line_color=color,
                                     text=[(round(trace[i, 3]/pi*180, 2), round(trace[i, 4], 2))
                                           for i in range(len(trace_y))],
                                     showlegend=False)
                          #  name='lines')
                          )
            i += 1
        queue += node.child
    # fig.update_traces(mode='lines')

    return fig
