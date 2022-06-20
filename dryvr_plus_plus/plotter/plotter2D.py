"""
This file consist main plotter code for DryVR reachtube output
"""

from __future__ import annotations
from audioop import reverse
# from curses import start_color
from re import A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import plotly.graph_objects as go
from typing import List
from PIL import Image, ImageDraw
import io
import copy
import operator
from collections import OrderedDict

from torch import layout
from dryvr_plus_plus.scene_verifier.analysis.analysis_tree_node import AnalysisTreeNode

colors = ['red', 'green', 'blue', 'yellow', 'black']


def plotly_plot(data,
                x_dim: int = 0,
                y_dim_list: List[int] = [1],
                color='blue',
                fig=None,
                x_lim=None,
                y_lim=None
                ):
    if fig is None:
        fig = plt.figure()

    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    for rect in data:
        lb = rect[0]
        ub = rect[1]
        for y_dim in y_dim_list:
            fig.add_shape(type="rect",
                          x0=lb[x_dim], y0=lb[y_dim], x1=ub[x_dim], y1=ub[y_dim],
                          line=dict(color=color),
                          fillcolor=color
                          )
            # rect_patch = patches.Rectangle(
            #     (lb[x_dim], lb[y_dim]), ub[x_dim]-lb[x_dim], ub[y_dim]-lb[y_dim], color=color)
            # ax.add_patch(rect_patch)
            x_min = min(lb[x_dim], x_min)
            y_min = min(lb[y_dim], y_min)
            x_max = max(ub[x_dim], x_max)
            y_max = max(ub[y_dim], y_max)
    fig.update_shapes(dict(xref='x', yref='y'))
    # ax.set_xlim([x_min-1, x_max+1])
    # ax.set_ylim([y_min-1, y_max+1])
    # fig.update_xaxes(range=[x_min-1, x_max+1], showgrid=False)
    # fig.update_yaxes(range=[y_min-1, y_max+1])
    return fig, (x_min, x_max), (y_min, y_max)


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


def generate_reachtube_anime(root, map=None, fig=None):
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
    print("reachtude")
    end_time = 0
    while stack != []:
        node = stack.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if trace[0][0] > 0:
                trace = trace[4:]
            # print(trace)
            end_time = trace[-1][0]
            for i in range(0, len(trace), 2):
                x_min = min(x_min, trace[i][1])
                x_max = max(x_max, trace[i][1])
                y_min = min(y_min, trace[i][2])
                y_max = max(y_max, trace[i][2])
                # if round(trace[i][0], 2) not in timed_point_dict:
                #     timed_point_dict[round(trace[i][0], 2)] = [
                #         trace[i][1:].tolist()]
                # else:
                #     init = False
                #     for record in timed_point_dict[round(trace[i][0], 2)]:
                #         if record == trace[i][1:].tolist():
                #             init = True
                #             break
                #     if init == False:
                #         timed_point_dict[round(trace[i][0], 2)].append(
                #             trace[i][1:].tolist())
                time_point = round(trace[i][0], 2)
                rect = [trace[i][1:].tolist(), trace[i+1][1:].tolist()]
                if time_point not in timed_point_dict:
                    timed_point_dict[time_point] = {agent_id: [rect]}
                else:
                    if agent_id in timed_point_dict[time_point].keys():
                        timed_point_dict[time_point][agent_id].append(rect)
                    else:
                        timed_point_dict[time_point][agent_id] = [rect]

        stack += node.child
    # fill in most of layout
    # print(end_time)
    duration = int(100/end_time)
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
        # "method": "update",
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    # make data
    agent_dict = timed_point_dict[0]  # {agent1:[rect1,..], ...}
    x_list = []
    y_list = []
    text_list = []
    for agent_id, rect_list in agent_dict.items():
        for rect in rect_list:
            # trace = list(data.values())[0]
            print(rect)
            x_list.append((rect[0][0]+rect[1][0])/2)
            y_list.append((rect[0][1]+rect[1][1])/2)
            text_list.append(
                ('{:.2f}'.format((rect[0][2]+rect[1][2])/pi*90), '{:.3f}'.format(rect[0][3]+rect[1][3])))
    # data_dict = {
    #     "x": x_list,
    #     "y": y_list,
    #     "mode": "markers + text",
    #     "text": text_list,
    #     "textposition": "bottom center",
    #     # "marker": {
    #     #     "sizemode": "area",
    #     #     "sizeref": 200000,
    #     #     "size": 2
    #     # },
    #     "name": "Current Position"
    # }
    # fig_dict["data"].append(data_dict)

    # make frames
    for time_point in timed_point_dict:
        frame = {"data": [], "layout": {
            "annotations": [], "shapes": []}, "name": str(time_point)}
        agent_dict = timed_point_dict[time_point]
        trace_x = []
        trace_y = []
        trace_theta = []
        trace_v = []
        for agent_id, rect_list in agent_dict.items():
            for rect in rect_list:
                trace_x.append((rect[0][0]+rect[1][0])/2)
                trace_y.append((rect[0][1]+rect[1][1])/2)
                trace_theta.append((rect[0][2]+rect[1][2])/2)
                trace_v.append((rect[0][3]+rect[1][3])/2)
                shape_dict = {
                    "type": 'rect',
                    "x0": rect[0][0],
                    "y0": rect[0][1],
                    "x1": rect[1][0],
                    "y1": rect[1][1],
                    "fillcolor": 'rgba(255,255,255,0.5)',
                    "line": dict(color='rgba(255,255,255,0)'),

                }
                frame["layout"]["shapes"].append(shape_dict)
        # data_dict = {
        #     "x": trace_x,
        #     "y": trace_y,
        #     "mode": "markers + text",
        #     "text": [('{:.2f}'.format(trace_theta[i]/pi*180), '{:.3f}'.format(trace_v[i])) for i in range(len(trace_theta))],
        #     "textposition": "bottom center",
        #     # "marker": {
        #     #     "sizemode": "area",
        #     #     "sizeref": 200000,
        #     #     "size": 2
        #     # },
        #     "name": "current position"
        # }
        # frame["data"].append(data_dict)
        # print(trace_x)
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
    # fig = plotly_map(map, 'g', fig)
    i = 1
    for agent_id in traces:
        fig = plotly_reachtube_tree_v2(root, agent_id, 1, [2], i, fig)
        i += 2

    return fig


def plotly_reachtube_tree(root, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='blue', fig=None, x_lim=None, y_lim=None):
    if fig is None:
        fig = go.Figure()

    # ax = fig.gca()
    # if x_lim is None:
    #     x_lim = ax.get_xlim()
    # if y_lim is None:
    #     y_lim = ax.get_ylim()

    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        # print(trace)
        data = []
        for i in range(0, len(trace)-1, 2):
            data.append([trace[i], trace[i+1]])
        fig, x_lim, y_lim = plotly_plot(
            data, x_dim, y_dim_list, color, fig, x_lim, y_lim)
        # print(data)
        queue += node.child

    return fig


def plotly_reachtube_tree_v2(root, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color=0, fig=None, x_lim=None, y_lim=None):
    if fig is None:
        fig = go.Figure()

    # ax = fig.gca()
    # if x_lim is None:
    #     x_lim = ax.get_xlim()
    # if y_lim is None:
    #     y_lim = ax.get_ylim()
    bg_color = ['rgba(31,119,180,1)', 'rgba(255,127,14,0.2)', 'rgba(44,160,44,0.2)', 'rgba(214,39,40,0.2)', 'rgba(148,103,189,0.2)',
                'rgba(140,86,75,0.2)', 'rgba(227,119,194,0.2)', 'rgba(127,127,127,0.2)', 'rgba(188,189,34,0.2)', 'rgba(23,190,207,0.2)']
    queue = [root]
    show_legend = False
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        # print(trace[0], trace[1], trace[-2], trace[-1])
        max_id = len(trace)-1
        # trace_x = np.zeros(max_id+1)
        # trace_x_2 = np.zeros(max_id+1)
        # trace_y = np.zeros(max_id+1)
        # trace_y_2 = np.zeros(max_id+1)
        # for y_dim in y_dim_list:
        #     for i in range(0, max_id, 2):
        #         id = int(i/2)
        #         trace_x[id] = trace[i+1][x_dim]
        #         trace_x[max_id-id] = trace[i][x_dim]
        #         trace_x_2[id] = trace[i][x_dim]
        #         trace_x_2[max_id-id] = trace[i+1][x_dim]
        #         trace_y[id] = trace[i][y_dim]
        #         trace_y[max_id-id] = trace[i+1][y_dim]
        # fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
        #                          fill='toself',
        #                          fillcolor='blue',
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=trace_x_2, y=trace_y,
        #                          fill='toself',
        #                          fillcolor='red',
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
        #                          mode='lines',
        #                          line_color="black",
        #                          text=[range(0, max_id+1)],
        #                          name='lines',
        #                          showlegend=False))
        trace_x_odd = np.array([trace[i][1] for i in range(0, max_id, 2)])
        trace_x_even = np.array([trace[i][1] for i in range(1, max_id+1, 2)])
        trace_y_odd = np.array([trace[i][2] for i in range(0, max_id, 2)])
        trace_y_even = np.array([trace[i][2] for i in range(1, max_id+1, 2)])
        fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend
                                 ))
        fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend))
        fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend
                                 ))
        fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist(), y=trace_y_odd.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist(), y=trace_y_odd.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist(), y=trace_y_even.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid",shape="spline"),
        #                          line_color=bg_color[0],
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist(), y=trace_y_even.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True))
        # fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
        #                          mode='markers',
        #                          #  fill='toself',
        #                          #  line=dict(dash="dot"),
        #                          line_color="black",
        #                          text=[range(0, max_id+1)],
        #                          name='lines',
        #                          showlegend=False))
        queue += node.child
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        # print(trace[0], trace[1], trace[-2], trace[-1])
        max_id = len(trace)-1
        # trace_x = np.zeros(max_id+1)
        # trace_x_2 = np.zeros(max_id+1)
        # trace_y = np.zeros(max_id+1)
        # trace_y_2 = np.zeros(max_id+1)
        # for y_dim in y_dim_list:
        #     for i in range(0, max_id, 2):
        #         id = int(i/2)
        #         trace_x[id] = trace[i+1][x_dim]
        #         trace_x[max_id-id] = trace[i][x_dim]
        #         trace_x_2[id] = trace[i][x_dim]
        #         trace_x_2[max_id-id] = trace[i+1][x_dim]
        #         trace_y[id] = trace[i][y_dim]
        #         trace_y[max_id-id] = trace[i+1][y_dim]
        # fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
        #                          fill='toself',
        #                          fillcolor='blue',
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=trace_x_2, y=trace_y,
        #                          fill='toself',
        #                          fillcolor='red',
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
        #                          mode='lines',
        #                          line_color="black",
        #                          text=[range(0, max_id+1)],
        #                          name='lines',
        #                          showlegend=False))
        # trace_x_odd = np.array([trace[i][1] for i in range(0, max_id, 2)])
        # trace_x_even = np.array([trace[i][1] for i in range(1, max_id+1, 2)])
        # trace_y_odd = np.array([trace[i][2] for i in range(0, max_id, 2)])
        # trace_y_even = np.array([trace[i][2] for i in range(1, max_id+1, 2)])
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=bg_color[0],
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=bg_color[0],
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=True))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=bg_color[0],
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=bg_color[0],
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=True))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist(), y=trace_y_odd.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist(), y=trace_y_odd.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist(), y=trace_y_even.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid",shape="spline"),
        #                          line_color=bg_color[0],
        #                          showlegend=True
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist(), y=trace_y_even.tolist(), mode='lines',
        #                          #  fill='toself',
        #                          #  fillcolor=bg_color[0],
        #                          #  line=dict(width=1, dash="solid"),
        #                          line_color=bg_color[0],
        #                          showlegend=True))
        fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
                                 mode='markers',
                                 #  fill='toself',
                                 #  line=dict(dash="dot"),
                                 line_color="black",
                                 marker={
            "sizemode": "area",
            "sizeref": 200000,
            "size": 2
        },
            text=[range(0, max_id+1)],
            name='lines',
            showlegend=False))
        queue += node.child
    # fig.update_traces(line_dash="dash")
    return fig


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
        # print(trace)
        data = []
        for i in range(0, len(trace)-1, 2):
            data.append([trace[i], trace[i+1]])
        fig, x_lim, y_lim = plot(
            data, x_dim, y_dim_list, color, fig, x_lim, y_lim)
        # print(data)
        queue += node.child

    return fig


def plotly_map(map, color='b', fig: go.Figure() = None, x_lim=None, y_lim=None):
    if fig is None:
        fig = go.Figure()
    all_x = []
    all_y = []
    all_v = []
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
                # fig = go.Figure().add_heatmap(x=)
                seg_x, seg_y, seg_v = lane_seg.get_all_speed()
                all_x += seg_x
                all_y += seg_y
                all_v += seg_v
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
    start_color = [0, 0, 255, 0.2]
    end_color = [255, 0, 0, 0.2]
    curr_color = copy.deepcopy(start_color)
    max_speed = max(all_v)
    min_speed = min(all_v)

    for i in range(len(all_v)):
        # print(all_x[i])
        # print(all_y[i])
        # print(all_v[i])
        curr_color = copy.deepcopy(start_color)
        for j in range(len(curr_color)-1):
            curr_color[j] += (all_v[i]-min_speed)/(max_speed -
                                                   min_speed)*(end_color[j]-start_color[j])
        fig.add_trace(go.Scatter(x=all_x[i], y=all_y[i],
                                 mode='lines',
                                 line_color='rgba(0,0,0,0)',
                                 fill='toself',
                                 fillcolor='rgba'+str(tuple(curr_color)),
                                 #  marker=dict(
                                 #     symbol='square',
                                 #     size=16,
                                 #     cmax=max_speed,
                                 #     cmin=min_speed,
                                 #     # color=all_v[i],
                                 #     colorbar=dict(
                                 #         title="Colorbar"
                                 #     ),
                                 #     colorscale=[
                                 #         [0, 'rgba'+str(tuple(start_color))], [1, 'rgba'+str(tuple(end_color))]]
                                 # ),
                                 showlegend=False,
                                 ))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='markers',
                             # fill='toself',
                             # fillcolor='rgba'+str(tuple(curr_color)),
                             marker=dict(
                                 symbol='square',
                                 size=16,
                                 cmax=max_speed,
                                 cmin=min_speed,
                                 color='rgba(0,0,0,0)',
                                 colorbar=dict(
                                        title="Speed Limit"
                                 ),
                                 colorscale=[
                                     [0, 'rgba'+str(tuple(start_color))], [1, 'rgba'+str(tuple(end_color))]]
    ),
        showlegend=False,
    ))
    # fig.update_coloraxes(colorbar=dict(title="Colorbar"), colorscale=[
    #                      [0, 'rgba'+str(tuple(start_color))], [1, 'rgba'+str(tuple(end_color))]])
    return fig


def plot_map(map, color='b', fig=None, x_lim=None, y_lim=None):
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


def plotly_simulation_tree(root: AnalysisTreeNode, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='b', fig=None, x_lim=None, y_lim=None):
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
        print(node.mode)
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


def plot_simulation_tree(root: AnalysisTreeNode, agent_id, x_dim: int = 0, y_dim_list: List[int] = [1], color='b', fig=None, x_lim=None, y_lim=None):
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
        print(node.mode)
        # [[time,x,y,theta,v]...]
        trace = np.array(traces[agent_id])
        # print(trace)
        for y_dim in y_dim_list:
            ax.plot(trace[:, x_dim], trace[:, y_dim], color)
            x_min = min(x_min, trace[:, x_dim].min())
            x_max = max(x_max, trace[:, x_dim].max())

            y_min = min(y_min, trace[:, y_dim].min())
            y_max = max(y_max, trace[:, y_dim].max())
        queue += node.child
    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
    # plt.show()
    # generate_simulation_anime(root, None, fig)
    return fig


def generate_simulation_anime(root, map=None, fig=None):
    if fig is None:
        fig = plt.figure()
    # fig = plot_map(map, 'g', fig)
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
        # plot_map(map, color='g', fig=fig)
        for data in point_list:
            point = data
            color = data[1]
            ax = plt.gca()
            ax.plot([point[0]], [point[1]], markerfacecolor=color,
                    markeredgecolor=color, marker='.', markersize=20)
            x_tail = point[0]
            y_tail = point[1]
            dx = np.cos(point[2])*point[3]
            dy = np.sin(point[2])*point[3]
            ax.arrow(x_tail, y_tail, dx, dy, head_width=1, head_length=0.5)
        plt.pause(0.005)
        plt.clf()
    return fig
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format = 'png')
    #     im = Image.open(img_buf)
    #     frames.append(im)
    #     plt.clf()
    # frame_one = frames[0]
    # frame_one.save(fn, format = "GIF", append_images = frames, save_all = True, duration = 100, loop = 0)


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
    print("plot")
    # print(root.mode)
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    # segment_start = set()
    # previous_mode = {}
    # for agent_id in root.mode:
    #     previous_mode[agent_id] = []

    while stack != []:
        node = stack.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            print(trace)
            # segment_start.add(round(trace[0][0], 2))
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][1])
                x_max = max(x_max, trace[i][1])
                y_min = min(y_min, trace[i][2])
                y_max = max(y_max, trace[i][2])
                # print(round(trace[i][0], 2))
                time_point = round(trace[i][0], 2)
                if time_point not in timed_point_dict:
                    timed_point_dict[time_point] = [
                        {agent_id: trace[i][1:].tolist()}]
                else:
                    init = False
                    for record in timed_point_dict[time_point]:
                        if list(record.values())[0] == trace[i][1:].tolist():
                            init = True
                            break
                    if init == False:
                        timed_point_dict[time_point].append(
                            {agent_id: trace[i][1:].tolist()})
            time = round(trace[i][0], 2)
        stack += node.child
    # fill in most of layout
    # print(segment_start)
    # print(timed_point_dict.keys())
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
    print(point_list)
    x_list = []
    y_list = []
    text_list = []
    for data in point_list:
        trace = list(data.values())[0]
        # print(trace)
        x_list.append(trace[0])
        y_list.append(trace[1])
        text_list.append(
            ('{:.2f}'.format(trace[2]/pi*180), '{:.3f}'.format(trace[3])))
    data_dict = {
        "x": x_list,
        "y": y_list,
        "mode": "markers + text",
        "text": text_list,
        "textfont": dict(size=14, color="black"),
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
        # print(time_point)
        frame = {"data": [], "layout": {
            "annotations": []}, "name": '{:.2f}'.format(time_point)}
        # print(timed_point_dict[time_point][0])
        point_list = timed_point_dict[time_point]
        # point_list = list(OrderedDict.fromkeys(timed_point_dict[time_point]))
        # todokeyi
        trace_x = []
        trace_y = []
        trace_theta = []
        trace_v = []
        for data in point_list:
            trace = list(data.values())[0]
            # print(trace)
            trace_x.append(trace[0])
            trace_y.append(trace[1])
            trace_theta.append(trace[2])
            trace_v.append(trace[3])
        data_dict = {
            "x": trace_x,
            "y": trace_y,
            "mode": "markers + text",
            # "text": [(round(trace_theta[i]/pi*180, 2), round(trace_v[i], 3)) for i in range(len(trace_theta))],
            "text": [('{:.2f}'.format(trace_theta[i]/pi*180), '{:.3f}'.format(trace_v[i])) for i in range(len(trace_theta))],
            "textfont": dict(size=14, color="black"),
            "textposition": "bottom center",
            # "marker": {
            #     "sizemode": "area",
            #     "sizeref": 200000,
            #     "size": 2
            # },
            "name": "current position",
            # "show_legend": False
        }
        frame["data"].append(data_dict)
        for i in range(len(trace_x)):
            ax = np.cos(trace_theta[i])*trace_v[i]
            ay = np.sin(trace_theta[i])*trace_v[i]
            # print(trace_x[i]+ax, trace_y[i]+ay)
            annotations_dict = {"x": trace_x[i]+ax, "y": trace_y[i]+ay,
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

            # if (time_point in segment_start) and (operator.ne(previous_mode[agent_id], node.mode[agent_id])):
            #     annotations_dict = {"x": trace_x[i], "y": trace_y[i],
            #                         # "xshift": ax, "yshift": ay,
            #                         # "ax": trace_x[i], "ay": trace_y[i],
            #                         # "arrowwidth": 2,
            #                         # "arrowside": 'end',
            #                         "showarrow": False,
            #                         # "arrowsize": 1,
            #                         # "xref": 'x', "yref": 'y',
            #                         # "axref": 'x', "ayref": 'y',
            #                         "text": str(node.mode[agent_id][0]),
            #                         # "arrowhead": 1,
            #                         # "arrowcolor": "black"
            #                         }
            #     frame["layout"]["annotations"].append(annotations_dict)
            #     print(frame["layout"]["annotations"])
            # i += 1
            # previous_mode[agent_id] = node.mode[agent_id]

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
    previous_mode = {}
    agent_list = []
    for agent_id in root.mode:
        previous_mode[agent_id] = []
        agent_list.append(agent_id)
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print(node.mode)
        # [[time,x,y,theta,v]...]
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            # print(trace)
            trace_y = trace[:, 2].tolist()
            trace_x = trace[:, 1].tolist()
            # theta = [i/pi*180 for i in trace[:, 3]]
            i = agent_list.index(agent_id)
            color = colors[i % 5]
            fig.add_trace(go.Scatter(x=trace[:, 1], y=trace[:, 2],
                                     mode='lines',
                                     line_color=color,
                                     text=[(round(trace[i, 3]/pi*180, 2), round(trace[i, 4], 3))
                                           for i in range(len(trace_y))],
                                     showlegend=False)
                          #  name='lines')
                          )
            if previous_mode[agent_id] != node.mode[agent_id]:
                theta = trace[0, 3]
                veh_mode = node.mode[agent_id][0]
                if veh_mode == 'Normal':
                    text_pos = 'middle center'
                elif veh_mode == 'Brake':
                    if theta >= -pi/2 and theta <= pi/2:
                        text_pos = 'middle left'
                    else:
                        text_pos = 'middle right'
                elif veh_mode == 'Accelerate':
                    if theta >= -pi/2 and theta <= pi/2:
                        text_pos = 'middle right'
                    else:
                        text_pos = 'middle left'
                elif veh_mode == 'SwitchLeft':
                    if theta >= -pi/2 and theta <= pi/2:
                        text_pos = 'top center'
                    else:
                        text_pos = 'bottom center'
                elif veh_mode == 'SwitchRight':
                    if theta >= -pi/2 and theta <= pi/2:
                        text_pos = 'bottom center'
                    else:
                        text_pos = 'top center'
                fig.add_trace(go.Scatter(x=[trace[0, 1]], y=[trace[0, 2]],
                                         mode='markers+text',
                                         line_color='rgba(255,255,255,0.3)',
                                         text=str(agent_id)+': ' +
                                         str(node.mode[agent_id][0]),
                                         textposition=text_pos,
                                         textfont=dict(
                    #  family="sans serif",
                    size=10,
                                             color="grey"),
                                         showlegend=False,
                                         ))
                # i += 1
                previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    fig.update_traces(showlegend=False)
    # fig.update_annotations(textfont=dict(size=14, color="black"))
    # print(fig.frames[0].layout["annotations"])
    return fig
    # fig.show()


# The 'color' property is a color and may be specified as:
#       - A hex string (e.g. '#ff0000')
#       - An rgb/rgba string (e.g. 'rgb(255,0,0)')
#       - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
#       - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
#       - A named CSS color:
#             aliceblue, antiquewhite, aqua, aquamarine, azure,
#             beige, bisque, black, blanchedalmond, blue,
#             blueviolet, brown, burlywood, cadetblue,
#             chartreuse, chocolate, coral, cornflowerblue,
#             cornsilk, crimson, cyan, darkblue, darkcyan,
#             darkgoldenrod, darkgray, darkgrey, darkgreen,
#             darkkhaki, darkmagenta, darkolivegreen, darkorange,
#             darkorchid, darkred, darksalmon, darkseagreen,
#             darkslateblue, darkslategray, darkslategrey,
#             darkturquoise, darkviolet, deeppink, deepskyblue,
#             dimgray, dimgrey, dodgerblue, firebrick,
#             floralwhite, forestgreen, fuchsia, gainsboro,
#             ghostwhite, gold, goldenrod, gray, grey, green,
#             greenyellow, honeydew, hotpink, indianred, indigo,
#             ivory, khaki, lavender, lavenderblush, lawngreen,
#             lemonchiffon, lightblue, lightcoral, lightcyan,
#             lightgoldenrodyellow, lightgray, lightgrey,
#             lightgreen, lightpink, lightsalmon, lightseagreen,
#             lightskyblue, lightslategray, lightslategrey,
#             lightsteelblue, lightyellow, lime, limegreen,
#             linen, magenta, maroon, mediumaquamarine,
#             mediumblue, mediumorchid, mediumpurple,
#             mediumseagreen, mediumslateblue, mediumspringgreen,
#             mediumturquoise, mediumvioletred, midnightblue,
#             mintcream, mistyrose, moccasin, navajowhite, navy,
#             oldlace, olive, olivedrab, orange, orangered,
#             orchid, palegoldenrod, palegreen, paleturquoise,
#             palevioletred, papayawhip, peachpuff, peru, pink,
#             plum, powderblue, purple, red, rosybrown,
#             royalblue, rebeccapurple, saddlebrown, salmon,
#             sandybrown, seagreen, seashell, sienna, silver,
#             skyblue, slateblue, slategray, slategrey, snow,
#             springgreen, steelblue, tan, teal, thistle, tomato,
#             turquoise, violet, wheat, white, whitesmoke,
#             yellow, yellowgreen
