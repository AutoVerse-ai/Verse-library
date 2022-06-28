"""
This file consist main plotter code for DryVR reachtube output
"""

from __future__ import annotations
# from audioop import reverse
# from curses import start_color
# from re import A
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
from math import pi
import plotly.graph_objects as go
from typing import List
# from PIL import Image, ImageDraw
# import io
import copy
# import operator
# from collections import OrderedDict

# from torch import layout
from dryvr_plus_plus.scene_verifier.analysis.analysis_tree_node import AnalysisTreeNode

colors = [['#CC0000', '#FF0000', '#FF3333', '#FF6666', '#FF9999'],
          ['#CC6600', '#FF8000', '#FF9933', '#FFB266', '#FFCC99'],
          ['#CCCC00', '#FFFF00', '#FFFF33', '#FFFF66', '#FFFF99'],
          ['#66CC00', '#80FF00', '#99FF33', '#B2FF66', '#CCFF99'],
          ['#00CC00', '#00FF00', '#33FF33', '#66FF66', '#99FF99'],
          ['#00CC66', '#00FF80', '#33FF99', '#66FFB2', '#99FFCC'],
          ['#00CCCC', '#00FFFF', '#33FFFF', '#66FFFF', '#99FFFF'],
          ['#0066CC', '#0080FF', '#3399FF', '#66B2FF', '#99CCFF'],
          ['#0000CC', '#0000FF', '#3333FF', '#6666FF', '#9999FF'],
          ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF'],
          ['#CC00CC', '#FF00FF', '#FF33FF', '#FF66FF', '#FF99FF'],
          ['#CC0066', '#FF007F', '#FF3399', '#FF66B2', '#FF99CC']
          ]
scheme_dict = {'red': 0, 'orange': 1, 'yellow': 2, 'yellowgreen': 3, 'lime': 4,
               'springgreen': 5, 'cyan': 6, 'cyanblue': 7, 'blue': 8, 'purple': 9, 'magenta': 10, 'pink': 11}
bg_color = ['rgba(31,119,180,1)', 'rgba(255,127,14,0.2)', 'rgba(44,160,44,0.2)', 'rgba(214,39,40,0.2)', 'rgba(148,103,189,0.2)',
            'rgba(140,86,75,0.2)', 'rgba(227,119,194,0.2)', 'rgba(127,127,127,0.2)', 'rgba(188,189,34,0.2)', 'rgba(23,190,207,0.2)']
color_cnt = 0
text_size = 8
scale_factor = 0.25
mode_color = 'rgba(0,0,0,0.5)'


def general_reachtube_anime(root, map=None, fig=None, x_dim: int = 1, y_dim: int = 2, map_type='lines'):
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
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
                x_min = min(x_min, trace[i][x_dim])
                x_max = max(x_max, trace[i][x_dim])
                y_min = min(y_min, trace[i][y_dim])
                y_max = max(y_max, trace[i][y_dim])
                time_point = round(trace[i][0], 2)
                rect = [trace[i][0:].tolist(), trace[i+1][0:].tolist()]
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
        # "range": [(x_min-10), (x_max+10)],
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
            # print(rect)
            x_list.append((rect[0][x_dim]+rect[1][x_dim])/2)
            y_list.append((rect[0][y_dim]+rect[1][y_dim])/2)
            text_list.append(
                ('{:.2f}'.format((rect[0][x_dim]+rect[1][x_dim])/2), '{:.3f}'.format(rect[0][y_dim]+rect[1][y_dim])/2))
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
        for agent_id, rect_list in agent_dict.items():
            for rect in rect_list:
                trace_x.append((rect[0][x_dim]+rect[1][x_dim])/2)
                trace_y.append((rect[0][y_dim]+rect[1][y_dim])/2)
                # trace_theta.append((rect[0][2]+rect[1][2])/2)
                # trace_v.append((rect[0][3]+rect[1][3])/2)
                shape_dict = {
                    "type": 'rect',
                    "x0": rect[0][x_dim],
                    "y0": rect[0][y_dim],
                    "x1": rect[1][x_dim],
                    "y1": rect[1][y_dim],
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
        fig = draw_reachtube_tree_v2(root, agent_id, 1, 2, i, fig)
        i += 2

    return fig


def draw_reachtube_tree_v2(root, agent_id, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, color_id=None, map_type='lines'):
    # if fig is None:
    #     fig = go.Figure()
    global color_cnt, bg_color
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    if color_id is None:
        color_id = color_cnt
    queue = [root]
    show_legend = False
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        # print(trace[0], trace[1], trace[-2], trace[-1])
        max_id = len(trace)-1
        trace_x_odd = np.array([trace[i][x_dim] for i in range(0, max_id, 2)])
        trace_x_even = np.array([trace[i][x_dim]
                                for i in range(1, max_id+1, 2)])
        trace_y_odd = np.array([trace[i][y_dim] for i in range(0, max_id, 2)])
        trace_y_even = np.array([trace[i][y_dim]
                                for i in range(1, max_id+1, 2)])
        fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color_id],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend
                                 ))
        fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color_id],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend))
        fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color_id],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend
                                 ))
        fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
                                 fill='toself',
                                 fillcolor=bg_color[color_id],
                                 line_color='rgba(255,255,255,0)',
                                 showlegend=show_legend))
        queue += node.child
        color_id = (color_id+1) % 10
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        # print(trace[0], trace[1], trace[-2], trace[-1])
        max_id = len(trace)-1
        fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
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
    color_cnt = color_id
    # fig.update_traces(line_dash="dash")
    return fig


def draw_map(map, color='rgba(0,0,0,1)', fig: go.Figure() = go.Figure(), fill_type='lines'):
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    if fill_type == 'detailed':
        speed_dict = map.get_all_speed_limit()
        speed_list = list(filter(None, speed_dict.values()))
        speed_min = min(speed_list)
        speed_max = max(speed_list)
        start_color = [255, 255, 255, 0.2]
        end_color = [0, 0, 0, 0.2]
        curr_color = [0, 0, 0, 0]
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        curr_color = [0, 0, 0, 0]
        if fill_type == 'detailed':
            speed_limit = speed_dict[lane_idx]
            if speed_limit is not None:
                lens = len(curr_color)-1
                for j in range(lens):
                    curr_color[j] = int(start_color[j]+(speed_limit-speed_min)/(speed_max -
                                                                                speed_min)*(end_color[j]-start_color[j]))
                curr_color[lens] = start_color[lens]+(speed_limit-speed_min)/(speed_max -
                                                                              speed_min)*(end_color[lens]-start_color[lens])
                # print(curr_color)
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                start1 = lane_seg.start + lane_seg.width/2 * lane_seg.direction_lateral
                end1 = lane_seg.end + lane_seg.width/2 * lane_seg.direction_lateral
                start2 = lane_seg.start - lane_seg.width/2 * lane_seg.direction_lateral
                end2 = lane_seg.end - lane_seg.width/2 * lane_seg.direction_lateral
                trace_x = [start1[0], end1[0], end2[0], start2[0], start1[0]]
                trace_y = [start1[1], end1[1], end2[1], start2[1], start1[1]]
                x_min = min(x_min, min(trace_x))
                y_min = min(y_min, min(trace_y))
                x_max = max(x_max, max(trace_x))
                y_max = max(y_max, max(trace_y))
                if fill_type == 'lines' or speed_limit is None:
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             #  fill='toself',
                                             #  fillcolor='rgba(255,255,255,0)',
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
                elif fill_type == 'detailed' and speed_limit is not None:
                    print(curr_color)
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             fillcolor='rgba' +
                                             str(tuple(curr_color)),
                                             showlegend=False,
                                             # text=theta,
                                             name='limit'))
                elif fill_type == 'fill':
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             #  fillcolor='rgba(255,255,255,0)',
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(
                    start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                r1 = lane_seg.radius - lane_seg.width/2
                x1 = (np.cos(phase_array)*r1 + lane_seg.center[0]).tolist()
                y1 = (np.sin(phase_array)*r1 + lane_seg.center[1]).tolist()
                # fig.add_trace(go.Scatter(x=x1, y=y1,
                #                          mode='lines',
                #                          line_color=color,
                #                          showlegend=False,
                #                          # text=theta,
                #                          name='lines'))
                r2 = lane_seg.radius + lane_seg.width/2
                x2 = (np.cos(phase_array)*r2 +
                      lane_seg.center[0]).tolist().reverse()
                y2 = (np.sin(phase_array)*r2 +
                      lane_seg.center[1]).tolist().reverse()
                trace_x = x1+x2+[x1[0]]
                trace_y = y1+y2+[y1[0]]
                x_min = min(x_min, min(trace_x))
                y_min = min(y_min, min(trace_y))
                x_max = max(x_max, max(trace_x))
                y_max = max(y_max, max(trace_y))
                if fill_type == 'lines':
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
                elif fill_type == 'detailed' and speed_limit != None:
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             fillcolor='rgba' +
                                             str(tuple(curr_color)),
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
                elif fill_type == 'fill':
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    if fill_type == 'detailed':
        fig.add_trace(go.Scatter(x=[0], y=[0],
                                 mode='markers',
                                 # fill='toself',
                                 # fillcolor='rgba'+str(tuple(curr_color)),
                                 marker=dict(
            symbol='square',
            size=16,
            cmax=speed_max,
            cmin=speed_min,
            color='rgba(0,0,0,0)',
            colorbar=dict(
                title="Speed Limit"
            ),
            colorscale=[
                [0, 'rgba'+str(tuple(start_color))], [1, 'rgba'+str(tuple(end_color))]]
        ),
            showlegend=False,
        ))
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def plotly_map(map, color='rgba(0,0,0,1)', fig: go.Figure() = go.Figure()):
    # if fig is None:
    #     fig = go.Figure()
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
                fig.add_trace(go.Scatter(x=[start1[0], end1[0], end2[0], start2[0], start1[0]], y=[start1[1], end1[1], end2[1], start2[1], start1[1]],
                                         mode='lines',
                                         line_color=color,
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
                                         line_color=color,
                                         showlegend=False,
                                         # text=theta,
                                         name='lines'))

                r2 = lane_seg.radius + lane_seg.width/2
                x = np.cos(phase_array)*r2 + lane_seg.center[0]
                y = np.sin(phase_array)*r2 + lane_seg.center[1]
                fig.add_trace(go.Scatter(x=x, y=y,
                                         mode='lines',
                                         line_color=color,
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


def draw_simulation_tree(root, map=None, fig=None, x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace'):
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    # mark
    agent_list = root.agent.keys()
    print(agent_list)
    scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = draw_simulation_tree_single(
            root, agent_id, fig, x_dim, y_dim, scheme_list[i], map_type)
        i = (i+5) % 12
    if scale_type == 'trace':
        queue = [root]
        x_min, x_max = float('inf'), -float('inf')
        y_min, y_max = float('inf'), -float('inf')
        # scale_factor = 0.25
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
            if scale_type == 'trace':
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            # print(trace)
            # trace_y = trace[:, y_dim].tolist()
            # trace_x = trace[:, x_dim].tolist()
            # theta = [i/pi*180 for i in trace[:, 3]]
            i = agent_list.index(agent_id)
            # color = colors[i % 5]
            # fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
            #                          mode='lines',
            #                          line_color=color,
            #                          text=[('{:.2f}'.format(trace_x[i]), '{:.2f}'.format(
            #                              trace_y[i])) for i in range(len(trace_x))],
            #                          showlegend=False)
            #               #  name='lines')
            #               )
            if previous_mode[agent_id] != node.mode[agent_id]:
                veh_mode = node.mode[agent_id][0]
                if veh_mode == 'Normal':
                    text_pos = 'middle center'
                elif veh_mode == 'Brake':
                    text_pos = 'middle left'
                elif veh_mode == 'Accelerate':
                    text_pos = 'middle right'
                elif veh_mode == 'SwitchLeft':
                    text_pos = 'top center'
                elif veh_mode == 'SwitchRight':
                    text_pos = 'bottom center'

                fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                         mode='markers+text',
                                         line_color=mode_color,
                                         text=str(agent_id)+': ' +
                                         str(node.mode[agent_id][0]),
                                         textposition=text_pos,
                                         textfont=dict(
                    #  family="sans serif",
                    size=text_size,
                                             color="grey"),
                                         showlegend=False,
                                         ))
                # i += 1
                previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    return fig


def draw_simulation_tree_single(root: AnalysisTreeNode, agent_id, fig: go.Figure() = go.Figure(), x_dim: int = 1, y_dim: int = 2, color=None, map_type='lines'):
    global color_cnt
    # fig = draw_map(map=map, fig=fig, fill_type=map_type)
    queue = [root]
    color_id = 0
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt+1) % 12
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print(node.mode)
        # [[time,x,y,theta,v]...]
        trace = np.array(traces[agent_id])
        # print(trace)
        # trace_y = trace[:, y_dim].tolist()
        # trace_x = trace[:, x_dim].tolist()
        # trace_x_rev = trace_x[::-1]
        # # print(trace_x)
        # trace_upper = [i+1 for i in trace_y]
        # trace_lower = [i-1 for i in trace_y]
        # trace_lower = trace_lower[::-1]
        # # print(trace_upper)
        # # print(trace[:, y_dim])
        # fig.add_trace(go.Scatter(x=trace_x+trace_x_rev, y=trace_upper+trace_lower,
        #                          fill='toself',
        #                          fillcolor=bg_color[color_id],
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=False))'
        trace_text = []
        for i in range(len(trace)):
            trace_text.append(['{:.2f}'.format(trace[i, j])
                              for j in range(trace.shape[1])])

        fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
                                 mode='lines',
                                 line_color=colors[scheme_dict[color]
                                                   ][color_id],
                                 #  text=[('{:.2f}'.format(trace[i, x_dim]), '{:.2f}'.format(
                                 #      trace[i, y_dim])) for i in range(len(trace))],
                                 text=trace_text,
                                 #  name='lines',
                                 showlegend=False))
        color_id = (color_id+4) % 5
        queue += node.child
    return fig


def draw_simulation_anime(root, map=None, fig=None):
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
        "textfont": dict(size=text_size, color="black"),
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
            "textfont": dict(size=text_size, color="black"),
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
                                         line_color=mode_color,
                                         text=str(agent_id)+': ' +
                                         str(node.mode[agent_id][0]),
                                         textposition=text_pos,
                                         textfont=dict(
                    #  family="sans serif",
                    size=text_size,
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


def general_simu_anime(root, map=None, fig=None, x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace'):
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    # fig = plot_map(map, 'g', fig)
    timed_point_dict = {}
    stack = [root]
    # print("plot")
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
            # print(trace)
            # segment_start.add(round(trace[0][0], 2))
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][x_dim])
                x_max = max(x_max, trace[i][x_dim])
                y_min = min(y_min, trace[i][y_dim])
                y_max = max(y_max, trace[i][y_dim])
                # print(round(trace[i][0], 2))
                time_point = round(trace[i][0], 2)
                if time_point not in timed_point_dict:
                    timed_point_dict[time_point] = [
                        {agent_id: trace[i][0:].tolist()}]
                else:
                    init = False
                    for record in timed_point_dict[time_point]:
                        if list(record.values())[0] == trace[i][0:].tolist():
                            init = True
                            break
                    if init == False:
                        timed_point_dict[time_point].append(
                            {agent_id: trace[i][0:].tolist()})
            time = round(trace[i][0], 2)
        stack += node.child
    # fill in most of layout
    # print(segment_start)
    # print(timed_point_dict.keys())
    duration = 10
    fig_dict["layout"]["xaxis"] = {
        "range": [x_min, x_max],
        "title": "x position"}
    fig_dict["layout"]["yaxis"] = {
        "range": [y_min, y_max],
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
    x_list = []
    y_list = []
    text_list = []
    for data in point_list:
        trace = list(data.values())[0]
        # print(trace)
        x_list.append(trace[x_dim])
        y_list.append(trace[y_dim])
        # text_list.append(
        #     ('{:.2f}'.format(trace[x_dim]), '{:.2f}'.format(trace[y_dim])))
        text_list.append(['{:.2f}'.format(trace[i])
                         for i in range(len(trace))])
    data_dict = {
        "x": x_list,
        "y": y_list,
        "mode": "markers + text",
        "text": text_list,
        "textfont": dict(size=text_size, color="black"),
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
        text_list = []
        for data in point_list:
            # todo
            trace = list(data.values())[0]
            # print(trace)
            trace_x.append(trace[x_dim])
            trace_y.append(trace[y_dim])
            # text_list.append(
            #     ('{:.2f}'.format(trace[x_dim]), '{:.2f}'.format(trace[y_dim])))
            text_list.append(['{:.2f}'.format(trace[i])
                              for i in range(len(trace))])
        data_dict = {
            "x": trace_x,
            "y": trace_y,
            "mode": "markers + text",
            "text": text_list,
            "marker": {
                # "color": color,
                # "opacity": opacity_step*(trail_len-id),
                # # "sizemode": "area",
                # # "sizeref": 200000,
                # "size": min_size + size_step*(trail_len-id)
            },
            "textfont": dict(size=text_size, color="black"),
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
        # for i in range(len(trace_x)):
        #     ax = np.cos(trace_theta[i])*trace_v[i]
        #     ay = np.sin(trace_theta[i])*trace_v[i]
        # print(trace_x[i]+ax, trace_y[i]+ay)
        # annotations_dict = {"x": trace_x[i]+ax, "y": trace_y[i]+ay,
        #                     # "xshift": ax, "yshift": ay,
        #                     "ax": trace_x[i], "ay": trace_y[i],
        #                     "arrowwidth": 2,
        #                     # "arrowside": 'end',
        #                     "showarrow": True,
        #                     # "arrowsize": 1,
        #                     "xref": 'x', "yref": 'y',
        #                     "axref": 'x', "ayref": 'y',
        #                     # "text": "erver",
        #                     "arrowhead": 1,
        #                     "arrowcolor": "black"}
        # frame["layout"]["annotations"].append(annotations_dict)

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
    # print(map)
    fig = draw_map(map, 'rgba(0,0,0,1)', fig, map_type)
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
            trace_y = trace[:, y_dim].tolist()
            trace_x = trace[:, x_dim].tolist()
            # theta = [i/pi*180 for i in trace[:, 3]]
            i = agent_list.index(agent_id)
            # color = colors[i % 5]
            # fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
            #                          mode='lines',
            #                          line_color=color,
            #                          text=[('{:.2f}'.format(trace_x[i]), '{:.2f}'.format(
            #                              trace_y[i])) for i in range(len(trace_x))],
            #                          showlegend=False)
            #               #  name='lines')
            #               )
            if previous_mode[agent_id] != node.mode[agent_id]:
                veh_mode = node.mode[agent_id][0]
                if veh_mode == 'Normal':
                    text_pos = 'middle center'
                elif veh_mode == 'Brake':
                    text_pos = 'middle left'
                elif veh_mode == 'Accelerate':
                    text_pos = 'middle right'
                elif veh_mode == 'SwitchLeft':
                    text_pos = 'top center'
                elif veh_mode == 'SwitchRight':
                    text_pos = 'bottom center'

                fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                         mode='markers+text',
                                         line_color=mode_color,
                                         text=str(agent_id)+': ' +
                                         str(node.mode[agent_id][0]),
                                         textposition=text_pos,
                                         opacity=0.5,
                                         textfont=dict(
                    #  family="sans serif",
                    size=text_size,
                                             color="grey"),
                                         showlegend=False,
                                         ))
                # i += 1
                previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    fig.update_traces(showlegend=False)
    # scale_factor = 0.25
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    # fig.update_annotations(textfont=dict(size=14, color="black"))
    # print(fig.frames[0].layout["annotations"])
    return fig


def test_simu_anime(root, map=None, fig=None, x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace'):
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
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][x_dim])
                x_max = max(x_max, trace[i][x_dim])
                y_min = min(y_min, trace[i][y_dim])
                y_max = max(y_max, trace[i][y_dim])
                time_point = round(trace[i][0], 2)
                tmp_trace = trace[i][0:].tolist()
                if time_point not in timed_point_dict:
                    timed_point_dict[time_point] = {agent_id: [tmp_trace]}
                else:
                    if agent_id not in timed_point_dict[time_point].keys():
                        timed_point_dict[time_point][agent_id] = [tmp_trace]
                    elif tmp_trace not in timed_point_dict[time_point][agent_id]:
                        timed_point_dict[time_point][agent_id].append(
                            tmp_trace)
            time = round(trace[i][0], 2)
        stack += node.child
    # fill in most of layout
    # print(segment_start)
    # print(timed_point_dict.keys())
    duration = int(100/time)
    duration = 1
    fig_dict["layout"]["xaxis"] = {
        # "range": [x_min, x_max],
        "title": "x position"}
    fig_dict["layout"]["yaxis"] = {
        # "range": [y_min, y_max],
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
    # # make data
    trace_dict = timed_point_dict[0]

    x_list = []
    y_list = []
    text_list = []
    for time_point in timed_point_dict:
        trace_dict = timed_point_dict[time_point]
        for agent_id, point_list in trace_dict.items():
            for point in point_list:
                x_list.append(point[x_dim])
                y_list.append(point[y_dim])
                # text_list.append(
                #     ('{:.2f}'.format(point[x_dim]), '{:.2f}'.format(point[y_dim])))
                text_list.append(['{:.2f}'.format(point[i])
                                  for i in range(len(point))])
        data_dict = {
            "x": x_list,
            "y": y_list,
            "mode": "markers",
            "text": text_list,
            "textfont": dict(size=text_size, color="black"),
            "visible": False,
            "textposition": "bottom center",
            # "marker": {
            #     "sizemode": "area",
            #     "sizeref": 200000,
            #     "size": 2
            # },
            "name": "Current Position"
        }
        fig_dict["data"].append(data_dict)
    time_list = list(timed_point_dict.keys())
    agent_list = list(trace_dict.keys())
    # start = time_list[0]
    # step = time_list[1]-start
    trail_limit = min(10, len(time_list))
    # print(agent_list)
    # make frames
    for time_point_id in range(trail_limit, len(time_list)):
        time_point = time_list[time_point_id]
        frame = {"data": [], "layout": {
            "annotations": []}, "name": '{:.2f}'.format(time_point)}
        # todokeyi
        trail_len = min(time_point_id+1, trail_limit)
        opacity_step = 1/trail_len
        size_step = 2/trail_len
        min_size = 5
        for agent_id in agent_list:
            color = colors[agent_list.index(agent_id) % 12][1]
            for id in range(0, trail_len, 2):
                tmp_point_list = timed_point_dict[time_list[time_point_id-id]][agent_id]
                trace_x = []
                trace_y = []
                text_list = []
                for point in tmp_point_list:
                    trace_x.append(point[x_dim])
                    trace_y.append(point[y_dim])
                    # text_list.append(
                    #     ('{:.2f}'.format(point[x_dim]), '{:.2f}'.format(point[y_dim])))
                    text_list.append(['{:.2f}'.format(point[i])
                                      for i in range(len(point))])
                #  print(trace_y)
                if id == 0:
                    data_dict = {
                        "x": trace_x,
                        "y": trace_y,
                        "mode": "markers+text",
                        "text": text_list,
                        "textfont": dict(size=text_size, color="black"),
                        "textposition": "bottom center",
                        "visible": True,
                        "marker": {
                            "color": color,
                            "opacity": opacity_step*(trail_len-id),
                            # "sizemode": "area",
                            # "sizeref": 200000,
                            "size": min_size + size_step*(trail_len-id)
                        },
                        "name": "current position",
                        # "show_legend": False
                    }
                else:
                    data_dict = {
                        "x": trace_x,
                        "y": trace_y,
                        "mode": "markers",
                        "text": text_list,
                        # "textfont": dict(size=6, color="black"),
                        # "textposition": "bottom center",
                        "visible": True,
                        "marker": {
                            "color": color,
                            "opacity": opacity_step*(trail_len-id),
                            # "sizemode": "area",
                            # "sizeref": 200000,
                            "size": min_size + size_step*(trail_len-id)
                        },
                        "name": "current position",
                        # "show_legend": False
                    }
                frame["data"].append(data_dict)

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
    fig = draw_map(map, 'rgba(0,0,0,1)', fig, map_type)
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
            trace_y = trace[:, y_dim].tolist()
            trace_x = trace[:, x_dim].tolist()
            # theta = [i/pi*180 for i in trace[:, 3]]
            i = agent_list.index(agent_id)
            # color = colors[i % 5]
            # fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
            #                          mode='lines',
            #                          line_color=color,
            #                          text=[('{:.2f}'.format(trace_x[i]), '{:.2f}'.format(
            #                              trace_y[i])) for i in range(len(trace_x))],
            #                          showlegend=False)
            #               #  name='lines')
            #               )
            if previous_mode[agent_id] != node.mode[agent_id]:
                veh_mode = node.mode[agent_id][0]
                if veh_mode == 'Normal':
                    text_pos = 'middle center'
                elif veh_mode == 'Brake':
                    text_pos = 'middle left'
                elif veh_mode == 'Accelerate':
                    text_pos = 'middle right'
                elif veh_mode == 'SwitchLeft':
                    text_pos = 'top center'
                elif veh_mode == 'SwitchRight':
                    text_pos = 'bottom center'

                fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                         mode='markers+text',
                                         line_color=mode_color,
                                         text=str(agent_id)+': ' +
                                         str(node.mode[agent_id][0]),
                                         opacity=0.5,
                                         textposition=text_pos,
                                         textfont=dict(
                    #  family="sans serif",
                    size=text_size,
                                             color="grey"),
                                         showlegend=False,
                                         ))
                # i += 1
                previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    fig.update_traces(showlegend=False)
    # scale_factor = 0.5
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    # fig.update_annotations(textfont=dict(size=14, color="black"))
    # print(fig.frames[0].layout["annotations"])
    return fig


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
