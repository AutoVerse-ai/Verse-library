"""
This file consist main plotter code for DryVR reachtube output
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union
from plotly.graph_objs.scatter import Marker
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode

colors = [['#CC0000', '#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC'],
          ['#CCCC00', '#FFFF00', '#FFFF33', '#FFFF66', '#FFFF99', '#FFE5CC'],
          ['#66CC00', '#80FF00', '#99FF33', '#B2FF66', '#CCFF99', '#FFFFCC'],
          ['#00CC00', '#00FF00', '#33FF33', '#66FF66', '#99FF99', '#E5FFCC'],
          ['#00CC66', '#00FF80', '#33FF99', '#66FFB2', '#99FFCC', '#CCFFCC'],
          ['#00CCCC', '#00FFFF', '#33FFFF', '#66FFFF', '#99FFFF', '#CCFFE5'],
          ['#0066CC', '#0080FF', '#3399FF', '#66B2FF', '#99CCFF', '#CCE5FF'],
          ['#0000CC', '#0000FF', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF'],
          ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'],
          ['#CC00CC', '#FF00FF', '#FF33FF', '#FF66FF', '#FF99FF', '#FFCCFF'],
          ['#CC0066', '#FF007F', '#FF3399', '#FF66B2', '#FF99CC', '#FFCCE5']
          ]
scheme_dict = {'red': 0, 'orange': 1, 'yellow': 2, 'yellowgreen': 3, 'lime': 4,
               'springgreen': 5, 'cyan': 6, 'cyanblue': 7, 'blue': 8, 'purple': 9, 'magenta': 10, 'pink': 11}
color_cnt = 0
text_size = 8
scale_factor = 0.25
mode_point_color = 'rgba(0,0,0,0.5)'
mode_text_color = 'black'
duration = 10


"""These 5 Functions below are high-level functions and are recommended to use."""
"""
API
These 5 functions share the same API.

- root: the root node of the trace, should be the return value of Scenario.verify() or Scenario.simulate().
- map: the map of the scenario, templates are in verse.example.example_map.simple_map2.py.
- fig: the object of the figure, its type should be plotly.graph_objects.Figure().
- x_dim: the dimension of x coordinate in the trace list of every time step. The default value is 1.
- y_dim: the dimension of y coordinate in the trace list of every time step. The default value is 2.
- map_type the way to draw the map. It should be 'lines' or 'fill' or 'detailed'. The default value is 'lines'.
--  For the 'lines' mode, map is only drawn by margins of lanes.
--  For the 'fill' mode, the lanes will be filled with semitransparent colors.
--  For the 'detailed' mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). 
    Otherwise, it is the same as the 'lines' mode.
- scale_type the way to scale the coordinate axises. It should be 'trace' or 'map'. The default value is 'trace'.
--  For the 'trace' mode, the traces will be in the center of the plot with an appropriate scale.
--  For the 'map' mode, the map will be in the center of the plot with an appropriate scale.
- print_dim_list the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. 
    The default value is None. And then all dimensions will be shown.
"""


def reachtube_anime(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, map_type='lines', scale_type='trace', print_dim_list=None, label_mode='None'):
    if isinstance(root, AnalysisTree):
        root = root.root
    """It gives the animation of the verfication."""
    agent_list = list(root.agent.keys())
    timed_point_dict = {}
    stack = [root]
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    # input check
    num_dim = np.array(root.trace[list(root.agent.keys())[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)
    scheme_list = list(scheme_dict.keys())

    while stack != []:
        node = stack.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if trace[0][0] > 0:
                trace = trace[4:]
            for i in range(0, len(trace)-1, 2):
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
    fig_dict, sliders_dict = create_anime_dict(duration)
    for time_point in timed_point_dict:
        frame = {"data": [], "layout": {
            "annotations": [], "shapes": []}, "name": str(time_point)}
        agent_dict = timed_point_dict[time_point]
        for agent_id, rect_list in agent_dict.items():
            for rect in rect_list:
                shape_dict = {
                    "type": 'rect',
                    "x0": rect[0][x_dim],
                    "y0": rect[0][y_dim],
                    "x1": rect[1][x_dim],
                    "y1": rect[1][y_dim],
                    "fillcolor": 'rgba(0,0,0,0.5)',
                    "line": dict(color='rgba(255,255,255,0)'),
                    "visible": True

                }
                frame["layout"]["shapes"].append(shape_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            ['{:.2f}'.format(time_point)],
            {"frame": {"duration": duration, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": duration}}
        ],
            "label": '{:.2f}'.format(time_point),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single(
            root, agent_id, fig, x_dim, y_dim, scheme_list[i], print_dim_list)
        i = (i+1) % 12
    if scale_type == 'trace':
        queue = [root]
        x_min, x_max = float('inf'), -float('inf')
        y_min, y_max = float('inf'), -float('inf')
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == 'trace':
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            if label_mode != 'None':
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    x0 = trace[0, x_dim]
                    x1 = trace[1, x_dim]
                    y0 = trace[0, y_dim]
                    y1 = trace[1, y_dim]
                    mode_point_color = colors[agent_list.index(
                        agent_id) % 12][0]
                    fig.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(y0+y1)/2],
                                             mode='markers+text',
                                             line_color=mode_point_color,
                                             text=str(agent_id)+': ' + text,
                                             textposition=text_pos,
                                             opacity=0.5,
                                             textfont=dict(
                        size=text_size,
                        color=mode_text_color),
                        showlegend=False,
                    ))
                    previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    return fig


def reachtube_tree(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace', print_dim_list=None, label_mode='None'):
    if isinstance(root, AnalysisTree):
        root = root.root
    """It statically shows all the traces of the verfication."""
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single(
            root, agent_id, fig, x_dim, y_dim, scheme_list[i], print_dim_list)
        i = (i+1) % 12
    if scale_type == 'trace':
        queue = [root]
        x_min, x_max = float('inf'), -float('inf')
        y_min, y_max = float('inf'), -float('inf')
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print({k: len(v) for k, v in traces.items()})
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == 'trace':
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            i = agent_list.index(agent_id)
            if label_mode != 'None':
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    mode_point_color = colors[agent_list.index(
                        agent_id) % 12][0]
                    fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                             mode='markers+text',
                                             line_color=mode_point_color,
                                             opacity=0.5,
                                             text=str(agent_id)+': ' + text,
                                             textposition=text_pos,
                                             textfont=dict(
                        size=text_size,
                        color=mode_text_color),
                        showlegend=False,
                    ))
                    previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    return fig


def simulation_tree(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=None, x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace', print_dim_list=None, label_mode='None'):
    root = root.root
    """It statically shows all the traces of the simulation."""
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = simulation_tree_single(
            root, agent_id, fig, x_dim, y_dim, scheme_list[i], print_dim_list)
        i = (i+1) % 12
    if scale_type == 'trace':
        x_min, x_max = float('inf'), -float('inf')
        y_min, y_max = float('inf'), -float('inf')
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == 'trace':
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            mode_point_color = colors[agent_list.index(agent_id) % 12][0]
            if previous_mode[agent_id] != node.mode[agent_id]:
                text_pos, text = get_text_pos(node.mode[agent_id][0])
                texts = [f"{agent_id}: {text}" for _ in trace]
                mark_colors = [mode_point_color for _ in trace]
                mark_sizes = [0 for _ in trace]
                if node.assert_hits != None and agent_id in node.assert_hits:
                    mark_colors[-1] = "black"
                    mark_sizes[-1] = 10
                    texts[-1] = "BOOM!!!\nAssertions hit:\n" + "\n".join("  " + a for a in node.assert_hits[agent_id])
                marker = Marker(color=mark_colors, size=mark_sizes)
                fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                         mode='markers+lines',
                                         line_color=mode_point_color,
                                         opacity=0.5,
                                         text=texts,
                                         marker=marker,
                                         textposition=text_pos,
                                         textfont=dict(
                                             size=text_size,
                                             color=mode_text_color
                                         ),
                                         showlegend=False,
                                         ))
                previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    fig.update_xaxes(title='x')
    fig.update_yaxes(title='y')
    fig.update_layout(legend_title_text='Agent list')
    return fig


def simulation_anime(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=None, x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace', print_dim_list=None, label_mode='None'):
    if isinstance(root, AnalysisTree):
        root = root.root
    """It gives the animation of the simulation without trail but is faster."""
    timed_point_dict = {}
    stack = [root]
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    # input check
    num_dim = np.array(root.trace[list(root.agent.keys())[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)
    agent_list = list(root.agent.keys())
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
    fig_dict, sliders_dict = create_anime_dict(duration)
    # make data
    trace_dict = timed_point_dict[0]
    for agent_id, trace_list in trace_dict.items():
        color = colors[agent_list.index(agent_id) % 12][1]
        x_list = []
        y_list = []
        text_list = []
        branch_cnt = 0
        for trace in trace_list:
            x_list.append(trace[x_dim])
            y_list.append(trace[y_dim])
            text_list.append(['{:.2f}'.format(trace[i])
                              for i in print_dim_list])
            branch_cnt += 1
        data_dict = {
            "x": x_list,
            "y": y_list,
            "text": text_list,
            "mode": "markers + text",
            "textfont": dict(size=text_size, color="black"),
            "textposition": "bottom center",
            "marker": {
                "color": color,
            },
            "name": agent_id,
            "showlegend": True
        }
        fig_dict["data"].append(data_dict)
    # make frames
    for time_point in timed_point_dict:
        frame = {"data": [], "layout": {
            "annotations": []}, "name": '{:.2f}'.format(time_point)}
        point_list = timed_point_dict[time_point]
        for agent_id, trace_list in point_list.items():
            color = colors[agent_list.index(agent_id) % 12][1]
            x_list = []
            y_list = []
            text_list = []
            branch_cnt = 0
            for trace in trace_list:
                x_list.append(trace[x_dim])
                y_list.append(trace[y_dim])
                text_list.append(['{:.2f}'.format(trace[i])
                                  for i in print_dim_list])
                branch_cnt += 1
            data_dict = {
                "x": x_list,
                "y": y_list,
                "text": text_list,
                "mode": "markers + text",
                "marker": {
                    "color": color,
                },
                "textfont": dict(size=text_size, color="black"),
                "textposition": "bottom center",
                # "name": "Branch-"+str(branch_cnt),
                "name": agent_id,
                "showlegend": True
            }
            frame["data"].append(data_dict)
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            ['{:.2f}'.format(time_point)],
            {"frame": {"duration": duration, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": duration}}
        ],
            "label": '{:.2f}'.format(time_point),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = draw_map(map, 'rgba(0,0,0,1)', fig, map_type)
    i = 0
    queue = [root]
    previous_mode = {}
    agent_list = list(root.agent.keys())
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            mode_point_color = colors[agent_list.index(agent_id) % 12][0]
            if label_mode != 'None':
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                             mode='markers+text',
                                             line_color=mode_point_color,
                                             text=str(agent_id)+': ' + text,
                                             textposition=text_pos,
                                             opacity=0.5,
                                             textfont=dict(
                        size=text_size,
                        color=mode_text_color),
                        showlegend=False,
                    ))
                    previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    fig.update_layout(legend_title_text='Agent list')
    return fig


def simulation_anime_trail(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim=2, map_type='lines', scale_type='trace', print_dim_list=None, label_mode='None'):
    """It gives the animation of the simulation with trail."""
    if isinstance(root, AnalysisTree):
        root = root.root
    timed_point_dict = {}
    stack = [root]
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    # input check
    num_dim = np.array(root.trace[list(root.agent.keys())[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

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
                time_point = round(trace[i][0], 3)
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
    fig_dict, sliders_dict = create_anime_dict(duration)
    time_list = list(timed_point_dict.keys())
    agent_list = list(root.agent.keys())
    trail_limit = min(10, len(time_list))
    trail_len = trail_limit
    opacity_step = 1/trail_len
    size_step = 2/trail_len
    min_size = 5
    step = 2
    # # make data
    trace_dict = timed_point_dict[0]

    for time_point in list(timed_point_dict.keys())[0:int(trail_limit/step)]:
        trace_dict = timed_point_dict[time_point]
        for agent_id, point_list in trace_dict.items():
            x_list = []
            y_list = []
            text_list = []
            for point in point_list:
                x_list.append(point[x_dim])
                y_list.append(point[y_dim])
                text_list.append(['{:.2f}'.format(point[i])
                                  for i in print_dim_list])
            data_dict = {
                "x": x_list,
                "y": y_list,
                "mode": "markers",
                "text": text_list,
                "textfont": dict(size=text_size, color="black"),
                "visible": False,
                "textposition": "bottom center",
                "name": agent_id
            }
            fig_dict["data"].append(data_dict)

    # make frames
    for time_point_id in range(trail_limit, len(time_list)):
        time_point = time_list[time_point_id]
        frame = {"data": [], "layout": {
            "annotations": []}, "name": '{:.2f}'.format(time_point)}
        for agent_id in agent_list:
            color = colors[agent_list.index(agent_id) % 12][1]
            for id in range(0, trail_len, step):
                tmp_point_list = timed_point_dict[time_list[time_point_id-id]][agent_id]
                trace_x = []
                trace_y = []
                text_list = []
                for point in tmp_point_list:
                    trace_x.append(point[x_dim])
                    trace_y.append(point[y_dim])
                    text_list.append(['{:.2f}'.format(point[i])
                                      for i in print_dim_list])
                if id == 0:
                    data_dict = {
                        "x": trace_x,
                        "y": trace_y,
                        "mode": "markers+text",
                        "text": text_list,
                        "textfont": dict(size=text_size, color="black"),
                        "textposition": "bottom center",
                        "visible": True,  # 'legendonly'
                        "marker": {
                            "color": color,
                            "opacity": opacity_step*(trail_len-id),
                            "size": min_size + size_step*(trail_len-id)
                        },
                        "name": agent_id,
                        "showlegend": True
                    }
                else:
                    data_dict = {
                        "x": trace_x,
                        "y": trace_y,
                        "mode": "markers",
                        "text": text_list,
                        "visible": True,  # 'legendonly'
                        "marker": {
                            "color": color,
                            "opacity": opacity_step*(trail_len-id),
                            "size": min_size + size_step*(trail_len-id)
                        },
                        "name": agent_id,
                        "showlegend": True
                    }
                frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            ['{:.2f}'.format(time_point)],
            {"frame": {"duration": duration, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": duration}}
        ],
            "label": '{:.2f}'.format(time_point),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = draw_map(map, 'rgba(0,0,0,1)', fig, map_type)
    i = 0
    queue = [root]
    previous_mode = {}
    agent_list = list(root.agent.keys())
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = 'middle center'
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            trace_y = trace[:, y_dim].tolist()
            trace_x = trace[:, x_dim].tolist()
            i = agent_list.index(agent_id)
            mode_point_color = colors[agent_list.index(agent_id) % 12][0]
            if label_mode != 'None':
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    fig.add_trace(go.Scatter(x=[trace[0, x_dim]], y=[trace[0, y_dim]],
                                             mode='markers+text',
                                             line_color=mode_point_color,
                                             text=str(agent_id)+': ' + text,
                                             opacity=0.5,
                                             textposition=text_pos,
                                             textfont=dict(
                        size=text_size,
                        color=mode_text_color
                    ),
                        showlegend=False,
                    ))
                    previous_mode[agent_id] = node.mode[agent_id]
        queue += node.child
    if scale_type == 'trace':
        fig.update_xaxes(
            range=[x_min-scale_factor*(x_max-x_min), x_max+scale_factor*(x_max-x_min)])
        fig.update_yaxes(
            range=[y_min-scale_factor*(y_max-y_min), y_max+scale_factor*(y_max-y_min)])
    fig.update_layout(legend_title_text='Agent list')
    return fig


"""Functions below are low-level functions and usually are not called outside this file."""


def reachtube_tree_single(root: Union[AnalysisTree, AnalysisTreeNode], agent_id, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, color=None, print_dim_list=None):
    if isinstance(root, AnalysisTree):
        root = root.root
    """It statically shows the verfication traces of one given agent."""
    global color_cnt
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt+1) % 12
    queue = [root]
    show_legend = False
    fillcolor = colors[scheme_dict[color]][5]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        max_id = len(trace)-1
        trace_x_odd = np.array([trace[i][x_dim] for i in range(0, max_id, 2)])
        trace_x_even = np.array([trace[i][x_dim]
                                for i in range(1, max_id+1, 2)])
        trace_y_odd = np.array([trace[i][y_dim] for i in range(0, max_id, 2)])
        trace_y_even = np.array([trace[i][y_dim]
                                for i in range(1, max_id+1, 2)])
        # trace_y_new = [0]*len(trace)
        # trace_y_new[0] = trace[0][y_dim]
        # trace_y_new[int(len(trace)/2)] = trace[-1][y_dim]
        # for i in range(1, max_id, 2):
        #     if trace[i][y_dim] > trace[i+1][y_dim]:
        #         trace_y_new[i] = trace[i][y_dim]
        #         trace_y_new[max_id+1-i] = trace[i+1][y_dim]
        #     else:
        #         trace_y_new[i] = trace[i+1][y_dim]
        #         trace_y_new[max_id+1-i] = trace[i][y_dim]
        # trace_y_new=np.array(trace_y_new)
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_even[::-1].tolist(), y=trace_y_new, mode='lines',
        #                          fill='toself',
        #                          fillcolor=fillcolor,
        #                          opacity=0.5,
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=show_legend
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=fillcolor,
        #                          opacity=0.5,
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=show_legend
        #                          ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_even[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=fillcolor,
        #                          opacity=0.5,
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=show_legend))
        fig.add_trace(go.Scatter(x=trace_x_odd.tolist()+trace_x_even[::-1].tolist()+[trace_x_odd[0]], y=trace_y_odd.tolist()+trace_y_even[::-1].tolist()+[trace_y_odd[0]], mode='markers+lines',
                                 fill='toself',
                                 fillcolor=fillcolor,
                                 #  opacity=0.5,
                                 marker={'size': 1},
                                 line_color=colors[scheme_dict[color]][2],
                                 line={'width': 1},
                                 showlegend=show_legend
                                 ))
        # fig.add_trace(go.Scatter(x=trace_x_even.tolist()+trace_x_odd[::-1].tolist(), y=trace_y_odd.tolist()+trace_y_even[::-1].tolist(), mode='lines',
        #                          fill='toself',
        #                          fillcolor=fillcolor,
        #                          opacity=0.5,
        #                          line_color='rgba(255,255,255,0)',
        #                          showlegend=show_legend))
        queue += node.child

    # queue = [root]
    # while queue != []:
    #     node = queue.pop(0)
    #     traces = node.trace
    #     trace = np.array(traces[agent_id])
    #     max_id = len(trace)-1
    #     fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
    #                              mode='markers',
    #                              text=[
    #                                  ['{:.2f}'.format(trace[i, j])for j in print_dim_list] for i in range(0, trace.shape[0])],
    #                              line_color=colors[scheme_dict[color]][0],
    #                              marker={
    #         "sizemode": "area",
    #         "sizeref": 200000,
    #         "size": 2
    #     },
    #         name='lines',
    #         showlegend=False))
    #     queue += node.child
    return fig


def simulation_tree_single(root: Union[AnalysisTree, AnalysisTreeNode], agent_id, fig: go.Figure = go.Figure(), x_dim: int = 1, y_dim: int = 2, color=None, print_dim_list=None):
    if isinstance(root, AnalysisTree):
        root = root.root
    """It statically shows the simulation traces of one given agent."""
    global color_cnt
    queue = [root]
    color_id = 0
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt+1) % 12
    start_list = []
    end_list = []
    count_dict = {}
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        if agent_id not in traces.keys():
            return fig
        trace = np.array(traces[agent_id])
        start = list(trace[0])
        end = list(trace[-1])
        if (start in start_list) and (end in end_list):
            continue
        time = tuple([round(start[0], 2), round(end[0], 2)])
        if time in count_dict.keys():
            count_dict[time] += 1
        else:
            count_dict[time] = 1
        start_list.append(start)
        end_list.append(end)

        fig.add_trace(go.Scatter(x=trace[:, x_dim], y=trace[:, y_dim],
                                 mode='lines',
                                 line_color=colors[scheme_dict[color]
                                                   ][color_id],
                                 text=[
            ['{:.2f}'.format(trace[i, j])for j in print_dim_list] for i in range(trace.shape[0])],
            legendgroup=agent_id,
            legendgrouptitle_text=agent_id,
            name=str(round(start[0], 2))+'-'+str(round(end[0], 2)) +
            '-'+str(count_dict[time]),
            showlegend=True))
        if node.assert_hits != None and agent_id in node.assert_hits:
            fig.add_trace(go.Scatter(x=[trace[-1, x_dim]], y=[trace[-1, y_dim]],
                                     mode='markers+text',
                                     #  line_color='grey',
                                     text=['HIT:\n' +
                                           a for a in node.assert_hits[agent_id]],
                                     textfont={'color': 'grey'},
                                     legendgroup=agent_id,
                                     marker={'size': 4, 'color': 'black'},
                                     legendgrouptitle_text=agent_id,
                                     name=str(round(start[0], 2))+'-'+str(round(end[0], 2)) +
                                     '-'+str(count_dict[time])+'hit',
                                     showlegend=True))

        color_id = (color_id+4) % 5
        queue += node.child
    fig.update_layout(legend=dict(
        groupclick="toggleitem",
        itemclick="toggle",
        itemdoubleclick="toggleothers"
    ))
    return fig


def draw_map(map, color='rgba(0,0,0,1)', fig: go.Figure() = go.Figure(), fill_type='lines'):
    """It draws the the map"""
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    if map is None:
        return fig
    if fill_type == 'detailed':
        speed_dict = map.get_all_speed_limit()
        speed_list = list(filter(None, speed_dict.values()))
        speed_min = min(speed_list)
        speed_max = max(speed_list)
        start_color = [255, 255, 255, 0.2]
        end_color = [0, 0, 0, 0.2]
        curr_color = [0, 0, 0, 0]
    speed_limit = None
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
                if fill_type == 'lines' or (fill_type == 'detailed' and speed_limit is None):
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             showlegend=False,
                                             name='lines'))
                elif fill_type == 'detailed' and speed_limit is not None:
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             fillcolor='rgba' +
                                             str(tuple(curr_color)),
                                             showlegend=False,
                                             name='limit'))
                elif fill_type == 'fill':
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             fillcolor='rgba(0,0,0,0.1)',
                                             showlegend=False,
                                             # text=theta,
                                             name='lines'))
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(
                    start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100)
                r1 = lane_seg.radius - lane_seg.width/2
                x1 = (np.cos(phase_array)*r1 + lane_seg.center[0]).tolist()
                y1 = (np.sin(phase_array)*r1 + lane_seg.center[1]).tolist()
                r2 = lane_seg.radius + lane_seg.width/2
                x2 = (np.cos(phase_array)*r2 +
                      lane_seg.center[0]).tolist()[::-1]
                y2 = (np.sin(phase_array)*r2 +
                      lane_seg.center[1]).tolist()[::-1]
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
                                             name='lines'))
                elif fill_type == 'detailed' and speed_limit != None:
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             fillcolor='rgba' +
                                             str(tuple(curr_color)),
                                             showlegend=False,
                                             name='lines'))
                elif fill_type == 'fill':
                    fig.add_trace(go.Scatter(x=trace_x, y=trace_y,
                                             mode='lines',
                                             line_color=color,
                                             fill='toself',
                                             showlegend=False,
                                             name='lines'))
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    if fill_type == 'detailed':
        fig.add_trace(go.Scatter(x=[0], y=[0],
                                 mode='markers',
                                 marker=dict(
            symbol='square',
            size=16,
            cmax=speed_max,
            cmin=speed_min,
            color='rgba(0,0,0,0)',
            colorbar=dict(
                title="Speed Limit",
                orientation='h'
            ),
            colorscale=[
                [0, 'rgba'+str(tuple(start_color))], [1, 'rgba'+str(tuple(end_color))]]
        ),
            showlegend=False,
        ))
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def check_dim(num_dim: int, x_dim: int = 1, y_dim: int = 2, print_dim_list: List(int) = None):
    if x_dim < 0 or x_dim >= num_dim:
        raise ValueError(f'wrong x dimension value {x_dim}')
    if y_dim < 0 or y_dim >= num_dim:
        raise ValueError(f'wrong y dimension value {y_dim}')
    if print_dim_list is None:
        return True
    for i in print_dim_list:
        if y_dim < 0 or y_dim >= num_dim:
            raise ValueError(f'wrong printed dimension value {i}')
    return True


def create_anime_dict(duration=10):
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"]["xaxis"] = {
        "title": "x position"}
    fig_dict["layout"]["yaxis"] = {
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
    return fig_dict, sliders_dict


def get_text_pos(veh_mode):
    if veh_mode == 'Normal':
        text_pos = 'middle center'
        text = 'N'
    elif veh_mode == 'Brake':
        text_pos = 'middle left'
        text = 'B'
    elif veh_mode == 'Accelerate':
        text_pos = 'middle right'
        text = 'A'
    elif veh_mode == 'SwitchLeft':
        text_pos = 'top center'
        text = 'SL'
    elif veh_mode == 'SwitchRight':
        text_pos = 'bottom center'
        text = 'SR'
    elif veh_mode == 'Stop':
        text_pos = 'middle center'
        text = 'S'
    else:
        text_pos = 'middle center'
        text = veh_mode
    return text_pos, text
