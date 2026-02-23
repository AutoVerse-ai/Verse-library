'''
    This file contains plotter code simulations and reachtubes
'''

from __future__ import annotations
import copy
import time
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union
import numbers
from plotly.graph_objs.scatter import Marker
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map import LaneMap
import os

colors = [
    ["#CC0000", "#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"],  # red
    ["#0000CC", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF"],  # blue
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#CCFFCC"],  # green
    ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],  # yellow
    #   ['#66CC00', '#80FF00', '#99FF33', '#B2FF66', '#CCFF99', '#FFFFCC'], # yellowgreen
    ["#CC00CC", "#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF", "#FFCCFF"],  # magenta
    #   ['#00CC66', '#00FF80', '#33FF99', '#66FFB2', '#99FFCC', '#CCFFCC'], # springgreen
    ["#00CCCC", "#00FFFF", "#33FFFF", "#66FFFF", "#99FFFF", "#CCFFE5"],  # cyan
    #   ['#0066CC', '#0080FF', '#3399FF', '#66B2FF', '#99CCFF', '#CCE5FF'], # cyanblue
    ["#CC6600", "#FF8000", "#FF9933", "#FFB266", "#FFCC99", "#FFE5CC"],  # orange
    #   ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'], # purple
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],  # lime
    ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],  # pink
]
scheme_dict = {
    "red": 0,
    "orange": 1,
    "green": 2,
    "yellow": 3,
    "yellowgreen": 4,
    "lime": 5,
    "springgreen": 6,
    "cyan": 7,
    "cyanblue": 8,
    "blue": 9,
    "purple": 10,
    "magenta": 11,
    "pink": 12,
}
scheme_list = list(scheme_dict.keys())
num_theme = len(colors)
color_cnt = 0
text_size = 8
marker_size = 4
scale_factor = 0.25
mode_point_color = "rgba(0,0,0,0.5)"
mode_text_color = "black"
duration = 1


'''The next 4 functions are the main functions to be used by users'''


def simulation_tree(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    plot_color=None,
):
    '''This function adds the traces of a simulation as a 2D plot to a plotly graph object.
        Parameters:
            root (Union[AnalysisTree, AnalysisTreeNode]): Root of the simulation tree to be plotted.
            map (Map): Map to be plotted in the background.
            fig (plotly.graph_objects): Input figure object in which the plot is added.
            x_dim (int): The state dimension to be plotted along x-axis.
            y_dim (int): The state dimension to be plotted in y-axis.
            scale_type (string): Only allowed value is "trace".
            label_mode (string): Only allowed value is "None".
        Returns:
            fig (plotly.graph_objects): Figure which includes the plots.
    '''
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    # scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = simulation_tree_single(
            root, agent_id, fig, x_dim, y_dim, scheme_list[i], print_dim_list, plot_color
        )
        i = (i + 1) % num_theme
    if scale_type == "trace":
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = "middle center"
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == "trace":
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            mode_point_color = colors[agent_list.index(agent_id) % num_theme][0]
            if label_mode != "None":
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[0, x_dim]],
                            y=[trace[0, y_dim]],
                            mode="markers+text",
                            line_color=mode_point_color,
                            text=str(agent_id) + ": " + text,
                            opacity=0.5,
                            textposition=text_pos,
                            marker={"size": marker_size, "color": mode_text_color},
                            showlegend=False,
                        )
                    )
                    previous_mode[agent_id] = node.mode[agent_id]
            if node.assert_hits != None and agent_id in node.assert_hits:
                fig.add_trace(
                    go.Scatter(
                        x=[trace[-1, x_dim]],
                        y=[trace[-1, y_dim]],
                        mode="markers+text",
                        text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                        # textfont={"color": "grey"},
                        marker={"size": marker_size, "color": "black"},
                        #  legendgroup=agent_id,
                        #  legendgrouptitle_text=agent_id,
                        #  name=str(round(start[0], 2))+'-'+str(round(end[0], 2)) +
                        #  '-'+str(count_dict[time])+'hit',
                        showlegend=False,
                    )
                )
        queue += node.child
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    # fig.update_xaxes(title='x')
    # fig.update_yaxes(title='y')
    fig.update_layout(legend_title_text="Agent list")
    fig = update_style(fig)
    return fig


def simulation_anime(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    time_step=None,
    speed_rate=1,
    anime_mode="normal",
    full_trace=False,
):
    """Normal: It gives the animation of the simulation without trail but is faster."""
    """Trail: It gives the animation of the simulation with trail."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if time_step != None:
        num_digit = num_digits(time_step)
    else:
        num_digit = 3
    org_root = copy.deepcopy(root)
    root = sample_trace(root, sample_rate)
    timed_point_dict = {}
    queue = [root]
    x_min, x_max = float("inf"), -float("inf")
    y_min, y_max = float("inf"), -float("inf")
    # input check
    num_dim = np.array(root.trace[list(root.agent.keys())[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)
    agent_list = list(root.agent.keys())
    num_points = 0
    while queue != []:
        node = queue.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            for i in range(len(trace)):
                x_min = min(x_min, trace[i][x_dim])
                x_max = max(x_max, trace[i][x_dim])
                y_min = min(y_min, trace[i][y_dim])
                y_max = max(y_max, trace[i][y_dim])
                time_point = round(trace[i][0], num_digit)
                tmp_trace = trace[i][0:].tolist()
                if time_point not in timed_point_dict:
                    num_points += 1
                    timed_point_dict[time_point] = {agent_id: [tmp_trace]}
                else:
                    if agent_id not in timed_point_dict[time_point].keys():
                        timed_point_dict[time_point][agent_id] = [tmp_trace]
                    elif tmp_trace not in timed_point_dict[time_point][agent_id]:
                        timed_point_dict[time_point][agent_id].append(tmp_trace)
        queue += node.child
    duration = int(5000 / num_points / speed_rate)
    fig_dict, sliders_dict = create_anime_dict(duration)
    # used for trail mode
    time_list = list(timed_point_dict.keys())
    agent_list = list(root.agent.keys())
    trail_limit = min(10, len(time_list))
    trail_len = trail_limit
    opacity_step = 1 / trail_len
    size_step = 2 / trail_len
    min_size = 5
    step = 2

    if anime_mode == "normal":
        # make data
        trace_dict = timed_point_dict[0]
        for agent_id, trace_list in trace_dict.items():
            color = colors[agent_list.index(agent_id) % num_theme][1]
            x_list = []
            y_list = []
            text_list = []
            branch_cnt = 0
            for trace in trace_list:
                x_list.append(trace[x_dim])
                y_list.append(trace[y_dim])
                text_list.append(["{:.2f}".format(trace[i]) for i in print_dim_list])
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
                "showlegend": False,
            }
            fig_dict["data"].append(data_dict)
        # make frames
        for time_point in timed_point_dict:
            frame = {"data": [], "layout": {"annotations": []}, "name": time_point}
            point_list = timed_point_dict[time_point]
            for agent_id, trace_list in point_list.items():
                color = colors[agent_list.index(agent_id) % num_theme][1]
                x_list = []
                y_list = []
                text_list = []
                branch_cnt = 0
                for trace in trace_list:
                    x_list.append(trace[x_dim])
                    y_list.append(trace[y_dim])
                    text_list.append(["{:.2f}".format(trace[i]) for i in print_dim_list])
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
                    "showlegend": False,
                }
                frame["data"].append(data_dict)
            fig_dict["frames"].append(frame)
            slider_step = {
                "args": [
                    [time_point],
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": duration},
                    },
                ],
                "label": time_point,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]
    else:
        # make data
        trace_dict = timed_point_dict[0]
        for time_point in list(timed_point_dict.keys())[0 : int(trail_limit / step)]:
            trace_dict = timed_point_dict[time_point]
            for agent_id, point_list in trace_dict.items():
                x_list = []
                y_list = []
                text_list = []
                for point in point_list:
                    x_list.append(point[x_dim])
                    y_list.append(point[y_dim])
                    text_list.append(["{:.2f}".format(point[i]) for i in print_dim_list])
                data_dict = {
                    "x": x_list,
                    "y": y_list,
                    "mode": "markers",
                    "text": text_list,
                    "textfont": dict(size=text_size, color="black"),
                    "visible": False,
                    "textposition": "bottom center",
                    "name": agent_id,
                }
                fig_dict["data"].append(data_dict)
        # make frames
        for time_point_id in range(trail_limit, len(time_list)):
            time_point = time_list[time_point_id]
            frame = {"data": [], "layout": {"annotations": []}, "name": time_point}
            for agent_id in agent_list:
                color = colors[agent_list.index(agent_id) % num_theme][1]
                for id in range(0, trail_len, step):
                    tmp_point_list = timed_point_dict[time_list[time_point_id - id]][agent_id]
                    trace_x = []
                    trace_y = []
                    text_list = []
                    for point in tmp_point_list:
                        trace_x.append(point[x_dim])
                        trace_y.append(point[y_dim])
                        text_list.append(["{:.2f}".format(point[i]) for i in print_dim_list])
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
                                "opacity": opacity_step * (trail_len - id),
                                "size": min_size + size_step * (trail_len - id),
                            },
                            "name": agent_id,
                            "showlegend": False,
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
                                "opacity": opacity_step * (trail_len - id),
                                "size": min_size + size_step * (trail_len - id),
                            },
                            "name": agent_id,
                            "showlegend": False,
                        }
                    frame["data"].append(data_dict)

            fig_dict["frames"].append(frame)
            slider_step = {
                "args": [
                    [time_point],
                    {
                        "frame": {"duration": duration, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": duration},
                    },
                ],
                "label": time_point,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = draw_map(map, "rgba(0,0,0,1)", fig, map_type)
    i = 0
    queue = [root]
    previous_mode = {}
    agent_list = list(root.agent.keys())
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = "middle center"
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            trace_y = trace[:, y_dim].tolist()
            trace_x = trace[:, x_dim].tolist()
            i = agent_list.index(agent_id)
            mode_point_color = colors[agent_list.index(agent_id) % num_theme][0]
            if label_mode != "None":
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[0, x_dim]],
                            y=[trace[0, y_dim]],
                            mode="markers+text",
                            line_color=mode_point_color,
                            text=str(agent_id) + ": " + text,
                            opacity=0.5,
                            textposition=text_pos,
                            marker={"size": marker_size, "color": mode_text_color},
                            showlegend=False,
                        )
                    )
                    previous_mode[agent_id] = node.mode[agent_id]
                if node.assert_hits != None and agent_id in node.assert_hits:
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[-1, x_dim]],
                            y=[trace[-1, y_dim]],
                            mode="markers+text",
                            text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                            # textfont={"color": "grey"},
                            marker={"size": marker_size, "color": "black"},
                            showlegend=False,
                        )
                    )
        queue += node.child
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    fig.update_layout(legend_title_text="Agent list")
    if full_trace == True:
        fig = simulation_tree(
            org_root,
            map,
            fig,
            x_dim,
            y_dim,
            print_dim_list,
            map_type,
            scale_type,
            label_mode,
            sample_rate,
        )
    fig = update_style(fig)
    return fig


def reachtube_tree(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    combine_rect=1,
    plot_color=None,
):
    """It statically shows all the traces of the verfication."""
    if plot_color is None:
        plot_color = colors
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    # scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single(
            root,
            agent_id,
            fig,
            x_dim,
            y_dim,
            scheme_list[i],
            print_dim_list,
            combine_rect,
            plot_color=plot_color,
        )
        i = (i + 1) % num_theme
    if scale_type == "trace":
        queue = [root]
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = "middle center"
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print({k: len(v) for k, v in traces.items()})
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == "trace":
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            i = agent_list.index(agent_id)
            if label_mode != "None":
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    mode_point_color = plot_color[agent_list.index(agent_id) % num_theme][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[0, x_dim]],
                            y=[trace[0, y_dim]],
                            mode="markers+text",
                            line_color=mode_point_color,
                            opacity=0.5,
                            text=str(agent_id) + ": " + text,
                            textposition=text_pos,
                            marker={"size": marker_size, "color": mode_text_color},
                            showlegend=False,
                        )
                    )
                    previous_mode[agent_id] = node.mode[agent_id]
                if node.assert_hits != None and agent_id in node.assert_hits:
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[-1, x_dim]],
                            y=[trace[-1, y_dim]],
                            mode="markers+text",
                            text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                            # textfont={"color": "grey"},
                            marker={"size": marker_size, "color": "black"},
                            showlegend=False,
                        )
                    )
        queue += node.child
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    fig = update_style(fig)
    return fig


def reachtube_anime(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    time_step=None,
    speed_rate=1,
    combine_rect=None,
):
    """It gives the animation of the verfication."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if time_step != None:
        num_digit = num_digits(time_step)
    else:
        num_digit = 3
    root = sample_trace(root, sample_rate)
    agent_list = list(root.agent.keys())
    timed_point_dict = {}
    queue = [root]
    x_min, x_max = float("inf"), -float("inf")
    y_min, y_max = float("inf"), -float("inf")
    # input check
    num_dim = np.array(root.trace[list(root.agent.keys())[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)
    # scheme_list = list(scheme_dict.keys())
    num_points = 0
    while queue != []:
        node = queue.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if trace[0][0] > 0:
                trace = trace[8:]
            for i in range(0, len(trace) - 1, 2):
                x_min = min(x_min, trace[i][x_dim])
                x_max = max(x_max, trace[i][x_dim])
                y_min = min(y_min, trace[i][y_dim])
                y_max = max(y_max, trace[i][y_dim])
                time_point = round(trace[i][0], num_digit)
                rect = [trace[i][0:].tolist(), trace[i + 1][0:].tolist()]
                if time_point not in timed_point_dict:
                    num_points += 1
                    timed_point_dict[time_point] = {agent_id: [rect]}
                else:
                    if agent_id in timed_point_dict[time_point].keys():
                        timed_point_dict[time_point][agent_id].append(rect)
                    else:
                        timed_point_dict[time_point][agent_id] = [rect]

        queue += node.child
    duration = int(5000 / num_points / speed_rate)
    fig_dict, sliders_dict = create_anime_dict(duration)
    for time_point in timed_point_dict:
        frame = {"data": [], "layout": {"annotations": [], "shapes": []}, "name": time_point}
        agent_dict = timed_point_dict[time_point]
        for agent_id, rect_list in agent_dict.items():
            for rect in rect_list:
                shape_dict = {
                    "type": "rect",
                    "x0": rect[0][x_dim],
                    "y0": rect[0][y_dim],
                    "x1": rect[1][x_dim],
                    "y1": rect[1][y_dim],
                    "fillcolor": "rgba(0,0,0,0.7)",
                    "line": dict(color="rgba(0,0,0,0.7)", width=5),
                    "visible": True,
                }
                frame["layout"]["shapes"].append(shape_dict)

        fig_dict["frames"].append(frame)
        slider_step = {
            "args": [
                [time_point],
                {
                    "frame": {"duration": duration, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": time_point,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single(
            root,
            agent_id,
            fig,
            x_dim,
            y_dim,
            scheme_list[i],
            print_dim_list,
            combine_rect=combine_rect,
        )
        i = (i + 1) % num_theme
    if scale_type == "trace":
        queue = [root]
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = "middle center"
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == "trace":
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            if label_mode != "None":
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    x0 = trace[0, x_dim]
                    x1 = trace[1, x_dim]
                    y0 = trace[0, y_dim]
                    y1 = trace[1, y_dim]
                    mode_point_color = colors[agent_list.index(agent_id) % num_theme][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[(x0 + x1) / 2],
                            y=[(y0 + y1) / 2],
                            mode="markers+text",
                            line_color=mode_point_color,
                            text=str(agent_id) + ": " + text,
                            textposition=text_pos,
                            opacity=0.5,
                            marker={"size": marker_size, "color": mode_text_color},
                            showlegend=False,
                        )
                    )
                    previous_mode[agent_id] = node.mode[agent_id]
                if node.assert_hits != None and agent_id in node.assert_hits:
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[-1, x_dim]],
                            y=[trace[-1, y_dim]],
                            mode="markers+text",
                            text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                            # textfont={"color": "grey"},
                            marker={"size": marker_size, "color": "black"},
                            showlegend=False,
                        )
                    )
        queue += node.child
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    fig = update_style(fig)
    return fig


def reachtube_tree_video(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    combine_rect=1,
    plot_color=None,
    time_step=None,
    speed_rate=1,
    output_path="reachtube_animation.html",
    max_slider_steps=100,
    max_frame_steps=None,
    video_config=None,
):
    """Build an export-oriented reachtube animation.

    This function aggregates reachsets across all tree nodes by rounded
    time and creates cumulative shape frames (frame at time ``t`` contains all rects
    for times ``<= t``). The resulting figure can be exported to HTML, GIF, or MP4.

    Parameters
    ----------
    root : AnalysisTree or AnalysisTreeNode
        Verification tree root.
    map, fig, x_dim, y_dim, print_dim_list, map_type, scale_type, label_mode,
    sample_rate, combine_rect, plot_color
        Same as reachtube_tree().
    time_step : float or None
        Time rounding precision source; None uses 3 decimal digits.
    speed_rate : float
        Playback speed factor (higher is faster).
    output_path : str or None
        Output path. Defaults to ``reachtube_animation.html``.
        ``.html`` writes an animation page; ``.gif``/``.mp4`` export rendered frames.
    max_slider_steps : int or None
        Maximum slider labels shown.
    max_frame_steps : int or None
        Optional cap on animation frame count. If None, HTML defaults to 150
        frames for faster browser loading, while GIF/MP4 keep all timesteps.
    video_config : dict or None
                Optional styling overrides for export output.

                For ``.html`` output:
                - ``layout`` (dict): passed to ``fig.update_layout(**layout)``.
                - ``xaxis``/``xaxes``/``x_axis``/``x_axes`` (dict): passed to Plotly x-axis layout.
                - ``yaxis``/``yaxes``/``y_axis``/``y_axes`` (dict): passed to Plotly y-axis layout.
                - top-level ``paper_bgcolor``, ``plot_bgcolor``, ``width``, ``height``,
                    ``margin``, ``font``: passed to ``fig.update_layout``.

                For ``.gif``/``.mp4`` direct export:
                - ``layout`` plus axis aliases above are read and interpreted by the rasterizer.
                - supported visual keys: ``paper_bgcolor``, ``plot_bgcolor``, ``width``,
                    ``height``, ``margin``, ``font.size``, ``xaxis.title``, ``yaxis.title``,
                    ``xaxis.range``, ``yaxis.range``, ``xaxis/yaxis.showline``, ``linewidth``,
                    ``linecolor``, ``showgrid``, ``gridwidth``, ``gridcolor``.
                - unsupported keys are ignored in direct export.

    Returns
    -------
    go.Figure
        Figure with cumulative animation frames.
    """
    # NOTE: Normalize root, time rounding, and sample the tree 
    if plot_color is None:
        plot_color = colors
    if isinstance(root, AnalysisTree):
        root = root.root
    if time_step is not None:
        num_digit = num_digits(time_step)
        if num_digit is False:
            num_digit = 3
    else:
        num_digit = 3
    root = sample_trace(root, sample_rate)
    agent_list = list(root.agent.keys())
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    output_ext = ""
    output_path = output_path.strip()
    if output_path:
        output_path = str(output_path)
        output_ext = os.path.splitext(output_path)[1].lower()

    effective_max_frame_steps = max_frame_steps
    if effective_max_frame_steps is None and output_ext == ".html":
        effective_max_frame_steps = 150

    # NOTE: Build time -> [(agent_id, rect), ...] and axis bounds 
    # Each rect is [lower_state, upper_state]; we store (agent_id, rect) so we
    # can assign per-agent colors when building frames.
    timed_point_dict = {}
    queue = [root]
    x_min, x_max = float("inf"), -float("inf")
    y_min, y_max = float("inf"), -float("inf")
    while queue:
        node = queue.pop(0)
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if len(trace) < 2:
                continue
            for i in range(0, len(trace) - 1, 2):
                x_min = min(x_min, trace[i][x_dim], trace[i + 1][x_dim]) # FIXME: bad code pattern from past plotting functions -- should instead make sure that reachtube format is correct so that even entries always min and odd entries always max
                x_max = max(x_max, trace[i][x_dim], trace[i + 1][x_dim])
                y_min = min(y_min, trace[i][y_dim], trace[i + 1][y_dim])
                y_max = max(y_max, trace[i][y_dim], trace[i + 1][y_dim])
                time_point = round(trace[i][0], num_digit)
                rect = [trace[i][0:].tolist(), trace[i + 1][0:].tolist()]
                if time_point not in timed_point_dict:
                    timed_point_dict[time_point] = []
                timed_point_dict[time_point].append((agent_id, rect))
        queue += node.child

    sorted_times = sorted(timed_point_dict.keys())
    if not sorted_times:
        fig = draw_map(map=map, fig=fig, fill_type=map_type)
        fig = update_style(fig)
        return fig

    if effective_max_frame_steps is None:
        frame_times = sorted_times
    elif len(sorted_times) > effective_max_frame_steps:
        indices = np.linspace(0, len(sorted_times) - 1, effective_max_frame_steps, dtype=int)
        frame_times = [sorted_times[i] for i in indices]
    else:
        frame_times = sorted_times

    if max_slider_steps is None:
        slider_times = frame_times
    elif len(frame_times) > max_slider_steps:
        indices = np.linspace(0, len(frame_times) - 1, max_slider_steps, dtype=int)
        slider_times = [frame_times[i] for i in indices]
    else:
        slider_times = frame_times

    duration = max(1, int(5000 / len(frame_times) / speed_rate))

    if output_ext in [".gif", ".mp4"]:
        _export_reachtube_video_direct(
            timed_point_dict=timed_point_dict,
            sorted_times=sorted_times,
            frame_times=frame_times,
            agent_list=agent_list,
            plot_color=plot_color,
            map=map,
            map_type=map_type,
            scale_type=scale_type,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_dim=x_dim,
            y_dim=y_dim,
            output_path=output_path,
            duration=duration,
            video_config=video_config,
        )
        return fig # NOTE: fig is not modified by reachtube_tree_video in this instance

    fig_dict, sliders_dict = create_anime_dict(duration)

    all_rects_data = []
    time_to_trace_indices = {}

    for time_point in sorted_times:
        if time_point not in time_to_trace_indices:
            time_to_trace_indices[time_point] = []
        for _, (agent_id, rect) in enumerate(timed_point_dict[time_point]):
            color_idx = agent_list.index(agent_id) % len(plot_color)
            linecolor = plot_color[color_idx][0]
            fillcolor = plot_color[color_idx][1]

            rect_trace = {
                "x": [rect[0][x_dim], rect[1][x_dim], rect[1][x_dim], rect[0][x_dim], rect[0][x_dim]],
                "y": [rect[0][y_dim], rect[0][y_dim], rect[1][y_dim], rect[1][y_dim], rect[0][y_dim]],
                "mode": "lines",
                "fill": "toself",
                "fillcolor": fillcolor,
                "line": {"color": linecolor, "width": 1},
                "visible": False,
                "showlegend": False,
                "hoverinfo": "none",
            }
            trace_idx = len(all_rects_data)
            all_rects_data.append(rect_trace)
            time_to_trace_indices[time_point].append(trace_idx)

    fig_dict["data"] = all_rects_data

    visible = [False] * len(all_rects_data)
    cursor = 0
    for time_point in frame_times:
        while cursor < len(sorted_times) and sorted_times[cursor] <= time_point:
            for trace_idx in time_to_trace_indices[sorted_times[cursor]]:
                visible[trace_idx] = True
            cursor += 1

        frame_data = [{"visible": v} for v in visible]
        frame = {
            "data": frame_data,
            "name": str(time_point)
        }
        fig_dict["frames"].append(frame)

    slider_time_set = set(slider_times)
    for time_point in frame_times:
        if time_point not in slider_time_set:
            continue
        slider_step = {
            "args": [
                [str(time_point)],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(time_point),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    if fig_dict["frames"]:
        first_frame_data = fig_dict["frames"][0]["data"]
        for idx, vis_update in enumerate(first_frame_data):
            fig_dict["data"][idx]["visible"] = vis_update.get("visible", False)

    fig = go.Figure(fig_dict)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    fig = update_style(fig)
    fig = _apply_video_config_to_plotly_fig(fig, video_config)

    if output_path:
        output_path = str(output_path)
        if output_path.lower().endswith(".html"):
            fig.write_html(output_path, auto_open=not os.path.exists('/.dockerenv'))
        # elif output_path.lower().endswith(".gif") or output_path.lower().endswith(".mp4"): # should never be hit
        #     _export_animation_video(fig, output_path, duration, len(frame_times))
        else:
            raise Exception(f'Unexpected output extension: {output_path.lower().split(".")[-1]}')

    return fig


def _apply_video_config_to_plotly_fig(fig, video_config):
    if not isinstance(video_config, dict):
        return fig

    layout_cfg = video_config.get("layout")
    if isinstance(layout_cfg, dict):
        fig.update_layout(**layout_cfg)

    xaxis_cfg = None
    for x_key in ["xaxis", "xaxes", "x_axis", "x_axes"]:
        if x_key in video_config:
            xaxis_cfg = video_config[x_key]
            break
    if isinstance(xaxis_cfg, dict):
        fig.update_layout(xaxis=xaxis_cfg)

    yaxis_cfg = None
    for y_key in ["yaxis", "yaxes", "y_axis", "y_axes"]:
        if y_key in video_config:
            yaxis_cfg = video_config[y_key]
            break
    if isinstance(yaxis_cfg, dict):
        fig.update_layout(yaxis=yaxis_cfg)

    top_layout_overrides = {}
    for top_key in ["paper_bgcolor", "plot_bgcolor", "width", "height", "margin", "font"]:
        if top_key in video_config:
            top_layout_overrides[top_key] = video_config[top_key]
    if top_layout_overrides:
        fig.update_layout(**top_layout_overrides)

    return fig

def _export_reachtube_video_direct(
    timed_point_dict,
    sorted_times,
    frame_times,
    agent_list,
    plot_color,
    map,
    map_type,
    scale_type,
    x_min,
    x_max,
    y_min,
    y_max,
    x_dim,
    y_dim,
    output_path,
    duration,
    video_config=None,
):
    """Export reachtube animation directly to GIF/MP4 without Plotly frame rendering.

    This path rasterizes map/axes/rectangles directly and writes frames incrementally,
    avoiding Plotly/Kaleido ``to_image`` per-frame overhead.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("GIF/MP4 export requires imageio. Install with: pip install imageio")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Direct GIF/MP4 export requires Pillow. Install with: pip install pillow")

    ext = output_path.lower().split(".")[-1]
    frame_duration_sec = duration / 1000.0
    total_frames = len(frame_times)
    if total_frames == 0:
        return
    export_start = time.perf_counter()
    print(
        f"[export] start direct export: times={len(sorted_times)} sampled_frames={total_frames}"
    )

    rect_specs = []
    time_to_trace_indices = {}
    agent_idx = {agent_id: idx for idx, agent_id in enumerate(agent_list)}

    def _hex_to_rgb(color: str):
        value = color.strip().lstrip("#")
        if len(value) != 6:
            return (0, 0, 255)
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))

    def _to_px(x_val, y_val, x_lo, x_hi, y_lo, y_hi, left, top, plot_w, plot_h):
        x_den = (x_hi - x_lo) if (x_hi - x_lo) != 0 else 1.0
        y_den = (y_hi - y_lo) if (y_hi - y_lo) != 0 else 1.0
        px = left + (x_val - x_lo) * (plot_w - 1) / x_den
        py = top + (plot_h - 1) - (y_val - y_lo) * (plot_h - 1) / y_den
        return int(round(px)), int(round(py))

    def _get_cfg(cfg, *keys, default=None):
        if not isinstance(cfg, dict):
            return default
        for key in keys:
            if key in cfg:
                return cfg[key]
        return default

    def _normalize_axis_title(title_cfg, fallback):
        if isinstance(title_cfg, str):
            return title_cfg
        if isinstance(title_cfg, dict):
            title_text = title_cfg.get("text")
            if title_text is not None:
                return str(title_text)
        return fallback

    def _coerce_int(value, default):
        if isinstance(value, numbers.Real):
            try:
                return int(value)
            except Exception:
                return default
        return default

    def _coerce_bool(value, default):
        if isinstance(value, bool):
            return value
        return default

    def _merge_style(default_style, overrides):
        merged = dict(default_style)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_style(merged[key], value)
            else:
                merged[key] = value
        return merged

    style_defaults = {
        "width": 1280,
        "height": 720,
        "margin": {"l": 100, "r": 40, "t": 40, "b": 80},
        "paper_bgcolor": "rgba(255,255,255,1)",
        "plot_bgcolor": "rgba(255,255,255,1)",
        "font": {"size": 32},
        "xaxis": {
            "title": "x position",
            "showline": True,
            "linewidth": 4,
            "linecolor": "Gray",
            "showgrid": False,
            "gridwidth": 2,
            "gridcolor": "LightGrey",
        },
        "yaxis": {
            "title": "y position",
            "showline": True,
            "linewidth": 4,
            "linecolor": "Gray",
            "showgrid": False,
            "gridwidth": 2,
            "gridcolor": "LightGrey",
        },
    }

    user_style = {}
    if isinstance(video_config, dict):
        layout_cfg = _get_cfg(video_config, "layout", default={})
        if isinstance(layout_cfg, dict):
            user_style = _merge_style(user_style, layout_cfg)

        xaxis_cfg = _get_cfg(video_config, "xaxis", "xaxes", "x_axis", "x_axes", default=None)
        yaxis_cfg = _get_cfg(video_config, "yaxis", "yaxes", "y_axis", "y_axes", default=None)
        if isinstance(xaxis_cfg, dict):
            user_style["xaxis"] = _merge_style(user_style.get("xaxis", {}), xaxis_cfg)
        if isinstance(yaxis_cfg, dict):
            user_style["yaxis"] = _merge_style(user_style.get("yaxis", {}), yaxis_cfg)

        for top_key in ["paper_bgcolor", "plot_bgcolor", "width", "height", "margin", "font"]:
            if top_key in video_config:
                user_style[top_key] = video_config[top_key]

    style = _merge_style(style_defaults, user_style)

    for time_point in sorted_times:
        if time_point not in time_to_trace_indices:
            time_to_trace_indices[time_point] = []
        for agent_id, rect in timed_point_dict[time_point]:
            color_idx = agent_idx[agent_id] % len(plot_color)
            linecolor = plot_color[color_idx][0]
            fillcolor = plot_color[color_idx][1]

            trace_idx = len(rect_specs)
            rect_specs.append(
                {
                    "x0": min(rect[0][x_dim], rect[1][x_dim]),
                    "y0": min(rect[0][y_dim], rect[1][y_dim]),
                    "x1": max(rect[0][x_dim], rect[1][x_dim]),
                    "y1": max(rect[0][y_dim], rect[1][y_dim]),
                    "line_rgb": _hex_to_rgb(linecolor),
                    "fill_rgb": _hex_to_rgb(fillcolor),
                }
            )
            time_to_trace_indices[time_point].append(trace_idx)

    prep_rects_time = time.perf_counter()
    print(
        f"[export] prepared traces: count={len(rect_specs)} "
        f"elapsed={prep_rects_time - export_start:.2f}s"
    )

    frame_new_trace_indices = []
    cursor = 0
    for time_point in frame_times:
        new_indices = []
        while cursor < len(sorted_times) and sorted_times[cursor] <= time_point:
            new_indices.extend(time_to_trace_indices[sorted_times[cursor]])
            cursor += 1
        frame_new_trace_indices.append(new_indices)

    prep_frames_time = time.perf_counter()
    print(
        f"[export] prepared frame deltas: elapsed={prep_frames_time - prep_rects_time:.2f}s"
    )

    width = _coerce_int(style.get("width"), 1280)
    height = _coerce_int(style.get("height"), 720)
    margin_cfg = style.get("margin", {}) if isinstance(style.get("margin"), dict) else {}
    left_pad = _coerce_int(margin_cfg.get("l", margin_cfg.get("left", 100)), 100)
    right_pad = _coerce_int(margin_cfg.get("r", margin_cfg.get("right", 40)), 40)
    top_pad = _coerce_int(margin_cfg.get("t", margin_cfg.get("top", 40)), 40)
    bottom_pad = _coerce_int(margin_cfg.get("b", margin_cfg.get("bottom", 80)), 80)
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    if scale_type == "trace":
        x_lo = x_min - scale_factor * (x_max - x_min)
        x_hi = x_max + scale_factor * (x_max - x_min)
        y_lo = y_min - scale_factor * (y_max - y_min)
        y_hi = y_max + scale_factor * (y_max - y_min)
    else:
        x_lo, x_hi, y_lo, y_hi = x_min, x_max, y_min, y_max

    xaxis_cfg = style.get("xaxis", {}) if isinstance(style.get("xaxis"), dict) else {}
    yaxis_cfg = style.get("yaxis", {}) if isinstance(style.get("yaxis"), dict) else {}

    x_range = xaxis_cfg.get("range")
    y_range = yaxis_cfg.get("range")
    if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
    if isinstance(y_range, (list, tuple)) and len(y_range) == 2:
        y_lo, y_hi = float(y_range[0]), float(y_range[1])

    if x_hi == x_lo:
        x_hi = x_lo + 1.0
    if y_hi == y_lo:
        y_hi = y_lo + 1.0

    from PIL import ImageColor

    def _to_rgba(color_value, default_rgba=(255, 255, 255, 255)):
        if color_value is None:
            return default_rgba
        if isinstance(color_value, (tuple, list)):
            if len(color_value) == 4:
                return tuple(int(c) for c in color_value)
            if len(color_value) == 3:
                return (int(color_value[0]), int(color_value[1]), int(color_value[2]), 255)
        if isinstance(color_value, str):
            color_str = color_value.strip()
            if color_str.startswith("rgba"):
                vals = color_str[color_str.find("(") + 1 : color_str.find(")")].split(",")
                if len(vals) == 4:
                    try:
                        r = int(float(vals[0]))
                        g = int(float(vals[1]))
                        b = int(float(vals[2]))
                        a_raw = float(vals[3])
                        a = int(255 * a_raw) if a_raw <= 1 else int(a_raw)
                        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, min(255, a)))
                    except Exception:
                        return default_rgba
            if color_str.startswith("rgb"):
                vals = color_str[color_str.find("(") + 1 : color_str.find(")")].split(",")
                if len(vals) == 3:
                    try:
                        r = int(float(vals[0]))
                        g = int(float(vals[1]))
                        b = int(float(vals[2]))
                        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)
                    except Exception:
                        return default_rgba
            try:
                rgb = ImageColor.getrgb(color_str)
                if isinstance(rgb, tuple) and len(rgb) == 3:
                    return (rgb[0], rgb[1], rgb[2], 255)
            except Exception:
                return default_rgba
        return default_rgba

    paper_bg = _to_rgba(style.get("paper_bgcolor"), default_rgba=(255, 255, 255, 255))
    plot_bg = _to_rgba(style.get("plot_bgcolor"), default_rgba=(255, 255, 255, 255))
    x_line_color = _to_rgba(xaxis_cfg.get("linecolor", "Gray"), default_rgba=(128, 128, 128, 255))
    y_line_color = _to_rgba(yaxis_cfg.get("linecolor", "Gray"), default_rgba=(128, 128, 128, 255))
    x_grid_color = _to_rgba(xaxis_cfg.get("gridcolor", "LightGrey"), default_rgba=(211, 211, 211, 255))
    y_grid_color = _to_rgba(yaxis_cfg.get("gridcolor", "LightGrey"), default_rgba=(211, 211, 211, 255))

    font_cfg = style.get("font", {}) if isinstance(style.get("font"), dict) else {}
    base_font_size = max(12, _coerce_int(font_cfg.get("size", 32), 32))
    tick_font_size = max(14, int(round(base_font_size * 0.6)))
    axis_title_font_size = max(16, int(round(base_font_size * 0.9)))

    def _load_font(size):
        for font_name in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    tick_font = _load_font(tick_font_size)
    axis_font = _load_font(axis_title_font_size)

    def _text_size(draw_obj, text, font_obj):
        bbox = draw_obj.textbbox((0, 0), str(text), font=font_obj)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    bg = Image.new("RGBA", (width, height), paper_bg)
    bg_draw = ImageDraw.Draw(bg, "RGBA")
    bg_draw.rectangle(
        [left_pad, top_pad, left_pad + plot_w, top_pad + plot_h],
        fill=plot_bg,
    )

    x_show_grid = _coerce_bool(xaxis_cfg.get("showgrid"), False)
    y_show_grid = _coerce_bool(yaxis_cfg.get("showgrid"), False)
    x_grid_width = max(1, _coerce_int(xaxis_cfg.get("gridwidth", 2), 2))
    y_grid_width = max(1, _coerce_int(yaxis_cfg.get("gridwidth", 2), 2))
    num_ticks = 6
    if x_show_grid:
        for tidx in range(1, num_ticks - 1):
            xv = left_pad + int(round(tidx * plot_w / (num_ticks - 1)))
            bg_draw.line([(xv, top_pad), (xv, top_pad + plot_h)], fill=x_grid_color, width=x_grid_width)
    if y_show_grid:
        for tidx in range(1, num_ticks - 1):
            yv = top_pad + int(round(tidx * plot_h / (num_ticks - 1)))
            bg_draw.line([(left_pad, yv), (left_pad + plot_w, yv)], fill=y_grid_color, width=y_grid_width)

    x_show_line = _coerce_bool(xaxis_cfg.get("showline"), True)
    y_show_line = _coerce_bool(yaxis_cfg.get("showline"), True)
    x_line_width = max(1, _coerce_int(xaxis_cfg.get("linewidth", 4), 4))
    y_line_width = max(1, _coerce_int(yaxis_cfg.get("linewidth", 4), 4))
    if x_show_line:
        bg_draw.line(
            [(left_pad, top_pad + plot_h), (left_pad + plot_w, top_pad + plot_h)],
            fill=x_line_color,
            width=x_line_width,
        )
    if y_show_line:
        bg_draw.line(
            [(left_pad, top_pad), (left_pad, top_pad + plot_h)],
            fill=y_line_color,
            width=y_line_width,
        )

    tick_color = _to_rgba("black", default_rgba=(0, 0, 0, 255))
    for tidx in range(num_ticks):
        xv = left_pad + int(round(tidx * plot_w / (num_ticks - 1)))
        xval = x_lo + (x_hi - x_lo) * tidx / (num_ticks - 1)
        xlbl = str(int(round(xval)))
        xlbl_w, xlbl_h = _text_size(bg_draw, xlbl, tick_font)
        bg_draw.line([(xv, top_pad + plot_h), (xv, top_pad + plot_h + 8)], fill=tick_color, width=1)
        bg_draw.text(
            (xv - xlbl_w / 2, top_pad + plot_h + 12),
            xlbl,
            fill=tick_color,
            font=tick_font,
        )

    for tidx in range(num_ticks):
        yv = top_pad + int(round(tidx * plot_h / (num_ticks - 1)))
        yval = y_hi - (y_hi - y_lo) * tidx / (num_ticks - 1)
        ylbl = str(int(round(yval)))
        ylbl_w, ylbl_h = _text_size(bg_draw, ylbl, tick_font)
        bg_draw.line([(left_pad - 8, yv), (left_pad, yv)], fill=tick_color, width=1)
        bg_draw.text(
            (left_pad - 12 - ylbl_w, yv - ylbl_h / 2),
            ylbl,
            fill=tick_color,
            font=tick_font,
        )

    x_title = _normalize_axis_title(xaxis_cfg.get("title"), "x position")
    y_title = _normalize_axis_title(yaxis_cfg.get("title"), "y position")
    if x_title:
        x_title_w, x_title_h = _text_size(bg_draw, str(x_title), axis_font)
        x_title_x = left_pad + (plot_w - x_title_w) / 2
        x_title_y = min(height - x_title_h - 4, top_pad + plot_h + 40)
        bg_draw.text((x_title_x, x_title_y), str(x_title), fill=tick_color, font=axis_font)
    if y_title:
        y_title_text = str(y_title)
        y_title_w, y_title_h = _text_size(bg_draw, y_title_text, axis_font)
        y_title_img = Image.new("RGBA", (y_title_w + 2, y_title_h + 2), (0, 0, 0, 0))
        y_title_draw = ImageDraw.Draw(y_title_img, "RGBA")
        y_title_draw.text((0, 0), y_title_text, fill=tick_color, font=axis_font)
        y_title_img = y_title_img.rotate(90, expand=True)
        y_title_x = max(2, left_pad - 70)
        y_title_y = top_pad + (plot_h - y_title_img.height) // 2
        bg.alpha_composite(y_title_img, (int(y_title_x), int(y_title_y)))

    bg_draw.rectangle(
        [left_pad, top_pad, left_pad + plot_w, top_pad + plot_h],
        outline=x_line_color,
        width=max(1, x_line_width),
    )

    if map is not None:
        for lane_idx in map.lane_dict:
            lane = map.lane_dict[lane_idx]
            for lane_seg in lane.segment_list:
                if lane_seg.type == "Straight":
                    start1 = lane_seg.start + lane_seg.width / 2 * lane_seg.direction_lateral
                    end1 = lane_seg.end + lane_seg.width / 2 * lane_seg.direction_lateral
                    start2 = lane_seg.start - lane_seg.width / 2 * lane_seg.direction_lateral
                    end2 = lane_seg.end - lane_seg.width / 2 * lane_seg.direction_lateral
                    trace_x = [start1[0], end1[0], end2[0], start2[0], start1[0]]
                    trace_y = [start1[1], end1[1], end2[1], start2[1], start1[1]]
                elif lane_seg.type == "Circular":
                    phase_array = np.linspace(
                        start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100
                    )
                    r1 = lane_seg.radius - lane_seg.width / 2
                    x1 = (np.cos(phase_array) * r1 + lane_seg.center[0]).tolist()
                    y1 = (np.sin(phase_array) * r1 + lane_seg.center[1]).tolist()
                    r2 = lane_seg.radius + lane_seg.width / 2
                    x2 = (np.cos(phase_array) * r2 + lane_seg.center[0]).tolist()[::-1]
                    y2 = (np.sin(phase_array) * r2 + lane_seg.center[1]).tolist()[::-1]
                    trace_x = x1 + x2 + [x1[0]]
                    trace_y = y1 + y2 + [y1[0]]
                else:
                    continue
                points = [
                    _to_px(xp, yp, x_lo, x_hi, y_lo, y_hi, left_pad, top_pad, plot_w, plot_h)
                    for xp, yp in zip(trace_x, trace_y)
                ]
                if len(points) > 1:
                    bg_draw.line(points, fill=(0, 0, 0, 90), width=1)

    bg_setup_time = time.perf_counter()
    print(f"[export] raster background setup elapsed={bg_setup_time - prep_frames_time:.2f}s")

    rect_pixels = []
    for spec in rect_specs:
        p0 = _to_px(spec["x0"], spec["y0"], x_lo, x_hi, y_lo, y_hi, left_pad, top_pad, plot_w, plot_h)
        p1 = _to_px(spec["x1"], spec["y1"], x_lo, x_hi, y_lo, y_hi, left_pad, top_pad, plot_w, plot_h)
        x0p, x1p = min(p0[0], p1[0]), max(p0[0], p1[0])
        y0p, y1p = min(p0[1], p1[1]), max(p0[1], p1[1])
        rect_pixels.append((x0p, y0p, x1p, y1p, spec["line_rgb"], spec["fill_rgb"]))

    pix_prep_time = time.perf_counter()
    print(f"[export] pixel projection setup elapsed={pix_prep_time - bg_setup_time:.2f}s")

    print(f"[export] writing {ext.upper()} to {output_path} ({total_frames} frames)")

    if ext == "gif":
        writer = imageio.get_writer(output_path, mode="I", duration=frame_duration_sec, loop=0)
    elif ext == "mp4":
        fps = 1.0 / frame_duration_sec if frame_duration_sec > 0 else 10
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    else:
        raise ValueError("output_path must end with .gif or .mp4 for video export.")

    progress_every = max(1, total_frames // 20)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    visible_count = 0
    try:
        for idx, new_indices in enumerate(frame_new_trace_indices, start=1):
            for trace_idx in new_indices:
                x0p, y0p, x1p, y1p, line_rgb, fill_rgb = rect_pixels[trace_idx]
                overlay_draw.rectangle(
                    [x0p, y0p, x1p, y1p],
                    fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], 120),
                    outline=(line_rgb[0], line_rgb[1], line_rgb[2], 220),
                    width=1,
                )
            visible_count += len(new_indices)

            frame_img = Image.alpha_composite(bg, overlay).convert("RGB")
            writer.append_data(np.array(frame_img, dtype=np.uint8))

            if idx % progress_every == 0 or idx == total_frames:
                elapsed = time.perf_counter() - export_start
                print(
                    f"[export] frame {idx}/{total_frames} "
                    f"visible={visible_count}/{len(rect_specs)} elapsed={elapsed:.2f}s"
                )
    finally:
        writer.close()

    total_elapsed = time.perf_counter() - export_start
    print(f"[export] finished {output_path} total_elapsed={total_elapsed:.2f}s")


"""Functions below are low-level functions and usually are not called outside this file."""


def reachtube_tree_single(
    root: Union[AnalysisTree, AnalysisTreeNode],
    agent_id,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    color=None,
    print_dim_list=None,
    combine_rect=1,
    plot_color=None,
):
    """It statically shows the verfication traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if plot_color is None:
        plot_color = colors
    global color_cnt
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt + 1) % num_theme
    queue = [root]
    show_legend = False
    fillcolor = plot_color[scheme_dict[color]][1]
    linecolor = plot_color[scheme_dict[color]][0]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        if agent_id not in traces:
            break
        trace = np.array(traces[agent_id])
        max_id = len(trace) - 1
        if (
            len(np.unique(np.array([trace[i][x_dim] for i in range(0, max_id)]))) == 1
            and len(np.unique(np.array([trace[i][y_dim] for i in range(0, max_id)]))) == 1
        ):
            fig.add_trace(
                go.Scatter(
                    x=[trace[0][x_dim]],
                    y=[trace[0][y_dim]],
                    mode="markers+lines",
                    #  fill='toself',
                    #  fillcolor=fillcolor,
                    #  opacity=0.5,
                    marker={"size": 5},
                    line_color=linecolor,
                    line={"width": 1},
                    showlegend=show_legend,
                )
            )
        elif combine_rect == None:
            max_id = len(trace) - 1
            trace_x_odd = np.array([trace[i][x_dim] for i in range(0, max_id, 2)])
            trace_x_even = np.array([trace[i][x_dim] for i in range(1, max_id + 1, 2)])
            trace_y_odd = np.array([trace[i][y_dim] for i in range(0, max_id, 2)])
            trace_y_even = np.array([trace[i][y_dim] for i in range(1, max_id + 1, 2)])
            fig.add_trace(
                go.Scatter(
                    x=trace_x_odd.tolist() + trace_x_even[::-1].tolist() + [trace_x_odd[0]],
                    y=trace_y_odd.tolist() + trace_y_even[::-1].tolist() + [trace_y_odd[0]],
                    mode="markers+lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    #  opacity=0.5,
                    marker={"size": 1},
                    line_color=linecolor,
                    line={"width": 2},
                    showlegend=show_legend,
                )
            )
        elif combine_rect <= 1:
            for idx in range(0, len(trace), 2):
                trace_x = np.array(
                    [
                        trace[idx][x_dim],
                        trace[idx + 1][x_dim],
                        trace[idx + 1][x_dim],
                        trace[idx][x_dim],
                        trace[idx][x_dim],
                    ]
                )
                trace_y = np.array(
                    [
                        trace[idx][y_dim],
                        trace[idx][y_dim],
                        trace[idx + 1][y_dim],
                        trace[idx + 1][y_dim],
                        trace[idx][y_dim],
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=trace_x,
                        y=trace_y,
                        mode="markers+lines",
                        fill="toself",
                        fillcolor=fillcolor,
                        #  opacity=0.5,
                        marker={"size": 1},
                        line_color=linecolor,
                        line={"width": 1},
                        showlegend=show_legend,
                    )
                )
        else:
            for idx in range(0, len(trace), combine_rect * 2):
                trace_seg = trace[idx : idx + combine_rect * 2]
                max_id = len(trace_seg - 1)
                if max_id <= 2:
                    trace_x = np.array(
                        [
                            trace_seg[0][x_dim],
                            trace_seg[0 + 1][x_dim],
                            trace_seg[0 + 1][x_dim],
                            trace_seg[0][x_dim],
                            trace_seg[0][x_dim],
                        ]
                    )
                    trace_y = np.array(
                        [
                            trace_seg[0][y_dim],
                            trace_seg[0][y_dim],
                            trace_seg[0 + 1][y_dim],
                            trace_seg[0 + 1][y_dim],
                            trace_seg[0][y_dim],
                        ]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="markers+lines",
                            fill="toself",
                            fillcolor=fillcolor,
                            #  opacity=0.5,
                            marker={"size": 1},
                            line_color=linecolor,
                            line={"width": 1},
                            showlegend=show_legend,
                        )
                    )
                else:
                    trace_x_odd = np.array([trace_seg[i][x_dim] for i in range(0, max_id, 2)])
                    trace_x_even = np.array([trace_seg[i][x_dim] for i in range(1, max_id + 1, 2)])

                    trace_y_odd = np.array([trace_seg[i][y_dim] for i in range(0, max_id, 2)])
                    trace_y_even = np.array([trace_seg[i][y_dim] for i in range(1, max_id + 1, 2)])

                    x_start = 0
                    x_end = 0
                    if trace_x_odd[-1] >= trace_x_odd[-2] and trace_x_even[-1] >= trace_x_even[-2]:
                        x_end = trace_x_even[-1]
                    elif (
                        trace_x_odd[-1] <= trace_x_odd[-2] and trace_x_even[-1] <= trace_x_even[-2]
                    ):
                        x_end = trace_x_odd[-1]
                    else:
                        x_end = trace_x_odd[-1]

                    if (
                        trace_x_odd[1 - 1] >= trace_x_odd[2 - 1]
                        and trace_x_even[1 - 1] >= trace_x_even[2 - 1]
                    ):
                        x_start = trace_x_even[1 - 1]
                    elif (
                        trace_x_odd[1 - 1] <= trace_x_odd[2 - 1]
                        and trace_x_even[1 - 1] <= trace_x_even[2 - 1]
                    ):
                        x_start = trace_x_odd[1 - 1]
                    else:
                        x_start = trace_x_odd[1 - 1]

                    y_start = 0
                    y_end = 0
                    if trace_y_odd[-1] >= trace_y_odd[-2] and trace_y_even[-1] >= trace_y_even[-2]:
                        y_end = trace_y_even[-1]
                        if (
                            trace_x_odd[-1] >= trace_x_odd[-2]
                            and trace_x_even[-1] >= trace_x_even[-2]
                        ):
                            x_end = trace_x_odd[-1]
                    elif (
                        trace_y_odd[-1] <= trace_y_odd[-2] and trace_y_even[-1] <= trace_y_even[-2]
                    ):
                        y_end = trace_y_odd[-1]
                    else:
                        y_end = trace_y_odd[-1]

                    if (
                        trace_y_odd[1 - 1] >= trace_y_odd[2 - 1]
                        and trace_y_even[1 - 1] >= trace_y_even[2 - 1]
                    ):
                        y_start = trace_y_even[1 - 1]
                    elif (
                        trace_y_odd[1 - 1] <= trace_y_odd[2 - 1]
                        and trace_y_even[1 - 1] <= trace_y_even[2 - 1]
                    ):
                        y_start = trace_y_odd[1 - 1]
                        if (
                            trace_x_odd[1 - 1] <= trace_x_odd[2 - 1]
                            and trace_x_even[1 - 1] <= trace_x_even[2 - 1]
                        ):
                            x_start = trace_x_even[1 - 1]
                    else:
                        y_start = trace_y_even[1 - 1]

                    trace_x = (
                        trace_x_odd.tolist()
                        + [x_end]
                        + trace_x_even[::-1].tolist()
                        + [x_start]
                        + [trace_x_odd[0]]
                    )
                    trace_y = (
                        trace_y_odd.tolist()
                        + [y_end]
                        + trace_y_even[::-1].tolist()
                        + [y_start]
                        + [trace_y_odd[0]]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="markers+lines",
                            fill="toself",
                            fillcolor=fillcolor,
                            #  opacity=0.5,
                            marker={"size": 1},
                            line_color=linecolor,
                            line={"width": 1},
                            showlegend=show_legend,
                        )
                    )
        queue += node.child
    return fig


def simulation_tree_single(
    root: Union[AnalysisTree, AnalysisTreeNode],
    agent_id,
    fig: go.Figure() = go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    color=None,
    print_dim_list=None,
    plot_color = None
):
    """It statically shows the simulation traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    global color_cnt
    queue = [root]
    color_id = 0
    if plot_color is None:
        plot_color = colors
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt + 1) % num_theme
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
        # if (start in start_list) and (end in end_list):
        #     continue
        time = tuple([round(start[0], 2), round(end[0], 2)])
        if time in count_dict.keys():
            count_dict[time] += 1
        else:
            count_dict[time] = 1
        start_list.append(start)
        end_list.append(end)

        fig.add_trace(
            go.Scatter(
                x=trace[:, x_dim],
                y=trace[:, y_dim],
                mode="lines",
                line_color=plot_color[scheme_dict[color]][color_id],
                text=[
                    ["{:.2f}".format(trace[i, j]) for j in print_dim_list]
                    for i in range(trace.shape[0])
                ],
                legendgroup=agent_id,
                legendgrouptitle_text=agent_id,
                name=str(round(start[0], 2))
                + "-"
                + str(round(end[0], 2))
                + "-"
                + str(count_dict[time]),
                showlegend=False,
            )
        )

        color_id = (color_id + 4) % 5
        queue += node.child
    fig.update_layout(
        legend=dict(groupclick="toggleitem", itemclick="toggle", itemdoubleclick="toggleothers")
    )
    return fig


def draw_map(
    map: LaneMap, color="rgba(0,0,0,0.5)", fig: go.Figure() = go.Figure(), fill_type="lines"
):
    """It draws the the map"""
    x_min, x_max = float("inf"), -float("inf")
    y_min, y_max = float("inf"), -float("inf")
    if map is None:
        return fig
    if fill_type == "detailed":
        speed_dict = map.get_all_speed_limit()
        speed_list = list(filter(None, speed_dict.values()))
        speed_min = min(speed_list)
        speed_max = max(speed_list)
        start_color = [255, 255, 255, 0.2]
        end_color = [0, 0, 0, 0.2]
        curr_color = [0, 0, 0, 0]
    speed_limit = None
    line_style = {"width": 0.5}
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        curr_color = [0, 0, 0, 0]
        if fill_type == "detailed":
            speed_limit = speed_dict[lane_idx]
            if speed_limit is not None:
                lens = len(curr_color) - 1
                for j in range(lens):
                    curr_color[j] = int(
                        start_color[j]
                        + (speed_limit - speed_min)
                        / (speed_max - speed_min)
                        * (end_color[j] - start_color[j])
                    )
                curr_color[lens] = start_color[lens] + (speed_limit - speed_min) / (
                    speed_max - speed_min
                ) * (end_color[lens] - start_color[lens])
        for lane_seg in lane.segment_list:
            if lane_seg.type == "Straight":
                start1 = lane_seg.start + lane_seg.width / 2 * lane_seg.direction_lateral
                end1 = lane_seg.end + lane_seg.width / 2 * lane_seg.direction_lateral
                start2 = lane_seg.start - lane_seg.width / 2 * lane_seg.direction_lateral
                end2 = lane_seg.end - lane_seg.width / 2 * lane_seg.direction_lateral
                trace_x = [start1[0], end1[0], end2[0], start2[0], start1[0]]
                trace_y = [start1[1], end1[1], end2[1], start2[1], start1[1]]
                x_min = min(x_min, min(trace_x))
                y_min = min(y_min, min(trace_y))
                x_max = max(x_max, max(trace_x))
                y_max = max(y_max, max(trace_y))
                if fill_type == "lines" or (fill_type == "detailed" and speed_limit is None):
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            showlegend=False,
                            hoverinfo=None,
                            name="lines",
                        )
                    )
                elif fill_type == "detailed" and speed_limit is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            fill="toself",
                            fillcolor="rgba" + str(tuple(curr_color)),
                            showlegend=False,
                            name="limit",
                        )
                    )
                elif fill_type == "fill":
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            fill="toself",
                            fillcolor="rgba(0,0,0,0.1)",
                            showlegend=False,
                            # text=theta,
                            name="lines",
                        )
                    )
            elif lane_seg.type == "Circular":
                phase_array = np.linspace(
                    start=lane_seg.start_phase, stop=lane_seg.end_phase, num=100
                )
                r1 = lane_seg.radius - lane_seg.width / 2
                x1 = (np.cos(phase_array) * r1 + lane_seg.center[0]).tolist()
                y1 = (np.sin(phase_array) * r1 + lane_seg.center[1]).tolist()
                r2 = lane_seg.radius + lane_seg.width / 2
                x2 = (np.cos(phase_array) * r2 + lane_seg.center[0]).tolist()[::-1]
                y2 = (np.sin(phase_array) * r2 + lane_seg.center[1]).tolist()[::-1]
                trace_x = x1 + x2 + [x1[0]]
                trace_y = y1 + y2 + [y1[0]]
                x_min = min(x_min, min(trace_x))
                y_min = min(y_min, min(trace_y))
                x_max = max(x_max, max(trace_x))
                y_max = max(y_max, max(trace_y))
                if fill_type == "lines":
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            showlegend=False,
                            name="lines",
                        )
                    )
                elif fill_type == "detailed" and speed_limit != None:
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            fill="toself",
                            fillcolor="rgba" + str(tuple(curr_color)),
                            showlegend=False,
                            name="lines",
                        )
                    )
                elif fill_type == "fill":
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="lines",
                            line_color=color,
                            line=line_style,
                            fill="toself",
                            showlegend=False,
                            name="lines",
                        )
                    )
            else:
                raise ValueError(f"Unknown lane segment type {lane_seg.type}")
    if fill_type == "detailed":
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    cmax=speed_max,
                    cmin=speed_min,
                    color="rgba(0,0,0,0)",
                    colorbar=dict(title="Speed Limit", orientation="h"),
                    colorscale=[
                        [0, "rgba" + str(tuple(start_color))],
                        [1, "rgba" + str(tuple(end_color))],
                    ],
                ),
                showlegend=False,
            )
        )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def check_dim(num_dim: int, x_dim: int = 1, y_dim: int = 2, print_dim_list: List(int) = None):
    if x_dim < 0 or x_dim >= num_dim:
        raise ValueError(f"wrong x dimension value {x_dim}")
    if y_dim < 0 or y_dim >= num_dim:
        raise ValueError(f"wrong y dimension value {y_dim}")
    if print_dim_list is None:
        return True
    for i in print_dim_list:
        if y_dim < 0 or y_dim >= num_dim:
            raise ValueError(f"wrong printed dimension value {i}")
    return True


def create_anime_dict(duration=10):
    fig_dict = {"data": [], "layout": {}, "frames": []}
    fig_dict["layout"]["xaxis"] = {"title": "x position"}
    fig_dict["layout"]["yaxis"] = {"title": "y position"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": duration, "easing": "quadratic-in-out"},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
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
            "xanchor": "right",
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
    return fig_dict, sliders_dict


def get_text_pos(veh_mode):
    if veh_mode == "Normal":
        text_pos = "middle center"
        text = "N"
    elif veh_mode == "Brake":
        text_pos = "middle left"
        text = "B"
    elif veh_mode == "Accelerate":
        text_pos = "middle right"
        text = "A"
    elif veh_mode == "SwitchLeft":
        text_pos = "top center"
        text = "SL"
    elif veh_mode == "SwitchRight":
        text_pos = "bottom center"
        text = "SR"
    elif veh_mode == "Stop":
        text_pos = "middle center"
        text = "S"
    else:
        text_pos = "middle center"
        text = veh_mode
    return text_pos, text


def sample_trace(root, sample_rate: int = 1):
    queue = [root]
    # print(root.trace)
    if root.type == "reachtube":
        sample_rate = sample_rate * 2
        while queue != []:
            node = queue.pop()
            for agent_id in node.trace:
                trace_length = len(node.trace[agent_id])
                tmp = []
                for i in range(0, trace_length, sample_rate):
                    if i + sample_rate - 1 < trace_length:
                        tmp.append(node.trace[agent_id][i])
                        tmp.append(node.trace[agent_id][i + sample_rate - 1])
                node.trace[agent_id] = tmp
            queue += node.child
    else:
        while queue != []:
            node: AnalysisTreeNode = queue.pop()
            for agent_id in node.trace:
                node.trace[agent_id] = [
                    node.trace[agent_id][i]
                    for i in range(0, len(node.trace[agent_id]), sample_rate)
                ]
            queue += node.child
    return root


def num_digits(val: float):
    val_str = str(val)
    digits_location = val_str.find(".")
    if digits_location:
        num = len(val_str[digits_location + 1 :])
        print(num)
        return num
    return False


# fig= go.Figure()
# fig.update_traces(line={'width':5})


def update_style(fig: go.Figure() = go.Figure()):
    # fig.update_traces(line={'width': 3})
    fig.update_layout(
        # paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(font={"size": 32})
    linewidth = 4
    gridwidth = 2
    fig.update_yaxes(
        showline=True,
        linewidth=linewidth,
        linecolor="Gray",
        showgrid=False,
        gridwidth=gridwidth,
        gridcolor="LightGrey",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=linewidth,
        linecolor="Gray",
        showgrid=False,
        gridwidth=gridwidth,
        gridcolor="LightGrey",
    )
    return fig

def reachtube_tree_slice(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    print_dim_list=None,
    map_type="lines",
    scale_type="trace",
    label_mode="None",
    sample_rate=1,
    combine_rect=1,
    plot_color=None,
    t_lower: float = None,
    t_upper: float = None
):
    """It statically shows all the traces of the verfication."""
    if plot_color is None:
        plot_color = colors
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    fig = draw_map(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    # scheme_list = list(scheme_dict.keys())
    i = 0
    if t_lower is None:
        t_lower = 0
    if t_upper is None:
        t_upper = np.inf
        
    for agent_id in agent_list:
        fig = reachtube_tree_single_slice(
            root,
            agent_id,
            fig,
            x_dim,
            y_dim,
            scheme_list[i],
            print_dim_list,
            combine_rect,
            plot_color=plot_color,
            t_lower=t_lower,
            t_upper=t_upper
        )
        i = (i + 1) % num_theme
    if scale_type == "trace":
        queue = [root]
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")
    i = 0
    queue = [root]
    previous_mode = {}
    for agent_id in root.mode:
        previous_mode[agent_id] = []
    text_pos = "middle center"
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        # print({k: len(v) for k, v in traces.items()})
        i = 0
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if scale_type == "trace":
                x_min = min(x_min, min(trace[:, x_dim]))
                x_max = max(x_max, max(trace[:, x_dim]))
                y_min = min(y_min, min(trace[:, y_dim]))
                y_max = max(y_max, max(trace[:, y_dim]))
            i = agent_list.index(agent_id)
            if label_mode != "None":
                if previous_mode[agent_id] != node.mode[agent_id]:
                    text_pos, text = get_text_pos(node.mode[agent_id][0])
                    mode_point_color = plot_color[agent_list.index(agent_id) % num_theme][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[0, x_dim]],
                            y=[trace[0, y_dim]],
                            mode="markers+text",
                            line_color=mode_point_color,
                            opacity=0.5,
                            text=str(agent_id) + ": " + text,
                            textposition=text_pos,
                            marker={"size": marker_size, "color": mode_text_color},
                            showlegend=False,
                        )
                    )
                    previous_mode[agent_id] = node.mode[agent_id]
                if node.assert_hits != None and agent_id in node.assert_hits:
                    fig.add_trace(
                        go.Scatter(
                            x=[trace[-1, x_dim]],
                            y=[trace[-1, y_dim]],
                            mode="markers+text",
                            text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                            # textfont={"color": "grey"},
                            marker={"size": marker_size, "color": "black"},
                            showlegend=False,
                        )
                    )
        queue += node.child
    if scale_type == "trace":
        fig.update_xaxes(
            range=[x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)]
        )
        fig.update_yaxes(
            range=[y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)]
        )
    fig = update_style(fig)
    return fig


### for this to actually take an arbitrary slice rather than just the last state, needs to take in two time parameters and check like reach_at
### raise exception if user's provided time interval does not enclose a hyperrectangle time interval
### write this out in math
def reachtube_tree_single_slice(
    root: Union[AnalysisTree, AnalysisTreeNode],
    agent_id,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    color=None,
    print_dim_list=None,
    combine_rect=1,
    plot_color=None,
    t_lower: float = None,
    t_upper: float = None
):
    """It statically shows the verfication traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if plot_color is None:
        plot_color = colors
    global color_cnt
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt + 1) % num_theme
    queue = [root]
    show_legend = False
    fillcolor = plot_color[scheme_dict[color]][1]
    linecolor = plot_color[scheme_dict[color]][0]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        max_id = len(trace) - 1 

        if (
            len(np.unique(np.array([trace[i][x_dim] for i in range(0, max_id)]))) == 1
            and len(np.unique(np.array([trace[i][y_dim] for i in range(0, max_id)]))) == 1
        ):
            fig.add_trace(
                go.Scatter(
                    x=[trace[0][x_dim]],
                    y=[trace[0][y_dim]],
                    mode="markers+lines",
                    #  fill='toself',
                    #  fillcolor=fillcolor,
                    #  opacity=0.5,
                    marker={"size": 5},
                    line_color=linecolor,
                    line={"width": 1},
                    showlegend=show_legend,
                )
            )
        elif combine_rect == None:
            ### may have to modify if statements
            max_id = len(trace) - 1
            trace_x_odd = np.array([trace[i][x_dim] for i in range(0, max_id, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
            trace_x_even = np.array([trace[i][x_dim] for i in range(1, max_id + 1, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
            trace_y_odd = np.array([trace[i][y_dim] for i in range(0, max_id, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
            trace_y_even = np.array([trace[i][y_dim] for i in range(1, max_id + 1, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
            fig.add_trace(
                go.Scatter(
                    x=trace_x_odd.tolist() + trace_x_even[::-1].tolist() + [trace_x_odd[0]],
                    y=trace_y_odd.tolist() + trace_y_even[::-1].tolist() + [trace_y_odd[0]],
                    mode="markers+lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    #  opacity=0.5,
                    marker={"size": 1},
                    line_color=linecolor,
                    line={"width": 2},
                    showlegend=show_legend,
                )
            )
        elif combine_rect <= 1:
            for idx in range(0, len(trace), 2):
                if trace[idx][0]<t_lower:
                    continue
                if trace[idx+1][0]>t_upper:
                    break
                trace_x = np.array(
                    [
                        trace[idx][x_dim],
                        trace[idx + 1][x_dim],
                        trace[idx + 1][x_dim],
                        trace[idx][x_dim],
                        trace[idx][x_dim],
                    ]
                )
                trace_y = np.array(
                    [
                        trace[idx][y_dim],
                        trace[idx][y_dim],
                        trace[idx + 1][y_dim],
                        trace[idx + 1][y_dim],
                        trace[idx][y_dim],
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=trace_x,
                        y=trace_y,
                        mode="markers+lines",
                        fill="toself",
                        fillcolor=fillcolor,
                        #  opacity=0.5,
                        marker={"size": 1},
                        line_color=linecolor,
                        line={"width": 1},
                        showlegend=show_legend,
                    )
                )
        else:
            for idx in range(0, len(trace), combine_rect * 2):
                trace_seg = trace[idx : idx + combine_rect * 2]
                max_id = len(trace_seg - 1)
                if max_id <= 2:
                    trace_x = np.array(
                        [
                            trace_seg[0][x_dim],
                            trace_seg[0 + 1][x_dim],
                            trace_seg[0 + 1][x_dim],
                            trace_seg[0][x_dim],
                            trace_seg[0][x_dim],
                        ]
                    )
                    trace_y = np.array(
                        [
                            trace_seg[0][y_dim],
                            trace_seg[0][y_dim],
                            trace_seg[0 + 1][y_dim],
                            trace_seg[0 + 1][y_dim],
                            trace_seg[0][y_dim],
                        ]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="markers+lines",
                            fill="toself",
                            fillcolor=fillcolor,
                            #  opacity=0.5,
                            marker={"size": 1},
                            line_color=linecolor,
                            line={"width": 1},
                            showlegend=show_legend,
                        )
                    )
                else:
                    ### assuming max_id is <=len(trace)-1, if not, modify the conditions
                    trace_x_odd = np.array([trace_seg[i][x_dim] for i in range(0, max_id, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
                    trace_x_even = np.array([trace_seg[i][x_dim] for i in range(1, max_id + 1, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])

                    trace_y_odd = np.array([trace_seg[i][y_dim] for i in range(0, max_id, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])
                    trace_y_even = np.array([trace_seg[i][y_dim] for i in range(1, max_id + 1, 2) if trace[i][0]>=t_lower and trace[i+1][0]<=t_upper])

                    x_start = 0
                    x_end = 0
                    if trace_x_odd[-1] >= trace_x_odd[-2] and trace_x_even[-1] >= trace_x_even[-2]:
                        x_end = trace_x_even[-1]
                    elif (
                        trace_x_odd[-1] <= trace_x_odd[-2] and trace_x_even[-1] <= trace_x_even[-2]
                    ):
                        x_end = trace_x_odd[-1]
                    else:
                        x_end = trace_x_odd[-1]

                    if (
                        trace_x_odd[1 - 1] >= trace_x_odd[2 - 1]
                        and trace_x_even[1 - 1] >= trace_x_even[2 - 1]
                    ):
                        x_start = trace_x_even[1 - 1]
                    elif (
                        trace_x_odd[1 - 1] <= trace_x_odd[2 - 1]
                        and trace_x_even[1 - 1] <= trace_x_even[2 - 1]
                    ):
                        x_start = trace_x_odd[1 - 1]
                    else:
                        x_start = trace_x_odd[1 - 1]

                    y_start = 0
                    y_end = 0
                    if trace_y_odd[-1] >= trace_y_odd[-2] and trace_y_even[-1] >= trace_y_even[-2]:
                        y_end = trace_y_even[-1]
                        if (
                            trace_x_odd[-1] >= trace_x_odd[-2]
                            and trace_x_even[-1] >= trace_x_even[-2]
                        ):
                            x_end = trace_x_odd[-1]
                    elif (
                        trace_y_odd[-1] <= trace_y_odd[-2] and trace_y_even[-1] <= trace_y_even[-2]
                    ):
                        y_end = trace_y_odd[-1]
                    else:
                        y_end = trace_y_odd[-1]

                    if (
                        trace_y_odd[1 - 1] >= trace_y_odd[2 - 1]
                        and trace_y_even[1 - 1] >= trace_y_even[2 - 1]
                    ):
                        y_start = trace_y_even[1 - 1]
                    elif (
                        trace_y_odd[1 - 1] <= trace_y_odd[2 - 1]
                        and trace_y_even[1 - 1] <= trace_y_even[2 - 1]
                    ):
                        y_start = trace_y_odd[1 - 1]
                        if (
                            trace_x_odd[1 - 1] <= trace_x_odd[2 - 1]
                            and trace_x_even[1 - 1] <= trace_x_even[2 - 1]
                        ):
                            x_start = trace_x_even[1 - 1]
                    else:
                        y_start = trace_y_even[1 - 1]

                    trace_x = (
                        trace_x_odd.tolist()
                        + [x_end]
                        + trace_x_even[::-1].tolist()
                        + [x_start]
                        + [trace_x_odd[0]]
                    )
                    trace_y = (
                        trace_y_odd.tolist()
                        + [y_end]
                        + trace_y_even[::-1].tolist()
                        + [y_start]
                        + [trace_y_odd[0]]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=trace_x,
                            y=trace_y,
                            mode="markers+lines",
                            fill="toself",
                            fillcolor=fillcolor,
                            #  opacity=0.5,
                            marker={"size": 1},
                            line_color=linecolor,
                            line={"width": 1},
                            showlegend=show_legend,
                        )
                    )
        queue += node.child
    return fig

def display_figure(fig: go.Figure):
    if os.path.exists('/.dockerenv'):
        try:
            import dash
            from dash import dcc, html
            print("\n" + "="*40)
            print("DOCKER VISUALIZATION STARTING")
            print("URL: http://localhost:8050")
            print("="*40 + "\n")
            
            app = dash.Dash(__name__)
            
            # Setup the layout to be full-screen
            app.layout = html.Div([
                dcc.Graph(
                    figure=fig, 
                    style={'height': '98vh', 'width': '100%'}
                )
            ], style={'margin': '0', 'padding': '0', 'backgroundColor': '#f0f0f0'})
            
            # This will "hold" the terminal until you Ctrl+C
            # host='0.0.0.0' is required to map to your Windows host
            # The modern way to start the server
            app.run(
                host='0.0.0.0', 
                port=8050, 
                debug=False,            # Disables reloader and dev tools by default
                dev_tools_ui=False,     # Hides the blue Dash debug button
                dev_tools_hot_reload=False # The direct analogue to use_reloader=False
            )
        except ImportError:
            print("\n[!] Dash not found. Saving to HTML instead.")
            fig.write_html("docking_results.html")
    else:
        fig.show()