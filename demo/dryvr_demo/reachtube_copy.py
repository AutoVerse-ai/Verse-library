'''
This file contains plotter code for DryVR reachtube output
'''

from __future__ import annotations
import copy
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union
from plotly.graph_objs.scatter import Marker
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map import LaneMap
from verse.plotter.plotter2D import *
### imports, stucture, and code from original reachtube_tree function
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
        if len(node.child):
            queue += node.child
            continue
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
            trace_x_odd = np.array([trace[i][x_dim] for i in range(max_id-2, max_id, 2)])
            trace_x_even = np.array([trace[i][x_dim] for i in range(max_id-1, max_id + 1, 2)])
            trace_y_odd = np.array([trace[i][y_dim] for i in range(max_id-2, max_id, 2)])
            trace_y_even = np.array([trace[i][y_dim] for i in range(max_id-1, max_id + 1, 2)])
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
            for idx in range(len(trace)-2, len(trace), 2):
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