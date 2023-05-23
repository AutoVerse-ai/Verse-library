import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Union
from math import pi, cos, sin, acos, asin
from plotly.subplots import make_subplots
from enum import Enum, auto

# import time
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d_v1

colors = [
    ["#CC0000", "#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"],
    ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],
    ["#66CC00", "#80FF00", "#99FF33", "#B2FF66", "#CCFF99", "#FFFFCC"],
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],
    ["#00CC66", "#00FF80", "#33FF99", "#66FFB2", "#99FFCC", "#CCFFCC"],
    ["#00CCCC", "#00FFFF", "#33FFFF", "#66FFFF", "#99FFFF", "#CCFFE5"],
    ["#0066CC", "#0080FF", "#3399FF", "#66B2FF", "#99CCFF", "#CCE5FF"],
    ["#0000CC", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF"],
    ["#6600CC", "#7F00FF", "#9933FF", "#B266FF", "#CC99FF", "#E5CCFF"],
    ["#CC00CC", "#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF", "#FFCCFF"],
    ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],
]
scheme_dict = {
    "red": 0,
    "springgreen": 5,
    "blue": 8,
    "orange": 1,
    "yellow": 2,
    "yellowgreen": 3,
    "lime": 4,
    "cyan": 6,
    "cyanblue": 7,
    "purple": 9,
    "magenta": 10,
    "pink": 11,
}


def simulation_tree_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    z_dim: int = 3,
    print_dim_list=None,
    map_type="outline",
    sample_rate=1,
):
    """It statically shows all the traces of the simulation."""
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    fig = draw_map_3d(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, z_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = simulation_tree_single_3d(
            root, agent_id, fig, x_dim, y_dim, z_dim, scheme_list[i], print_dim_list
        )
        i = (i + 1) % 12

    # fig.update_xaxes(title='x')
    # fig.update_yaxes(title='y')
    # fig.update_layout(legend_title_text='Agent list')
    fig.update_layout(
        scene=dict(
            # xaxis = dict(nticks=4, range=[-100,100],),
            # yaxis = dict(nticks=4, range=[-50,100],),
            zaxis=dict(
                nticks=4,
                range=[-15, 15],
            )
        )
    )
    fig = update_style(fig)
    return fig


def reachtube_tree_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    z_dim: int = 3,
    print_dim_list=None,
    map_type="outline",
    sample_rate=1,
    combine_rect=None,
    xrange=[],
    yrange=[],
    zrange=[],
):
    """It statically shows all the traces of the verfication."""
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    fig = draw_map_3d(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, z_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    scheme_list = list(scheme_dict.keys())
    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single_3d(
            root, agent_id, fig, x_dim, y_dim, z_dim, scheme_list[i], print_dim_list, combine_rect
        )
        i = (i + 1) % 12

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=xrange,
            ),
            yaxis=dict(
                nticks=4,
                range=yrange,
            ),
            zaxis=dict(
                nticks=4,
                range=zrange,
            ),
        )
    )
    fig = update_style(fig)
    return fig


def draw_map_3d(map: LaneMap_3d, fig=go.Figure(), fill_type="outline", color="rgba(0,0,0,1)"):
    if map is None:
        return fig
    num = 100
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        opacity = 0.2
        start_color = end_color = "grey"
        for lane_seg in lane.segment_list:
            if lane_seg.type == "Straight":
                lane_seg: StraightLane_3d = lane_seg
                oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                fig.add_trace(
                    go.Scatter3d(
                        x=oc_x,
                        y=oc_y,
                        z=oc_z,
                        # opacity=opacity,
                        showlegend=False,
                        mode="lines",
                        marker=dict(
                            color="rgb(255,255,255)",
                        ),
                        line=dict(color="rgb(0,0,0)", width=10),
                    )
                )

            elif lane_seg.type == "Circular":
                lane_seg = lane_seg
                oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                fig.add_trace(
                    go.Scatter3d(
                        x=oc_x,
                        y=oc_y,
                        z=oc_z,
                        # opacity=opacity,
                        showlegend=False,
                        mode="lines",
                        marker=dict(
                            color="rgb(255,255,255)",
                        ),
                        line=dict(color="rgb(0,0,0)", width=10),
                    )
                )
            else:
                raise ValueError(f"Unknown lane segment type {lane_seg.type}")
    # for agent_id, wps in map.wps.items():
    #     if agent_id != 'test3':
    #         continue
    #     points = [wp[:3] for wp in wps]
    #     if points == []:
    #         return fig
    #     fig.add_trace(go.Scatter3d(
    #         x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
    #         # opacity=opacity,
    #         showlegend=False,
    #         mode='markers',
    #         marker=dict(color='rgb(127,127,127)', size=5),
    #         # line=dict(color='rgb(0,0,0)', width=10
    #         #             )
    #     ))
    return fig


def simulation_tree_single_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    agent_id,
    fig: go.Figure() = go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    z_dim: int = 3,
    color=None,
    print_dim_list=None,
):
    """It statically shows the simulation traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    global color_cnt
    queue = [root]
    color_id = 0
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt + 1) % 12
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
            go.Scatter3d(
                x=trace[:, x_dim],
                y=trace[:, y_dim],
                z=trace[:, z_dim],
                mode="lines",
                marker=dict(color=colors[scheme_dict[color]][color_id]),
                line=dict(color=colors[scheme_dict[color]][color_id], width=18),
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
                showlegend=True,
            )
        )

        color_id = (color_id + 4) % 5
        queue += node.child
    # fig.update_layout(legend=dict(
    #     groupclick="toggleitem",
    #     itemclick="toggle",
    #     itemdoubleclick="toggleothers"
    # ))
    return fig


def reachtube_tree_single_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    agent_id,
    fig=go.Figure(),
    x_dim: int = 1,
    y_dim: int = 2,
    z_dim: int = 3,
    color=None,
    print_dim_list=None,
    combine_rect=None,
):
    """It statically shows the verfication traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    global color_cnt
    if color == None:
        color = list(scheme_dict.keys())[color_cnt]
        color_cnt = (color_cnt + 1) % 12
    queue = [root]
    show_legend = True
    fillcolor = colors[scheme_dict[color]][5]
    linecolor = colors[scheme_dict[color]][4]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        # print(trace)
        max_id = len(trace) - 1
        if combine_rect == None:
            max_id = len(trace) - 1
            num_cube = 0
            i = [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7]
            j = [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3]
            k = [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2]
            x_total = []
            y_total = []
            z_total = []
            i_total = []
            j_total = []
            k_total = []
            for n in range(0, max_id, 2):
                px = [trace[n][x_dim], trace[n + 1][x_dim]]
                py = [trace[n][y_dim], trace[n + 1][y_dim]]
                pz = [trace[n][z_dim], trace[n + 1][z_dim]]
                n = 0
                for z in pz:
                    for y in py:
                        for x in px:
                            x_total.append(x)
                            y_total.append(y)
                            z_total.append(z)
                for n in range(12):
                    i_total.append(i[n] + num_cube * 8)
                    j_total.append(j[n] + num_cube * 8)
                    k_total.append(k[n] + num_cube * 8)
                num_cube += 1
            fig.add_trace(
                go.Mesh3d(
                    # vertices of cubes
                    x=x_total,
                    y=y_total,
                    z=z_total,
                    # i, j and k give the vertices of triangles
                    i=i_total,
                    j=j_total,
                    k=k_total,
                    name="y",
                    opacity=0.20,
                    color=colors[scheme_dict[color]][1],
                    flatshading=True,
                )
            )

            # trace_x_odd = np.array([trace[i][x_dim]
            #                         for i in range(0, max_id, 2)])
            # trace_x_even = np.array([trace[i][x_dim]
            #                          for i in range(1, max_id+1, 2)])
            # trace_y_odd = np.array([trace[i][y_dim]
            #                         for i in range(0, max_id, 2)])
            # trace_y_even = np.array([trace[i][y_dim]
            #                          for i in range(1, max_id+1, 2)])
            # trace_z_odd = np.array([trace[i][z_dim]
            #                         for i in range(0, max_id, 2)])
            # trace_z_even = np.array([trace[i][z_dim]
            #                          for i in range(1, max_id+1, 2)])
            # fig.add_trace(go.Scatter3d(x=trace_x_odd,
            #                            y=trace_y_odd,
            #                            z=trace_z_odd,
            #                            mode='lines',
            #                            #  opacity=0.5,
            #                            marker={
            #                                'size': 1, 'color': colors[scheme_dict[color]][0]},
            #                            line={
            #                                'width': 10, 'color': colors[scheme_dict[color]][0]},
            #                            showlegend=show_legend
            #                            ))
            # fig.add_trace(go.Scatter3d(x=trace_x_even,
            #                            y=trace_y_even,
            #                            z=trace_z_even,
            #                            mode='lines',
            #                            #  opacity=0.5,
            #                            marker={'size': 1, 'color': linecolor},
            #                            line={'width': 10, 'color': linecolor},
            #                            showlegend=show_legend
            #                            ))
        queue += node.child
    return fig


def check_dim(num_dim: int, x_dim: int = 1, y_dim: int = 2, z_dim: int = 3, print_dim_list=None):
    if x_dim < 0 or x_dim >= num_dim:
        raise ValueError(f"wrong x dimension value {x_dim}")
    if y_dim < 0 or y_dim >= num_dim:
        raise ValueError(f"wrong y dimension value {y_dim}")
    if z_dim < 0 or z_dim >= num_dim:
        raise ValueError(f"wrong z dimension value {z_dim}")
    if print_dim_list is None:
        return True
    for i in print_dim_list:
        if y_dim < 0 or y_dim >= num_dim:
            raise ValueError(f"wrong printed dimension value {i}")
    return True


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
            node = queue.pop()
            for agent_id in node.trace:
                node.trace[agent_id] = [
                    node.trace[agent_id][i]
                    for i in range(0, len(node.trace[agent_id]), sample_rate)
                ]
            queue += node.child
    return root


def update_style(fig: go.Figure() = go.Figure()):
    # fig.update_traces(line={'width': 3})
    fig.update_layout(
        # paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(font={"size": 18})
    linewidth = 4
    gridwidth = 2
    fig.update_yaxes(
        showline=True,
        linewidth=linewidth,
        linecolor="Gray",
        showgrid=True,
        gridwidth=gridwidth,
        gridcolor="LightGrey",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=linewidth,
        linecolor="Gray",
        showgrid=True,
        gridwidth=gridwidth,
        gridcolor="LightGrey",
    )
    return fig
