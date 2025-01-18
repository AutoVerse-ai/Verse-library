import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Union
from math import pi, cos, sin, acos, asin
from plotly.subplots import make_subplots
from enum import Enum, auto

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import math

# import time
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d_v1

color_array_def = [
    ["#CC0000", "#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"],
    ["#0066CC", "#0080FF", "#3399FF", "#66B2FF", "#99CCFF", "#CCE5FF"],
    ["#00CCCC", "#00FFFF", "#33FFFF", "#66FFFF", "#99FFFF", "#CCFFE5"],
    ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],
    ["#66CC00", "#80FF00", "#99FF33", "#B2FF66", "#CCFF99", "#FFFFCC"],
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],
    ["#00CC66", "#00FF80", "#33FF99", "#66FFB2", "#99FFCC", "#CCFFCC"],
    ["#0000CC", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF"],
    ["#6600CC", "#7F00FF", "#9933FF", "#B266FF", "#CC99FF", "#E5CCFF"],
    ["#CC00CC", "#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF", "#FFCCFF"],
    ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],
]


def simulation_tree_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    fig=go.Figure(),
    x_dim: int = 1,
    x_title: str = None, 
    y_dim: int = 2,
    y_title: str = None, 
    z_dim: int = 3,
    z_title: str = None, 
    print_dim_list=None,
    map=None,
    color_array=None,
    map_type="outline",
    sample_rate=1,
    xrange=[],
    yrange=[],
    zrange=[],
):
    """It statically shows all the traces of the simulation."""
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    if color_array is None:
        color_array = color_array_def
    num_theme = len(color_array)
    fig = draw_map_3d(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, z_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    i = 0
    for agent_id in agent_list:
        fig = simulation_tree_single_3d(
            root, agent_id, fig, x_dim, y_dim, z_dim, color_array, i, print_dim_list
        )
        i = (i + 1) % num_theme

    # fig.update_xaxes(title='x')
    # fig.update_yaxes(title='y')
    # fig.update_layout(legend_title_text='Agent list')
    # fig.update_layout(
    #     scene=dict(
    #         # xaxis = dict(nticks=4, range=[-100,100],),
    #         # yaxis = dict(nticks=4, range=[-50,100],),
    #         zaxis=dict(
    #             nticks=4,
    #             range=[-15, 15],
    #         )
    #     )
    # )

    if xrange:
        fig.update_scenes(xaxis_range=xrange) 
    if yrange:
        fig.update_scenes(yaxis_range=yrange) 
    if zrange:
        fig.update_scenes(zaxis_range=zrange) 

    fig = update_style(fig, x_title, y_title, z_title)
    return fig


def reachtube_tree_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    fig=go.Figure(),
    x_dim: int = 1,
    x_title: str = None, 
    y_dim: int = 2,
    y_title: str = None, 
    z_dim: int = 3,
    z_title: str = None, 
    print_dim_list=None,
    map=None,
    color_array=None,
    map_type="outline",
    sample_rate=1,
    xrange=[],
    yrange=[],
    zrange=[],
    combine_rect=None,
):
    """It statically shows all the traces of the verfication."""
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, sample_rate)
    if color_array is None:
        color_array = color_array_def
    num_theme = len(color_array)
    fig = draw_map_3d(map=map, fig=fig, fill_type=map_type)
    agent_list = list(root.agent.keys())
    # input check
    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, z_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)

    i = 0
    for agent_id in agent_list:
        fig = reachtube_tree_single_3d(
            root, agent_id, fig, x_dim, y_dim, z_dim, color_array, i, print_dim_list, combine_rect
        )
        i = (i + 1) % num_theme

    if xrange:
        fig.update_scenes(xaxis_range=xrange) 
    if yrange:
        fig.update_scenes(yaxis_range=yrange) 
    if zrange:
        fig.update_scenes(zaxis_range=zrange) 

    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             nticks=4,
    #             range=xrange,
    #         ),
    #         yaxis=dict(
    #             nticks=4,
    #             range=yrange,
    #         ),
    #         zaxis=dict(
    #             nticks=4,
    #             range=zrange,
    #         ),
    #     )
    # )
    fig = update_style(fig, x_title, y_title, z_title)
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
    color_array=None,
    theme_id=None,
    print_dim_list=None,
):
    """It statically shows the simulation traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if color_array is None:
        color_array = color_array_def
    num_theme = len(color_array)
    global color_cnt
    queue = [root]
    color_id = 0
    if theme_id == None:
        theme_id = color_cnt % num_theme
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
            go.Scatter3d(
                x=trace[:, x_dim],
                y=trace[:, y_dim],
                z=trace[:, z_dim],
                mode="lines",
                marker=dict(color=color_array[theme_id][color_id]),
                line=dict(color=color_array[theme_id][color_id], 
                          width=18
                          ),
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
                # showlegend=True,
                showlegend=False,
            )
        )
        if node.assert_hits != None and agent_id in node.assert_hits:
            fig.add_trace(
                go.Scatter3d(
                    x=[trace[-1, x_dim]],
                    y=[trace[-1, y_dim]],
                    z=[trace[-1, z_dim]],
                    mode="markers+text",
                    text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                    # textfont={"color": "grey"},
                    marker={"size": 4, "color": "black"},
                    #  legendgroup=agent_id,
                    #  legendgrouptitle_text=agent_id,
                    #  name=str(round(start[0], 2))+'-'+str(round(end[0], 2)) +
                    #  '-'+str(count_dict[time])+'hit',
                    showlegend=False,
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
    color_array=None,
    theme_id=None,
    print_dim_list=None,
    combine_rect=None,
):
    """It statically shows the verfication traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
    if color_array is None:
        color_array = color_array_def
    num_theme = len(color_array)    
    global color_cnt
    if theme_id == None:
        theme_id = color_cnt % num_theme
        color_cnt = (color_cnt + 1) % num_theme
    queue = [root]
    show_legend = True
    fillcolor = color_array[theme_id][5]
    linecolor = color_array[theme_id][4]
    i = [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7]
    j = [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3]
    k = [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        max_id = len(trace) - 1
        if combine_rect == None:
            num_cube = 0

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
                fig.add_trace(go.Scatter3d(                    
                    x=[px[0],px[1],px[1],px[0],px[0],px[0],px[1],px[1],px[1],px[1],px[1],px[1],px[0],px[0],px[0],px[0]],
                    y=[py[0],py[0],py[1],py[1],py[0],py[0],py[0],py[0],py[0],py[1],py[1],py[1],py[1],py[1],py[1],py[0]],
                    z=[pz[0],pz[0],pz[0],pz[0],pz[0],pz[1],pz[1],pz[0],pz[1],pz[1],pz[0],pz[1],pz[1],pz[0],pz[1],pz[1]],
                    mode='lines',
                    # name='',
                    showlegend=False,
                    line=dict(color= 'rgb(70,70,70)', width=0.1))  )
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
                    color=color_array[theme_id][1],
                    flatshading=True,
                )
            )
            if node.assert_hits != None and agent_id in node.assert_hits:
                fig.add_trace(
                    go.Scatter3d(
                        x=[trace[-1, x_dim]],
                        y=[trace[-1, y_dim]],
                        z=[trace[-1, z_dim]],
                        mode="markers+text",
                        text=["HIT:\n" + a for a in node.assert_hits[agent_id]],
                        # textfont={"color": "grey"},
                        marker={"size": 4, "color": "black"},
                        showlegend=False,
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
            #                                'size': 1, 'color': color_array[theme_id][0]},
            #                            line={
            #                                'width': 10, 'color': color_array[theme_id][0]},
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


def update_style(fig: go.Figure() = go.Figure(), x_title=None, y_title=None, z_title=None):
    # fig.update_traces(line={'width': 3})
    fig.update_layout(
        # paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(font={"size": 14})
    linewidth = 4
    gridwidth = 2
    fig.update_layout(scene = dict(
                        xaxis = dict(
                        title=x_title if x_title is not None else None,
                        gridcolor="white",
                        showgrid=True,
                        showbackground=True,
                        zerolinecolor="white",),
                        yaxis = dict(
                        title=y_title if y_title is not None else None,
                        gridcolor="white",
                        showgrid=True,
                        showbackground=True,
                        zerolinecolor="white",),
                        zaxis = dict(
                        title=z_title if z_title is not None else None,
                        gridcolor="white",
                        showgrid=True,
                        showbackground=True,
                        zerolinecolor="white",),
                        ))
    # fig.update_yaxes(
    #     showline=True,
    #     linewidth=linewidth,
    #     linecolor="Gray",
    #     showgrid=True,
    #     gridwidth=gridwidth,
    #     gridcolor="LightGrey",
    # )
    # fig.update_xaxes(
    #     showline=True,
    #     linewidth=linewidth,
    #     linecolor="Gray",
    #     showgrid=True,
    #     gridwidth=gridwidth,
    #     gridcolor="LightGrey",
    # )
    return fig

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
    "blue": 1,
    "green": 2,
    "yellow": 3,
    "yellowgreen": 4,
    "lime": 5,
    "springgreen": 6,
    "cyan": 7,
    "cyanblue": 8,
    "orange": 9,
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

def reachtube_anime_3d(
    root: Union[AnalysisTree, AnalysisTreeNode],
    fig=go.Figure(),
    x_dim: int = 1,
    x_title: str = None, 
    y_dim: int = 2,
    y_title: str = None, 
    z_dim: int = 3,
    z_title: str = None, 
    print_dim_list=None,
    color_array=scheme_list,
    sample_rate=1,
    combine_rect=None,
    save = False,
    name = "animation.gif"
):
    """It gives the animation of the verification traces in 3D."""
    if isinstance(root, AnalysisTree):
        root = root.root

   
    num_digit = 3
    root = sample_trace(root, sample_rate)
    agent_list = list(root.agent.keys())
    timed_point_dict = {}
    queue = [root]
    

    num_dim = np.array(root.trace[agent_list[0]]).shape[1]
    check_dim(num_dim, x_dim, y_dim, z_dim, print_dim_list)
    if print_dim_list is None:
        print_dim_list = range(0, num_dim)
    
    num_points = 0
    while queue:
        node = queue.pop()
        traces = node.trace
        for agent_id in traces:
            trace = np.array(traces[agent_id])
            if trace[0][0] > 0:
                trace = trace[8:]
            for i in range(0, len(trace) - 1, 2):
                time_point = round(trace[i][0], num_digit)
                
                rect = [trace[i][0:].tolist(), trace[i + 1][0:].tolist()]
                if time_point not in timed_point_dict:
                    num_points += 1
                    timed_point_dict[time_point] = {agent_id: [rect]}
                else:
                    if agent_id in timed_point_dict[time_point]:
                        timed_point_dict[time_point][agent_id].append(rect)
                    else:
                        timed_point_dict[time_point][agent_id] = [rect]

        queue += node.child

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    time_steps = sorted(timed_point_dict.keys())

    #init agent colors
    time_step = time_steps[0]
    step_info = timed_point_dict[time_step]
    agents = list(step_info.keys())
    agent_colors = {}
    for i,agent in enumerate(agents):
        agent_colors[agent] = color_array[i]

    # init plot
    def init():
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_zlabel(z_title)    
        ax.set_xlim([0, 55])
        ax.set_ylim([-10, 260])
        ax.set_zlim([-50, 100])
        ax.set_title('3D Rectangle Animation')

    def update(frame):
        time_step = time_steps[frame]
        step_info = timed_point_dict[time_step]
        agents = list(step_info.keys())

        for agent in agents:
            agent_rectangles = step_info[agent]
            for rect in agent_rectangles:
                x_min = rect[0][x_dim]
                y_min = rect[0][y_dim]
                z_min = rect[0][z_dim]

                x_max = rect[1][x_dim]
                y_max = rect[1][y_dim]
                z_max = rect[1][z_dim]

                vertices = np.array([
                    [x_min, y_min, z_min],  # Bottom-left-front
                    [x_max, y_min, z_min],  # Bottom-right-front
                    [x_max, y_max, z_min],  # Bottom-right-back
                    [x_min, y_max, z_min],  # Bottom-left-back
                    [x_min, y_min, z_max],  # Top-left-front
                    [x_max, y_min, z_max],  # Top-right-front
                    [x_max, y_max, z_max],  # Top-right-back
                    [x_min, y_max, z_max]   # Top-left-back
                ])

                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
                ]

                # Plot the rectangle
                cuboid = Poly3DCollection(faces, facecolors=agent_colors[agent], linewidths=1, edgecolors=agent_colors[agent], alpha=.125)
                ax.add_collection3d(cuboid)
                ax.view_init(elev=15, azim=frame*0.75)


    # Create animation
    ani = FuncAnimation(fig, update, frames=len(time_steps), init_func=init, interval = 25, repeat=True)
    if save:
        ani.save(name)
    # plt.show()
    return fig