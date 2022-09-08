import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Union
from math import pi, cos, sin, acos, asin
from plotly.subplots import make_subplots
from enum import Enum, auto
# import time
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d

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


def simulation_tree_3d(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, z_dim: int = 3, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1):
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
            root, agent_id, fig, x_dim, y_dim, z_dim, scheme_list[i], print_dim_list)
        i = (i+1) % 12

    # fig.update_xaxes(title='x')
    # fig.update_yaxes(title='y')
    # fig.update_layout(legend_title_text='Agent list')
    return fig


def draw_map_3d(map: LaneMap_3d, fig=go.Figure(), fill_type='lines', color='rgba(0,0,0,1)'):
    if map is None:
        return fig
    num = 100
    for lane_idx in map.lane_dict:
        lane = map.lane_dict[lane_idx]
        curr_color = [0, 0, 0, 0]
        opacity = 0.5
        for lane_seg in lane.segment_list:
            if lane_seg.type == 'Straight':
                lane_seg: StraightLane_3d = lane_seg
                if fill_type == 'lines':
                    x, y, z = lane_seg.get_sample_points(num, num)
                    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=opacity,
                                             colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,255,255)']]))
                    fig.update_traces(showscale=False)
                    # oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                    # fig.add_trace(go.Scatter3d(
                    #     x=oc_x, y=oc_y, z=oc_z, opacity=opacity,
                    #     mode='markers',
                    #     marker=dict(color='rgb(255,255,255)')))

            elif lane_seg.type == "Circular":
                lane_seg: CircularLane_3d = lane_seg
                # thetas, ls = np.mgrid[0:2*pi:num*1j, 0:lane_seg.length:num*1j]
                # outer_centers = lane_seg.get_outer_center_vec(ls)
                oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                # print(oc)
                if fill_type == 'lines':
                    # fig.add_trace(go.Surface(x=oc_x, y=oc_y, z=oc_z, opacity=0.1,
                    #                          colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,255,255)']]))
                    # fig.update_traces(showscale=False)
                    fig.add_trace(go.Scatter3d(
                        x=oc_x, y=oc_y, z=oc_z,
                        # opacity=opacity,
                        showlegend=False,
                        mode='lines',
                        marker=dict(color='rgb(255,255,255)',
                                    ),
                        line=dict(color='rgb(0,0,0)', width=10
                                  )))
            else:
                raise ValueError(f'Unknown lane segment type {lane_seg.type}')
    return fig


def simulation_tree_single_3d(root: Union[AnalysisTree, AnalysisTreeNode], agent_id, fig: go.Figure() = go.Figure(), x_dim: int = 1, y_dim: int = 2, z_dim: int = 3, color=None, print_dim_list=None):
    """It statically shows the simulation traces of one given agent."""
    if isinstance(root, AnalysisTree):
        root = root.root
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

        fig.add_trace(go.Scatter3d(x=trace[:, x_dim], y=trace[:, y_dim], z=trace[:, z_dim],
                                   mode='lines',
                                   marker=dict(
            color=colors[scheme_dict[color]][color_id]),
            line=dict(
            color=colors[scheme_dict[color]][color_id], width=10),
            text=[
            ['{:.2f}'.format(trace[i, j])for j in print_dim_list] for i in range(trace.shape[0])],
            legendgroup=agent_id,
            legendgrouptitle_text=agent_id,
            name=str(round(start[0], 2))+'-'+str(round(end[0], 2)) +
            '-'+str(count_dict[time]),
            showlegend=True))

        color_id = (color_id+4) % 5
        queue += node.child
    # fig.update_layout(legend=dict(
    #     groupclick="toggleitem",
    #     itemclick="toggle",
    #     itemdoubleclick="toggleothers"
    # ))
    return fig


def check_dim(num_dim: int, x_dim: int = 1, y_dim: int = 2, z_dim: int = 3, print_dim_list=None):
    if x_dim < 0 or x_dim >= num_dim:
        raise ValueError(f'wrong x dimension value {x_dim}')
    if y_dim < 0 or y_dim >= num_dim:
        raise ValueError(f'wrong y dimension value {y_dim}')
    if z_dim < 0 or z_dim >= num_dim:
        raise ValueError(f'wrong z dimension value {z_dim}')
    if print_dim_list is None:
        return True
    for i in print_dim_list:
        if y_dim < 0 or y_dim >= num_dim:
            raise ValueError(f'wrong printed dimension value {i}')
    return True


def sample_trace(root, sample_rate: int = 1):
    queue = [root]
    # print(root.trace)
    if root.type == 'reachtube':
        sample_rate = sample_rate*2
        while queue != []:
            node = queue.pop()
            for agent_id in node.trace:
                trace_length = len(node.trace[agent_id])
                tmp = []
                for i in range(0, trace_length, sample_rate):
                    if i+sample_rate-1 < trace_length:
                        tmp.append(node.trace[agent_id][i])
                        tmp.append(node.trace[agent_id][i+sample_rate-1])
                node.trace[agent_id] = tmp
            queue += node.child
    else:
        while queue != []:
            node = queue.pop()
            for agent_id in node.trace:
                node.trace[agent_id] = [node.trace[agent_id][i]
                                        for i in range(0, len(node.trace[agent_id]), sample_rate)]
            queue += node.child
    return root
