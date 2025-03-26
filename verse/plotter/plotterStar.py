from __future__ import annotations
import matplotlib.pyplot as plt 

import copy
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union
from plotly.graph_objs.scatter import Marker
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map import LaneMap

colors = ['blue', 'green', 'red', 'yellow', 'purple', 'teal','orange']

def plot_reachtube_stars(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    x_dim: int = 0,
    y_dim: int = 1,
    filter: int = 100
):
    print("graphing")
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, filter)
    agent_list = list(root.agent.keys())
    i = 0
    for agent_id in agent_list:
        #print(agent_id)
        reachtube_tree_single(
            root,
            agent_id,
            x_dim,
            y_dim,
            colors[i])
        i = (i+1)%7
    plt.show()

def plot_reach_tube(traces, agent_id, freq = 100):
    trace = np.array(traces[agent_id], freq)
    plot_agent_trace(trace)

def plot_agent_trace(trace, freq=100):
    for i in range(0, len(trace)):
        if i%freq == 0:
            x, y = np.array(trace[i][1].get_verts())
            plt.plot(x, y, lw = 1, color = colors[i%7])
            centerx, centery = trace[i][1].get_center_pt(0, 1)
            plt.plot(centerx, centery, 'o', color = colors[i%7])
    plt.show()

def reachtube_tree_single(root,agent_id,x_dim,y_dim, color):
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        #print(traces)
        if agent_id in traces.keys():
            trace = np.array(traces[agent_id])
            #plot the trace
            for i in range(0, len(trace)):
                #trace[i][1].show()
                #print(trace[i][1])
                x, y = np.array(trace[i][1].get_verts(x_dim, y_dim))
                if len(set(x)) == 1 and len(set(y)) == 1:
                    plt.scatter(x[0], y[0], s=1, color = color)
                    continue
                #print(verts)
                #x=[verts[:,0]]
                #print(x)
                #y=[verts[:,1]]
                #print(y)
                plt.plot(x, y, lw = 1, color = color)
                # plt.scatter(x, y, color=color)
                # x, y = np.array(trace[i][1].get_verts_dim())
                # plt.plot(x, y, 'o', color = 'red')
                #plt.show()
            queue += node.child
        else:
            print("KB: concerning issue, where is an agent??")

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

'''Spend some time to add the sampled points here too'''
def plot_stars_time(root: AnalysisTree, dim: int=0, title: str = 'Star Set Reachtube', scenario_agent = None, lane_map = None, num_samples: int = 100,**kwargs) -> None:
    stars = []
    modes = []
    post_points = []
    agent_list = list(root.root.agent.keys())
    
    j = 1
    for agent in agent_list:
        for node in root.nodes:
            s_mode = []
            if not node.trace:
                continue
            for star in node.trace[agent]:
                s_mode.append(star)
            stars.append(s_mode)
            # modes.append(list(node.mode.values())[0]) # only considering nodes with one mode
            modes.append(node.mode[agent]) # only considering nodes with one mode
        verts = []

        for i in range(len(stars)): # per mode
            s_mode = stars[i] # all the stars in current mdoe
            mode = modes[i]
            v_mode = [] # get the vertices
            star_0 = None # star set
            points = []
            post_points = []
            for star in s_mode:
                if star_0 is None:
                    star_0 = star[1]
                v_mode.append([star[0], *star[1].get_max_min(dim)]) # each vertex is actually the time index and the min/max of that dimension at that time
            v_mode = np.array(v_mode)
            verts.append(v_mode)

            # if scenario_agent is not None:
            #     points = star_0.sample(num_samples)
            #     ts = v_mode[1][0]-v_mode[0][0]
            #     t0 = v_mode[0][0]
            #     T = v_mode[-1][0]-t0
            #     # print(t0, T)
            #     for point in points:
            #         post_points.append(scenario_agent.TC_simulate(mode, point, T, ts, lane_map).tolist())
            #     post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
            #     for t in range(len(post_points[0])):
            #         plt.scatter(np.ones(len(post_points))*post_points[0,t,0]+t0, post_points[:,t,dim+1], color=colors[j%7]) 

        # for i in range(len(verts)):
        #     v_mode = verts[i]
        #     plt.fill_between(v_mode[:, 0], v_mode[:, 1], v_mode[:, 2], color=colors[j%7], alpha=0.5, label=f'Agent {agent}, Mode {mode}')
        # j+=1
        color_ind = 0
        for i in range(len(verts)):
            v_mode = verts[i]
            if i>0 and modes[i]!=modes[i-1]:
                color_ind+=1
            plt.fill_between(v_mode[:, 0], v_mode[:, 1], v_mode[:, 2], color=colors[color_ind*j%7], alpha=0.5, label=f'Agent {agent}, Mode: {modes[i]}')
        j+=1
        
        stars = []
        modes = []
    
        
        
    plt.title(title)
    plt.ylabel(f'Dimension {dim}')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()