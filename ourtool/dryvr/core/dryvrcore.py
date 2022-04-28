"""
This file contains core functions used by DryVR
"""
from __future__ import print_function
import random

import networkx as nx
import numpy as np
import igraph


from ourtool.dryvr.common.constant import *
from ourtool.dryvr.common.io import writeReachTubeFile
from ourtool.dryvr.common.utils import randomPoint, calcDelta, calcCenterPoint, trimTraces
from ourtool.dryvr.discrepancy.Global_Disc import get_reachtube_segment
# from ourtool.dryvr.tube_computer.backend.reachabilityengine import ReachabilityEngine
# from ourtool.dryvr.tube_computer.backend.initialset import InitialSet

def build_graph(vertex, edge, guards, resets):
    """
    Build graph object using given parameters
    
    Args:
        vertex (list): list of vertex with mode name
        edge (list): list of edge that connects vertex
        guards (list): list of guard corresponding to each edge
        resets (list): list of reset corresponding to each edge

    Returns:
        graph object

    """
    g = igraph.Graph(directed=True)
    g.add_vertices(len(vertex))
    g.add_edges(edge)

    g.vs['label'] = vertex
    g.vs['name'] = vertex
    labels = []
    for i in range(len(guards)):
        cur_guard = guards[i]
        cur_reset = resets[i]
        if not cur_reset:
            labels.append(cur_guard)
        else:
            labels.append(cur_guard + '|' + cur_reset)

    g.es['label'] = labels
    g.es['guards'] = guards
    g.es['resets'] = resets

    # if PLOTGRAPH:
    #     graph = igraph.plot(g, GRAPHOUTPUT, margin=40)
    #     graph.save()
    return g


def build_rrt_graph(modes, traces, is_ipynb):
    """
    Build controller synthesis graph object using given modes and traces.
    Note this function is very different from buildGraph function.
    This is white-box transition graph learned from controller synthesis algorithm
    The reason to build it is to output the transition graph to file
    
    Args:
        modes (list): list of mode name
        traces (list): list of trace corresponding to each mode
        is_ipynb (bool): check if it's in Ipython notebook environment

    Returns:
        None

    """
    if is_ipynb:
        vertex = []
        # Build unique identifier for a vertex and mode name
        for idx, v in enumerate(modes):
            vertex.append(v + "," + str(idx))

        edge_list = []
        edge_label = {}
        for i in range(1, len(modes)):
            edge_list.append((vertex[i - 1], vertex[i]))
            lower = traces[i - 1][-2][0]
            upper = traces[i - 1][-1][0]
            edge_label[(vertex[i - 1], vertex[i])] = "[" + str(lower) + "," + str(upper) + "]"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(edge_list)
        pos = nx.spring_layout(nx_graph)
        colors = ['green'] * len(nx_graph.nodes())
        fig.suptitle('transition graph', fontsize=10)
        nx.draw_networkx_labels(nx_graph, pos)
        options = {
            'node_color': colors,
            'node_size': 1000,
            'cmap': plt.get_cmap('jet'),
            'arrowstyle': '-|>',
            'arrowsize': 50,
        }
        nx.draw_networkx(nx_graph, pos, arrows=True, **options)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_label)
        fig.canvas.draw()

    else:
        g = igraph.Graph(directed=True)
        g.add_vertices(len(modes))
        edges = []
        for i in range(1, len(modes)):
            edges.append([i - 1, i])
        g.add_edges(edges)

        g.vs['label'] = modes
        g.vs['name'] = modes

        # Build guard
        guard = []
        for i in range(len(traces) - 1):
            lower = traces[i][-2][0]
            upper = traces[i][-1][0]
            guard.append("And(t>" + str(lower) + ", t<=" + str(upper) + ")")
        g.es['label'] = guard
        graph = igraph.plot(g, RRTGRAPHPOUTPUT, margin=40)
        graph.save()


def simulate(g, init_condition, time_horizon, guard, sim_func, reset, init_vertex, deterministic):
    """
    This function does a full hybrid simulation

    Args:
        g (obj): graph object
        init_condition (list): initial point
        time_horizon (float): time horizon to simulate
        guard (src.core.guard.Guard): list of guard string corresponding to each transition
        sim_func (function): simulation function
        reset (src.core.reset.Reset): list of reset corresponding to each transition
        init_vertex (int): initial vertex that simulation starts
        deterministic (bool) : enable or disable must transition

    Returns:
        A dictionary obj contains simulation result.
        Key is mode name and value is the simulation trace.

    """

    ret_val = igraph.defaultdict(list)
    # If you do not declare initialMode, then we will just use topological sort to find starting point
    if init_vertex == -1:
        computer_order = g.topological_sorting(mode=igraph.OUT)
        cur_vertex = computer_order[0]
    else:
        cur_vertex = init_vertex
    remain_time = time_horizon
    cur_time = 0

    # Plus 1 because we need to consider about time
    dimensions = len(init_condition) + 1

    sim_result = []
    # Avoid numeric error
    while remain_time > 0.01:

        if DEBUG:
            print(NEWLINE)
            print((cur_vertex, remain_time))
            print('Current State', g.vs[cur_vertex]['label'], remain_time)

        if init_condition is None:
            # Ideally this should not happen
            break

        cur_successors = g.successors(cur_vertex)
        transit_time = remain_time
        cur_label = g.vs[cur_vertex]['label']

        cur_sim_result = sim_func(cur_label, init_condition, transit_time)
        if isinstance(cur_sim_result, np.ndarray):
            cur_sim_result = cur_sim_result.tolist()

        if len(cur_successors) == 0:
            # Some model return numpy array, convert to list
            init_condition, truncated_result = guard.guard_sim_trace(
                cur_sim_result,
                ""
            )
            cur_successor = None

        else:
            # First find all possible transition
            # Second randomly pick a path and time to transit
            next_modes = []
            for cur_successor in cur_successors:
                edge_id = g.get_eid(cur_vertex, cur_successor)
                cur_guard_str = g.es[edge_id]['guards']
                cur_reset_str = g.es[edge_id]['resets']

                next_init, truncated_result = guard.guard_sim_trace(
                    cur_sim_result,
                    cur_guard_str
                )

                next_init = reset.reset_point(cur_reset_str, next_init)
                # If there is a transition
                if next_init:
                    next_modes.append((cur_successor, next_init, truncated_result))
            if next_modes:
                # It is a non-deterministic system, randomly choose next state to transit
                if not deterministic:
                    cur_successor, init_condition, truncated_result = random.choice(next_modes)
                # This is deterministic system, choose earliest transition
                else:
                    shortest_time = float('inf')
                    for s, i, t in next_modes:
                        cur_tube_time = t[-1][0]
                        if cur_tube_time < shortest_time:
                            cur_successor = s
                            init_condition = i
                            truncated_result = t
                            shortest_time = cur_tube_time
            else:
                cur_successor = None
                init_condition = None

        # Get real transit time from truncated result
        transit_time = truncated_result[-1][0]
        ret_val[cur_label] += truncated_result
        sim_result.append(cur_label)
        for simRow in truncated_result:
            simRow[0] += cur_time
            sim_result.append(simRow)

        remain_time -= transit_time
        print("transit time", transit_time, "remain time", remain_time)
        cur_time += transit_time
        cur_vertex = cur_successor

    writeReachTubeFile(sim_result, SIMRESULTOUTPUT)
    return ret_val


def calc_bloated_tube(
        mode_label,
        initial_set,
        time_horizon,
        sim_func,
        bloating_method,
        kvalue,
        sim_trace_num,
        guard_checker=None,
        guard_str="",
        lane_map = None
    ):
    """
    This function calculate the reach tube for single given mode

    Args:
        mode_label (str): mode name
        initial_set (list): a list contains upper and lower bound of the initial set
        time_horizon (float): time horizon to simulate
        sim_func (function): simulation function
        bloating_method (str): determine the bloating method for reach tube, either GLOBAL or PW
        sim_trace_num (int): number of simulations used to calculate the discrepancy
        kvalue (list): list of float used when bloating method set to PW
        guard_checker (src.core.guard.Guard or None): guard check object
        guard_str (str): guard string
       
    Returns:
        Bloated reach tube

    """
    random.seed(4)
    cur_center = calcCenterPoint(initial_set[0], initial_set[1])
    cur_delta = calcDelta(initial_set[0], initial_set[1])
    traces = [sim_func(mode_label, cur_center, time_horizon, lane_map)]
    # Simulate SIMTRACENUM times to learn the sensitivity
    for _ in range(sim_trace_num):
        new_init_point = randomPoint(initial_set[0], initial_set[1])
        traces.append(sim_func(mode_label, new_init_point, time_horizon, lane_map))

    # Trim the trace to the same length
    traces = trimTraces(traces)
    if guard_checker is not None:
        # pre truncated traces to get better bloat result
        max_idx = -1
        for trace in traces:
            ret_idx = guard_checker.guard_sim_trace_time(trace, guard_str)
            max_idx = max(max_idx, ret_idx + 1)
        for i in range(len(traces)):
            traces[i] = traces[i][:max_idx]

    # The major
    if bloating_method == GLOBAL:
        # TODO: Replace this with ReachabilityEngine.get_reachtube_segment
        cur_reach_tube: np.ndarray = get_reachtube_segment(np.array(traces), np.array(cur_delta), "PWGlobal")
        # cur_reach_tube: np.ndarray = ReachabilityEngine.get_reachtube_segment_wrapper(np.array(traces), np.array(cur_delta))
    elif bloating_method == PW:
        # TODO: Replace this with ReachabilityEngine.get_reachtube_segment
        cur_reach_tube: np.ndarray = get_reachtube_segment(np.array(traces), np.array(cur_delta), "PW")
        # cur_reach_tube: np.ndarray = ReachabilityEngine.get_reachtube_segment_wrapper(np.array(traces), np.array(cur_delta))
    else:
        raise ValueError("Unsupported bloating method '" + bloating_method + "'")
    final_tube = np.zeros((cur_reach_tube.shape[0]*2, cur_reach_tube.shape[2]))
    final_tube[0::2, :] = cur_reach_tube[:, 0, :]
    final_tube[1::2, :] = cur_reach_tube[:, 1, :]
    print(final_tube.tolist()[-2], final_tube.tolist()[-1])
    return final_tube.tolist()
