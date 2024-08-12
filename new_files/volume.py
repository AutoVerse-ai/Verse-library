import numpy as np 
from scipy.integrate import ode 
from verse import BaseAgent, Scenario 
from verse.analysis.analysis_tree import AnalysisTree 
from verse.analysis import AnalysisTreeNode, AnalysisTree
from new_files.fp_points import *


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))


#using Alex's reach_at_fix function
def time_step_diameter(tree: AnalysisTree, time_horizon, time_step):
    time_steps = np.arange(0, time_horizon, time_step)
    diameters = []
    for i in range(len(time_steps) - 1):
        curr_diam = []
        reach_tubes = reach_at_fix(tree, time_steps[i], time_steps[i+1])
        nodes = list(reach_tubes.keys())
        for node in nodes:
            curr_diam.append(manhattan(reach_tubes[node][0], reach_tubes[node][1]))
        diameters.append(max(curr_diam))
    return diameters


#using Alex's reach_at_fix function
def time_step_volume(tree: AnalysisTree, time_horizon, time_step):
    time_steps = np.arange(0, time_horizon, time_step)
    volumes = []
    for i in range(len(time_steps) - 1):
        curr_vol = []
        reach_tubes = reach_at_fix(tree, time_steps[i], time_steps[i+1])
        nodes = list(reach_tubes.keys())
        for node in nodes:
            curr_vol.append(rect_volume(reach_tubes[node][0], reach_tubes[node][1]))
        volumes.append(max(curr_vol))
    return volumes
        


def rect_volume(lower, upper):
    vol = 1
    for i in range(len(upper)):
        vol = vol*abs(upper[i] - lower[i])
    return vol    


#make one big hyperrectangle that includes all agents
def node_volume(node: AnalysisTreeNode):
    agents = list(node.trace.keys())
    trace_len = len(node.trace[agents[0]])
    total_volume = 0
    for i in range(0,trace_len,2):
        time_step_volume = 1
        for agent in agents:
            time_step_volume = time_step_volume*rect_volume(node.trace[agent][i][1:], node.trace[agent][i+1][1:])
        total_volume += time_step_volume
    return total_volume

def tree_volume(tree: AnalysisTree):
    nodes = tree.nodes
    volume = 0
    for node in nodes:
        volume = volume + node_volume(node)
    return volume

def root_volume(tree: AnalysisTree):
    return node_volume(tree.root)

def average_tree_volume(tree: AnalysisTree):
    return tree_volume(tree)/len(tree.nodes)

def leaf_tree_volume(tree: AnalysisTree):
    nodes = tree.get_all_nodes(tree.root)
    leaf_nodes = []
    volume = 0
    for node in nodes:
        if len(node.child) == 0:
            leaf_nodes.append(node)
            volume += node_volume(node)
    return volume


