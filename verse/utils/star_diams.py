import numpy as np
from verse.stars.starset import StarSet, HalfSpace
from typing_extensions import List, Callable, Dict
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from z3 import *
from verse.utils.star_manhattan import *
from verse.utils.fixed_points import *

import numpy as np 
from scipy.integrate import ode 
from verse import BaseAgent, Scenario 
from verse.analysis.analysis_tree import AnalysisTree 
from verse.analysis import AnalysisTreeNode, AnalysisTree


def over_approx_rectangle(trace, time_horizon, time_step):
    time_steps = np.linspace(0, time_horizon, int(time_horizon/time_step) + 1, endpoint= True)
    for i in range(len(time_steps) - 1):
        reach_tubes = reach_star(trace, time_steps[i] - time_step*.05, time_steps[i+1] + time_step*.05)
        agents = list(reach_tubes.keys())
        for agent in agents:
            if reach_tubes[agent] is not None:
                if len(reach_tubes[agent]) > 0:
                    star = reach_tubes[agent][0]
                    rect = [time_steps[i], star.overapprox_rectangle()]
                    print(rect)


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

#using Alex's reach_at_fix function
def time_step_diameter(trace, time_horizon, time_step):
    time_steps = np.linspace(0, time_horizon, int(time_horizon/time_step) + 1, endpoint= True)
    time_steps = np.append(time_steps, [time_steps[-1] + time_step])
    diameters = []
    #breakpoint()
    for i in range(len(time_steps) - 1):
        #breakpoint()
        curr_diam = []
        reach_tubes = reach_star(trace, time_steps[i] - time_step*.1, time_steps[i+1] - time_step*.1)
        agents = list(reach_tubes.keys())
        #breakpoint()
        for agent in agents:
            if reach_tubes[agent] is not None:
                if len(reach_tubes[agent]) > 0:
                    for j in range(0, len(reach_tubes[agent])):
                        star = reach_tubes[agent][j]
                        diam, result = star_manhattan_distance(star.center, star.basis, star.C, star.g)
                        if(result == 1):
                            curr_diam.append(diam)
                        else:
                            rect = star.overapprox_rectangle()
                            curr_diam.append(manhattan(rect[0], rect[1]))
                    # breakpoint()
                    # print(time_steps[i])
                    # print(time_steps[i+1])
                    # print(reach_tubes)
                    # print(star.overapprox_rectangle())
        if len(curr_diam) > 0:
            #print(curr_diam)
            diameters.append(max(curr_diam))
        #breakpoint()
        #print({"timesteps": time_steps[i]})

    #print(diameters)
    return diameters

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def time_step_diameter_rect(tree: AnalysisTree, time_horizon, time_step):
    time_steps = np.linspace(0, time_horizon, int(time_horizon/time_step) + 1, endpoint= True)
    diameters = []
    for i in range(len(time_steps) - 1):
        curr_diam = []
        reach_tubes = reach_at_fix(tree, time_steps[i] - time_step*.05, time_steps[i+1] + time_step*.05)
        nodes = list(reach_tubes.keys())
        for node in nodes:
            if reach_tubes[node] is not None:
                curr_diam.append(manhattan(reach_tubes[node][0], reach_tubes[node][1]))
        if len(curr_diam) > 0:
            diameters.append(max(curr_diam))
    return diameters

### assuming mode is not a parameter
def reach_star(traces, t_l: float = 0, t_u: float = None) -> Dict[str, List[StarSet]]: 
    reach = {}

    nodes = traces.nodes 
    agents = list(nodes[0].trace.keys())

    # if t_u is None:
    #     for agent in trace:
    #         t_u = trace[agent][-1][0] # T
    #         break

    for node in nodes:
        trace = node.trace
        for agent in agents:
            length = len(trace[agent])
            for i in range(length):
                cur = trace[agent][i]
                if cur[0]<t_l:
                    continue
                if cur[0]>t_u:
                    break
                if agent not in reach:
                    reach[agent] = []
                reach[agent].append(cur[1]) # just store the star set
    
    return reach