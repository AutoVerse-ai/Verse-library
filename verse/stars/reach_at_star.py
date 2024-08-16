import numpy as np
from starset import StarSet, HalfSpace
from verse.utils.utils import sample_rect
from typing_extensions import List, Callable, Dict
from scipy.integrate import ode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy as cp
from z3 import *
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
from verse.utils.star_manhattan import *

def time_step_diameter(trace, time_horizon, time_step):
    time_steps = np.linspace(0, time_horizon, int(time_horizon/time_step) + 1, endpoint= True)
    time_steps = np.append(time_steps, [time_steps[-1] + time_step])
    diameters = []
    #breakpoint()
    for i in range(len(time_steps) - 1):
        curr_diam = []
        reach_tubes = reach_star(trace, time_steps[i] - time_step*.05, time_steps[i+1] - time_step*.05)
        agents = list(reach_tubes.keys())
        #breakpoint()
        for agent in agents:
            if reach_tubes[agent] is not None:
                if len(reach_tubes[agent]) > 0:
                    for j in range(0, len(reach_tubes[agent])):
                        star = reach_tubes[agent][j]
                        curr_diam.append(star_manhattan_distance(star.center, star.basis, star.C, star.g))
                    # breakpoint()
                    # print(time_steps[i])
                    # print(time_steps[i+1])
                    # print(reach_tubes)
                    # print(star.overapprox_rectangle())
        if len(curr_diam) > 0:
            #print(curr_diam)
            diameters.append(max(curr_diam))
        #breakpoint()

    #print(diameters)
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
            for i in range(len(trace[agent])):
                cur = trace[agent][i]
                if cur[0]<t_l:
                    continue
                if cur[0]>t_u:
                    break
                if agent not in reach:
                    reach[agent] = []
                reach[agent].append(cur[1]) # just store the star set
    
    return reach

if __name__ == "__main__":
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    example = {'agent': [[0, StarSet(np.array([0, 0]), np.array([[1, 0], [0, 1]]), C, np.array([1, 1, 1, 1]))]]}
    print(reach_star(example))