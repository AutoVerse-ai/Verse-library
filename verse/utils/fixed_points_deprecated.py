from typing import Tuple, List 
import numpy as np 
from scipy.integrate import ode 
from verse import BaseAgent, Scenario,  ScenarioConfig
from verse.analysis.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy
from typing import Dict, Sequence
from z3 import *
from verse.analysis.simulator import convertStrToEnum
import types

### returns whether hyperrectangle 1 is contained within hyperrectangle 2
### expects to be given two states (lower and upper bounds for each state) from a trace/set of traces
def contained_single(h1: List[np.array], h2: List[np.array]) -> bool: 
    lower: List[bool] = h2[0][1:] <= h1[0][1:] # consider after first index, time -- move to top
    upper: List[bool] = h1[1][1:] <= h2[1][1:]
    return all(lower) and all(upper)

### extracts nodes reached between a certain lower time bound and upper time bound
### by default, will only extract the states from the very last
def reach_at(trace: AnalysisTree, t_lower: float = None, t_upper: float = None) -> Dict[str, Dict[str, List[List[np.array]]]]: 
    nodes = trace.nodes 
    agents = nodes[0].agent.keys() # list of agents\
    reached: Dict[str, Dict[str, List[List[np.array]]]] = {}
    for agent in agents:
        reached[agent] = {}
    T = 0 # we know time horizon occurs in the leaf node, by definition, shouldn't matter which leaf node we check
    if t_lower is None or t_upper is None:
        last: list[AnalysisTreeNode] = trace.get_leaf_nodes(trace.root)
        T = last[0].trace[list(agents)[0]][-1][0] # using list casting since the agent doesn't matter
    for node in nodes:
        modes = node.mode 
        for agent in agents:
            ### make error message for t_lower and t_upper
            if t_lower is not None and t_upper is not None: ### may want to seperate this out into a seperate function
                for i in range(0, len(node.trace[agent]), 2): # just check the upper bound, time difference between lower and upper known to be time step of scenario
                    lower = node.trace[agent][i]
                    upper = node.trace[agent][i+1]
                    if upper[0]>=t_lower and upper[0]<=t_upper: ### think about why cut off here explicitly -- includes T-delta T but not T if upper bound is T-delta T
                        state = [lower, upper]
                        if modes[agent] not in reached[agent]: #initializing mode of agent in dict if not already there
                            reached[agent][modes[agent]] = []
                        reached[agent][modes[agent]].append(state) #finally, add state to reached
            else: ### for now, just assume if t_lower and t_upper not supplied, we just want last node 
                if node.trace[agent][-1][0]==T:
                    if modes[agent] not in reached[agent]: #initializing mode of agent in dict if not already there
                        reached[agent][modes[agent]] = []
                    reached[agent][modes[agent]].append([node.trace[agent][-2], node.trace[agent][-1]]) 
    return reached

### assuming getting inputs from reach_at calls
### this algorithm shouldn't work, it doesn't consider the composition of all agents
def contain_all(reach1: Dict[str, Dict[str, List[List[np.array]]]], reach2: Dict[str, Dict[str, List[List[np.array]]]], state_len: int = None) -> bool:
    if state_len is None: ### taking from input now, could also derive from either reach1 or reach2
        agents = list(reach1.keys()) # if this works 100%, then get rid of if and option to pass state_len as input
        state_len = len(reach1[agents[0]][list(reach1[agents[0]].keys())[0]][0][0])-1 # strip one dimension to account for ignoring time
    P = RealVector('p', state_len) 
    in_r1 = Or() #just set this boolean statement to just Or for now, can sub with Or(False) if having issues
    in_r2 = Or() #as above, sub with Or(False) if having issues
    s = Solver()
    ### eventually want to check sat of (in_r1 and not in_r2)
    for agent in reach1.keys():
        for mode in reach1[agent].keys():
            for hr in reach1[agent][mode]: # hr is a hyperrectangle defined by lower and upper bounds stored in a list
                includes = And()
                for i in range(state_len):
                    includes = And(includes, P[i] >= hr[0][i+1], P[i] <= hr[1][i+1]) ### include the previous conditions and new bounds for the new dimension
                in_r1 = Or(in_r1, includes) ### in_r1 should check to if there exists a point that can be any hyperrectangle from set r1
    ### could probably seperate out above and below loops into subroutine
    for agent in reach2.keys():
        for mode in reach2[agent].keys():
            for hr in reach2[agent][mode]:
                includes = And()
                for i in range(state_len):
                    includes = And(includes, P[i] >= hr[0][i+1], P[i] <= hr[1][i+1]) ### include the previous conditions and new bounds for the new dimension
                in_r2 = Or(in_r2, includes) ### in_r2 should check to if there exists a point that can be any hyperrectangle from set r2
    
    s.add(in_r1, Not(in_r2))
    return s.check() == unsat

### assuming user has T and t_step from previous scenario definition 
def fixed_points_sat(tree: AnalysisTree, T: float = 40, t_step: float = 0.01) -> bool: 
    reach_end = reach_at(tree)
    reach_else = reach_at(tree, 0, T-t_step+t_step*.01) # need to add some offset because of some potential decimical diffs  
    return contain_all(reach_end, reach_else)

def pp_old(reach_set: Dict[str, Dict[str, List[List[np.array]]]]) -> None:
    for agent in reach_set:
        for mode in reach_set[agent]:
            for state in reach_set[agent][mode]:
                print(f'Agent {agent} in mode {mode}: {state}')