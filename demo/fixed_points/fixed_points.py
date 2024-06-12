from typing import Tuple, List 
import numpy as np 
from scipy.integrate import ode 
from verse import BaseAgent, Scenario,  ScenarioConfig
from verse.analysis.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from vehicle_controller import VehicleMode
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

from vehicle_controller import VehicleMode, TLMode # this line is not id of controller, but fixed_points should be

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

### revised version of top function
### will now return a list of the vertices of composed hyperrectangles (2 vertices per rect) index by node number
def reach_at_fix(tree: AnalysisTree, t_lower: float = None, t_upper: float = None) -> List[List[float]]:
    nodes: List[AnalysisTreeNode] = tree.nodes 
    agents = nodes[0].agent.keys() # list of agents
    reached: Dict[int, List[List[float]]] = {}
    node_counter = 0
    sim_enum: Dict[str, Dict[str, int]] = {} #indexed by agent and then by mode -- simulates the enum class 
    for agent in agents:
        sim_enum[agent] = {'_index': 0} # initial start with an index of 0 
    T = 0 # we know time horizon occurs in the leaf node, by definition, shouldn't matter which leaf node we check
    if t_lower is None or t_upper is None:
        last: list[AnalysisTreeNode] = tree.get_leaf_nodes(tree.root)
        T = last[0].trace[list(agents)[0]][-1][0] # using list casting since the agent doesn't matter
    for node in nodes:
        reached[node_counter] = None
        modes = node.mode 
        reach_node: Dict[str, List[List[float]]] = {} # will store ordered list of hyperrectangle vertices indexed by agents
        for agent in agents:
            reach_node[agent] = []
            # could turn this into a subroutine
            for mode in modes[agent]:
                if mode not in sim_enum[agent]:
                    sim_enum[agent][mode] = sim_enum[agent]['_index']
                    sim_enum[agent]['_index'] += 1

            ### make error message for t_lower and t_upper
            if t_lower is not None and t_upper is not None: ### may want to seperate this out into a seperate function
                for i in range(0, len(node.trace[agent]), 2): # just check the upper bound, time difference between lower and upper known to be time step of scenario
                    lower = list(node.trace[agent][i][1:]) #strip out time and add agent's mode(s)
                    upper = list(node.trace[agent][i+1][1:])
                    for mode in modes[agent]: # assume that size of modes[agent] doesn't change between nodes
                        ### for now, use this simulated enum instead of trying to figure out how to access enum class
                        lower.append(sim_enum[agent][mode])
                        upper.append(sim_enum[agent][mode]) 
                    if upper[0]>=t_lower and upper[0]<=t_upper: ### think about why cut off here explicitly -- includes T-delta T but not T if upper bound is T-delta T
                        reach_node[agent] += [lower, upper]
            else: ### for now, just assume if t_lower and t_upper not supplied, we just want last node 
                if node.trace[agent][-1][0]==T:
                    lower = list(node.trace[agent][-2][1:])
                    upper = list(node.trace[agent][-1][1:])
                    for mode in modes[agent]: # assume that size of modes[agent] doesn't change between nodes
                        ### for now, use this simulated enum instead of trying to figure out how to access enum class
                        lower.append(sim_enum[agent][mode])
                        upper.append(sim_enum[agent][mode]) 
                    reach_node[agent] += [lower, upper]
        ### loop through each agent and add its state value to an existing composed state
        for agent in agents:
            for i in range(len(reach_node[agent])):
                if reached[node_counter] is None:
                    reached[node_counter] = [[] for _ in range(len(reach_node[agent]))] 
                # print(len(reach_node[agent]))
                reached[node_counter][i] += reach_node[agent][i]
        if reached[node_counter] is not None:
            node_counter += 1
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
    # print(reach_end, reach_else)
    return contain_all(reach_end, reach_else)

### should we getting a scenario or is a trace enough? --shouldn't matter, but should switch to trace to be a bit faster
def fixed_points(scenario: Scenario, agent: str, tol: float = 0.01, t: float = 160) -> bool: 
    #what algorithm should do: 
    #1: compute reachable states for t
    #2: compute reachables states for t+dt
    #3: repeat until either Reach_t+d(n-1)t = Reach_t_t+dnt or no fixed points exist
    # **should be using verify instead of simulate
    # **ask about or find scenarios where loops possible/not possible

    ## v2
    # just check if fixed points for a given time horizon
    # can make helper function that calls this function with different time horizons if necessary
    # should also make agent mode a parameter if only checking for a single agent (doesn't really make sense, should be checking for all, only makes sense for scenarios with only one non-stationary agent)
    # see if rounded to nearest hundredth helps or using a tolerance value -- otherwise need too many simulations  
    cur : AnalysisTree = scenario.verify(t, 0.1)
    all : list[AnalysisTreeNode] = cur.nodes
    last : AnalysisTreeNode = cur.get_leaf_nodes(cur.root)[0] # check to branching structures
    ### present with better example -- non-trivial branching scenario, understand at variable level
    check = list(map(lambda row: np.array(row[1:]), last.trace[agent][-2:])) # lambda function converts to np array and cuts off first col
    reached = list(map(lambda row: np.array(row[1:]), last.trace[agent][:-2][1:]))
    # # print(reached, check)
    for node in all:
        if node.mode == last.mode and node!=last:
            # for state in node.trace[agent]:
            reached += list(map(lambda row: np.array(row[1:]), node.trace[agent][:][1:]))
    # print(len(reached))
    # print(reached)
    # print(check)

    return contained(check, reached, tol)
    # for state in check:
    #     flag = False
    #     # mini = np.inf
    #     # r: np.array
    #     for reach in reached: 
    #         # if np.linalg.norm(state-reach)<mini:
    #         #     mini = np.linalg.norm(state-reach)
    #         #     r = reach
    #         if np.linalg.norm(state-reach)<=tol:
    #             flag = True
    #             break
    #     if not flag:
    #         # print(mini, r, state)
    #         return False

### function that indicates whether some goal state(s) are in the current reachset based on l2 norm of diff between part of goal state(s) and reached states
### revise variable names to be more general
def contained(goal: List[np.array], reached: List[np.array], tol: float = 0.01) -> bool:
    for part in goal:
        flag = False
        for reached_state in reached:
            if np.linalg.norm(part-reached_state)<=tol:
                flag = True
                break
        if not flag:
            return False
    return True

def fixed_points_aa_branching(scenario: Scenario, tol: float = 0.01, t: float = 160) -> bool:
    cur : AnalysisTree = scenario.verify(t, 0.1)
    all : list[AnalysisTreeNode] = cur.nodes
    last : list[AnalysisTreeNode] = cur.get_leaf_nodes(cur.root) # check to branching structures
    ### present with better example -- non-trivial branching scenario, understand at variable level
    agents = last[0].agent.keys() # list of agents
    check: dict[str, list[list[np.array]]] = {} # store each pair as its own list
    for agent in agents:
        for leaf in last:
            if agent not in check:
                check[agent] = [list(map(lambda row: np.array(row[1:]), leaf.trace[agent][-2:]))] # lambda function converts to np array and cuts off first col
            else:
                check[agent].append(list(map(lambda row: np.array(row[1:]), leaf.trace[agent][-2:])))
    reached: dict[str, list[np.array]] = {}
    for agent in agents:
        for leaf in last:
            if agent not in reached: # necessary because not using default dict
                reached[agent] = list(map(lambda row: np.array(row[1:]), last[0].trace[agent][:-2][1:]))
            else: 
                reached[agent] += list(map(lambda row: np.array(row[1:]), last[0].trace[agent][:-2][1:]))
    for node in all:
        # if node.mode == last.mode and node not in last:
        for leaf in last:
            if node.mode == leaf.mode and node not in last:
                for agent in agents:
                    reached[agent] += list(map(lambda row: np.array(row[1:]), node.trace[agent][:][1:]))
                break
    # for agent in agents:
    #     for goal in check[agent]:
    #         print(goal)
    for agent in agents:
        for goal in check[agent]:
            if not contained(goal, reached[agent], tol):
                return False
    return True

### instead of checking each goal state for each agent, makes sure that for each branch, all agents reach fixed points
### use other aa_branching algorithm if only checking one agent
### verified works on easy branching ball scenario, same with above alg
def fixed_points_aa_branching_composed(scenario: Scenario, tol: float = 0.01, t: float = 160) -> bool:
    cur : AnalysisTree = scenario.verify(t, 0.1)
    all : list[AnalysisTreeNode] = cur.nodes
    last : list[AnalysisTreeNode] = cur.get_leaf_nodes(cur.root) # now grabs 
    agents = last[0].agent.keys() # list of agents
    check: dict[str, list[list[np.array]]] = {} # store each pair (min/max of goal hyperrectangle) as its own list
    for agent in agents:
        for leaf in last:
            if agent not in check:
                check[agent] = [list(map(lambda row: np.array(row[1:]), leaf.trace[agent][-2:]))] # lambda function converts to np array and cuts off first col
            else:
                check[agent].append(list(map(lambda row: np.array(row[1:]), leaf.trace[agent][-2:])))
    ### should reached instead also consider the branch that the states came from? i.e. reached[branch][agent] = [reached states] ?  
    reached: dict[str, list[np.array]] = {}
    # initializing reached with the leaf nodes first since unlike other nodes, can't take all states
    for agent in agents:
        for leaf in last:
            if agent not in reached: # necessary because not using default dict
                reached[agent] = list(map(lambda row: np.array(row[1:]), last[0].trace[agent][:-2][1:]))
            else: 
                reached[agent] += list(map(lambda row: np.array(row[1:]), last[0].trace[agent][:-2][1:]))
    for node in all:
        for leaf in last:
            if node.mode == leaf.mode and node not in last:
                for agent in agents:
                    reached[agent] += list(map(lambda row: np.array(row[1:]), node.trace[agent][:][1:]))
                break

    first_agent = list(agents)[0]
    def within_tol(agent: str, ind: int, br: int):
        if np.linalg.norm(reached[agent][ind]-check[agent][i][0])<=tol and np.linalg.norm(reached[agent][ind+1]-check[agent][i][1])<=tol:
            return True
        if np.linalg.norm(reached[agent][ind]-check[agent][i][1])<=tol and np.linalg.norm(reached[agent][ind+1]-check[agent][i][0])<=tol:
            return True
        return False
    for i in range(len(check[first_agent])):
        flag = False #gets set to false at start of each branch/leaf, will only get set to true if we check that final state of all agents contained at some point within reached states
        mins = (np.inf, np.inf, 0, 0, check[first_agent][i][0], check[first_agent][i][1])
        for j in range(0, len(reached[first_agent]), 2): # these two loops should iterate through each branch and every state -- should I be checking all of reach or just the branch's portion of reach?
            # if np.linalg.norm(reached[first_agent][j]-check[first_agent][0])<=tol and np.linalg.norm(reached[first_agent][j+1]-check[first_agent][1])<=tol:                
            if within_tol(first_agent, j, i):
                flag = True #tentatively set flag to true if we found fixed point for first agent
                for agent in agents: #iterate through the other agents and guarentee that 
                    if agent==first_agent:
                        continue
                    # if either min or max of the rectangle isn't within the current state, set flag
                    # if np.linalg.norm(reached[agent][j]-check[agent][0])>tol or np.linalg.norm(reached[agent][j+1]-check[agent][1])>tol:
                    if within_tol(agent, j, i):
                        flag = False
                        break
        if not flag:
            return False
    return True