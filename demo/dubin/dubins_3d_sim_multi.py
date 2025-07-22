from dubins_3d_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import numpy as np 
from collections import deque
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time
import plotly.io as pio



class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

means_for_scaling = torch.FloatTensor([19791.091, 0.0, 0.0, 650.0, 600.0])
range_for_scaling = torch.FloatTensor([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
tau_list = [0, 1, 5, 10, 20, 50, 60, 80, 100] 
# tau = -(z_int-z_own)/(vz_int-vz_own)
# corresponds to last index (1,2,...,9), if >100 return index of 100, if <0 can return index of 0 but also means no chance of collision -- for ref, SB just stops simulating past tau<0
# if between two taus (e.g., 60 and 80), then choose the closer one, rounding down following SB's examples (e.g. if at tau=70, choose index of 60 instead of 80)

# recall new_state is [x,y,z,th,psi,v]
def get_acas_state(own_state: np.ndarray, int_state: np.ndarray) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[3])+np.arctan2(int_state[1]-own_state[1], int_state[0]-own_state[0]))
    psi = wrap_to_pi(int_state[3]-own_state[3])
    return torch.tensor([dist, theta, psi, own_state[-1], int_state[-1]])

def wtp(x: float): 
    return torch.remainder((x + torch.pi), (2 * torch.pi)) - torch.pi

def get_acas_state_torch(own_state: torch.Tensor, int_state: torch.Tensor) -> torch.Tensor:
    dist = torch.sqrt((own_state[:,0:1]-int_state[:,0:1])**2+(own_state[:,1:2]-int_state[:,1:2])**2)
    theta = wtp((2*torch.pi-own_state[:,3:4])+torch.arctan2(int_state[:,1:2], int_state[:,0:1]))
    # theta = wtp((2*torch.pi-own_state[:,2:3])+torch.arctan(int_state[:,1:2]/int_state[:,0:1]))
    psi = wtp(int_state[:,3:4]-own_state[:,3:4])
    # return torch.cat([dist, own_state[:,3:4], psi, own_state[:,3:4], int_state[:,3:4]], dim=1)
    return torch.cat([dist, theta, psi, own_state[:,5:6], int_state[:,5:6]], dim=1)

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_states = [n.trace['car2'][-1], n.trace['car3'][-1]] # eventually do something like [n.trace[f'car{i}'][-1] for i in range(2,k)]
    return own_state, int_states

def get_point_tau(own_state: np.ndarray, int_state: np.ndarray) -> float:
    z_own, z_int = own_state[2], int_state[2]
    vz_own, vz_int = own_state[-1]*np.sin(own_state[-2]), int_state[-1]*np.sin(int_state[-2])
    return -(z_int-z_own)/(vz_int-vz_own) # will be negative when z and vz are not aligned, which is fine

def get_tau_idx(own_state: np.ndarray, int_state: np.ndarray) -> int:
    tau = get_point_tau(own_state, int_state)
    # print(tau)
    if tau<0:
        return 0 # following Stanley Bak, if tau<0, return 0 -- note that Stanley Bak also ends simulation if tau<0
    if tau>tau_list[-1]:
        return len(tau_list)-1 
    for i in range(len(tau_list)-1):
        tau_low, tau_up = tau_list[i], tau_list[i+1]
        if tau_low <= tau <= tau_up:
            if np.abs(tau-tau_low)<=np.abs(tau-tau_up):
                return i
            else:
                return i+1
            
    return len(tau_list)-1 # this should be unreachable


if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller_3d.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    car3 = NPCAgent('car3')
    scenario = Scenario(ScenarioConfig(parallel=False))
    car.set_initial(
        # initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
        initial_state=[[-1, -1010, 0, np.pi/3, np.pi/6, 100], [1, -990, 0, np.pi/3, np.pi/6, 100]],
        # initial_state=[[0, -1001, np.pi/3, 100], [0, -999, np.pi/3, 100]],
        # initial_state=[[0, -1000, np.pi/3, 100], [0, -1000, np.pi/3, 100]],
        initial_mode=(AgentMode.COC,  )
    )
    car2.set_initial(
        # initial_state=[[15, 15, 0, 0.5], [15, 15, 0, 0.5]],
        # initial_state=[[-2000, 0, 1000, 0,0, 100], [-2000, 0, 1000, 0,0, 100]],
        initial_state=[[-2001, -1, 0, 0,0, 100], [-1999, 1, 0, 0,0, 100]],
        initial_mode=(AgentMode.COC,  )
    )
    T = 50
    Tv = 1
    ts = 0.01
    N = 1
    # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

    scenario.config.print_level = 0
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    scenario.add_agent(car)
    scenario.add_agent(car2)
    scenario.add_agent(car3)
    start = time.perf_counter()
    # id = 1+trace.root.id
    # net = 0 # eventually this could be modified in the loop by some cmd_list var
    # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
    models = [[torch.load(f"/Users/bachhoang/Verse-library/auto_LiRPA/examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
    norm = float("inf")

    ### begin looping
    traces = []
    for i in range(N):
        scenario.set_init(
            [[[-100, -1000, 1000, np.pi/3, np.pi/6, 100], [100, -900, 1000, np.pi/3, np.pi/6, 100]],
              [[-2000, 100, 1000, 0,0, 100], [-200, 100, 1000, 0,0, 100]],
              [[2000, 0, 1000, np.pi,0, 100], [2000, 1, 1000, np.pi,0, 100]]],
            [(AgentMode.COC,  ), (AgentMode.COC,  ),(AgentMode.COC,  )]
        )
        trace = scenario.verify(Tv, ts) # this is the root
        id = 1+trace.root.id
        # net = 0 # eventually this could be modified in the loop by some cmd_list var
        # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
        queue = deque()
        queue.append(trace.root) # queue should only contain ATNs  
        while len(queue):
            cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
            own_state, int_states = get_final_states_sim(cur_node)
            acas_states = [get_acas_state(own_state[1:], int_state[1:]).float() for int_state in int_states]
            acas_state = acas_states[0] if acas_states[0][0]<acas_states[1][0] else acas_states[1]
            closer_idx = 0 if acas_states[0][0]<acas_states[1][0] else 1
            acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
            # ads = model(acas_state.view(1,5)).detach().numpy()
            last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string
            tau_idx = get_tau_idx(own_state[1:], int_states[closer_idx][1:]) # point estimate
            # print(f'Last Command: {last_cmd}, Tau Index: {tau_idx}')
            ads = models[last_cmd-1][tau_idx](acas_state.view(1,5)).detach().numpy()
            new_mode = np.argmin(ads[0])+1 # will eventually be a list
            scenario.set_init(
                [[own_state[1:], own_state[1:]], [int_states[0][1:], int_states[0][1:]],
                 [int_states[1][1:], int_states[1][1:]]], # this should eventually be a range 
                [(AgentMode(new_mode),  ),(AgentMode.COC,  ),(AgentMode.COC,  )]
            )
            id += 1
            new_trace = scenario.verify(Tv, ts)
            temp_root = new_trace.root
            new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
            cur_node.child.append(new_node)
            if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                continue
            queue.append(new_node)
        
        trace.nodes = trace._get_all_nodes(trace.root)
        traces.append(trace)

    print(f'Total {N} simulations: {(time.perf_counter()-start):.2f} s')
    trace.nodes = trace._get_all_nodes(trace.root)
    # for node in trace.nodes:
    #     print(f'Start time: {node.start_time}, Mode: ', node.mode['car1'][0])
    fig = go.Figure()
    print(len(traces))
    for trace in traces:
        fig = simulation_tree_3d(trace, fig,1,'x', 2,'y',3,'z')
    # fig = simulation_tree_3d(trace, fig,1,'x', 2,'y',3,'z')
    pio.renderers.default = 'browser'
    fig.show()