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
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stanleybak_closed_loop.acasxu_dubins import State, state7_to_state5

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

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

def get_point_tau(own_state: np.ndarray, int_state: np.ndarray) -> float:
    z_own, z_int = own_state[2], int_state[2]
    vz_own, vz_int = own_state[-1]*np.sin(own_state[-2]), int_state[-1]*np.sin(int_state[-2])
    if (vz_own == vz_int) and abs(z_int-z_int)<100:
        return 0
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
    car2 = CarAgent('car2',file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(parallel=False))
    T = 10
    Tv = 1
    ts = 0.01
    N = 1
    # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

    scenario.config.print_level = 0
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    scenario.add_agent(car)
    scenario.add_agent(car2)
    start = time.perf_counter()
    # trace = scenario.simulate(Tv, ts)
    # id = 1+trace.root.id
    # net = 0 # eventually this could be modified in the loop by some cmd_list var
    # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
    models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
    norm = float("inf")

    # queue = deque()
    # queue.append(trace.root) # queue should only contain ATNs  
    ### begin looping
    traces = []
    for i in range(N):
        scenario.set_init(
            # [[[-2, -1, -2, np.pi, np.pi/6, 100], [-1,1, -1, np.pi, np.pi/6, 100]],
            # [[-1001, -1, 499, 0,0, 100], [-999, 1, 500, 0,0, 100]]],
            [[[-1, -1, -1, np.pi, np.pi/6, 100], [-1,-1, -1, np.pi, np.pi/6, 100]],
            [[-1000, 1, 500, 0,0, 100], [-1000, 1, 500, 0,0, 100]]],
            # [[[0, 0, 0, np.pi, np.pi/6, 100], [0, 0, 0, np.pi, np.pi/6, 100]],
            # [[[-100, -100, 0, np.pi, np.pi/6, 100], [100, 100, 0, np.pi, np.pi/6, 100]],
            # #   [[-2001, -1, 999, 0,0, 100], [-1999, 1, 1001, 0,0, 100]]],
            #     [[-4000, 0, 1000, 0,0, 100], [-4000, 0, 1000, 0,0, 100]]],
            [(AgentMode.COC,  ), (AgentMode.COC,  )]
        )
        trace = scenario.simulate(Tv, ts) # this is the root
        id = 1+trace.root.id
        # net = 0 # eventually this could be modified in the loop by some cmd_list var
        # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
        queue = deque()
        
        queue.append(trace.root) # queue should only contain ATNs  
        while len(queue):
            cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
            own_state, int_state = get_final_states_sim(cur_node)
            acas_state = get_acas_state(own_state[1:], int_state[1:]).float()
            '''
            Weirdness with ACAS: due to wrapping, slight deviations of theta around pi lead to huge changes in the advisories of int and own:
            Since theta_int = -theta_own (acas-wise), a slight deviation in theta around pi leads to stuff like theta_int ~= pi and theta_own ~= -pi,
            Which leads to drastically different advisories
            '''
            # print(f'Own cas state: {acas_state}')
            acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
            # ads = model(acas_state.view(1,5)).detach().numpy()
            last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string
            tau_idx = get_tau_idx(own_state[1:], int_state[1:])
            ads = models[last_cmd-1][tau_idx](acas_state.view(1,5)).detach().numpy()

            sb_state = [own_state[i+1] for i in [0,1,3]]+[int_state[i+1] for i in [0,1,3]]+[0]
            s = State(sb_state, tau_idx, -1, 100, 100, last_cmd-1)
            sb_ads, sb_norm_acas = s.update_command()
            print(f'Tau ind: {tau_idx}')
            # print(f'own sb acas : {state7_to_state5(s.vec, s.v_own, s.v_int)}')
            print(f'Own acas state {get_acas_state(own_state[1:], int_state[1:]).numpy()}')
            # print(f'own sb acas norm: {sb_norm_acas}')
            # print(f'Own acas state norm: {acas_state}\n')
            # print(f'Own sb advisory scores: {sb_ads}')
            # print(f'Own advisory scores: {ads}\n\n')
            new_mode = np.argmin(ads[0])+1 # will eventually be a list

            last_cmd_2 = getattr(AgentMode, cur_node.mode['car2'][0]).value
            tau_idx_2 = get_tau_idx(int_state[1:], own_state[1:]) 
            acas_state_2 = get_acas_state(int_state[1:], own_state[1:]).float()
            print(f'Int acas state {acas_state_2.numpy()}\n')
            # print(f'\n_______________\nInt acas state: {acas_state_2}')
            acas_state_2 = (acas_state_2-means_for_scaling)/range_for_scaling
            ads_2 = models[last_cmd_2-1][tau_idx_2](acas_state_2.view(1,5)).detach().numpy()
            # print(f'Int advisory scores: {ads_2}')
            # print(f'Int advisory score using same net as own: {models[last_cmd-1][tau_idx](acas_state_2.view(1,5)).detach().numpy()}')
            new_mode_2 = np.argmin(ads_2[0])+1 # will eventually be a list
            # print(tau_idx, tau_idx_2)
            # print(f'\n_______________\nLast Command: {last_cmd}, Tau Index: {tau_idx}\n Last Command Int: {last_cmd_2}, Tau Index Int: {tau_idx_2}')
            scenario.set_init(
                [[own_state[1:], own_state[1:]], [int_state[1:], int_state[1:]]], # this should eventually be a range 
                [(AgentMode(new_mode),  ),(AgentMode(new_mode_2),  )]
            )
            id += 1
            new_trace = scenario.simulate(Tv, ts)
            temp_root = new_trace.root
            new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
            cur_node.child.append(new_node)
            if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                continue
            queue.append(new_node)
            # print(f'New modes at {cur_node.start_time + Tv} s:', (AgentMode(new_mode),  ),(AgentMode(new_mode_2),))
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
    fig.show()