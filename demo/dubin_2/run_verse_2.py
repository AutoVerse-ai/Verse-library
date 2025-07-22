
import time
import os
import numpy as np
from enum import Enum,auto
from typing import Tuple, List, Dict

# Define whatever here
#=====================================================================================

from dubins_agent import CarAgent, NPCAgent
from verse import Scenario, ScenarioConfig

from verse.analysis.verifier import ReachabilityMethod
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
from collections import deque
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from verse.analysis.analysis_tree import AnalysisTreeNode
import itertools
from verse.plotter.plotter3D import plotRemaining

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()


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

# recall new_state is [x,y,z,th,psi,v]
### expects some 2x5 np arrays for both sets
def get_acas_reach(own_set: np.ndarray, int_set: np.ndarray) -> List[Tuple[torch.Tensor]]: 
    def dist(pnt1, pnt2):
        return np.linalg.norm(
            np.array(pnt1) - np.array(pnt2)
        )

    def get_extreme(rect1, rect2):
        lb11 = rect1[0]
        lb12 = rect1[1]
        ub11 = rect1[2]
        ub12 = rect1[3]

        lb21 = rect2[0]
        lb22 = rect2[1]
        ub21 = rect2[2]
        ub22 = rect2[3]

        # Using rect 2 as reference
        left = lb21 > ub11 
        right = ub21 < lb11 
        bottom = lb22 > ub12
        top = ub22 < lb12

        if top and left: 
            dist_min = dist((ub11, lb12),(lb21, ub22))
            dist_max = dist((lb11, ub12),(ub21, lb22))
        elif bottom and left:
            dist_min = dist((ub11, ub12),(lb21, lb22))
            dist_max = dist((lb11, lb12),(ub21, ub22))
        elif top and right:
            dist_min = dist((lb11, lb12), (ub21, ub22))
            dist_max = dist((ub11, ub12), (lb21, lb22))
        elif bottom and right:
            dist_min = dist((lb11, ub12),(ub21, lb22))
            dist_max = dist((ub11, lb12),(lb21, ub22))
        elif left:
            dist_min = lb21 - ub11 
            dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
        elif right: 
            dist_min = lb11 - ub21 
            dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
        elif top: 
            dist_min = lb12 - ub22
            dist_max = np.sqrt((ub12 - lb22)**2 + max((ub21-lb11)**2, (ub11-lb21)**2))
        elif bottom: 
            dist_min = lb22 - ub12 
            dist_max = np.sqrt((ub22 - lb12)**2 + max((ub21-lb11)**2, (ub11-lb21)**2)) 
        else: 
            dist_min = 0 
            dist_max = max(
                dist((lb11, lb12), (ub21, ub22)),
                dist((lb11, ub12), (ub21, lb22)),
                dist((ub11, lb12), (lb21, ub12)),
                dist((ub11, ub12), (lb21, lb22))
            )
        return dist_min, dist_max

    own_rect = [own_set[i//2][i%2] for i in range(4)]
    int_rect = [int_set[i//2][i%2] for i in range(4)]
    d_min, d_max = get_extreme(own_rect, int_rect)

    own_ext = [(own_set[i%2][0], own_set[i//2][1]) for i in range(4)] # will get ll, lr, ul, ur in order
    int_ext = [(int_set[i%2][0], int_set[i//2][1]) for i in range(4)] 

    arho_min = np.inf # does this make sense
    arho_max = -np.inf
    arho_pi_wrap = []
    arho_origin_wrap = []
    for own_vert in own_ext:
        for int_vert in int_ext:
            # arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi) 
            # arho_max = max(arho_max, arho)
            # arho_min = min(arho_min, arho)
            arho_pi_wrap.append(wrap_to_pi(np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0])))
            arho_origin_wrap.append(np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi))
    if max(arho_origin_wrap)-min(arho_origin_wrap)<max(arho_pi_wrap)-min(arho_pi_wrap):
        arho_min, arho_max = min(arho_origin_wrap), max(arho_origin_wrap)
    else:
        arho_min, arho_max = min(arho_pi_wrap), max(arho_pi_wrap)

    theta_min = wrap_to_pi((2*np.pi-own_set[1][3])+arho_min)
    theta_max = wrap_to_pi((2*np.pi-own_set[0][3])+arho_max) 
    theta_maxs = []
    theta_mins = []
    if theta_max<theta_min: # bound issue due to wrapping
        theta_mins = [-np.pi, theta_min]
        theta_maxs = [theta_max, np.pi]
    else:
        theta_mins = [theta_min]
        theta_maxs = [theta_max]

    psi_min = wrap_to_pi(int_set[0][3]-own_set[1][3])
    psi_max = wrap_to_pi(int_set[1][3]-own_set[0][3])
    psi_maxs = []
    psi_mins = []
    if psi_max<psi_min: # bound issue due to wrapping
        psi_mins = [-np.pi, psi_min]
        psi_maxs = [psi_max, np.pi]
    else:
        psi_mins = [psi_min]
        psi_maxs = [psi_max]

    sets = [(torch.tensor([d_min, theta_mins[i], psi_mins[j], own_set[0][-1], int_set[0][-1]]), 
             torch.tensor([d_max, theta_maxs[i], psi_maxs[j], own_set[1][-1], int_set[1][-1]])) for i in range(len(theta_mins)) for j in range(len(psi_mins))]
    
    return sets

def get_final_states_sim(n, agent_ids: List) -> Dict[str, List]: 
    states = {id: n.trace[id][-1][1:] for id in agent_ids}
    return states

def get_final_states_verify(n: 'AnalysisTreeNode', agent_ids: List) -> Dict[str, List]: 
    states = {id: np.array(n.trace[id][-2:])[:,1:] for id in agent_ids}
    return states

def get_point_tau(own_state: np.ndarray, int_state: np.ndarray) -> float:
    z_own, z_int = own_state[2], int_state[2]
    vz_own, vz_int = own_state[-1]*np.sin(own_state[-2]), int_state[-1]*np.sin(int_state[-2])
    if (vz_own == vz_int) and abs(z_own-z_int)<100:
        return 0
    elif vz_own == vz_int:
        return np.inf
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

#=====================================================================================





class VerseBridge():

    def __init__(self,ax):
        self.agents = {}
        self.agent_dict = {"Car": CarAgent, "NPC": NPCAgent}

        self.mode_dict = {"Car": AgentMode.COC, "NPC": AgentMode.COC}
        self.plotter = ax
    #uses input from from GUI
    def updatePlane(self, id="", agent_type=None,  dl=None, x=0, y=0, z=0, radius=0, yaw=0, pitch=0, v=0   ):

        init_set = [[x - radius ,y -radius,z-radius,yaw , pitch , v  ],[x + radius ,y + radius,z+radius,yaw,  pitch , v]]

        self.agents[id] = {"init_set":init_set, 
                           "init_mode": self.mode_dict[agent_type], 
                           "agent_type": self.agent_dict[agent_type],
                            "dl": dl  }
         

    # To set the initial set directly 
    def addInitialSet(self, id, initialSet):
        
        if id not in self.agents:
            self.agents[id] ={}
        
        self.agents[id]["init_set"] = initialSet


    # removes plane
    def removePlane(self, id):
        if(id in self.agents):
            self.agents.pop(id)

    def run_verse(self, ax= None, time_horizon=20, time_step=50,  x_dim=1, y_dim=2, z_dim=3, num_sims= 0, verify=True, noise=False):
        scenario = Scenario(ScenarioConfig( parallel=False))
        agent_ids = list(self.agents.keys())
        acas_agent_ids = []
        script_dir = os.path.realpath(os.path.dirname(__file__))

        for id, val in self.agents.items():

            dl = val["dl"]
            agent_type = val["agent_type"]
            init_mode = val["init_mode"]
            init_set = val["init_set"]
            
            
            if(dl != None):
                input_code_name = os.path.join(script_dir, dl )
                plane = agent_type(id, file_name=input_code_name)
            else:
                plane = agent_type(id)

            if self.agents[id]['agent_type']==CarAgent:
                acas_agent_ids.append(id)

            scenario.add_agent(plane) 

            scenario.set_init_single(
                id, init_set, (init_mode,)
            )
            print(init_set)

        self.plotter.clear()
        self.plotter.show_grid()
        # T = 10
        T = time_horizon
        #should define in plotter config instead
        Tv = 1
        ts = 0.01
        scenario.config.print_level = 0
        start = time.perf_counter()
        # models = [[torch.load(f"/Users/bachhoang/Verse-library/auto_LiRPA/examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
        norm = float("inf")
        # above is agnostic of verify/simulate
        #=====================================================================================================================
        if verify:
            scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC

            # self.plotter.render_window.GetRenderers().GetFirstRenderer().SetInteractive(False)
            # self.plotter.interactor.SetInteractorStyle(None)
            #self.plotter.render_window.SetAbortRender(True)    

            trace = scenario.verify(Tv,ts, self.plotter) # this is the root
            node_id = 1+trace.root.id


            queue = deque()
            queue.append(trace.root) # queue should only contain ATNs  
            ### begin looping
            while len(queue):
                cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
                states = get_final_states_verify(cur_node, agent_ids)
                # print(states)
                all_modes = {}
                for own_id in acas_agent_ids:
                    modes = set()
                    tau_idx_min = {int_id: min(get_tau_idx(states[own_id][1], states[int_id][0]), get_tau_idx(states[own_id][0], states[int_id][1])) for int_id in agent_ids if int_id != own_id}
                    tau_idx_max = {int_id: max(get_tau_idx(states[own_id][1], states[int_id][0]), get_tau_idx(states[own_id][0], states[int_id][1])) for int_id in agent_ids if int_id != own_id}
                    reachsets = {int_id: get_acas_reach(states[own_id], states[int_id]) for int_id in agent_ids if int_id != own_id}
                    closest_ids = []
                    closest_id = min(reachsets, key=lambda k:reachsets[k][0][0][0])
                    closest_dist_upper = reachsets[closest_id][0][1][0]

                    for id in reachsets:
                        if reachsets[id][0][0][0]<= closest_dist_upper:
                            closest_ids.append(id)

                    for id in closest_ids: # iterate over closest intruder(s) 
                        for reachset in reachsets[id]: # iterate over reachsets
                            if len(modes)==5: # if all modes are possible, stop iterating
                                break 
                            acas_min, acas_max = reachset
                            acas_min, acas_max = (acas_min-means_for_scaling)/range_for_scaling, (acas_max-means_for_scaling)/range_for_scaling
                            x_l, x_u = torch.tensor(acas_min).float().view(1,5), torch.tensor(acas_max).float().view(1,5)
                            x = (x_l+x_u)/2
                            # print(f'{own_id}-{id} {reachset}')
                            last_cmd = getattr(AgentMode, cur_node.mode[own_id][0]).value  # cur_mode.mode[.] is some string 
                            for tau_idx in range(tau_idx_min[id], tau_idx_max[id]+1):
                                lirpa_model = BoundedModule(models[last_cmd-1][tau_idx], (torch.empty_like(x))) 
                                # lirpa_model = BoundedModule(model, (torch.empty_like(x))) 
                                ptb_x = PerturbationLpNorm(norm = norm, x_L=x_l, x_U=x_u)
                                bounded_x = BoundedTensor(x, ptb=ptb_x)
                                lb, ub = lirpa_model.compute_bounds(bounded_x, method='alpha-CROWN')

                                # print(f'\n {own_id} Advisory ranges:', lb, ub,'\n')
                                new_mode = np.argmin(lb.numpy())+1                             
                                new_modes = []
                                for i in range(len(ub.numpy()[0])):
                                    lower = lb.numpy()[0][i]
                                    if lower<=ub.numpy()[0][new_mode-1]:
                                        new_modes.append(i+1)
                                modes.update(new_modes)

                    all_modes.update({own_id: modes})
                # print(modes, cur_node.start_time) # at 15 s, all modes possible -- investigate why
                
                all_modes_list = [all_modes[own_id] for own_id in acas_agent_ids]
                for nm in itertools.product(*all_modes_list):
                    cur_modes = dict(zip(acas_agent_ids, nm))
                    for id in agent_ids:
                        cur_mode = AgentMode(cur_modes[id]) if id in acas_agent_ids else AgentMode.COC
                        scenario.set_init_single(
                            id, states[id], (cur_mode,) # np array may not work
                        )
                    node_id += 1
                    # new_trace = scenario.simulate(Tv, ts)
                    
                    # start_ver = time.perf_counter() 
                    new_trace = scenario.verify(Tv, ts, self.plotter)
                    
                    # print(f'Verification time: {time.perf_counter()-start_ver:.2f} s')

                    temp_root = new_trace.root
                    new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, node_id)
                    cur_node.child.append(new_node)
                    print(f'Start time: {new_node.start_time}\nNode ID: {node_id}\nNew modes: {cur_modes}')
                        
                    if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                        continue
                    queue.append(new_node)

            trace.nodes = trace._get_all_nodes(trace.root)
            print(f'Verification time: {time.perf_counter()-start}')

        else:
            N = num_sims
            for __ in range(N):
                for id, val in self.agents.items():
                    init_mode = val["init_mode"]
                    init_set = val["init_set"]
                    scenario.set_init_single(
                        id, init_set, (init_mode,)
                    )
                trace = scenario.simulate(Tv,ts, self.plotter) # this is the root
                node_id = 1+trace.root.id
                # models = [[torch.load(f"/Users/bachhoang/Verse-library/auto_LiRPA/examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
                norm = float("inf")

                queue = deque()
                queue.append(trace.root) # queue should only contain ATNs  
                ### begin looping
                while len(queue):
                    cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
                    states = get_final_states_sim(cur_node, agent_ids)
                    all_modes = {}
                    for own_id in acas_agent_ids:
                        tau_idxs = {int_id: get_tau_idx(states[own_id], states[int_id]) for int_id in agent_ids if int_id != own_id}
                        acas_states = {int_id: get_acas_state(states[own_id], states[int_id]) for int_id in agent_ids if int_id != own_id}                    
                        closest_id = min(acas_states, key=lambda k:acas_states[k][0])
                        tau_idx = tau_idxs[closest_id]
                        # print('Tau idx:',tau_idx)
                        acas_state = acas_states[closest_id]
                        acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
                        last_cmd = getattr(AgentMode, cur_node.mode[own_id][0]).value  # cur_mode.mode[.] is some string
                        
                         
                        ads = models[last_cmd-1][tau_idx](acas_state.float().view(1,5)).detach().numpy()
                        
                        
                        new_mode = np.argmin(ads[0])+1 # will eventually be a list
                        all_modes[own_id] = new_mode
                    
                    
                    for id in agent_ids:
                        cur_mode = AgentMode(all_modes[id]) if id in acas_agent_ids else AgentMode.COC
                        scenario.set_init_single(
                            id, [states[id] for _ in range(2)], (cur_mode,) # np array may not work
                        )
                        ### ADDS POINT LABELS
                        if id in acas_agent_ids and getattr(AgentMode, cur_node.mode[id][0]) != cur_mode:
                            # print(f'{id} Modes {getattr(AgentMode, cur_node.mode[own_id][0])}, {cur_mode}')
                            self.plotter.add_point_labels(states[id][0:3], [cur_mode.name], always_visible=True, font_size=10)
                    node_id += 1
                    new_trace = scenario.simulate(Tv, ts, self.plotter)
                    temp_root = new_trace.root
                    new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, node_id)
                    cur_node.child.append(new_node)                    
                    if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                        continue
                    queue.append(new_node)

                # HAVE TO CALL THIS AFTER EACH SIMULATION TO UPDATE THE PLOTTER (THIS IS THE ONLY PLACE WITH THIS INFORMATION)
                plotRemaining(self.plotter, False)


                trace.nodes = trace._get_all_nodes(trace.root)
            print(f'Simulation time for {N} simulations: {time.perf_counter()-start}')