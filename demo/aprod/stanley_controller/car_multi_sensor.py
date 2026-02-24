import numpy as np
from scipy.optimize import minimize, OptimizeResult
# from prox_error_all_bounds import box_extreme_error, angular_span_between_rects, angular_bounds_diff # correct import paths
# from distance_bounds import dist_extrema
import pickle
from pathlib import Path
from verse import LaneMap, Lane
import sys
sys.path.insert(0, r'c:\Users\alexy\git repos\Verse-library')  # Hacky: Add repo root to sys.path
from verse.analysis.parsed_wrap import get_heading_bounds_optimized, get_lateral_distance_bounds_optimized

epsilon = 0.05
# epsilon = 0
epsilon_vel = 0.005
# epsilon_vel = 0

# ep_rho = 2.5
# ep_rho = 0.25
ep_rho = 0.01
# ep_angle = 0.006 # radians
ep_angle = 1e-6
# ep_angle = 0.01
# ep_rho_v = 0.000001
ep_rho_v = 1e-8
ep_ao = 0.006
D = 1e+10 # basically infinite

ts = 0.1 # time step size of the scenario 

def prox_rand_error(pos: np.ndarray):
    # pos = np.array([state_dict[cur_agent][0][i] for i in range(1,4)])
    rho = np.linalg.norm(pos) + ep_rho*(np.random.uniform(-1,1))
    rho = rho if rho > 0 else 0
    theta = np.arctan2(pos[1], pos[0]) # azimuth between ego and chief
    psi = np.arctan(pos[2]/(np.linalg.norm(pos[:2])))
    theta = theta + np.random.uniform(-1,1)*ep_angle
    psi = psi + np.random.uniform(-1,1)*ep_angle
    return np.array([pos[0]-rho*np.cos(theta)*np.cos(psi), pos[1]-rho*np.sin(theta)*np.cos(psi), pos[2]-rho*np.sin(psi)])

def _get_cache_path():
    """Get path to sensor cache directory"""
    cache_dir = Path(__file__).parent / "sensor_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def _get_cache_filename(ego_id):
    """Generate cache filename for given ego ID"""
    return _get_cache_path() / f"sensor_value_{ego_id}.pkl"

def cache_sensor_value(ego_id, sensor_value):
    """Cache sensor value for given ego ID"""
    cache_file = _get_cache_filename(ego_id)
    with open(cache_file, 'wb') as f:
        pickle.dump(sensor_value, f)

def get_cached_value(ego_id):
    """Get cached sensor value for given ego ID"""
    cache_file = _get_cache_filename(ego_id)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def clear_sensor_cache():
    """Clear all cached sensor values"""
    cache_dir = _get_cache_path()
    if cache_dir.exists():
        for f in cache_dir.glob("sensor_value_*.pkl"):
            f.unlink()

def get_heading_bounds_optimized_prev(lane_map, lane_idx, x_bounds, y_bounds):
    """
    More precise bound computation using optimization
    """
    def neg_heading(pos):
        return -lane_map.get_lane_heading(lane_idx, pos)
    
    def pos_heading(pos):
        return lane_map.get_lane_heading(lane_idx, pos)
    
    # Find minimum heading
    res_min = minimize(pos_heading, 
                      x0=[(x_bounds[0]+x_bounds[1])/2, (y_bounds[0]+y_bounds[1])/2],
                      bounds=[x_bounds, y_bounds])
    
    # Find maximum heading  
    res_max = minimize(neg_heading,
                      x0=[(x_bounds[0]+x_bounds[1])/2, (y_bounds[0]+y_bounds[1])/2],
                      bounds=[x_bounds, y_bounds])
    
    return [res_min.fun, -res_max.fun]

def get_lateral_distance_bounds_optimized_prev(lane_map, lane_idx, x_bounds, y_bounds):
    """
    More precise bound computation using optimization
    """
    def neg_distance(pos):
        return -lane_map.get_lateral_distance(lane_idx, pos)
    
    def pos_distance(pos):
        return lane_map.get_lateral_distance(lane_idx, pos)
    
    # Find minimum distance
    res_min = minimize(pos_distance,
                      x0=[(x_bounds[0]+x_bounds[1])/2, (y_bounds[0]+y_bounds[1])/2],
                      bounds=[x_bounds, y_bounds])
    
    # Find maximum distance
    res_max = minimize(neg_distance,
                      x0=[(x_bounds[0]+x_bounds[1])/2, (y_bounds[0]+y_bounds[1])/2],
                      bounds=[x_bounds, y_bounds])
    
    return [res_min.fun, -res_max.fun]

class CarSensor:
    def __init__(
        self, 
        ep_d: float = 0, ep_psi: float = 0
    ):
        self.ep_d = ep_d
        self.ep_psi = ep_psi

    def sense(self, agent, state_dict, lane_map: LaneMap = None, simulate = True):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        num_agents = len(state_dict)
        phi = min(np.pi/num_agents, np.arccos((num_agents-1)/(num_agents)))*0.9 # default value of (half) FOV of sensor, must be \in (0, min(.,.))

        if simulate: # just start with simulation 
            all_agents = 0
            for cur_agent in state_dict: # assuming all agents are active, otherwise do the same thing as before and give nominal ids to other agents
                all_agents += 2**state_dict[cur_agent][0][14]

            for cur_agent in state_dict:
                if cur_agent == agent.id:
                    # real continuous states
                    cont['ego.x'] = state_dict[cur_agent][0][1]
                    cont['ego.y'] = state_dict[cur_agent][0][2]
                    cont['ego.theta'] = state_dict[cur_agent][0][3]
                    cont['ego.v'] = state_dict[cur_agent][0][4]

                    # auxillary states
                    cont['ego.d'] = D
                    cont['ego.psi'] = 0 # by default, do nothing
                    cont['ego.timer'] = state_dict[cur_agent][0][7] # to trigger estimated state updates
                    
                    # discrete states
                    disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.track_mode'] = state_dict[cur_agent][1][1] # should have modes assigned, waiting (possible also assigned_location_unknown if not currently in FOV, then transition to assigned if in FOV)

        else:                   
            # TODO: make sure agents only update their distances when they are allowed to; need a priority parameters as in rdvz -- one parameter corresponding to constant priority, and another to current update status
            # TODO CONT: modes should be something like Ready, Update


            for cur_agent in state_dict:
                # if agent.id == 'car1':
                if cur_agent == agent.id:
                    cont['ego.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                    cont['ego.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] 
                    cont['ego.theta'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] 
                    cont['ego.v'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]] 
                    
                    cont['ego.d'] = [state_dict[cur_agent][0][0][5], state_dict[cur_agent][0][0][5]] 
                    cont['ego.psi'] = [state_dict[cur_agent][0][0][6], state_dict[cur_agent][0][0][6]] 
                    cont['ego.timer'] = [state_dict[cur_agent][0][0][7], state_dict[cur_agent][0][0][7]] 
                    cont['ego.priority'] = [state_dict[cur_agent][0][0][8], state_dict[cur_agent][0][0][8]] 
                    own_priority = cont['ego.priority'][0]

                    disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.track_mode'] = state_dict[cur_agent][1][1]
                    own_sensor_mode = disc['ego.sensor_mode'] = state_dict[cur_agent][1][2] 

                    has_priority = True # by default, agents have priority to update
                    # TODO: check each other agent, if agent has lower priority number and has not yet updated mode, then should not have priority
                    for other_agent in state_dict:
                        if other_agent == cur_agent:
                            continue
                        other_priority, other_sensor_mode = state_dict[other_agent][0][0][8], state_dict[other_agent][1][2] # NOTE: assume these exist, should for the most part 
                        # if other agent has lower (favored) priority and has the same mode, do not update
                        if other_priority < own_priority and other_sensor_mode == own_sensor_mode:
                            has_priority = False
                            break
                        # if other agent has higher (unfavored) priority and has different mode, do not update, wait until they've updated
                        if other_priority > own_priority and other_sensor_mode != own_sensor_mode:
                            has_priority = False
                            break

                    # just always compute the new distances, psis
                    # NOTE: this is not necessary in most cases
                    d_min, d_max = get_lateral_distance_bounds_optimized(lane_map, disc['ego.track_mode'], cont['ego.x'], cont['ego.y'])
                    psi_min, psi_max = get_heading_bounds_optimized(lane_map, disc['ego.track_mode'], cont['ego.x'], cont['ego.y'])
                    cont['ego.d'] = [d_min-self.ep_d, d_max+self.ep_d]
                    cont['ego.psi'] = [psi_min-self.ep_psi, psi_max+self.ep_psi] # technically just heading

                    if has_priority:
                        cont['other.sensor_update'] = [1, 1]

                    else:
                        cont['other.sensor_update'] = [0,0]

                else: # NOTE: not sure this is necessary, all relevant computations are occur in main 'if' block
                    if 'others.x' not in cont: # for now just consider perfect other info scenario
                        cont['others.x'] = [[state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]]
                        cont['others.y'] = [[state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] ]
                        cont['others.theta'] = [[state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] ]
                        cont['others.v'] = [[state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] ]

                        disc['others.agent_mode'] = [state_dict[cur_agent][1][0]]
                        disc['others.track_mode'] = [state_dict[cur_agent][1][1]]
                    else:
                        cont['others.x'].append([state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]])
                        cont['others.y'].append([state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] )
                        cont['others.theta'].append([state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] )
                        cont['others.v'].append([state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] )
                        
                        disc['others.agent_mode'].append(state_dict[cur_agent][1][0])
                        disc['others.track_mode'].append(state_dict[cur_agent][1][1])

        return cont, disc, len_dict