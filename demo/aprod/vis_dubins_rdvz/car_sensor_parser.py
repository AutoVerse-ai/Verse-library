import numpy as np
from scipy.optimize import minimize, OptimizeResult
# from prox_error_all_bounds import box_extreme_error, angular_span_between_rects, angular_bounds_diff # correct import paths
# from distance_bounds import dist_extrema
import pickle
from pathlib import Path
from distance_bounds import dist_extrema, psi_extrema
from parser_wrapper import parse_function

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

def vis_sensor_piecewise(psi, phi):
    if psi >= 0 and psi <= phi:
        return -1
    elif psi < 0 and psi >= -phi:
        return 1
    elif psi > phi:
        # return -2 
        return -2 # parser should be able to parse out arbitrary functions
    else:
        return 2

def vis_sensor(x,y,x_other,y_other, phi, theta):
    rel_x = x_other - x
    rel_y = y_other - y
    eta = atan2(rel_y, rel_x) # FIXME: bounded_arctan2 has wrong form, this is equivalent to atan2(y,x)
    psi = minus_angular(eta,theta) # eta-theta
    obs = vis_sensor_piecewise(psi, phi)
    return obs

class CarSensor:
    def sense(self, agent, state_dict, lane_map = None, simulate = True):
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
                    cont['ego.hx'] = state_dict[cur_agent][0][5]
                    cont['ego.hy'] = state_dict[cur_agent][0][6]
                    cont['ego.htheta'] = state_dict[cur_agent][0][7]
                    cont['ego.hv'] = state_dict[cur_agent][0][8]
                    cont['ego.ex'] = state_dict[cur_agent][0][9]
                    cont['ego.ey'] = state_dict[cur_agent][0][10]
                    cont['ego.etheta'] = state_dict[cur_agent][0][11]
                    cont['ego.ev'] = state_dict[cur_agent][0][12]

                    # auxillary states
                    cont['ego.timer'] = state_dict[cur_agent][0][13] # to trigger estimated state updates
                    cont['ego._id'] = state_dict[cur_agent][0][14] # this should be constant and unique per agent; >= 0; treat as priority with lower being higher priority
                    cont['ego.connected_ids'] = state_dict[cur_agent][0][15] # 'list' of merged agents; this should only be changed by decision logic; should be of form 2**agent_id_i for all i merged agents
                    cont['ego.assigned_id'] = state_dict[cur_agent][0][16] # can be changed here and propagated by reset DL -- this and the above two states must be singletons even in verification
                    
                    # relative states -- need to create placeholder states for these 
                    cont['ego.dist'] = None
                    cont['ego.prev_sense'] = state_dict[cur_agent][0][18] # the previous sensor value, taken from cache if it exists or keep default value if cache doesn't exist yet
                    cont['ego.cur_sense'] = None # current sensor reading 
                    
                    # discrete states
                    disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.assign_mode'] = state_dict[cur_agent][1][1] # should have modes assigned, waiting (possible also assigned_location_unknown if not currently in FOV, then transition to assigned if in FOV)
        
                    ### computation
                    assigned_agent = next((k for k in state_dict if k != agent.id and state_dict[k][0][14] == cont['ego.assigned_id']), None) # returns assigned_agent or None
                    if assigned_agent is None: # before this, do a check to see if all agents have been merged
                        assigned_agent = cur_agent # this should never happen in an actual run, but just keep it like this for toy simulation
                        # raise Exception(f'No other agent with assigned_id: {cont['ego.assigned_id']} found')

                    other_x, other_y = state_dict[assigned_agent][0][1], state_dict[assigned_agent][0][2]
                    other_hx, other_hy = state_dict[assigned_agent][0][5], state_dict[assigned_agent][0][6] # may use this
                    rel_x, rel_y = other_x-cont['ego.x'], other_y-cont['ego.y']
                    cont['ego.dist'] = np.linalg.norm([rel_x, rel_y])
                    psi = ((np.arctan2(rel_y, rel_x)-cont['ego.theta'] + np.pi) % (2*np.pi)) - np.pi  # corrected relative heading of the assigned agent 
                    if psi<phi and psi>0:
                        cont['ego.cur_sense'] = -1 # in left sector
                    elif psi>phi or psi<-phi:
                        cont['ego.cur_sense'] = -2 # of out view
                    else: # in right sector
                        cont['ego.cur_sense'] = 1 # in right sector

                    # if state_dict[cur_agent][0][0] == 0: #if we're at the first time step
                    #     cont['ego.prev_sense'] = cont['ego.cur_sense'] # this is a hack 
                    # cont['ego.prev_sense'] = get_cached_value(cont['ego._id']) if get_cached_value(cont['ego._id']) is not None else cont['ego.cur_sense'] # in the beginning just don't switch modes
                    # cache_sensor_value(cont['ego._id'], cont['ego.cur_sense']) # need to think about how this will work for verification and branching specifically -- prev_sense should always be a singleton
                    cont['other.has_priority'] = 1

                    if disc['ego.assign_mode'] == 'Waiting':
                        if int(cont['ego.connected_ids']) == int(all_agents):
                            cont['other.next_mode'] = 1 # set the next mode to be completed 
                        else:
                            # TODO: create algorithm to get next assigned_id, then add other assign_mode corresponding to assigned_agent not yet having been spotted yet (so just keep turning until we see it)
                            pass
                        
                        # has_priority = 1 # 1 if we do have priority
                        # for agent_id in state_dict:
                        #     if agent_id == cur_agent:
                        #         continue
                        #     if state_dict[agent_id][0][14]<cont['ego._id'] and state_dict[agent_id][1][1] == 'Waiting': # or whatever way we compute priority 
                        #         has_priority = 0

                        # cont['other.has_priority'] = has_priority
                    
                    if (cont['ego.timer'] in [6.7, 6.9]) and cur_agent=='car1':
                        pass
        else:               
            all_agents = 0
            for cur_agent in state_dict: # assuming all agents are active, otherwise do the same thing as before and give nominal ids to other agents
                all_agents += 2**state_dict[cur_agent][0][0][14]    
            for cur_agent in state_dict:
                if cur_agent == agent.id:
                    cont['ego.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                    cont['ego.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]]
                    cont['ego.theta'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]]
                    cont['ego.v'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]]
                    cont['ego.hx'] = [state_dict[cur_agent][0][0][5], state_dict[cur_agent][0][1][5]]
                    cont['ego.hy'] = [state_dict[cur_agent][0][0][6], state_dict[cur_agent][0][1][6]]
                    cont['ego.htheta'] = [state_dict[cur_agent][0][0][7], state_dict[cur_agent][0][1][7]]
                    cont['ego.hv'] = [state_dict[cur_agent][0][0][8], state_dict[cur_agent][0][1][8]]
                    cont['ego.ex'] = [state_dict[cur_agent][0][0][9], state_dict[cur_agent][0][1][9]]
                    cont['ego.ey'] = [state_dict[cur_agent][0][0][10], state_dict[cur_agent][0][1][10]]
                    cont['ego.etheta'] = [state_dict[cur_agent][0][0][11], state_dict[cur_agent][0][1][11]]
                    cont['ego.ev'] = [state_dict[cur_agent][0][0][12], state_dict[cur_agent][0][1][12]]

                    # auxillary states
                    cont['ego.timer'] = [state_dict[cur_agent][0][0][13], state_dict[cur_agent][0][0][13]] # to trigger estimated state updates
                    cont['ego._id'] = [state_dict[cur_agent][0][0][14], state_dict[cur_agent][0][1][14]] # this should be constant and unique per agent; >= 0; treat as priority with lower being higher priority
                    cont['ego.connected_ids'] = [state_dict[cur_agent][0][0][15], state_dict[cur_agent][0][1][15]] # 'list' of merged agents; this should only be changed by decision logic; should be of form 2**agent_id_i for all i merged agents
                    cont['ego.assigned_id'] = [state_dict[cur_agent][0][0][16], state_dict[cur_agent][0][1][16]] # can be changed here and propagated by reset DL -- this and the above two states must be singletons even in verification
                    
                    # relative states -- need to create placeholder states for these 
                    cont['ego.dist'] = None
                    cont['ego.prev_sense'] = [state_dict[cur_agent][0][0][18], state_dict[cur_agent][0][1][18]] # the previous sensor value, taken from cache if it exists or keep default value if cache doesn't exist yet
                    cont['ego.cur_sense'] = None # current sensor reading 
                    
                    # discrete states
                    disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.assign_mode'] = state_dict[cur_agent][1][1] # should have modes assigned, waiting (possible also assigned_location_unknown if not currently in FOV, then transition to assigned if in FOV)
        
                    ### computation
                    assigned_agent = next((k for k in state_dict if k != agent.id and state_dict[k][0][0][14] == cont['ego.assigned_id'][0]), None) # returns assigned_agent or None
                    if assigned_agent is None: # before this, do a check to see if all agents have been merged
                        assigned_agent = cur_agent # this should never happen in an actual run, but just keep it like this for toy simulation
                        raise Exception(f"No other agent with assigned_id: {cont['ego.assigned_id']} found")

                    pos_min, pos_max = np.array([state_dict[cur_agent][0][0][i] for i in range(1,3)]+[0]), np.array([state_dict[cur_agent][0][1][i] for i in range(1,3)] + [0])
                    obstacle_cont = state_dict[assigned_agent][0]
                    obstacle_pos_min, obstacle_pos_max = np.array([obstacle_cont[0][i] for i in range(1,3)]+[0]), np.array([obstacle_cont[1][i] for i in range(1,3)] + [0])
                    pos_bounds, obstacle_bounds = np.vstack([pos_min, pos_max]).T, np.vstack([obstacle_pos_min, obstacle_pos_max]).T
                    dist_min, dist_max = dist_extrema(pos_bounds, obstacle_bounds)
                    cont['ego.dist'] = [dist_min, dist_max]

                    x_other, y_other = (obstacle_cont[0][1], obstacle_cont[1][1]), (obstacle_cont[0][2], obstacle_cont[1][2])
                    input_bounds = {'x': [tuple(cont['ego.x'])], 'y':[tuple(cont['ego.y'])], 
                                    'x_other': [x_other], 'y_other':[y_other], 'phi':[(1,1)], 'theta':[tuple(cont['ego.theta'])]}
                    output_bounds, _ = parse_function(vis_sensor, input_bounds, [vis_sensor_piecewise])
                    cont['ego.cur_sense'] = list(output_bounds['obs'][0]) # since everything is a list of tuples

                    if cur_agent == 'car2' and cont['ego.timer'][0] < 6.7 and cont['ego.timer'][0] > 6.5:
                        pass

                    cont['other.has_priority'] = [1,1] # sentinel value
                    ### NOTE: what was I trying to do here that isn't being accomplished below?
                    # if cont['ego.prev_sense'][0]==-2:
                    #     has_priority = 1
                    #     if state_dict[assigned_agent][0][0][18]==-2 and state_dict[assigned_agent][0][0][14]<cont['ego._id'][0]:
                    #         has_priority = 0
                    #     cont['other.has_priority'] = [has_priority, has_priority]

                    if cont['ego.cur_sense'][0]!=cont['ego.cur_sense'][1] or cont['ego.cur_sense'][0]!=cont['ego.prev_sense'][0]:
                        has_priority = 1 # 1 if we do have priority
                        if state_dict[assigned_agent][0][0][14]<cont['ego._id'][0] and state_dict[assigned_agent][1][0] == state_dict[cur_agent][1][0]: # or whatever way we compute priority 
                            # NOTE: not sure this works in 3+ agent scenarios where symmetry breaks down -- maybe need to run through twice, once to see if which agents want to update, and another to determine which agent (at most 1) gets to update
                            has_priority = 0

                        if cur_agent == 'car2':
                            pass

                        cont['other.has_priority'] = [has_priority, has_priority]

                    # if state_dict[cur_agent][0][0] == 0: #if we're at the first time step
                    #     cont['ego.prev_sense'] = cont['ego.cur_sense'] # this is a hack 
                    # cont['ego.prev_sense'] = get_cached_value(cont['ego._id']) if get_cached_value(cont['ego._id']) is not None else cont['ego.cur_sense'] # in the beginning just don't switch modes
                    # cache_sensor_value(cont['ego._id'], cont['ego.cur_sense']) # need to think about how this will work for verification and branching specifically -- prev_sense should always be a singleton
        
                    if disc['ego.assign_mode'] == 'Waiting':
                        if int(cont['ego.connected_ids'][0]) == int(all_agents):
                            cont['other.next_mode'] = [1,1] # set the next mode to be completed 
                        else:
                            # TODO: create algorithm to get next assigned_id, then add other assign_mode corresponding to assigned_agent not yet having been spotted yet (so just keep turning until we see it)
                            pass
                        
                        has_priority = 1 # 1 if we do have priority
                        for agent_id in state_dict:
                            if agent_id == cur_agent:
                                continue
                            if state_dict[agent_id][0][0][14]<cont['ego._id'][0] and state_dict[agent_id][1][1] == 'Waiting': # or whatever way we compute priority 
                                has_priority = 0

                        cont['other.has_priority'] = [has_priority, has_priority]

        return cont, disc, len_dict