import numpy as np
from math import floor
from dubins_3d_agent import NPCAgent, CarAgent
from verse.utils.utils import wrap_to_pi


def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})


def adds(d, thing, attrs, vals):
    for k, v in zip(attrs, vals):
        if thing + '.' + k not in d:
            d[thing + '.' + k] = [v]
        else:
            d[thing + '.' + k].append(v)


def set_states_2d_ball(cnts, disc, thing, val):
    state, mode, static = val
    sets(cnts, thing, ['x', 'y', 'z', 'vx', 'vy', 'vz',
         'waypoint_index', 'done_flag'], state[1:9])
    sets(disc, thing, ["craft_mode", "lane_mode"], mode)


def set_states_3d_ball(cnts, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:9])
    assert len(transp) == 8
    sets(cnts, thing, ['x', 'y', 'z', 'vx', 'vy', 'vz',
         'waypoint_index', 'done_flag'], transp)
    sets(disc, thing, ["craft_mode", "lane_mode"], mode)


def add_states_2d_ball(cont, disc, thing, val):
    state, mode, static = val
    adds(cont, thing, ['x', 'y', 'z', 'vx', 'vy', 'vz',
         'waypoint_index', 'done_flag'], state[1:9])
    adds(disc, thing, ["craft_mode", "lane_mode"], mode)


def add_states_3d_ball(cont, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:9])
    assert len(transp) == 8
    adds(cont, thing, ['x', 'y', 'z', 'vx', 'vy',
         'vz', 'waypoint_index', 'done_flag'], transp)
    adds(disc, thing, ["craft_mode", "lane_mode"], mode)
    
    



# class QuadrotorSensor():
#     def sense(self, agent, state_dict, lane_map):
#         cont = {}
#         disc = {}
#         len_dict = {'others': len(state_dict)-1}
#         tmp = np.array(list(state_dict.values())[0][0])
#         if tmp.ndim < 2:
#             for agent_id in state_dict:
#                 if agent_id == agent.id:
#                     set_states_2d_ball(cont, disc, 'ego', state_dict[agent_id])
#                 else:
#                     add_states_2d_ball(cont, disc, 'others',
#                                        state_dict[agent_id])
#         else:
#             for agent_id in state_dict:
#                 if agent_id == agent.id:
#                     set_states_3d_ball(cont, disc, "ego", state_dict[agent_id])
#                 else:
#                     add_states_3d_ball(cont, disc, 'others',
#                                        state_dict[agent_id])
#         return cont, disc, len_dict


class DubinSensor():
    def sense(self, agent: CarAgent, state_dict, lane_map, simulate = False):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if simulate: #tmp.ndim < 2:
            if agent.id == "car1":
                len_dict['others'] = 1 
                
                ###########
                # New stuff
                ###########
                print(state_dict)
                curr_x = state_dict['car1'][0][1]
                curr_y = state_dict['car1'][0][2]
                curr_theta = state_dict['car1'][0][4]
                curr_v = state_dict['car1'][0][6]
                obstacle_x = state_dict['car2'][0][1]
                obstacle_y = state_dict['car2'][0][2]
                obs_theta = state_dict['car2'][0][4]
                obs_v = state_dict['car2'][0][6]
                
                ego_time = state_dict['car1'][0][8]
                
                curr_vx = curr_v * np.cos(curr_theta)
                curr_vy = curr_v * np.sin(curr_theta)
                
                obs_vx = obs_v * np.cos(obs_theta)
                obs_vy = obs_v * np.sin(obs_theta)
                
                #ego_time = state_dict["aircraft1"][0][-2]
                # transition_flag = state_dict['aircraft1'][0][-2]
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy = obstacle_y - curr_y
                dx = obstacle_x - curr_x
                    
                rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2) # Just considering 2-D distance right now (neglect z)
                v_own = np.sqrt(curr_vx**2 + curr_vy**2)
                v_int = np.sqrt((obs_vx)**2 + (obs_vy)**2) #0 # static obstacle, so hardcode zero velocity
                    
                theta = np.arctan2(dy, dx) - curr_theta

                psi = np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                    
                ## Ensure that angular states are between -pi and pi
                '''while theta < -np.pi:
                    theta += 2 * np.pi

                while theta > np.pi:
                    theta -= 2 * np.pi

                while psi < -np.pi:
                    psi += 2 * np.pi

                while psi > np.pi:
                    psi -= 2 * np.pi'''
                
                # make sure theta and psi are within [-pi, pi]
                theta = np.arctan2(np.sin(theta),np.cos(theta))
                psi = np.arctan2(np.sin(psi),np.cos(psi))

                #state = [rho, theta, psi, v_own, v_int]
                    
                cont['ego.rho'] = rho
                cont['ego.theta'] = theta
                cont['ego.psi'] = psi
                cont['ego.v_own'] = v_own
                cont['ego.v_int'] = v_int
                cont['ego.timer_DL'] = ego_time
                
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                #disc['ego.track_mode'] = state_dict['car1'][1][1]
        
                ##########
                # End of new stuff
                ##########
        else:
            if agent.id == 'car1':
                len_dict['others'] = 1 
                # dist_min, dist_max = get_extreme(
                #     (state_dict['car'][0][0][1],state_dict['car'][0][0][2],state_dict['car'][0][1][1],state_dict['car'][0][1][2]),
                #     (state_dict['pedestrian'][0][0][1],state_dict['pedestrian'][0][0][2],state_dict['pedestrian'][0][1][1],state_dict['pedestrian'][0][1][2]),
                # )
                
                #########
                # New Stuff
                #########
                # lower bound states
                curr_x_min = state_dict['car1'][0][0][1]
                curr_y_min = state_dict['car1'][0][0][2]
                curr_theta_min = state_dict['car1'][0][0][4]
                curr_v_min = state_dict['car1'][0][0][6]
                obstacle_x_min = state_dict['car2'][0][0][1]
                obstacle_y_min = state_dict['car2'][0][0][2]
                obs_theta_min = state_dict['car2'][0][0][4]
                obs_v_min = state_dict['car2'][0][0][6]  
                
                # Timer variable, no uncertainty
                ego_time = state_dict['car1'][0][0][8] # update
                
                # upper bound states
                curr_x_max = state_dict['car1'][0][1][1]
                curr_y_max = state_dict['car1'][0][1][2]
                curr_theta_max = state_dict['car1'][0][1][4]
                curr_v_max = state_dict['car1'][0][1][6]
                obstacle_x_max = state_dict['car2'][0][1][1]
                obstacle_y_max = state_dict['car2'][0][1][2]
                obs_theta_max = state_dict['car2'][0][1][4]
                obs_v_max = state_dict['car2'][0][1][6]   
                
                curr_vx_max = curr_v_max * np.max([np.cos(curr_theta_max), np.cos(curr_theta_min)])
                curr_vy_max = curr_v_max * np.max([np.sin(curr_theta_max), np.sin(curr_theta_min)])
                
                curr_vx_min = curr_v_min * np.min([np.cos(curr_theta_max), np.cos(curr_theta_min)])
                curr_vy_min = curr_v_min * np.min([np.sin(curr_theta_max), np.sin(curr_theta_min)])
                
                #obs_vx = obs_v * np.cos(obs_theta)
                #obs_vy = obs_v * np.sin(obs_theta)             
                
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy_max = np.max(np.abs([obstacle_y_max - curr_y_min, curr_y_max - obstacle_y_min]))
                dx_max = np.max(np.abs([obstacle_x_max - curr_x_min, curr_x_max - obstacle_x_min]))
                
                dy_min = np.min(np.abs([obstacle_y_max - curr_y_min, curr_y_max - obstacle_y_min]))
                dx_min = np.min(np.abs([obstacle_x_max - curr_x_min, curr_x_max - obstacle_x_min]))
                
                rho_max = np.sqrt(dx_max**2 + dy_max**2) # Just considering 2-D distance right now (neglect z)
                rho_min = np.sqrt(dx_min**2 + dy_min**2)
                
                # Range for velocities
                v_own_max = curr_v_max
                v_int_max = obs_v_max #0 # static obstacle, so hardcode zero velocity
                
                v_own_min = curr_v_min
                v_int_min = obs_v_min
                
                # Angular ranges
                sign_dy_max = np.max([obstacle_y_max - curr_y_min, obstacle_y_min - curr_y_max])
                sign_dx_max = np.max([obstacle_x_max - curr_x_min, obstacle_x_min - curr_x_max])
                
                sign_dy_min = np.min([obstacle_y_max - curr_y_min, obstacle_y_min - curr_y_max])
                sign_dx_min = np.min([obstacle_x_max - curr_x_min, obstacle_x_min - curr_x_max])
                
                theta_max = np.max([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_min
                theta_min = np.min([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_max

                psi_max = obs_theta_max - curr_theta_min #np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                psi_min = obs_theta_min - curr_theta_max
                
                #######
                # New wrap angles
                #######
                arho_min = np.inf # does this make sense
                arho_max = -np.inf
                
                own_ext = [(curr_x_min, curr_y_min), (curr_x_max, curr_y_max), (curr_x_min, curr_y_max), (curr_x_max, curr_y_min)]
                int_ext = [(obstacle_x_min, obstacle_y_min), (obstacle_x_max, obstacle_y_max), (obstacle_x_min, obstacle_y_max), (obstacle_x_max, obstacle_y_min)]
                for own_vert in own_ext:
                    for int_vert in int_ext:
                        arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi)
                        arho_max = max(arho_max, arho)
                        arho_min = min(arho_min, arho)
                
                theta_min = wrap_to_pi((2*np.pi-curr_theta_max)+arho_min)
                theta_max = wrap_to_pi((2*np.pi-curr_theta_min)+arho_max) 
                # theta_maxs = []
                # theta_mins = []
                if theta_max<theta_min: # bound issue due to wrapping
                    # theta_mins = [-np.pi, theta_min]
                    # theta_maxs = [theta_max, np.pi]
                    cont['ego.theta'] = [-np.pi, theta_min]
                    cont['ego.theta2'] = [theta_max, np.pi]
                else:
                    cont['ego.theta'] = [theta_min, theta_max]
                    cont['ego.theta2'] = [theta_min, theta_max]

                # Psi wrap
                psi_max = wrap_to_pi(obs_theta_max - curr_theta_min) #np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                psi_min = wrap_to_pi(obs_theta_min - curr_theta_max)
                
                if psi_max<psi_min: # bound issue due to wrapping
                    # theta_mins = [-np.pi, theta_min]
                    # theta_maxs = [theta_max, np.pi]
                    cont['ego.psi'] = [-np.pi, psi_min]
                    cont['ego.psi2'] = [psi_max, np.pi]
                else:
                    cont['ego.psi'] = [psi_min, psi_max]
                    cont['ego.psi2'] = [psi_min, psi_max]
                '''while psi_max < -np.pi:      
                    psi_max += 2 * np.pi

                while psi_max > np.pi:
                    psi_max -= 2 * np.pi
                # Wrap psi_min 
                while psi_min < -np.pi:
                    psi_min += 2 * np.pi

                while psi_min > np.pi:
                    psi_min -= 2 * np.pi'''
                
                #psi_max = np.arctan2(np.sin(psi_max),np.cos(psi_max))
                #psi_min = np.arctan2(np.sin(psi_min),np.cos(psi_min))
                
                cont['ego.rho'] = [
                    rho_min, rho_max
                    # First dimension is 0 or 1 (0 is cont, 1 for mode)
                    # Second dimension is bound (0 for lower, 1 for upper)
                    # Third dimesnion is state index (same indices as the state_dict)
                ]
                #cont['ego.theta'] = [
                #    theta_min, theta_max
                #]
                #cont['ego.psi'] = [
                #    psi_min, psi_max #state_dict['car'][0][0][3], state_dict['car'][0][1][3]
                #]
                cont['ego.v_own'] = [
                    v_own_min, v_own_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                cont['ego.v_int'] = [
                    v_int_min, v_int_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                
                
                '''transition_sample_period = 3.9
                if abs((ego_time / transition_sample_period) - round(ego_time / transition_sample_period)) < 0.02 and ego_time > 0.05:
                    transition_flag = 1
                else:
                    transition_flag = 0'''
                cont['ego.timer_DL'] = [ego_time, ego_time]
                #print(f"sensor ego r: {rho_min}\nsensor ego t:{ego_time}")
                
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                #disc['ego.track_mode'] = state_dict['car1'][1][1]
                '''if dist_min<self.sensor_distance:
                    cont['other.dist'] = [
                        dist_min, dist_max
                    ]
                    # cont['other.x'] = [
                    #     state_dict['pedestrian'][0][0][1], state_dict['pedestrian'][0][1][1]
                    # ]
                    # cont['other.y'] = [
                    #     state_dict['pedestrian'][0][0][2], state_dict['pedestrian'][0][1][2]
                    # ]
                    # cont['other.v'] = [
                    #     state_dict['pedestrian'][0][0][4], state_dict['pedestrian'][0][1][4]
                    # ]
                else:
                    cont['other.dist'] = [
                        1000, 1000
                    ]
                    # cont['other.x'] = [
                    #     1000, 1000
                    # ]
                    # cont['other.y'] = [
                    #     1000, 1000
                    # ]
                    # cont['other.v'] = [
                    #     1000, 1000
                    # ]
                    '''


        return cont, disc, len_dict

'''
class GUAMSensor:
    #def __init__(self):
        #self.time = 0

    # The baseline sensor is omniscient. Each agent can get the state of all other agents
    def sense(self, agent: AircraftAgent, state_dict, lane_map):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            if agent.id == "aircraft1":
                len_dict['others'] = 1 
                
                
                ###########
                # New stuff
                ###########
                
                curr_x = state_dict["aircraft1"][0][13]
                curr_y = state_dict["aircraft1"][0][14]
                obstacle_x = state_dict["aircraft2"][0][13]
                obstacle_y = state_dict["aircraft2"][0][14]
                curr_vx = state_dict["aircraft1"][0][7]
                curr_vy = state_dict["aircraft1"][0][8]
                obs_vx = state_dict["aircraft2"][0][7]
                obs_vy = state_dict["aircraft2"][0][8]
                
                ego_time = state_dict["aircraft1"][0][-2]
                # transition_flag = state_dict['aircraft1'][0][-2]
                #transition_sample_period = 10
                #if abs((ego_time / transition_sample_period) - round(ego_time / transition_sample_period)) < 0.02 and ego_time > 0.05:
                #    transition_flag = 1
                #else:
                #    transition_flag = 0
                print(f'ego time: {ego_time}')
                
                disc['ego.agent_mode'] = state_dict["aircraft1"][1][0]
                #cont['ego.transition_flag'] = transition_flag
                cont['ego.timer_DL'] = ego_time
                # disc['ego.transition_flag'] = transition_flag
                
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy = obstacle_y - curr_y
                dx = obstacle_x - curr_x
                
                rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2) # Just considering 2-D distance right now (neglect z)
                v_own = np.sqrt(curr_vx**2 + curr_vy**2)
                v_int = np.sqrt((obs_vx)**2 + (obs_vy)**2) #0 # static obstacle, so hardcode zero velocity
                
                theta = np.arctan2(dy, dx)

                psi = np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                
                ## Ensure that angular states are between -pi and pi
                while theta < -np.pi:
                    theta += 2 * np.pi

                while theta > np.pi:
                    theta -= 2 * np.pi

                while psi < -np.pi:
                    psi += 2 * np.pi

                while psi > np.pi:
                    psi -= 2 * np.pi

                #state = [rho, theta, psi, v_own, v_int]
                
                cont['ego.rho'] = rho
                cont['ego.theta'] = theta
                cont['ego.psi'] = psi
                cont['ego.v_own'] = v_own
                cont['ego.v_int'] = v_int
    
                ##########
                # End of new stuff
                ##########
        else:
            if agent.id == 'aircraft1':
                len_dict['others'] = 1 
                # dist_min, dist_max = get_extreme(
                #     (state_dict['car'][0][0][1],state_dict['car'][0][0][2],state_dict['car'][0][1][1],state_dict['car'][0][1][2]),
                #     (state_dict['pedestrian'][0][0][1],state_dict['pedestrian'][0][0][2],state_dict['pedestrian'][0][1][1],state_dict['pedestrian'][0][1][2]),
                # )
                
                #########
                # New Stuff
                #########
                # lower bound states
                curr_x_min = state_dict["aircraft1"][0][0][12]
                curr_y_min = state_dict["aircraft1"][0][0][13]
                obstacle_x_min = state_dict["aircraft2"][0][0][12]
                obstacle_y_min = state_dict["aircraft2"][0][0][13]
                curr_vx_min = state_dict["aircraft1"][0][0][6]
                curr_vy_min = state_dict["aircraft1"][0][0][7]
                obs_vx_min = state_dict["aircraft2"][0][0][6]
                obs_vy_min = state_dict["aircraft2"][0][0][7]  
                
                # upper bound states
                curr_x_max = state_dict["aircraft1"][0][1][12]
                curr_y_max = state_dict["aircraft1"][0][1][13]
                obstacle_x_max = state_dict["aircraft2"][0][1][12]
                obstacle_y_max = state_dict["aircraft2"][0][1][13]
                curr_vx_max = state_dict["aircraft1"][0][1][6]
                curr_vy_max = state_dict["aircraft1"][0][1][7]
                obs_vx_max = state_dict["aircraft2"][0][1][6]
                obs_vy_max = state_dict["aircraft2"][0][1][7]                
                
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy_max = obstacle_y_max - curr_y_min
                dx = obstacle_x_max - curr_x_min
                
                rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2) # Just considering 2-D distance right now (neglect z)
                v_own = np.sqrt(curr_vx**2 + curr_vy**2)
                v_int = np.sqrt((obs_vx)**2 + (obs_vy)**2) #0 # static obstacle, so hardcode zero velocity
                
                theta = np.arctan2(dy, dx)

                psi = np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                
                
                cont['ego.x'] = [
                    state_dict['car'][0][0][1], state_dict['car'][0][1][1] 
                    # First dimension is 0 or 1 (0 is cont, 1 for mode)
                    # Second dimension is bound (0 for lower, 1 for upper)
                    # Third dimesnion is state index (same indices as the state_dict)
                ]
                cont['ego.y'] = [
                    state_dict['car'][0][0][2], state_dict['car'][0][1][2]
                ]
                cont['ego.theta'] = [
                    state_dict['car'][0][0][3], state_dict['car'][0][1][3]
                ]
                cont['ego.v'] = [
                    state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                cont['other.dist'] = [
                    dist_min, dist_max
                ]
                disc['ego.agent_mode'] = state_dict['aircraft1'][1][0]
                if dist_min<self.sensor_distance:
                    cont['other.dist'] = [
                        dist_min, dist_max
                    ]
                    # cont['other.x'] = [
                    #     state_dict['pedestrian'][0][0][1], state_dict['pedestrian'][0][1][1]
                    # ]
                    # cont['other.y'] = [
                    #     state_dict['pedestrian'][0][0][2], state_dict['pedestrian'][0][1][2]
                    # ]
                    # cont['other.v'] = [
                    #     state_dict['pedestrian'][0][0][4], state_dict['pedestrian'][0][1][4]
                    # ]
                else:
                    cont['other.dist'] = [
                        1000, 1000
                    ]
                    # cont['other.x'] = [
                    #     1000, 1000
                    # ]
                    # cont['other.y'] = [
                    #     1000, 1000
                    # ]
                    # cont['other.v'] = [
                    #     1000, 1000
                    # ]


        return cont, disc, len_dict
'''
