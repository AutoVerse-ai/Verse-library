import numpy as np
from scipy.optimize import minimize, OptimizeResult
# from prox_error_all_bounds import box_extreme_error, angular_span_between_rects, angular_bounds_diff # correct import paths
# from distance_bounds import dist_extrema

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

class CarSensor:
    def sense(self, agent, state_dict, lane_map = None, simulate = True):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            cur_agent = agent.id
            cont['ego.x'] = state_dict[cur_agent][0][1]
            cont['ego.y'] = state_dict[cur_agent][0][2]
            cont['ego.theta'] = state_dict[cur_agent][0][3]
            cont['ego.v'] = state_dict[cur_agent][0][4]

            disc['ego.go_mode'] = state_dict[cur_agent][1][0]
            disc['ego.po_mode'] = state_dict[cur_agent][1][1]
            disc['ego.priority_mode'] = state_dict[cur_agent][1][2]
            disc['ego.move_mode'] = state_dict[cur_agent][1][3]
        
        else:
            for cur_agent in state_dict:
                # if agent.id == 'car1':
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
                    cont['ego.timer'] = [state_dict[cur_agent][0][0][13], state_dict[cur_agent][0][0][13]]
                    
                    disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.track_mode'] = state_dict[cur_agent][1][1]
                    disc['ego.gps_mode'] = state_dict[cur_agent][1][2]

                    if disc['ego.gps_mode'] == 'Active':
                        cont['ego.ex'], cont['ego.ey'] = [-epsilon, epsilon], [-epsilon, epsilon]
                        cont['ego.hx'], cont['ego.hy'] = [cont['ego.x'][0]-epsilon, cont['ego.x'][1]+epsilon], [cont['ego.y'][0]-epsilon, cont['ego.y'][1]+epsilon]

                else:
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