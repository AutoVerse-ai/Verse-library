import numpy as np
from scipy.optimize import minimize, OptimizeResult
from prox_error_all_bounds import box_extreme_error, angular_span_between_rects, angular_bounds_diff
from distance_bounds import dist_extrema

epsilon = 0.5
epsilon_vel = 0.00001

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

class OrbitalSensor:
    def sense(self, agent, state_dict, lane_map = None, simulate = True):
        """
        Only consider single agent scenarios, hard-code just to get things working
        """
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            cur_agent = agent.id
            cont['ego.x'] = state_dict[cur_agent][0][1]
            cont['ego.y'] = state_dict[cur_agent][0][2]
            cont['ego.z'] = state_dict[cur_agent][0][3]
            cont['ego.vx'] = state_dict[cur_agent][0][4]
            cont['ego.vy'] = state_dict[cur_agent][0][5]
            cont['ego.vz'] = state_dict[cur_agent][0][6]
            cont['ego.hx'] = state_dict[cur_agent][0][7]
            cont['ego.hy'] = state_dict[cur_agent][0][8]
            cont['ego.hz'] = state_dict[cur_agent][0][9]
            cont['ego.hvx'] = state_dict[cur_agent][0][10]
            cont['ego.hvy'] = state_dict[cur_agent][0][11]
            cont['ego.hvz'] = state_dict[cur_agent][0][12]
            cont['ego.ex'] = state_dict[cur_agent][0][13]
            cont['ego.ey'] = state_dict[cur_agent][0][14]
            cont['ego.ez'] = state_dict[cur_agent][0][15]
            cont['ego.evx'] = state_dict[cur_agent][0][16]
            cont['ego.evy'] = state_dict[cur_agent][0][17]
            cont['ego.evz'] = state_dict[cur_agent][0][18]
            cont['ego.timer'] = state_dict[cur_agent][0][19]
            cont['ego.po_timer'] = state_dict[cur_agent][0][20]
            disc['ego.go_mode'] = state_dict[cur_agent][1][0]
            disc['ego.po_mode'] = state_dict[cur_agent][1][1]
            disc['ego.priority_mode'] = state_dict[cur_agent][1][2]
            disc['ego.move_mode'] = state_dict[cur_agent][1][3]
        


            if disc['ego.go_mode'] == 'Active' and disc['ego.po_mode'] == 'Active':
                dir = np.random.normal(size=3)
                dir /= np.linalg.norm(dir)
                rad: float = epsilon * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                err_pos = rad*dir
                err_pos_prox = prox_rand_error(np.array([state_dict[cur_agent][0][i] for i in range(1,4)]))
                cont['ego.ex'], cont['ego.ey'], cont['ego.ez'] = np.mean([err_pos, err_pos_prox], axis=0)

                dir_vel = np.random.normal(size=3)
                dir_vel /= np.linalg.norm(dir_vel)
                rad_vel = epsilon_vel * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                err_vel = rad_vel*dir_vel
                err_vel_prox = prox_rand_error(np.array([state_dict[cur_agent][0][i] for i in range(4,7)]))
                cont['ego.evx'], cont['ego.evy'], cont['ego.evz'] = np.mean([err_vel, err_vel_prox], axis=0)

            elif disc['ego.go_mode'] == 'Active':
                # true_pos = np.array([state_dict[cur_agent][0][i] for i in range(1,4)])
                dir = np.random.normal(size=3)
                dir /= np.linalg.norm(dir)
                rad = epsilon * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                err_pos = rad*dir
                cont['ego.ex'] = err_pos[0]
                cont['ego.ey'] = err_pos[1]
                cont['ego.ez'] = err_pos[2]
                
                # true_vel = np.array([state_dict[cur_agent][0][i] for i in range(4,7)])
                dir_vel = np.random.normal(size=3)
                dir_vel /= np.linalg.norm(dir_vel)
                rad_vel = epsilon_vel * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                err_vel = rad_vel*dir_vel
                cont['ego.evx'] = err_vel[0]
                cont['ego.evy'] = err_vel[1]
                cont['ego.evz'] = err_vel[2]

            elif disc['ego.po_mode'] == 'Active':
                pos = np.array([state_dict[cur_agent][0][i] for i in range(1,4)])
                rho = np.linalg.norm(pos) + ep_rho*(np.random.uniform(-1,1))
                rho = rho if rho > 0 else 0
                theta = np.arctan2(pos[1], pos[0]) # azimuth between ego and chief
                psi = np.arctan(pos[2]/(np.linalg.norm(pos[:2])))
                theta = theta + np.random.uniform(-1,1)*ep_angle
                psi = psi + np.random.uniform(-1,1)*ep_angle
                cont['ego.ex'] = pos[0]-rho*np.cos(theta)*np.cos(psi) 
                cont['ego.ey'] = pos[1]-rho*np.sin(theta)*np.cos(psi)
                cont['ego.ez'] = pos[2]-rho*np.sin(psi)

                vel = np.array([state_dict[cur_agent][0][i] for i in range(4,7)])
                rho_v = np.linalg.norm(vel) + ep_rho_v*(np.random.uniform(-1,1))
                rho_v = rho_v if rho_v > 0 else 0
                theta_v = np.arctan2(vel[1], vel[0]) # azimuth between ego and chief
                psi_v = np.arctan(vel[2]/(np.linalg.norm(vel[:2])))
                theta_v = theta_v + np.random.uniform(-1,1)*ep_angle
                psi = psi_v + np.random.uniform(-1,1)*ep_angle
                cont['ego.evx'] = vel[0]-rho_v*np.cos(theta_v)*np.cos(psi_v) 
                cont['ego.evy'] = vel[1]-rho_v*np.sin(theta_v)*np.cos(psi_v)
                cont['ego.evz'] = vel[2]-rho_v*np.sin(psi_v)
                # cont['other.x'] = state_dict['car2'][0][1] # dummy assignments
                # cont['other.y'] = state_dict['car2'][0][2]
                # disc['other.track_mode'] = state_dict['car2'][1][1]
        else:
            for cur_agent in state_dict:
                if cur_agent == "obs":
                    cont['obs.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                    cont['obs.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] 
                    cont['obs.z'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] 
                    cont['obs.vx'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]] 
                    cont['obs.vy'] = [state_dict[cur_agent][0][0][5], state_dict[cur_agent][0][1][5]] 
                    cont['obs.vz'] = [state_dict[cur_agent][0][0][6], state_dict[cur_agent][0][1][6]] 
                    disc['obs.move_mode'] = state_dict[cur_agent][1][0]

                elif cur_agent == agent.id:
                    cont['ego.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                    cont['ego.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] 
                    cont['ego.z'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] 
                    cont['ego.vx'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]] 
                    cont['ego.vy'] = [state_dict[cur_agent][0][0][5], state_dict[cur_agent][0][1][5]] 
                    cont['ego.vz'] = [state_dict[cur_agent][0][0][6], state_dict[cur_agent][0][1][6]] 
                    cont['ego.hx'] = [state_dict[cur_agent][0][0][7], state_dict[cur_agent][0][1][7]]
                    cont['ego.hy'] = [state_dict[cur_agent][0][0][8], state_dict[cur_agent][0][1][8]]
                    cont['ego.hz'] = [state_dict[cur_agent][0][0][9], state_dict[cur_agent][0][1][9]]
                    cont['ego.hvx'] = [state_dict[cur_agent][0][0][10], state_dict[cur_agent][0][1][10]]
                    cont['ego.hvy'] = [state_dict[cur_agent][0][0][11], state_dict[cur_agent][0][1][11]]
                    cont['ego.hvz'] = [state_dict[cur_agent][0][0][12], state_dict[cur_agent][0][1][12]]
                    cont['ego.ex'] = [state_dict[cur_agent][0][0][13], state_dict[cur_agent][0][1][13]]
                    cont['ego.ey'] = [state_dict[cur_agent][0][0][14], state_dict[cur_agent][0][1][14]]
                    cont['ego.ez'] = [state_dict[cur_agent][0][0][15], state_dict[cur_agent][0][1][15]]
                    cont['ego.evx'] = [state_dict[cur_agent][0][0][16], state_dict[cur_agent][0][1][16]]
                    cont['ego.evy'] = [state_dict[cur_agent][0][0][17], state_dict[cur_agent][0][1][17]]
                    cont['ego.evz'] = [state_dict[cur_agent][0][0][18], state_dict[cur_agent][0][1][18]]
                    cont['ego.timer'] = [state_dict[cur_agent][0][0][19], state_dict[cur_agent][0][1][19]]
                    cont['ego.po_timer'] = [state_dict[cur_agent][0][0][20], state_dict[cur_agent][0][1][20]]
                    # cont['ego.time'] = [state_dict[cur_agent][0][0][21], state_dict[cur_agent][0][1][21]]
                    disc['ego.go_mode'] = state_dict[cur_agent][1][0]
                    disc['ego.po_mode'] = state_dict[cur_agent][1][1]
                    disc['ego.priority_mode'] = state_dict[cur_agent][1][2]
                    disc['ego.move_mode'] = state_dict[cur_agent][1][3]

                    pos_min = np.array([state_dict[cur_agent][0][0][i] for i in range(1,4)])
                    pos_max = np.array([state_dict[cur_agent][0][1][i] for i in range(1,4)]) 
                    
                    obs_name = 'deputy_ahead' if agent.id == 'deputy' else 'deputy' # make this more generalizable in the future
                    obstacle_cont = state_dict[obs_name][0]
                    obstacle_pos_min, obstacle_pos_max = np.array([obstacle_cont[0][i] for i in range(1,4)]), np.array([obstacle_cont[1][i] for i in range(1,4)])
                    pos_bounds, obstacle_bounds = np.vstack([pos_min, pos_max]).T, np.vstack([obstacle_pos_min, obstacle_pos_max]).T
                    dist_min, dist_max = dist_extrema(pos_bounds, obstacle_bounds)
                    cont['ego.dist'] = [dist_min, dist_max]
                    cont['ego.hdist'] = [D,D]

                    own_bounds = cont['ego.x'] + cont['ego.y']
                    obs_bounds = [obstacle_cont[0][1]] + [obstacle_cont[1][1]] + [obstacle_cont[0][2]] + [obstacle_cont[1][2]]
                    theta_min, theta_max = angular_span_between_rects(own_bounds, obs_bounds)

                    vel_bounds = cont['ego.vx'] + cont['ego.vy']
                    theta_v_min, theta_v_max = angular_span_between_rects(np.zeros(4), vel_bounds)
                    diff_min, diff_max = angular_bounds_diff([theta_min, theta_max],[theta_v_min, theta_v_max])
                    # TODO: fix way noise is being added, some weird things will occur in current naive implementation
                    if diff_max < diff_min: # if 2nd/3rd quadrant were both crossed
                        cont['ego.angle_minus'] = [diff_min-ep_ao, np.pi]
                        cont['ego.angle_plus'] = [-np.pi, diff_max+ep_ao]
                    else:
                        cont['ego.angle_minus'] = cont['ego.angle_plus'] = [diff_min-ep_ao, diff_max+ep_ao]

                    if disc['ego.go_mode'] == 'Active': # prop ground/linear sensor error
                        cont['ego.hx'] = [cont['ego.x'][0]-epsilon, cont['ego.x'][1]+epsilon] # just need to be here to not mess up cur_delta
                        cont['ego.hy'] = [cont['ego.y'][0]-epsilon, cont['ego.y'][1]+epsilon]
                        cont['ego.hz'] = [cont['ego.z'][0]-epsilon, cont['ego.z'][1]+epsilon]
                        cont['ego.hvx'] = [cont['ego.vx'][0]-epsilon_vel, cont['ego.vx'][1]+epsilon_vel]
                        cont['ego.hvy'] = [cont['ego.vy'][0]-epsilon_vel, cont['ego.vy'][1]+epsilon_vel]
                        cont['ego.hvz'] = [cont['ego.vz'][0]-epsilon_vel, cont['ego.vz'][1]+epsilon_vel]
                        cont['ego.ex'] = [-epsilon, epsilon]
                        cont['ego.ey'] = [-epsilon, epsilon]
                        cont['ego.ez'] = [-epsilon, epsilon]
                        cont['ego.evx'] = [-epsilon_vel, epsilon_vel]
                        cont['ego.evy'] = [-epsilon_vel, epsilon_vel]
                        cont['ego.evz'] = [-epsilon_vel, epsilon_vel]
                    if disc['ego.po_mode'] == 'Active':
                        cont['ego.hdist'] = [max(dist_min-ep_rho, 0), dist_max+ep_rho]

                else:
                    cont['other.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                    cont['other.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]] 
                    cont['other.z'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]] 
                    cont['other.vx'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]] 
                    cont['other.vy'] = [state_dict[cur_agent][0][0][5], state_dict[cur_agent][0][1][5]] 
                    cont['other.vz'] = [state_dict[cur_agent][0][0][6], state_dict[cur_agent][0][1][6]] 
                    cont['other.hx'] = [state_dict[cur_agent][0][0][7], state_dict[cur_agent][0][1][7]]
                    cont['other.hy'] = [state_dict[cur_agent][0][0][8], state_dict[cur_agent][0][1][8]]
                    cont['other.hz'] = [state_dict[cur_agent][0][0][9], state_dict[cur_agent][0][1][9]]
                    cont['other.hvx'] = [state_dict[cur_agent][0][0][10], state_dict[cur_agent][0][1][10]]
                    cont['other.hvy'] = [state_dict[cur_agent][0][0][11], state_dict[cur_agent][0][1][11]]
                    cont['other.hvz'] = [state_dict[cur_agent][0][0][12], state_dict[cur_agent][0][1][12]]
                    cont['other.ex'] = [state_dict[cur_agent][0][0][13], state_dict[cur_agent][0][1][13]]
                    cont['other.ey'] = [state_dict[cur_agent][0][0][14], state_dict[cur_agent][0][1][14]]
                    cont['other.ez'] = [state_dict[cur_agent][0][0][15], state_dict[cur_agent][0][1][15]]
                    cont['other.evx'] = [state_dict[cur_agent][0][0][16], state_dict[cur_agent][0][1][16]]
                    cont['other.evy'] = [state_dict[cur_agent][0][0][17], state_dict[cur_agent][0][1][17]]
                    cont['other.evz'] = [state_dict[cur_agent][0][0][18], state_dict[cur_agent][0][1][18]]
                    cont['other.timer'] = [state_dict[cur_agent][0][0][19], state_dict[cur_agent][0][1][19]]
                    cont['other.po_timer'] = [state_dict[cur_agent][0][0][20], state_dict[cur_agent][0][1][20]]
                    cont['other.time'] = [state_dict[cur_agent][0][0][21], state_dict[cur_agent][0][1][21]]
                    disc['other.go_mode'] = state_dict[cur_agent][1][0]
                    disc['other.po_mode'] = state_dict[cur_agent][1][1]
                    disc['other.priority_mode'] = state_dict[cur_agent][1][2]
                    disc['other.move_mode'] = state_dict[cur_agent][1][3]
                # elif disc['ego.po_mode'] == 'Active':
                #     """Testing to see why having proximity sensor active is increasing uncertainty"""
                #     ex_min, ex_max = 0, 0
                #     cont['ego.ex'] = [-ex_min, -ex_max] # analogous to -epsilon, epsilon except no longer symmetric
                #     ey_min, ey_max = 0, 0
                #     cont['ego.ey'] = [-ey_min, -ey_max]
                #     ez_min, ez_max = 0, 0
                #     cont['ego.ez'] = [-ez_min, -ez_max]

                #     evx_min, evx_max = 0, 0
                #     cont['ego.evx'] = [-evx_min, -evx_max]
                #     evy_min, evy_max = 0, 0
                #     cont['ego.evy'] = [-evy_min, -evy_max]
                #     evz_min, evz_max = 0, 0
                #     cont['ego.evz'] = [-evz_min, -evz_max]
                #     cont['ego.hx'] = [cont['ego.x'][0]-ex_min, cont['ego.x'][1]-ex_max]
                #     cont['ego.hy'] = [cont['ego.y'][0]-ey_min, cont['ego.y'][1]-ey_max]
                #     cont['ego.hz'] = [cont['ego.z'][0]-ez_min, cont['ego.z'][1]-ez_max]
                #     cont['ego.hvx'] = [cont['ego.vx'][0]-evx_min, cont['ego.vx'][1]-evx_max]
                #     cont['ego.hvy'] = [cont['ego.vy'][0]-evy_min, cont['ego.vy'][1]-evy_max]
                #     cont['ego.hvz'] = [cont['ego.vz'][0]-evz_min, cont['ego.vz'][1]-evz_max]
                # cont['ego.x'] = [state_dict[cur_agent][0][0][1], state_dict[cur_agent][0][1][1]]
                # cont['ego.y'] = [state_dict[cur_agent][0][0][2], state_dict[cur_agent][0][1][2]]
                # cont['ego.theta'] = [state_dict[cur_agent][0][0][3], state_dict[cur_agent][0][1][3]]
                # cont['ego.v'] = [state_dict[cur_agent][0][0][4], state_dict[cur_agent][0][1][4]]
                # disc['ego.agent_mode'] = state_dict[cur_agent][1][0]
                # disc['ego.track_mode'] = state_dict[cur_agent][1][1]

        return cont, disc, len_dict