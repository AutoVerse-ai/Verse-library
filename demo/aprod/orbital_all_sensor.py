import numpy as np

epsilon = 2.5
epsilon_vel = 0.00001

ep_rho = 3
ep_angle = 0.01 # radians
class OrbitalAllSensor:
    def sense(self, agent, state_dict, lane_map = None, simulate = True):
        """
        Only consider single agent scenarios, hard-code just to get things working
        """
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            if agent.id == 'deputy':
                cont['ego.x'] = state_dict['deputy'][0][1]
                cont['ego.y'] = state_dict['deputy'][0][2]
                cont['ego.z'] = state_dict['deputy'][0][3]
                cont['ego.vx'] = state_dict['deputy'][0][4]
                cont['ego.vy'] = state_dict['deputy'][0][5]
                cont['ego.vz'] = state_dict['deputy'][0][6]
                cont['ego.hx'] = state_dict['deputy'][0][7]
                cont['ego.hy'] = state_dict['deputy'][0][8]
                cont['ego.hz'] = state_dict['deputy'][0][9]
                cont['ego.hvx'] = state_dict['deputy'][0][10]
                cont['ego.hvy'] = state_dict['deputy'][0][11]
                cont['ego.hvz'] = state_dict['deputy'][0][12]
                cont['ego.ex'] = state_dict['deputy'][0][13]
                cont['ego.ey'] = state_dict['deputy'][0][14]
                cont['ego.ez'] = state_dict['deputy'][0][15]
                cont['ego.evx'] = state_dict['deputy'][0][16]
                cont['ego.evy'] = state_dict['deputy'][0][17]
                cont['ego.evz'] = state_dict['deputy'][0][18]
                cont['ego.timer'] = state_dict['deputy'][0][19]
                cont['ego.time'] = state_dict['deputy'][0][20]
                disc['ego.orbital_mode'] = state_dict['deputy'][1][0]
               
                if disc['ego.orbital_mode'] == 'GroundSensor':
                    # true_pos = np.array([state_dict['deputy'][0][i] for i in range(1,4)])
                    dir = np.random.normal(size=3)
                    dir /= np.linalg.norm(dir)
                    rad = epsilon * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                    err_pos = rad*dir
                    cont['ego.ex'] = err_pos[0]
                    cont['ego.ey'] = err_pos[1]
                    cont['ego.ez'] = err_pos[2]
                    
                    # true_vel = np.array([state_dict['deputy'][0][i] for i in range(4,7)])
                    dir_vel = np.random.normal(size=3)
                    dir_vel /= np.linalg.norm(dir_vel)
                    rad_vel = epsilon_vel * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                    err_vel = rad_vel*dir_vel
                    cont['ego.evx'] = err_vel[0]
                    cont['ego.evy'] = err_vel[1]
                    cont['ego.evz'] = err_vel[2]

                # elif disc['ego.orbital_mode'] == 'ProximitySensor':
                #     pos = np.array([state_dict['deputy'][0][i] for i in range(1,4)])
                #     rho = np.linalg.norm(pos, 2) + ep_rho*(np.random.uniform(-1,1))
                #     rho = rho if rho > 0 else 0
                #     theta = np.arctan2()


                # cont['other.x'] = state_dict['car2'][0][1] # dummy assignments
                # cont['other.y'] = state_dict['car2'][0][2]
                # disc['other.track_mode'] = state_dict['car2'][1][1]
        else:
            if agent.id == 'deputy':
                cont['ego.x'] = [state_dict['deputy'][0][0][1], state_dict['deputy'][0][1][1]]
                cont['ego.y'] = [state_dict['deputy'][0][0][2], state_dict['deputy'][0][1][2]] 
                cont['ego.z'] = [state_dict['deputy'][0][0][3], state_dict['deputy'][0][1][3]] 
                cont['ego.vx'] = [state_dict['deputy'][0][0][4], state_dict['deputy'][0][1][4]] 
                cont['ego.vy'] = [state_dict['deputy'][0][0][5], state_dict['deputy'][0][1][5]] 
                cont['ego.vz'] = [state_dict['deputy'][0][0][6], state_dict['deputy'][0][1][6]] 
                cont['ego.hx'] = [state_dict['deputy'][0][0][7], state_dict['deputy'][0][1][7]]
                cont['ego.hy'] = [state_dict['deputy'][0][0][8], state_dict['deputy'][0][1][8]]
                cont['ego.hz'] = [state_dict['deputy'][0][0][9], state_dict['deputy'][0][1][9]]
                cont['ego.hvx'] = [state_dict['deputy'][0][0][10], state_dict['deputy'][0][1][10]]
                cont['ego.hvy'] = [state_dict['deputy'][0][0][11], state_dict['deputy'][0][1][11]]
                cont['ego.hvz'] = [state_dict['deputy'][0][0][12], state_dict['deputy'][0][1][12]]
                cont['ego.ex'] = [state_dict['deputy'][0][0][13], state_dict['deputy'][0][1][13]]
                cont['ego.ey'] = [state_dict['deputy'][0][0][14], state_dict['deputy'][0][1][14]]
                cont['ego.ez'] = [state_dict['deputy'][0][0][15], state_dict['deputy'][0][1][15]]
                cont['ego.evx'] = [state_dict['deputy'][0][0][16], state_dict['deputy'][0][1][16]]
                cont['ego.evy'] = [state_dict['deputy'][0][0][17], state_dict['deputy'][0][1][17]]
                cont['ego.evz'] = [state_dict['deputy'][0][0][18], state_dict['deputy'][0][1][18]]
                cont['ego.timer'] = [state_dict['deputy'][0][0][19], state_dict['deputy'][0][1][19]]
                cont['ego.time'] = [state_dict['deputy'][0][0][20], state_dict['deputy'][0][1][20]]
                disc['ego.orbital_mode'] = state_dict['deputy'][1][0]

                if disc['ego.orbital_mode'] == 'GroundSensor':
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
                    
                # cont['ego.x'] = [state_dict['deputy'][0][0][1], state_dict['deputy'][0][1][1]]
                # cont['ego.y'] = [state_dict['deputy'][0][0][2], state_dict['deputy'][0][1][2]]
                # cont['ego.theta'] = [state_dict['deputy'][0][0][3], state_dict['deputy'][0][1][3]]
                # cont['ego.v'] = [state_dict['deputy'][0][0][4], state_dict['deputy'][0][1][4]]
                # disc['ego.agent_mode'] = state_dict['deputy'][1][0]
                # disc['ego.track_mode'] = state_dict['deputy'][1][1]

        return cont, disc, len_dict