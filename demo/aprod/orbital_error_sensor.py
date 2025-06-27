import numpy as np

epsilon = 0.05
epsilon_vel = 0.00001
class OrbitalErrorSensor:
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
                cont['ego.ex'] = state_dict['deputy'][0][7]
                cont['ego.ey'] = state_dict['deputy'][0][8]
                cont['ego.ez'] = state_dict['deputy'][0][9]
                cont['ego.evx'] = state_dict['deputy'][0][10]
                cont['ego.evy'] = state_dict['deputy'][0][11]
                cont['ego.evz'] = state_dict['deputy'][0][12]
                cont['ego.timer'] = state_dict['deputy'][0][13]
                cont['ego.time'] = state_dict['deputy'][0][14]
                disc['ego.orbital_mode'] = state_dict['deputy'][1][0]
               
                if disc['ego.orbital_mode'] == 'GroundSensor':
                    true_pos = np.array([state_dict['deputy'][0][i] for i in range(1,4)])
                    dir = np.random.normal(size=3)
                    dir /= np.linalg.norm(dir)
                    rad = epsilon * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                    err_pos = rad*dir
                    cont['ego.ex'] = err_pos[0]
                    cont['ego.ey'] = err_pos[1]
                    cont['ego.ez'] = err_pos[2]
                    
                    true_vel = np.array([state_dict['deputy'][0][i] for i in range(4,7)])
                    dir_vel = np.random.normal(size=3)
                    dir_vel /= np.linalg.norm(dir_vel)
                    rad_vel = epsilon_vel * np.cbrt(np.random.uniform(0, 1)) # uniform sampling in volume apparently 
                    err_vel = rad_vel*dir_vel
                    cont['ego.evx'] = err_vel[0]
                    cont['ego.evy'] = err_vel[1]
                    cont['ego.evz'] = err_vel[2]

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
                cont['ego.ex'] = [state_dict['deputy'][0][0][7], state_dict['deputy'][0][1][7]]
                cont['ego.ey'] = [state_dict['deputy'][0][0][8], state_dict['deputy'][0][1][8]]
                cont['ego.ez'] = [state_dict['deputy'][0][0][9], state_dict['deputy'][0][1][9]]
                cont['ego.evx'] = [state_dict['deputy'][0][0][10], state_dict['deputy'][0][1][10]]
                cont['ego.evy'] = [state_dict['deputy'][0][0][11], state_dict['deputy'][0][1][11]]
                cont['ego.evz'] = [state_dict['deputy'][0][0][12], state_dict['deputy'][0][1][12]]
                cont['ego.timer'] = [state_dict['deputy'][0][0][13], state_dict['deputy'][0][1][13]]
                cont['ego.time'] = [state_dict['deputy'][0][0][14], state_dict['deputy'][0][1][14]]
                disc['ego.orbital_mode'] = state_dict['deputy'][1][0]

                if disc['ego.orbital_mode'] == 'GroundSensor':
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