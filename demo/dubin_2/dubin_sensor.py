import numpy as np
from math import floor
from dubins_agent import NPCAgent, CarAgent

from verse.utils.utils import wrap_to_pi
    


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

def isIntersect(rect1, rect2) -> bool:
    lb11 = rect1[0]
    lb12 = rect1[1]
    ub11 = rect1[2]
    ub12 = rect1[3]

    lb21 = rect2[0]
    lb22 = rect2[1]
    ub21 = rect2[2]
    ub22 = rect2[3]
    return not (lb11 > ub21 or lb21 > ub11 or (lb12 > ub22) or (lb22 > ub22))

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
                #print(state_dict['car1'])
                curr_x = state_dict['car1'][0][1]
                curr_y = state_dict['car1'][0][2]
                curr_theta = state_dict['car1'][0][3]
                curr_v = state_dict['car1'][0][4]
                obstacle_x = state_dict['car2'][0][1]
                obstacle_y = state_dict['car2'][0][2]
                obs_theta = state_dict['car2'][0][3]
                obs_v = state_dict['car2'][0][4]
                
                ego_time = state_dict['car1'][0][5]
                
                curr_vx = curr_v * np.cos(curr_theta)
                curr_vy = curr_v * np.sin(curr_theta)
                
                obs_vx = obs_v * np.cos(obs_theta)
                obs_vy = obs_v * np.sin(obs_theta)
                
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy = obstacle_y - curr_y
                dx = obstacle_x - curr_x
                    
                rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2) # Just considering 2-D distance right now (neglect z)

                #print(rho)

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
            if agent.id == "car2":
                len_dict['others'] = 1
                
                curr_x = state_dict['car2'][0][1]
                curr_y = state_dict['car2'][0][2]
                curr_theta = state_dict['car2'][0][3]
                curr_v = state_dict['car2'][0][4]
                obstacle_x = state_dict['car1'][0][1]
                obstacle_y = state_dict['car1'][0][2]
                obs_theta = state_dict['car1'][0][3]
                obs_v = state_dict['car1'][0][4]
                
                ego_time = state_dict['car1'][0][5]
                
                curr_vx = curr_v * np.cos(curr_theta)
                curr_vy = curr_v * np.sin(curr_theta)
                
                obs_vx = obs_v * np.cos(obs_theta)
                obs_vy = obs_v * np.sin(obs_theta)
                
                # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
                dy = obstacle_y - curr_y
                dx = obstacle_x - curr_x
                    
                rho = np.sqrt((curr_x - obstacle_x)**2 + (curr_y - obstacle_y)**2) # Just considering 2-D distance right now (neglect z)

                #print(rho)

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
                
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                
        else:
            #print(state_dict['car1'])
            if agent.id == 'car1':
                len_dict['others'] = 1 
                #########
                # New Stuff
                #########
                # lower bound states
                curr_x_min = state_dict['car1'][0][0][1]
                curr_y_min = state_dict['car1'][0][0][2]
                curr_theta_min = state_dict['car1'][0][0][3]
                curr_v_min = state_dict['car1'][0][0][4]
                obstacle_x_min = state_dict['car2'][0][0][1]
                obstacle_y_min = state_dict['car2'][0][0][2]
                obs_theta_min = state_dict['car2'][0][0][3]
                obs_v_min = state_dict['car2'][0][0][4]  
                
                # Timer variable, no uncertainty
                ego_time = state_dict['car1'][0][0][5]
                curr_x_max = state_dict['car1'][0][1][1]
                curr_y_max = state_dict['car1'][0][1][2]
                curr_theta_max = state_dict['car1'][0][1][3]
                curr_v_max = state_dict['car1'][0][1][4]
                obstacle_x_max = state_dict['car2'][0][1][1]
                obstacle_y_max = state_dict['car2'][0][1][2]
                obs_theta_max = state_dict['car2'][0][1][3]
                obs_v_max = state_dict['car2'][0][1][4]   
                # print("curr_x: ", curr_x_min, curr_x_max)
                # print("curr_y: ", curr_y_min, curr_y_max)
                # print("obstacle_x: ", obstacle_x_min, obstacle_x_max)
                # print("obstacle_y: ", obstacle_y_min, obstacle_y_max)
                
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

                # Calculate rho_min and rho_max using the get_extreme function
                # ego_rect = [curr_x_min, curr_y_min, curr_x_max, curr_y_max]
                # obstacle_rect = [obstacle_x_min, obstacle_y_min, obstacle_x_max, obstacle_y_max]
                # rho_min, rho_max = get_extreme(ego_rect, obstacle_rect)
                

                #print("rho_min: ", rho_min, "rho_max: ", rho_max)
                
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
                # arho_min = np.inf # does this make sense
                # arho_max = -np.inf
                
                # own_ext = [(curr_x_min, curr_y_min), (curr_x_max, curr_y_max), (curr_x_min, curr_y_max), (curr_x_max, curr_y_min)]
                # int_ext = [(obstacle_x_min, obstacle_y_min), (obstacle_x_max, obstacle_y_max), (obstacle_x_min, obstacle_y_max), (obstacle_x_max, obstacle_y_min)]
                # for own_vert in own_ext:
                #     for int_vert in int_ext:
                #         arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi)
                #         arho_max = max(arho_max, arho)
                #         arho_min = min(arho_min, arho)
                
                # theta_min = wrap_to_pi((2*np.pi-curr_theta_max)+arho_min)
                # theta_max = wrap_to_pi((2*np.pi-curr_theta_min)+arho_max) 

                

                if theta_max<theta_min: 
                    cont['ego.theta'] = [-np.pi, theta_min]
                else:
                    cont['ego.theta'] = [theta_min, theta_max]
                
                psi_max = wrap_to_pi(obs_theta_max - curr_theta_min) #np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                psi_min = wrap_to_pi(obs_theta_min - curr_theta_max)
                
                if psi_max<psi_min: # bound issue due to wrapping
                    cont['ego.psi'] = [-np.pi, psi_min]
                else:
                    cont['ego.psi'] = [psi_min, psi_max]
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
                cont['ego.v_own'] = [
                    v_own_min, v_own_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                cont['ego.v_int'] = [
                    v_int_min, v_int_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                
                
                cont['ego.timer_DL'] = [ego_time, ego_time]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
            if agent.id == 'car2':
                len_dict['others'] = 1 
                #########
                # New Stuff
                #########
                # lower bound states
                curr_x_min = state_dict['car2'][0][0][1]
                curr_y_min = state_dict['car2'][0][0][2]
                curr_theta_min = state_dict['car2'][0][0][3]
                curr_v_min = state_dict['car2'][0][0][4]
                obstacle_x_min = state_dict['car1'][0][0][1]
                obstacle_y_min = state_dict['car1'][0][0][2]
                obs_theta_min = state_dict['car1'][0][0][3]
                obs_v_min = state_dict['car1'][0][0][4]  
                
                # Timer variable, no uncertainty
                ego_time = state_dict['car2'][0][0][5]
                curr_x_max = state_dict['car2'][0][1][1]
                curr_y_max = state_dict['car2'][0][1][2]
                curr_theta_max = state_dict['car2'][0][1][3]
                curr_v_max = state_dict['car2'][0][1][4]
                obstacle_x_max = state_dict['car1'][0][1][1]
                obstacle_y_max = state_dict['car1'][0][1][2]
                obs_theta_max = state_dict['car1'][0][1][3]
                obs_v_max = state_dict['car1'][0][1][4]   
                # print("curr_x: ", curr_x_min, curr_x_max)
                # print("curr_y: ", curr_y_min, curr_y_max)
                # print("obstacle_x: ", obstacle_x_min, obstacle_x_max)
                # print("obstacle_y: ", obstacle_y_min, obstacle_y_max)
                
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

                # Calculate rho_min and rho_max using the get_extreme function
                # ego_rect = [curr_x_min, curr_y_min, curr_x_max, curr_y_max]
                # obstacle_rect = [obstacle_x_min, obstacle_y_min, obstacle_x_max, obstacle_y_max]
                # rho_min, rho_max = get_extreme(ego_rect, obstacle_rect)
                

                #print("rho_min: ", rho_min, "rho_max: ", rho_max)
                
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
                # arho_min = np.inf # does this make sense
                # arho_max = -np.inf
                
                # own_ext = [(curr_x_min, curr_y_min), (curr_x_max, curr_y_max), (curr_x_min, curr_y_max), (curr_x_max, curr_y_min)]
                # int_ext = [(obstacle_x_min, obstacle_y_min), (obstacle_x_max, obstacle_y_max), (obstacle_x_min, obstacle_y_max), (obstacle_x_max, obstacle_y_min)]
                # for own_vert in own_ext:
                #     for int_vert in int_ext:
                #         arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi)
                #         arho_max = max(arho_max, arho)
                #         arho_min = min(arho_min, arho)
                
                # theta_min = wrap_to_pi((2*np.pi-curr_theta_max)+arho_min)
                # theta_max = wrap_to_pi((2*np.pi-curr_theta_min)+arho_max) 

                

                if theta_max<theta_min: 
                    cont['ego.theta'] = [-np.pi, theta_min]
                else:
                    cont['ego.theta'] = [theta_min, theta_max]
                
                psi_max = wrap_to_pi(obs_theta_max - curr_theta_min) #np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
                psi_min = wrap_to_pi(obs_theta_min - curr_theta_max)
                
                if psi_max<psi_min: # bound issue due to wrapping
                    cont['ego.psi'] = [-np.pi, psi_min]
                else:
                    cont['ego.psi'] = [psi_min, psi_max]
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
                cont['ego.v_own'] = [
                    v_own_min, v_own_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                cont['ego.v_int'] = [
                    v_int_min, v_int_max#state_dict['car'][0][0][4], state_dict['car'][0][1][4]
                ]
                
                
                cont['ego.timer_DL'] = [ego_time, ego_time]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
        
        return cont, disc, len_dict
    

class MultiAgentsDubinSensor:

    def heuristics(self, list_of_range):
        if len(list_of_range) == 0:
            raise Exception("This should not be empty")

        res = list_of_range[0]
        def func(range1, range2):
            x1, y1 = range1 # Need to ensure that x1 <= y1
            x2, y2 = range2 # Need to ensure that x2 <= y2
            if x1 > y2:
                return [x2, y1]
            if x2 > y1:
                return [x1, y2]

            # Now we have x1 <= y2 and x2 <= y1

            return [min(x1, x2), max(y1, y2)]
        for interval in list_of_range:
            res = func(res, interval)
        return res
    
    

    def get_info_for_verify(self, curr_x_range, curr_y_range, curr_theta_range, curr_v_range, obs_x_range, obs_y_range, obs_theta_range, obs_v_range):
        curr_x_min = curr_x_range[0]
        curr_x_max = curr_x_range[1]

        curr_y_min = curr_y_range[0]
        curr_y_max = curr_y_range[1]

        curr_theta_min = curr_theta_range[0]
        curr_theta_max = curr_theta_range[1]

        curr_v_min = curr_v_range[0]
        curr_v_max = curr_v_range[1]

        obs_x_min = obs_x_range[0]
        obs_x_max = obs_x_range[1]

        obs_y_min = obs_y_range[0]
        obs_y_max = obs_y_range[1]

        obs_theta_min = obs_theta_range[0]
        obs_theta_max = obs_theta_range[1]

        obs_v_min = obs_v_range[0]
        obs_v_max = obs_v_range[1]


        curr_vx_max = curr_v_max * np.max([np.cos(curr_theta_max), np.cos(curr_theta_min)])
        curr_vy_max = curr_v_max * np.max([np.sin(curr_theta_max), np.sin(curr_theta_min)])
        
        curr_vx_min = curr_v_min * np.min([np.cos(curr_theta_max), np.cos(curr_theta_min)])
        curr_vy_min = curr_v_min * np.min([np.sin(curr_theta_max), np.sin(curr_theta_min)])
        
        #obs_vx = obs_v * np.cos(obs_theta)
        #obs_vy = obs_v * np.sin(obs_theta)             
        
        # Calcs from Stanley Bak (acasxu_closed_loop_sim/acasxu_dubins/acasxu_dubins.py state7_to_state5() function)
        dy_max = np.max(np.abs([obs_y_max - curr_y_min, curr_y_max - obs_y_min]))
        dx_max = np.max(np.abs([obs_x_max - curr_x_min, curr_x_max - obs_x_min]))
        
        dy_min = np.min(np.abs([obs_y_max - curr_y_min, curr_y_max - obs_y_min]))
        dx_min = np.min(np.abs([obs_x_max - curr_x_min, curr_x_max - obs_x_min]))
        
        rho_max = np.sqrt(dx_max**2 + dy_max**2) # Just considering 2-D distance right now (neglect z)
        rho_min = np.sqrt(dx_min**2 + dy_min**2)
        
        # Range for velocities
        v_own_max = curr_v_max
        v_int_max = obs_v_max #0 # static obstacle, so hardcode zero velocity
        
        v_own_min = curr_v_min
        v_int_min = obs_v_min
        
        # Angular ranges
        sign_dy_max = np.max([obs_y_max - curr_y_min, obs_y_min - curr_y_max])
        sign_dx_max = np.max([obs_x_max - curr_x_min, obs_x_min - curr_x_max])
        
        sign_dy_min = np.min([obs_y_max - curr_y_min, obs_y_min - curr_y_max])
        sign_dx_min = np.min([obs_x_max - curr_x_min, obs_x_min - curr_x_max])
        
        theta_max = np.max([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_min
        theta_min = np.min([np.arctan2(sign_dy_max, sign_dx_min), np.arctan2(sign_dy_min, sign_dx_max), np.arctan2(sign_dy_max, sign_dx_max), np.arctan2(sign_dy_min, sign_dx_min)]) - curr_theta_max

        psi_max = obs_theta_max - curr_theta_min #np.arctan2(obs_vy, obs_vx) - np.arctan2(curr_vy, curr_vx)
        psi_min = obs_theta_min - curr_theta_max


        rho_range = [rho_min, rho_max]
        v_own_range = [v_own_min, v_own_max]
        v_int_range = [v_int_min, v_int_max]
        theta_range = []
        if psi_max<psi_min: # bound issue due to wrapping
            theta_range = [-np.pi, psi_min]
        else:
            theta_range = [psi_min, psi_max]
        psi_range = []
        if psi_max<psi_min: # bound issue due to wrapping
            psi_range = [-np.pi, psi_min]
        else:
            psi_range = [psi_min, psi_max]

        return rho_range, v_own_range, v_int_range, theta_range, psi_range
    
    def sense(self, agent : CarAgent, state_dict, lane_map, simulate : False):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        
        if simulate:
            pass
        else:
            others = {}
            for k, v in state_dict.items():
                if k != agent.id:
                    others[k] = v
            



        pass