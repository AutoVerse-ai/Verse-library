import numpy as np 
from verse.agents.base_agent import BaseAgent 
import numpy as np 

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
        dist_min = ub21 - lb11 
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

class VehiclePedestrainSensor:
    # The baseline sensor is omniscient. Each agent can get the state of all other agents
    def sense(self, agent: BaseAgent, state_dict, lane_map):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            if agent.id == 'car':
                len_dict['others'] = 1 
                cont['ego.x'] = state_dict['car'][0][1]
                cont['ego.y'] = state_dict['car'][0][2]
                cont['ego.theta'] = state_dict['car'][0][3]
                cont['ego.v'] = state_dict['car'][0][4]
                cont['ego.dist'] = np.sqrt(
                    (state_dict['car'][0][1]-state_dict['pedestrain'][0][1])**2+\
                    (state_dict['car'][0][2]-state_dict['pedestrain'][0][2])**2
                )
                disc['ego.agent_mode'] = state_dict['car'][1][0]
        else:
            if agent.id == 'car':
                len_dict['others'] = 1 
                cont['ego.x'] = [
                    state_dict['car'][0][0][1], state_dict['car'][0][1][1]
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
                dist_min, dist_max = get_extreme(
                    (state_dict['car'][0][0][1],state_dict['car'][0][0][2],state_dict['car'][0][1][1],state_dict['car'][0][1][2]),
                    (state_dict['pedestrain'][0][0][1],state_dict['pedestrain'][0][0][2],state_dict['pedestrain'][0][1][1],state_dict['pedestrain'][0][1][2]),
                )
                # dist_list = []
                # if state_dict['car'][0][0][0] < state_dict['pedestrain'][0][0][0]
                # for a in range(2):
                #     for b in range(2):
                #         for c in range(2):
                #             for d in range(2):
                #                 dist = np.sqrt(
                #                     (state_dict['car'][0][a][0] - state_dict['pedestrain'][0][b][0])**2+\
                #                     (state_dict['car'][0][c][1] - state_dict['pedestrain'][0][d][1])**2
                #                 )
                #                 dist_list.append(dist)
                cont['ego.dist'] = [
                    dist_min, dist_max
                ]
                print(dist_min)
                disc['ego.agent_mode'] = state_dict['car'][1][0]
        return cont, disc, len_dict
