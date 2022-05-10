import numpy as np

class FakeSensor1:
    def sense(self, scenario, agent, state_dict, lane_map):
        cnts = {}
        disc = {}
        state = state_dict['ego'][0]
        mode = state_dict['ego'][1].split(',')
        cnts['ego.x'] = state[1]
        cnts['ego.y'] = state[2]
        cnts['ego.theta'] = state[3]
        cnts['ego.v'] = state[4]
        disc['ego.vehicle_mode'] = mode[0]
        disc['ego.lane_mode'] = mode[1]
        return cnts, disc


class FakeSensor2:
    def sense(self, scenario, agent, state_dict, lane_map):
        cnts = {}
        disc = {}
        tmp = np.array(state_dict['car1'][0])
        if tmp.ndim < 2:
            if agent.id == 'car1':
                state = state_dict['car1'][0]
                mode = state_dict['car1'][1].split(',')
                cnts['ego.x'] = state[1]
                cnts['ego.y'] = state[2]
                cnts['ego.theta'] = state[3]
                cnts['ego.v'] = state[4]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['other.x'] = state[1]
                cnts['other.y'] = state[2]
                cnts['other.theta'] = state[3]
                cnts['other.v'] = state[4]
                disc['other.vehicle_mode'] = mode[0]
                disc['other.lane_mode'] = mode[1]
            elif agent.id == 'car2':
                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['ego.x'] = state[1]
                cnts['ego.y'] = state[2]
                cnts['ego.theta'] = state[3]
                cnts['ego.v'] = state[4]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car1'][0]
                mode = state_dict['car1'][1].split(',')
                cnts['other.x'] = state[1]
                cnts['other.y'] = state[2]
                cnts['other.theta'] = state[3]
                cnts['other.v'] = state[4]
                disc['other.vehicle_mode'] = mode[0]
                disc['other.lane_mode'] = mode[1]
            return cnts, disc
        else:
            if agent.id == 'car1':
                state = state_dict['car1'][0]
                mode = state_dict['car1'][1].split(',')
                cnts['ego.x'] = [state[0][1], state[1][1]]
                cnts['ego.y'] = [state[0][2], state[1][2]]
                cnts['ego.theta'] = [state[0][3], state[1][3]]
                cnts['ego.v'] = [state[0][4], state[1][4]]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['other.x'] = [state[0][1], state[1][1]]
                cnts['other.y'] = [state[0][2], state[1][2]]
                cnts['other.theta'] = [state[0][3], state[1][3]]
                cnts['other.v'] = [state[0][4], state[1][4]]
                disc['other.vehicle_mode'] = mode[0]
                disc['other.lane_mode'] = mode[1]
            elif agent.id == 'car2':
                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['ego.x'] = [state[0][1], state[1][1]]
                cnts['ego.y'] = [state[0][2], state[1][2]]
                cnts['ego.theta'] = [state[0][3], state[1][3]]
                cnts['ego.v'] = [state[0][4], state[1][4]]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car1'][0]
                mode = state_dict['car1'][1].split(',')
                cnts['other.x'] = [state[0][1], state[1][1]]
                cnts['other.y'] = [state[0][2], state[1][2]]
                cnts['other.theta'] = [state[0][3], state[1][3]]
                cnts['other.v'] = [state[0][4], state[1][4]]
                disc['other.vehicle_mode'] = mode[0]
                disc['other.lane_mode'] = mode[1]
            return cnts, disc
