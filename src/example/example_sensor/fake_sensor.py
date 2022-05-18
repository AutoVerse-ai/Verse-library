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

def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})

def set_states_2d(cnts, disc, thing, val):
    state, mode = val
    sets(cnts, thing, ["x", "y", "theta", "v"], state[1:5])
    sets(disc, thing, ["vehicle_mode", "lane_mode"], mode.split(","))

def set_states_3d(cnts, disc, thing, val):
    state, mode = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    sets(cnts, thing, ["x", "y", "theta", "v"], transp)
    sets(disc, thing, ["vehicle_mode", "lane_mode"], mode.split(","))

class FakeSensor2:
    def sense(self, scenario, agent, state_dict, lane_map):
        cnts = {}
        disc = {}
        tmp = np.array(state_dict['car1'][0])
        if tmp.ndim < 2:
            if agent.id == 'car1':
                set_states_2d(cnts, disc, "ego", state_dict["car1"])
                set_states_2d(cnts, disc, "other", state_dict["car2"])
                set_states_2d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'car2':
                set_states_2d(cnts, disc, "other", state_dict["car1"])
                set_states_2d(cnts, disc, "ego", state_dict["car2"])
                set_states_2d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'sign':
                set_states_2d(cnts, disc, "ego", state_dict["sign"])
                set_states_2d(cnts, disc, "other", state_dict["car2"])
                set_states_2d(cnts, disc, "sign", state_dict["sign"])
            return cnts, disc
        else:
            if agent.id == 'car1':
                set_states_3d(cnts, disc, "ego", state_dict["car1"])
                set_states_3d(cnts, disc, "other", state_dict["car2"])
                set_states_3d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'car2':
                set_states_3d(cnts, disc, "other", state_dict["car1"])
                set_states_3d(cnts, disc, "ego", state_dict["car2"])
                set_states_3d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'sign':
                set_states_3d(cnts, disc, "ego", state_dict["sign"])
                set_states_3d(cnts, disc, "other", state_dict["car2"])
                set_states_3d(cnts, disc, "sign", state_dict["sign"])
            return cnts, disc
