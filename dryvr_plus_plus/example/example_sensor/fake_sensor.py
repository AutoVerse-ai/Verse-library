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
        return cnts, disc, {}


def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})


def adds(d, thing, attrs, vals):
    for k, v in zip(attrs, vals):
        if thing + '.' + k not in d:
            d[thing + '.' + k] = [v]
        else:
            d[thing + '.' + k].append(v)


def add_states_2d(cont, disc, thing, val):
    state, mode, static = val
    adds(cont, thing, ['x', 'y', 'theta', 'v'], state[1:5])
    adds(disc, thing, ["vehicle_mode", "lane_mode"], mode)
    adds(disc, thing, ['type'], static)


def add_states_3d(cont, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    adds(cont, thing, ["x", "y", "theta", "v"], transp)
    adds(disc, thing, ["vehicle_mode", "lane_mode"], mode)
    adds(disc, thing, ['type'], static)


def set_states_2d(cnts, disc, thing, val):
    state, mode, static = val
    sets(cnts, thing, ["x", "y", "theta", "v"], state[1:5])
    sets(disc, thing, ["vehicle_mode", "lane_mode"], mode)
    sets(disc, thing, ['type'], static)


def set_states_3d(cnts, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    sets(cnts, thing, ["x", "y", "theta", "v"], transp)
    sets(disc, thing, ["vehicle_mode", "lane_mode"], mode)
    sets(disc, thing, ['type'], static)


class FakeSensor2:
    def sense(self, scenario, agent, state_dict, lane_map):
        cnts = {}
        disc = {}
        tmp = np.array(state_dict['car1'][0])
        if tmp.ndim < 2:
            if agent.id == 'car1':
                set_states_2d(cnts, disc, "ego", state_dict["car1"])
                set_states_2d(cnts, disc, "other", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_2d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'car2':
                set_states_2d(cnts, disc, "other", state_dict["car1"])
                set_states_2d(cnts, disc, "ego", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_2d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'sign':
                set_states_2d(cnts, disc, "ego", state_dict["sign"])
                set_states_2d(cnts, disc, "other", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_2d(cnts, disc, "sign", state_dict["sign"])
            return cnts, disc, {}
        else:
            if agent.id == 'car1':
                set_states_3d(cnts, disc, "ego", state_dict["car1"])
                set_states_3d(cnts, disc, "other", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_3d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'car2':
                set_states_3d(cnts, disc, "other", state_dict["car1"])
                set_states_3d(cnts, disc, "ego", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_3d(cnts, disc, "sign", state_dict["sign"])
            elif agent.id == 'sign':
                set_states_3d(cnts, disc, "ego", state_dict["sign"])
                set_states_3d(cnts, disc, "other", state_dict["car2"])
                if "sign" in state_dict:
                    set_states_3d(cnts, disc, "sign", state_dict["sign"])
            return cnts, disc, {}


class FakeSensor3:
    def sense(self, scenario, agent, state_dict, lane_map):
        cont = {}
        disc = {}
        len_dict = {'others': len(state_dict)-1}
        tmp = np.array(state_dict['car1'][0])
        if tmp.ndim < 2:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_2d(cont, disc, 'ego', state_dict[agent_id])
                else:
                    add_states_2d(cont, disc, 'others', state_dict[agent_id])
        else:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_3d(cont, disc, "ego", state_dict[agent_id])
                else:
                    add_states_3d(cont, disc, 'others', state_dict[agent_id])
        return cont, disc, len_dict


def set_states_2d_ball(cnts, disc, thing, val):
    state, mode = val
    sets(cnts, thing, ["x", "y", "vx", "vy"], state[1:5])
    sets(disc, thing, ["ball_mode", "lane_mode"], mode)


def set_states_3d_ball(cnts, disc, thing, val):
    state, mode = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    sets(cnts, thing, ["x", "y", "vx", "vy"], transp)
    sets(disc, thing, ["ball_mode", "lane_mode"], mode)


def add_states_2d_ball(cont, disc, thing, val):
    state, mode = val
    adds(cont, thing, ['x', 'y', 'vx', 'vy'], state[1:5])
    adds(disc, thing, ["ball_mode", "lane_mode", "type"], mode)


def add_states_3d_ball(cont, disc, thing, val):
    state, mode = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    adds(cont, thing, ['x', 'y', 'vx', 'vy'], transp)
    adds(disc, thing, ["ball_mode", "lane_mode", "type"], mode)


class FakeSensor4:
    def sense(self, scenario, agent, state_dict, lane_map):
        cont = {}
        disc = {}
        len_dict = {'others': len(state_dict)-1}
        tmp = np.array(list(state_dict.values())[0])
        if tmp.ndim < 2:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_2d_ball(cont, disc, 'ego', state_dict[agent_id])
                else:
                    add_states_2d_ball(cont, disc, 'others',
                                       state_dict[agent_id])
        else:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_3d_ball(cont, disc, "ego", state_dict[agent_id])
                else:
                    add_states_3d_ball(cont, disc, 'others',
                                       state_dict[agent_id])
        return cont, disc, len_dict
