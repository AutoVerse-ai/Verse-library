import numpy as np


def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})


def adds(d, thing, attrs, vals):
    for k, v in zip(attrs, vals):
        if thing + "." + k not in d:
            d[thing + "." + k] = [v]
        else:
            d[thing + "." + k].append(v)


def set_states_2d_ball(cnts, disc, thing, val):
    state, mode, static = val
    sets(cnts, thing, ["x", "y", "z", "vx", "vy", "vz", "waypoint_index", "done_flag"], state[1:9])
    sets(disc, thing, ["craft_mode", "lane_mode"], mode)


def set_states_3d_ball(cnts, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:9])
    assert len(transp) == 8
    sets(cnts, thing, ["x", "y", "z", "vx", "vy", "vz", "waypoint_index", "done_flag"], transp)
    sets(disc, thing, ["craft_mode", "lane_mode"], mode)


def add_states_2d_ball(cont, disc, thing, val):
    state, mode, static = val
    adds(cont, thing, ["x", "y", "z", "vx", "vy", "vz", "waypoint_index", "done_flag"], state[1:9])
    adds(disc, thing, ["craft_mode", "lane_mode"], mode)


def add_states_3d_ball(cont, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:9])
    assert len(transp) == 8
    adds(cont, thing, ["x", "y", "z", "vx", "vy", "vz", "waypoint_index", "done_flag"], transp)
    adds(disc, thing, ["craft_mode", "lane_mode"], mode)


class QuadrotorSensor:
    def sense(self, agent, state_dict, lane_map):
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_2d_ball(cont, disc, "ego", state_dict[agent_id])
                else:
                    add_states_2d_ball(cont, disc, "others", state_dict[agent_id])
        else:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    set_states_3d_ball(cont, disc, "ego", state_dict[agent_id])
                else:
                    add_states_3d_ball(cont, disc, "others", state_dict[agent_id])
        return cont, disc, len_dict
