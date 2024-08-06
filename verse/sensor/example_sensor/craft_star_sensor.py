import numpy as np


def sets(d, thing, attrs, vals, stars):
    if stars:
        d.update({thing + "." + k: v for k, v in zip(attrs, [vals for x in range(len(attrs))])})
    else:
        d.update({thing + "." + k: v for k, v in zip(attrs, vals)})

#def sets_disc(d, thing, attrs, vals)
#    if len(vals) > 0 and type(vals[0]) == StarSet:
#        d.update({thing + "." + k: v for k, v in zip(attrs, [vals[0] for x in range(len(attrs))])})
    #else:
    #    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})
    #d.update({thing + "." + k: v for k, v in zip(attrs, vals)})


def adds(d, thing, attrs, vals, stars):
    #TODO: how does this respond with more than 2 agents
    #    d.update({thing + "." + k: v for k, v in zip(attrs, [vals[0] for x in range(len(attrs))])})
    if stars:
        for k, v in zip(attrs, [vals for x in range(len(attrs))]):
            if thing + "." + k not in d:
                d[thing + "." + k] = [v]
            else:
                d[thing + "." + k].append(v)
    else:
        for k, v in zip(attrs, vals):
            if thing + "." + k not in d:
                d[thing + "." + k] = [v]
            else:
                d[thing + "." + k].append(v)


def set_states_2d_ball(cnts, disc, thing, val):
    state, mode, static = val
    sets(cnts, thing, ["xp", "yp", "xd", "yd", "total_time", "cycle_time"], state[1], True)
    sets(disc, thing, ["craft_mode"], mode, False)


def set_states_3d_ball(cnts, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:7])
    #assert len(transp) == 6
    sets(cnts, thing, ["xp", "yp", "xd", "yd", "total_time", "cycle_time"], state[1], True)
    sets(disc, thing, ["craft_mode"], mode, False)


def add_states_2d_ball(cont, disc, thing, val):
    state, mode, static = val
    adds(cont, thing, ["xp", "yp", "xd", "yd", "total_time", "cycle_time"], state[1], True)
    adds(disc, thing, ["craft_mode"], mode, False)


def add_states_3d_ball(cont, disc, thing, val):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:7])
    #assert len(transp) == 6
    adds(cont, thing, ["xp", "yp", "xd", "yd", "total_time", "cycle_time"], state[1], True)
    adds(disc, thing, ["craft_mode"], mode, False)


class CraftStarSensor:
    def sense(self, agent, state_dict, lane_map, simulate= True):
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if simulate:
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