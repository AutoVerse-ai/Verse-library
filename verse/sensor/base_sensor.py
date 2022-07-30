import numpy as np
from verse.agents.base_agent import BaseAgent


def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})


def adds(d, thing, attrs, vals):
    for k, v in zip(attrs, vals):
        if thing + '.' + k not in d:
            d[thing + '.' + k] = [v]
        else:
            d[thing + '.' + k].append(v)


def set_states_2d(cnts, disc, thing, val, cont_var, disc_var, stat_var):
    state, mode, static = val
    sets(cnts, thing, cont_var, state[1:])
    sets(disc, thing, disc_var, mode)
    sets(disc, thing, stat_var, static)


def set_states_3d(cnts, disc, thing, val, cont_var, disc_var, stat_var):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:])
    # assert len(transp) == 4
    sets(cnts, thing, cont_var, transp)
    sets(disc, thing, disc_var, mode)
    sets(disc, thing, stat_var, static)


def add_states_2d(cont, disc, thing, val, cont_var, disc_var, stat_var):
    state, mode, static = val
    adds(cont, thing, cont_var, state[1:])
    adds(disc, thing, disc_var, mode)
    adds(disc, thing, stat_var, static)


def add_states_3d(cont, disc, thing, val, cont_var, disc_var, stat_var):
    state, mode, static = val
    transp = np.transpose(np.array(state)[:, 1:])
    assert len(transp) == 4
    adds(cont, thing, cont_var, transp)
    adds(disc, thing, disc_var, mode)
    adds(disc, thing, stat_var, static)

# TODO-PARSER: Update base sensor


class BaseSensor():
    # The baseline sensor is omniscient. Each agent can get the state of all other agents
    def sense(self, scenario, agent: BaseAgent, state_dict, lane_map):
        cont = {}
        disc = {}
        len_dict = {'others': len(state_dict)-1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    # Get type of ego
                    controller_args = agent.controller.args
                    arg_type = None
                    for arg in controller_args:
                        if arg.name == 'ego':
                            arg_type = arg.typ
                            break 
                    if arg_type is None:
                        continue
                        raise ValueError(f"Invalid arg for ego")
                    cont_var = agent.controller.state_defs[arg_type].cont
                    disc_var = agent.controller.state_defs[arg_type].disc
                    stat_var = agent.controller.state_defs[arg_type].static
                    set_states_2d(
                        cont, disc, 'ego', state_dict[agent_id], cont_var, disc_var, stat_var)
                else:
                    controller_args = agent.controller.args
                    arg_type = None
                    arg_name = None
                    for arg in controller_args:
                        if arg.name != 'ego' and 'map' not in arg.name:
                            arg_name = arg.name
                            arg_type = arg.typ
                            break 
                    if arg_type is None:
                        continue
                        raise ValueError(f"Invalid arg for others")
                    cont_var = agent.controller.state_defs[arg_type].cont
                    disc_var = agent.controller.state_defs[arg_type].disc
                    stat_var = agent.controller.state_defs[arg_type].static
                    add_states_2d(
                        cont, disc, arg_name, state_dict[agent_id], cont_var, disc_var, stat_var)

        else:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    # Get type of ego
                    controller_args = agent.controller.args
                    arg_type = None
                    for arg in controller_args:
                        if arg.name == 'ego':
                            arg_type = arg.typ
                            break 
                    if arg_type is None:
                        raise ValueError(f"Invalid arg for ego")
                    cont_var = agent.controller.state_defs[arg_type].cont
                    disc_var = agent.controller.state_defs[arg_type].disc
                    stat_var = agent.controller.state_defs[arg_type].static
                    set_states_3d(
                        cont, disc, 'ego', state_dict[agent_id], cont_var, disc_var, stat_var)
                else:
                    controller_args = agent.controller.args
                    arg_type = None
                    arg_name = None
                    for arg in controller_args:
                        if arg.name != 'ego' and 'map' not in arg.name:
                            arg_name = arg.name
                            arg_type = arg.typ
                            break 
                    if arg_type is None:
                        raise ValueError(f"Invalid arg for others")
                    cont_var = agent.controller.state_defs[arg_type].cont
                    disc_var = agent.controller.state_defs[arg_type].disc
                    stat_var = agent.controller.state_defs[arg_type].static
                    add_states_3d(cont, disc, arg_name, state_dict[agent_id], cont_var, disc_var, stat_var)
                
        return cont, disc, len_dict
