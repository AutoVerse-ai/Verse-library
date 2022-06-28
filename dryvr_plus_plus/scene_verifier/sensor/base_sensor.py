import numpy as np

def sets(d, thing, attrs, vals):
    d.update({thing + "." + k: v for k, v in zip(attrs, vals)})

def adds(d, thing, attrs, vals):
    for k, v in zip(attrs, vals):
        if thing + '.' + k not in d:
            d[thing + '.' + k] = [v] 
        else:
            d[thing + '.' + k].append(v) 

def set_states_2d(cnts, disc, thing, val, cont_var, disc_var):
    state, mode = val
    sets(cnts, thing, cont_var, state[1:5])
    sets(disc, thing, disc_var, mode)

def set_states_3d(cnts, disc, thing, val, cont_var, disc_var):
    state, mode = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    sets(cnts, thing, cont_var, transp)
    sets(disc, thing, disc_var, mode)

def add_states_2d(cont, disc, thing, val, cont_var, disc_var):
    state, mode = val
    adds(cont, thing, cont_var, state[1:5])
    adds(disc, thing, disc_var, mode)

def add_states_3d(cont, disc, thing, val, cont_var, disc_var):
    state, mode = val
    transp = np.transpose(np.array(state)[:, 1:5])
    assert len(transp) == 4
    adds(cont, thing, cont_var, transp)
    adds(disc, thing, disc_var, mode)

class BaseSensor():
    # The baseline sensor is omniscient. Each agent can get the state of all other agents
    def sense(self, scenario, agent, state_dict, lane_map):
        cont = {}
        disc = {}
        len_dict = {'others':len(state_dict)-1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    if agent.controller.vars_dict:
                        cont_var = agent.controller.vars_dict['ego']['cont']
                        disc_var = agent.controller.vars_dict['ego']['disc']
                        set_states_2d(cont, disc, 'ego', state_dict[agent_id], cont_var, disc_var)
                else:
                    if agent.controller.vars_dict:
                        cont_var = agent.controller.vars_dict['others']['cont']
                        disc_var = agent.controller.vars_dict['others']['disc']
                        add_states_2d(cont, disc, 'others', state_dict[agent_id], cont_var, disc_var)
        else:
            for agent_id in state_dict:
                if agent_id == agent.id:
                    if agent.controller.vars_dict:
                        cont_var = agent.controller.vars_dict['ego']['cont']
                        disc_var = agent.controller.vars_dict['ego']['disc']
                        set_states_3d(cont, disc, "ego", state_dict[agent_id], cont_var, disc_var)
                else:
                    if agent.controller.vars_dict:
                        cont_var = agent.controller.vars_dict['others']['cont']
                        disc_var = agent.controller.vars_dict['others']['disc']
                        add_states_3d(cont, disc, 'others', state_dict[agent_id], cont_var, disc_var)
        return cont, disc, len_dict