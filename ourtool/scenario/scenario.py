from typing import Dict, List
import copy

from ourtool.agents.base_agent import BaseAgent
from ourtool.automaton.guard import GuardExpression
from pythonparser import Guard
from pythonparser import Reset
from ourtool.simulator.simulator import Simulator

class Scenario:
    def __init__(self):
        self.agent_dict = {}
        self.simulator = Simulator()
        self.init_dict = {}
        self.init_mode_dict = {}

    def add_agent(self, agent:BaseAgent):
        self.agent_dict[agent.id] = agent

    def set_init(self, init_list, init_mode_list):
        assert len(init_list) == len(self.agent_dict)
        assert len(init_mode_list) == len(self.agent_dict)
        for i,agent_id in enumerate(self.agent_dict.keys()):
            self.init_dict[agent_id] = copy.deepcopy(init_list[i])
            self.init_mode_dict[agent_id] = copy.deepcopy(init_mode_list[i])

    def simulate(self, time_horizon):
        init_list = []
        init_mode_list = []
        agent_list = []
        for agent_id in self.agent_dict:
            init_list.append(self.init_dict[agent_id])
            init_mode_list.append(self.init_mode_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        return self.simulator.simulate(init_list, init_mode_list, agent_list, self, time_horizon)

    def get_all_transition(self, state_dict):
        guard_hit = False
        satisfied_guard = []
        for agent_id in state_dict:
            agent:BaseAgent = self.agent_dict[agent_id]
            agent_state, agent_mode = state_dict[agent_id]
            t = agent_state[0]
            agent_state = agent_state[1:]
            paths = agent.controller.getNextModes(agent_mode)
            for path in paths:
                guard_list = []
                reset_list = []
                for item in path:
                    if isinstance(item, Guard):
                        guard_list.append("(" + item.code + ")")
                    elif isinstance(item, Reset):
                        reset_list.append(item.code)
                guard_str = ' and '.join(guard_list)
                guard_expression = GuardExpression(logic_str = guard_str)
                # print(guard_expression.generate_guard_string_python())
                discrete_variable_dict = {}
                agent_mode_split = agent_mode.split(',')
                assert len(agent_mode_split)==len(agent.controller.discrete_variables)
                for dis_var,dis_val in zip(agent.controller.discrete_variables, agent_mode_split):
                    for key in agent.controller.modes:
                        if dis_val in agent.controller.modes[key]:
                            tmp = key+'.'+dis_val
                            break
                    discrete_variable_dict[dis_var] = tmp
                guard_can_satisfy = guard_expression.execute_guard(discrete_variable_dict)
                if guard_can_satisfy:
                    dryvr_guard_string = guard_expression.generate_guard_string_python()
                    # Substitute value into dryvr guard string
                    for i, variable in enumerate(agent.controller.variables):
                        dryvr_guard_string = dryvr_guard_string.replace(variable, str(agent_state[i]))
                    # Evaluate the guard strings 
                    res = eval(dryvr_guard_string)
                    # if result is true, check reset and construct next mode             
                    if res:
                        next_init = agent_state
                        dest = agent_mode.split(',')
                        for reset in reset_list:
                            # Specify the destination mode
                            if "mode" in reset:
                                for i, discrete_variable in enumerate(agent.controller.discrete_variables):
                                    if discrete_variable in reset:
                                        break
                                tmp = reset.split('=')
                                tmp = tmp[1].split('.')
                                if tmp[0].strip(' ') in agent.controller.modes:
                                    dest[i] = tmp[1]
                                
                            else: 
                                for i, cts_variable in enumerate(agent.controller.variables):
                                    if cts_variable in reset:
                                        break 
                                tmp = reset.split('=')
                                next_init[i] = float(tmp[1])
                        dest = ','.join(dest)
                        next_transition = (
                            agent_id, agent_mode, dest, next_init, 
                        )
                        satisfied_guard.append(next_transition)
        return satisfied_guard