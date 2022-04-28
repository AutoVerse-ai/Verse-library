from typing import Dict, List
import copy
import itertools
import ast

import numpy as np
from rsa import verify

from ourtool.agents.base_agent import BaseAgent
from ourtool.automaton.guard import GuardExpressionAst
from pythonparser import Guard
from pythonparser import Reset
from ourtool.analysis.simulator import Simulator
from ourtool.analysis.verifier import Verifier
from ourtool.map.lane_map import LaneMap

class FakeSensor:
    def sense(self, scenario, agent, state_dict, lane_map):
        cnts = {}
        disc = {}
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

class Scenario:
    def __init__(self):
        self.agent_dict = {}
        self.simulator = Simulator()
        self.verifier = Verifier()
        self.init_dict = {}
        self.init_mode_dict = {}
        self.map = None
        self.sensor = FakeSensor()

    def add_map(self, lane_map:LaneMap):
        self.map = lane_map

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
        return self.simulator.simulate(init_list, init_mode_list, agent_list, self, time_horizon, self.map)

    def verify(self, time_horizon):
        init_list = []
        init_mode_list = []
        agent_list = []
        for agent_id in self.agent_dict:
            init = self.init_dict[agent_id]
            tmp = np.array(init)
            if tmp.ndim < 2:
                init = [init, init]
            init_list.append(init)
            init_mode_list.append(self.init_mode_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        return self.verifier.compute_full_reachtube(init_list, init_mode_list, agent_list, self, time_horizon, self.map)

    def get_all_transition(self, state_dict):
        lane_map = self.map
        satisfied_guard = []
        for agent_id in state_dict:
            agent:BaseAgent = self.agent_dict[agent_id]
            agent_state, agent_mode = state_dict[agent_id]
            t = agent_state[0]
            agent_state = agent_state[1:]
            paths = agent.controller.getNextModes(agent_mode)
            for path in paths:
                # Construct the guard expression
                guard_list = []
                reset_list = []
                for item in path:
                    if isinstance(item, Guard):
                        guard_list.append(item)
                    elif isinstance(item, Reset):
                        reset_list.append(item.code)
                # guard_expression = GuardExpression(guard_list=guard_list)
                guard_expression = GuardExpressionAst(guard_list)
                # Map the values to variables using sensor
                continuous_variable_dict, discrete_variable_dict = self.sensor.sense(self, agent, state_dict, self.map)
                
                '''Execute functions related to map to see if the guard can be satisfied'''
                '''Check guards related to modes to see if the guards can be satisfied'''
                '''Actually plug in the values to see if the guards can be satisfied'''
                # Check if the guard can be satisfied
                guard_satisfied = guard_expression.evaluate_guard(agent, continuous_variable_dict, discrete_variable_dict, self.map)
                if guard_satisfied:
                    # If the guard can be satisfied, handle resets
                    next_init = agent_state
                    dest = agent_mode.split(',')
                    possible_dest = [[elem] for elem in dest]
                    for reset in reset_list:
                        # Specify the destination mode
                        if "mode" in reset:
                            for i, discrete_variable_ego in enumerate(agent.controller.vars_dict['ego']['disc']):
                                if discrete_variable_ego in reset:
                                    break
                            tmp = reset.split('=')
                            if 'map' in tmp[1]:
                                tmp = tmp[1]
                                for var in discrete_variable_dict:
                                    tmp = tmp.replace(var, f"'{discrete_variable_dict[var]}'")
                                possible_dest[i] = eval(tmp)
                            else:
                                tmp = tmp[1].split('.')
                                if tmp[0].strip(' ') in agent.controller.modes:
                                    possible_dest[i] = [tmp[1]]                            
                        else: 
                            for i, cts_variable in enumerate(agent.controller.vars_dict['ego']['cont']):
                                if "output."+cts_variable in reset:
                                    break 
                            tmp = reset.split('=')
                            tmp = tmp[1]
                            for cts_variable in continuous_variable_dict:
                                tmp = tmp.replace(cts_variable, str(continuous_variable_dict[cts_variable]))
                            next_init[i] = eval(tmp)
                    all_dest = itertools.product(*possible_dest)
                    for dest in all_dest:
                        dest = ','.join(dest)
                        next_transition = (
                            agent_id, agent_mode, dest, next_init, 
                        )
                        satisfied_guard.append(next_transition)

        return satisfied_guard