from typing import Dict, List, Tuple
import copy
import itertools
import ast

import numpy as np

from ourtool.agents.base_agent import BaseAgent
from ourtool.automaton.guard import GuardExpressionAst
from ourtool.automaton.reset import ResetExpression
from pythonparser import Guard
from pythonparser import Reset
from ourtool.analysis.simulator import Simulator
from ourtool.analysis.verifier import Verifier
from ourtool.map.lane_map import LaneMap

class FakeSensor:
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
                cnts['ego.x'] = [state[0][1],state[1][1]]
                cnts['ego.y'] = [state[0][2],state[1][2]]
                cnts['ego.theta'] = [state[0][3],state[1][3]]
                cnts['ego.v'] = [state[0][4],state[1][4]]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['other.x'] = [state[0][1],state[1][1]]
                cnts['other.y'] = [state[0][2],state[1][2]]
                cnts['other.theta'] = [state[0][3],state[1][3]]
                cnts['other.v'] = [state[0][4],state[1][4]]
                disc['other.vehicle_mode'] = mode[0]
                disc['other.lane_mode'] = mode[1]
            elif agent.id == 'car2':
                state = state_dict['car2'][0]
                mode = state_dict['car2'][1].split(',')
                cnts['ego.x'] = [state[0][1],state[1][1]]
                cnts['ego.y'] = [state[0][2],state[1][2]]
                cnts['ego.theta'] = [state[0][3],state[1][3]]
                cnts['ego.v'] = [state[0][4],state[1][4]]
                disc['ego.vehicle_mode'] = mode[0]
                disc['ego.lane_mode'] = mode[1]

                state = state_dict['car1'][0]
                mode = state_dict['car1'][1].split(',')
                cnts['other.x'] = [state[0][1],state[1][1]]
                cnts['other.y'] = [state[0][2],state[1][2]]
                cnts['other.theta'] = [state[0][3],state[1][3]]
                cnts['other.v'] = [state[0][4],state[1][4]]
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

    def check_guard_hit(self, state_dict):
        lane_map = self.map 
        guard_hits = []
        is_contained = False        # TODO: Handle this
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
                        reset_list.append(item)
                # guard_expression = GuardExpression(guard_list=guard_list)
                guard_expression = GuardExpressionAst(guard_list)
                # Map the values to variables using sensor
                continuous_variable_dict, discrete_variable_dict = self.sensor.sense(self, agent, state_dict, self.map)
                
                '''Execute functions related to map to see if the guard can be satisfied'''
                '''Check guards related to modes to see if the guards can be satisfied'''
                '''Actually plug in the values to see if the guards can be satisfied'''
                # Check if the guard can be satisfied
                guard_can_satisfied = guard_expression.evaluate_guard_disc(agent, discrete_variable_dict, self.map)
                if not guard_can_satisfied:
                    continue
                guard_satisfied, is_contained = guard_expression.evaluate_guard_cont(agent, continuous_variable_dict, self.map)
                if guard_satisfied:
                    guard_hits.append((agent_id, guard_list, reset_list))
        return guard_hits, is_contained

    def get_all_transition_set(self, node):
        possible_transitions = []
        trace_length = int(len(list(node.trace.values())[0])/2)
        guard_hits = []
        guard_hit_bool = False

        # TODO: can add parallalization for this loop
        for idx in range(0,trace_length):
            # For each trace, check with the guard to see if there's any possible transition
            # Store all possible transition in a list
            # A transition is defined by (agent, src_mode, dest_mode, corresponding reset, transit idx)
            # Here we enforce that only one agent transit at a time
            all_agent_state = {}
            for agent_id in node.agent:
                all_agent_state[agent_id] = (node.trace[agent_id][idx*2:idx*2+2], node.mode[agent_id])
            hits, is_contain = self.check_guard_hit(all_agent_state)
            if hits != []:
                guard_hits.append((hits, all_agent_state, idx))
                guard_hit_bool = True
            if hits == [] and guard_hit_bool:
                break
            if is_contain:
                break

        reset_dict = {}
        reset_idx_dict = {}
        for hits, all_agent_state, hit_idx in guard_hits:
            for agent_id, guard_list, reset_list in hits:
                dest_list,reset_rect = self.apply_reset(node.agent[agent_id], reset_list, all_agent_state)
                if agent_id not in reset_dict:
                    reset_dict[agent_id] = {}
                    reset_idx_dict[agent_id] = {}
                for dest in dest_list:
                    if dest not in reset_dict[agent_id]:
                        reset_dict[agent_id][dest] = []
                        reset_idx_dict[agent_id][dest] = []
                    reset_dict[agent_id][dest].append(reset_rect)
                    reset_idx_dict[agent_id][dest].append(hit_idx)
            
        # Combine reset rects and construct transitions
        for agent in reset_dict:
            for dest in reset_dict[agent]:
                combined_rect = None 
                for rect in reset_dict[agent][dest]:
                    rect = np.array(rect)
                    if combined_rect is None:
                        combined_rect = rect 
                    else:
                        combined_rect[0,:] = np.minimum(combined_rect[0,:], rect[0,:])
                        combined_rect[1,:] = np.maximum(combined_rect[1,:], rect[1,:])
                combined_rect = combined_rect.tolist()
                min_idx = min(reset_idx_dict[agent][dest])
                max_idx = max(reset_idx_dict[agent][dest])
                transition = (agent, node.mode[agent], dest, combined_rect, (min_idx, max_idx))
                possible_transitions.append(transition)
        # Return result
        return possible_transitions

    def apply_reset(self, agent, reset_list, all_agent_state) -> Tuple[str, np.ndarray]:
        reset_expr = ResetExpression(reset_list)
        continuous_variable_dict, discrete_variable_dict = self.sensor.sense(self, agent, all_agent_state, self.map)
        dest = reset_expr.get_dest(agent, all_agent_state[agent.id], discrete_variable_dict, self.map)
        rect = reset_expr.apply_reset_continuous(agent, continuous_variable_dict, self.map)
        return dest, rect

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
                        reset_list.append(item)
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
                        reset = reset.code
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