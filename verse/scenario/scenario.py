from typing import DefaultDict, Optional, Tuple, List, Dict, Any
import copy
import itertools
import warnings
from collections import defaultdict, namedtuple
import ast

import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.automaton import GuardExpressionAst, ResetExpression
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree
from verse.analysis.utils import find, sample_rect
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

EGO, OTHERS = "ego", "others"


class Scenario:
    def __init__(self):
        self.agent_dict = {}
        self.simulator = Simulator()
        self.verifier = Verifier()
        self.init_dict = {}
        self.init_mode_dict = {}
        self.static_dict = {}
        self.static_dict = {}
        self.map = LaneMap()
        self.sensor = BaseSensor()

        # Parameters
        self.init_seg_length = 1000
        self.verify_method = 'PW'

    def set_sensor(self, sensor):
        self.sensor = sensor

    def set_map(self, lane_map: LaneMap):
        self.map = lane_map
        # Update the lane mode field in the agent
        for agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            self.update_agent_lane_mode(agent, lane_map)

    def add_agent(self, agent: BaseAgent):
        if self.map is not None:
            # Update the lane mode field in the agent
            self.update_agent_lane_mode(agent, self.map)
        self.agent_dict[agent.id] = agent

    # TODO-PARSER: update this function
    def update_agent_lane_mode(self, agent: BaseAgent, lane_map: LaneMap):
        for lane_id in lane_map.lane_dict:
            if 'LaneMode' in agent.controller.mode_defs and lane_id not in agent.controller.mode_defs['LaneMode'].modes:
                agent.controller.mode_defs['LaneMode'].modes.append(lane_id)
        # mode_vals = list(agent.controller.modes.values())
        # agent.controller.vertices = list(itertools.product(*mode_vals))
        # agent.controller.vertexStrings = [','.join(elem) for elem in agent.controller.vertices]

    def set_init(self, init_list, init_mode_list, static_list=[]):
        assert len(init_list) == len(self.agent_dict)
        assert len(init_mode_list) == len(self.agent_dict)
        assert len(static_list) == len(
            self.agent_dict) or len(static_list) == 0
        print(init_mode_list)
        print(type(init_mode_list))
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.init_dict[agent_id] = copy.deepcopy(init_list[i])
            self.init_mode_dict[agent_id] = copy.deepcopy(init_mode_list[i])
            if static_list:
                self.static_dict[agent_id] = copy.deepcopy(static_list[i])
            else:
                self.static_dict[agent_id] = []

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step) -> AnalysisTree:
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id]))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        return self.simulator.simulate(init_list, init_mode_list, static_list, agent_list, self, time_horizon, time_step, self.map)

    def verify(self, time_horizon, time_step) -> AnalysisTree:
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        for agent_id in self.agent_dict:
            init = self.init_dict[agent_id]
            tmp = np.array(init)
            if tmp.ndim < 2:
                init = [init, init]
            init_list.append(init)
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        return self.verifier.compute_full_reachtube(init_list, init_mode_list, static_list, agent_list, self, time_horizon, time_step, self.map, self.init_seg_length, self.verify_method)

    def apply_reset(self, agent: BaseAgent, reset_list, all_agent_state) -> Tuple[str, np.ndarray]:
        lane_map = self.map
        dest = []
        rect = []

        agent_state, agent_mode, agent_static = all_agent_state[agent.id]

        dest = copy.deepcopy(agent_mode)
        possible_dest = [[elem] for elem in dest]
        ego_type = find(agent.controller.args, lambda a: a.name == EGO).typ
        rect = copy.deepcopy([agent_state[0][1:], agent_state[1][1:]])

        # The reset_list here are all the resets for a single transition. Need to evaluate each of them
        # and then combine them together
        for reset_tuple in reset_list:
            reset, disc_var_dict, cont_var_dict, _ = reset_tuple
            reset_variable = reset.var
            expr = reset.expr
            # First get the transition destinations
            if "mode" in reset_variable:
                found = False
                for var_loc, discrete_variable_ego in enumerate(agent.controller.state_defs[ego_type].disc):
                    if discrete_variable_ego == reset_variable:
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f'Reset discrete variable {discrete_variable_ego} not found')
                if isinstance(reset.val_ast, ast.Constant):
                    val = eval(expr)
                    possible_dest[var_loc] = [val]
                else:
                    tmp = expr.split('.')
                    if 'map' in tmp[0]:
                        for var in disc_var_dict:
                            expr = expr.replace(var, f"'{disc_var_dict[var]}'")
                        res = eval(expr)
                        if not isinstance(res, list):
                            res = [res]
                        possible_dest[var_loc] = res
                    else:
                        expr = tmp
                        if expr[0].strip(' ') in agent.controller.mode_defs:
                            possible_dest[var_loc] = [expr[1]]

            # Assume linear function for continuous variables
            else:
                lhs = reset_variable
                rhs = expr
                found = False
                for lhs_idx, cts_variable in enumerate(agent.controller.state_defs[ego_type].cont):
                    if cts_variable == lhs:
                        found = True
                        break
                if not found:
                    raise ValueError(f'Reset continuous variable {cts_variable} not found')
                # substituting low variables

                symbols = []
                for var in cont_var_dict:
                    if var in expr:
                        symbols.append(var)

                # TODO: Implement this function
                # The input to this function is a list of used symbols and the cont_var_dict
                # The ouput of this function is a list of tuple of values for each variable in the symbols list
                # The function will explor all possible combinations of low bound and upper bound for the variables in the symbols list
                comb_list = self._get_combinations(symbols, cont_var_dict)

                lb = float('inf')
                ub = -float('inf')

                for comb in comb_list:
                    val_dict = {}
                    tmp = copy.deepcopy(expr)
                    for symbol_idx, symbol in enumerate(symbols):
                        tmp = tmp.replace(symbol, str(comb[symbol_idx]))
                    res = eval(tmp, {}, val_dict)
                    lb = min(lb, res)
                    ub = max(ub, res)

                rect[0][lhs_idx] = lb
                rect[1][lhs_idx] = ub

        all_dest = itertools.product(*possible_dest)
        dest = []
        for tmp in all_dest:
            dest.append(tmp)

        return dest, rect

    def _get_combinations(self, symbols, cont_var_dict):
        data_list = []
        for symbol in symbols:
            data_list.append(cont_var_dict[symbol])
        comb_list = list(itertools.product(*data_list))
        return comb_list

    def apply_cont_var_updater(self, cont_var_dict, updater):
        for variable in updater:
            for unrolled_variable, unrolled_variable_index in updater[variable]:
                cont_var_dict[unrolled_variable] = cont_var_dict[variable][unrolled_variable_index]

    # def apply_disc_var_updater(self,disc_var_dict, updater):
    #     for variable in updater:
    #         unrolled_variable, unrolled_variable_index = updater[variable]
    #         disc_var_dict[unrolled_variable] = disc_var_dict[variable][unrolled_variable_index]

    def get_transition_simulate_new(self, node: AnalysisTreeNode) -> Tuple[Optional[Dict[str, List[str]]], Dict[str, List[Tuple[float]]], float]:
        lane_map = self.map
        trace_length = len(list(node.trace.values())[0])

        # For each agent
        agent_guard_dict = defaultdict(list)

        for agent_id in node.agent:
            # Get guard
            agent: BaseAgent = self.agent_dict[agent_id]
            agent_mode = node.mode[agent_id]
            if len(agent.controller.args) == 0:
                continue
            state_dict = {}
            for tmp in node.agent:
                state_dict[tmp] = (node.trace[tmp][0], node.mode[tmp], node.static[tmp])
            cont_var_dict_template, discrete_variable_dict, len_dict = self.sensor.sense(self, agent, state_dict, self.map)
            paths = agent.controller.paths
            for path in paths:
                agent_guard_dict[agent_id].append((path.cond, discrete_variable_dict, path.var, path.val))

        transitions = defaultdict(list)
        # TODO: We can probably rewrite how guard hit are detected and resets are handled for simulation
        for idx in range(trace_length):
            satisfied_guard = []
            asserts = defaultdict(list)
            for agent_id in agent_guard_dict:
                agent: BaseAgent = self.agent_dict[agent_id]
                state_dict = {}
                for tmp in node.agent:
                    state_dict[tmp] = (node.trace[tmp][idx],
                                       node.mode[tmp], node.static[tmp])
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                agent_state = agent_state[1:]
                continuous_variable_dict, orig_disc_vars, _ = self.sensor.sense(
                    self, agent, state_dict, self.map)
                # Unsafety checking
                ego_ty_name = find(agent.controller.args, lambda a: a.name == EGO).typ
                def pack_env(agent: BaseAgent, cont, disc, map):
                    env = copy.deepcopy(cont)
                    env.update(disc)

                    state_ty = namedtuple(
                        ego_ty_name, agent.controller.state_defs[ego_ty_name].all_vars())
                    packed: DefaultDict[str, Any] = defaultdict(dict)
                    for k, v in env.items():
                        k = k.split(".")
                        packed[k[0]][k[1]] = v
                    for arg in agent.controller.args:
                        if arg.name != EGO and 'map' not in arg.name:
                            other = arg.name
                            if other in packed:
                                others_keys = list(packed[other].keys())
                                packed[other] = [state_ty(**{k: packed[other][k][i] for k in others_keys}) for i in range(len(packed[other][others_keys[0]]))]
                                if not arg.is_list:
                                    packed[other] = packed[other][0]
                            else:
                                if arg.is_list:
                                    packed[other] = []
                                else:
                                    raise ValueError(f"Expected one {ego_ty_name} for {other}, got none")

                    packed[EGO] = state_ty(**packed[EGO])
                    map_var = find(agent.controller.args, lambda a: "map" in a.name)
                    if map_var != None:
                        packed[map_var.name] = map
                    packed: Dict[str, Any] = dict(packed.items())
                    # packed.update(env)
                    return packed
                packed_env = pack_env(agent, continuous_variable_dict, orig_disc_vars, self.map)

                # Check safety conditions
                for assertion in agent.controller.asserts:
                    if eval(assertion.pre, packed_env):
                        if not eval(assertion.cond, packed_env):
                            del packed_env["__builtins__"]
                            print(f"assert hit for {agent_id}: \"{assertion.label}\" @ {packed_env}")
                            asserts[agent_id].append(assertion.label)
                if agent_id in asserts:
                    continue

                all_resets = defaultdict(list)
                for guard_comp, discrete_variable_dict, var, reset in agent_guard_dict[agent_id]:
                    new_cont_var_dict = copy.deepcopy(continuous_variable_dict)
                    env = pack_env(agent, new_cont_var_dict, discrete_variable_dict, self.map)

                    # Collect all the hit guards for this agent at this time step
                    if eval(guard_comp, env):
                        # If the guard can be satisfied, handle resets
                        all_resets[var].append(reset)

                iter_list = []
                for reset_var in all_resets:
                    iter_list.append(range(len(all_resets[reset_var])))
                pos_list = list(itertools.product(*iter_list))
                if len(pos_list) == 1 and pos_list[0] == ():
                    continue
                for i in range(len(pos_list)):
                    pos = pos_list[i]
                    next_init = copy.deepcopy(agent_state)
                    dest = copy.deepcopy(agent_mode)
                    possible_dest = [[elem] for elem in dest]
                    for j, reset_idx in enumerate(pos):
                        reset_variable = list(all_resets.keys())[j]
                        res = eval(all_resets[reset_variable][reset_idx], packed_env)
                        ego_type = agent.controller.state_defs[ego_ty_name]
                        if "mode" in reset_variable:
                            var_loc = ego_type.disc.index(reset_variable)
                            if not isinstance(res, list):
                                res = [res]
                            possible_dest[var_loc] = res
                        else:
                            var_loc = ego_type.cont.index(reset_variable)
                            next_init[var_loc] = res
                    all_dest = list(itertools.product(*possible_dest))
                    if not all_dest:
                        warnings.warn(
                            f"Guard hit for mode {agent_mode} for agent {agent_id} without available next mode")
                        all_dest.append(None)
                    for dest in all_dest:
                        satisfied_guard.append(
                            (agent_id, agent_mode, dest, next_init))
            if len(asserts) > 0:
                return asserts, transitions, idx
            if len(satisfied_guard) > 0:
                for agent_idx, src_mode, dest_mode, next_init in satisfied_guard:
                    transitions[agent_idx].append(
                        (agent_idx, src_mode, dest_mode, next_init, idx))
                break
        return None, transitions, idx

    def get_transition_verify_new(self, node:AnalysisTreeNode):
        lane_map = self.map

        agent_guard_dict = defaultdict(list)
        for agent_id in node.agent:
            agent:BaseAgent = self.agent_dict[agent_id]
            if len(agent.controller.args) == 0:
                continue
            agent_mode = node.mode[agent_id]
            state_dict = {}
            for tmp in node.agent:
                state_dict[tmp] = (node.trace[tmp][0*2:0*2+2], node.mode[tmp], node.static[tmp])

            cont_var_dict_template, discrete_variable_dict, length_dict = self.sensor.sense(self, agent, state_dict, self.map)
            # TODO-PARSER: Get equivalent for this function
            paths = agent.controller.paths
            for guard_idx, path in enumerate(paths):
                # Construct the guard expression
                reset = (path.var, path.val_veri)
                guard_expression = GuardExpressionAst([path.cond_veri], guard_idx)

                cont_var_updater = guard_expression.parse_any_all_new(cont_var_dict_template, discrete_variable_dict, length_dict)
                self.apply_cont_var_updater(cont_var_dict_template, cont_var_updater)
                guard_can_satisfied = guard_expression.evaluate_guard_disc(agent, discrete_variable_dict, cont_var_dict_template, self.map)
                if not guard_can_satisfied:
                    continue
                agent_guard_dict[agent_id].append((guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), reset))

        trace_length = int(len(list(node.trace.values())[0])/2)
        guard_hits = []
        guard_hit = False
        for idx in range(0,trace_length):
            any_contained = False
            hits = []
            state_dict = {}
            for tmp in node.agent:
                state_dict[tmp] = (node.trace[tmp][idx*2:idx*2+2], node.mode[tmp], node.static[tmp])

            asserts = defaultdict(list)
            for agent_id in self.agent_dict.keys():
                agent:BaseAgent = self.agent_dict[agent_id]
                if len(agent.controller.args) == 0:
                    continue
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                agent_state = agent_state[1:]
                cont_vars, disc_vars, len_dict = self.sensor.sense(self, agent, state_dict, self.map)
                resets = defaultdict(list)
                # Check safety conditions
                for i, a in enumerate(agent.controller.asserts_veri):
                    pre_expr = a.pre
                    def eval_expr(expr):
                        ge = GuardExpressionAst([copy.deepcopy(expr)])
                        cont_var_updater = ge.parse_any_all_new(cont_vars, disc_vars, len_dict)
                        self.apply_cont_var_updater(cont_vars, cont_var_updater)
                        sat = ge.evaluate_guard_disc(agent, disc_vars, cont_vars, self.map)
                        if sat:
                            sat = ge.evaluate_guard_hybrid(agent, disc_vars, cont_vars, self.map)
                            if sat:
                                sat, contained = ge.evaluate_guard_cont(agent, cont_vars, self.map)
                                sat = sat and contained
                        return sat
                    if eval_expr(pre_expr):
                        if not eval_expr(a.cond):
                            label = a.label if a.label != None else f"<assert {i}>"
                            print(f"assert hit for {agent_id}: \"{label}\"")
                            print(idx)
                            asserts[agent_id].append(label)
                if agent_id in asserts:
                    continue
                if agent_id not in agent_guard_dict:
                    continue

                for guard_expression, continuous_variable_updater, discrete_variable_dict, reset in agent_guard_dict[agent_id]:
                    new_cont_var_dict = copy.deepcopy(cont_vars)
                    one_step_guard:GuardExpressionAst = copy.deepcopy(guard_expression)

                    self.apply_cont_var_updater(
                        new_cont_var_dict, continuous_variable_updater)
                    guard_can_satisfied = one_step_guard.evaluate_guard_hybrid(
                        agent, discrete_variable_dict, new_cont_var_dict, self.map)
                    if not guard_can_satisfied:
                        continue
                    guard_satisfied, is_contained = one_step_guard.evaluate_guard_cont(
                        agent, new_cont_var_dict, self.map)
                    any_contained = any_contained or is_contained
                    # TODO: Can we also store the cont and disc var dict so we don't have to call sensor again?
                    if guard_satisfied:
                        reset_expr = ResetExpression(reset)
                        resets[reset_expr.var].append(
                            (reset_expr, discrete_variable_dict, new_cont_var_dict, guard_expression.guard_idx)
                        )
                # Perform combination over all possible resets to generate all possible real resets
                combined_reset_list = list(itertools.product(*resets.values()))
                if len(combined_reset_list) == 1 and combined_reset_list[0] == ():
                    continue
                for i in range(len(combined_reset_list)):
                    # Compute reset_idx 
                    reset_idx = []
                    for reset_info in combined_reset_list[i]:
                        reset_idx.append(reset_info[3])
                    # a list of reset expression
                    hits.append((agent_id, tuple(reset_idx), 
                                combined_reset_list[i]))
            if len(asserts) > 0:
                return (asserts, idx), None
            if hits != []:
                guard_hits.append((hits, state_dict, idx))
                guard_hit = True
            elif guard_hit:
                break
            if any_contained:
                break

        reset_dict = {}#defaultdict(lambda: defaultdict(list))
        reset_idx_dict = {}#defaultdict(lambda: defaultdict(list))
        for hits, all_agent_state, hit_idx in guard_hits:
            for agent_id, reset_idx, reset_list in hits:
                # TODO: Need to change this function to handle the new reset expression and then I am done
                dest_list,reset_rect = self.apply_reset(node.agent[agent_id], reset_list, all_agent_state)
                if agent_id not in reset_dict:
                    reset_dict[agent_id] = {}
                    reset_idx_dict[agent_id] = {}
                if not dest_list:
                    warnings.warn(
                        f"Guard hit for mode {node.mode[agent_id]} for agent {agent_id} without available next mode")
                    dest_list.append(None)
                if reset_idx not in reset_dict[agent_id]:
                    reset_dict[agent_id][reset_idx] = {}
                    reset_idx_dict[agent_id][reset_idx] = {}
                for dest in dest_list:
                    if dest not in reset_dict[agent_id][reset_idx]:
                        reset_dict[agent_id][reset_idx][dest] = []
                        reset_idx_dict[agent_id][reset_idx][dest] = []
                    reset_dict[agent_id][reset_idx][dest].append(reset_rect)
                    reset_idx_dict[agent_id][reset_idx][dest].append(hit_idx)

        possible_transitions = []
        # Combine reset rects and construct transitions
        for agent in reset_dict:
            for reset_idx in reset_dict[agent]:
                for dest in reset_dict[agent][reset_idx]:
                    transition = (
                        agent, node.mode[agent], dest, reset_dict[agent][reset_idx][dest], reset_idx_dict[agent][reset_idx][dest])
                    possible_transitions.append(transition)
        # Return result
        return None, possible_transitions
