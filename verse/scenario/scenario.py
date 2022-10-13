from pprint import pp
from typing import DefaultDict, NamedTuple, Optional, Tuple, List, Dict, Any
import copy
import itertools
import warnings
from collections import defaultdict, namedtuple
import ast
from dataclasses import dataclass

import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.analysis.dryvr import _EPSILON
from verse.analysis.incremental import CachedRTTrans, CachedSegment, combine_all, reach_trans_suit, sim_trans_suit
from verse.analysis.simulator import PathDiffs
from verse.automaton import GuardExpressionAst, ResetExpression
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree
from verse.analysis.utils import dedup, sample_rect
from verse.parser import astunparser
from verse.parser.parser import ControllerIR, ModePath, find
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

EGO, OTHERS = "ego", "others"

def red(s):
    return "\x1b[31m" + s + "\x1b[0m"

def pack_env(agent: BaseAgent, ego_ty_name: str, cont: Dict[str, float], disc: Dict[str, str], lane_map):
    state_ty = None #namedtuple(ego_ty_name, agent.controller.state_defs[ego_ty_name].all_vars())
    packed: DefaultDict[str, Any] = defaultdict(dict)
    # packed = {}
    for e in [cont, disc]:
        for k, v in e.items():
            k1, k2 = k.split(".")
            packed[k1][k2] = v
    for arg in agent.controller.args:
        if "map" in arg.name:
            packed[arg.name] = lane_map
        elif arg.name != EGO:
            other = arg.name
            if other in packed:
                other_keys, other_vals = tuple(map(list, zip(*packed[other].items())))
                state_ty = namedtuple(ego_ty_name, other_keys)
                packed[other] = list(map(lambda v: state_ty(*v), zip(*other_vals)))
                if not arg.is_list:
                    packed[other] = packed[other][0]
            else:
                if arg.is_list:
                    packed[other] = []
                else:
                    raise ValueError(f"Expected one {ego_ty_name} for {other}, got none")

    packed[EGO] = state_ty(**packed[EGO])
    return dict(packed.items())

def check_sim_transitions(agent: BaseAgent, guards: List[Tuple], cont, disc, map, state, mode):
    asserts = []
    satisfied_guard = []
    agent_id = agent.id
    # Unsafety checking
    ego_ty_name = find(agent.controller.args, lambda a: a.name == EGO).typ
    packed_env = pack_env(agent, ego_ty_name, cont, disc, map)

    # Check safety conditions
    for assertion in agent.controller.asserts:
        if eval(assertion.pre, packed_env):
            if not eval(assertion.cond, packed_env):
                del packed_env["__builtins__"]
                print(f"assert hit for {agent_id}: \"{assertion.label}\" @ {packed_env}")
                asserts.append(assertion.label)
    if len(asserts) != 0:
        return asserts, satisfied_guard

    all_resets = defaultdict(list)
    for path, disc_vars in guards:
        env = pack_env(agent, ego_ty_name, cont, disc, map)    # TODO: diff disc -> disc_vars?

        # Collect all the hit guards for this agent at this time step
        if eval(path.cond, env):
            # If the guard can be satisfied, handle resets
            all_resets[path.var].append((path.val, path))

    iter_list = []
    for vals in all_resets.values():
        paths = [p for _, p in vals]
        iter_list.append(zip(range(len(vals)), paths))
    pos_list = list(itertools.product(*iter_list))
    if len(pos_list) == 1 and pos_list[0] == ():
        return None, satisfied_guard
    for pos in pos_list:
        next_init = copy.deepcopy(state)
        dest = copy.deepcopy(mode)
        possible_dest = [[elem] for elem in dest]
        for j, (reset_idx, path) in enumerate(pos):
            reset_variable = list(all_resets.keys())[j]
            res = eval(all_resets[reset_variable][reset_idx][0], packed_env)
            ego_type = agent.controller.state_defs[ego_ty_name]
            if "mode" in reset_variable:
                var_loc = ego_type.disc.index(reset_variable)
                assert not isinstance(res, list), res
                possible_dest[var_loc] = [(res, path)]
            else:
                var_loc = ego_type.cont.index(reset_variable)
                next_init[var_loc] = res
        all_dest = list(itertools.product(*possible_dest))
        if not all_dest:
            warnings.warn(
                f"Guard hit for mode {mode} for agent {agent_id} without available next mode")
            all_dest.append(None)
        for dest in all_dest:
            assert isinstance(dest, tuple)
            paths = []
            pure_dest = []
            for d in dest:
                if isinstance(d, tuple):
                    pure_dest.append(d[0])
                    paths.append(d[1])
                else:
                    pure_dest.append(d)
            satisfied_guard.append((agent_id, pure_dest, next_init, paths))
    return None, satisfied_guard

@dataclass
class ScenarioConfig:
    incremental: bool = True
    unsafe_continue: bool = False
    init_seg_length: int = 1000
    reachability_method: str = 'DRYVR'

class Scenario:
    def __init__(self, config=ScenarioConfig()):
        self.agent_dict: Dict[str, BaseAgent] = {}
        self.simulator = Simulator(config)
        self.verifier = Verifier(config)
        self.init_dict = {}
        self.init_mode_dict = {}
        self.static_dict = {}
        self.uncertain_param_dict = {}
        self.map = LaneMap()
        self.sensor = BaseSensor()
        self.past_runs = []

        # Parameters
        self.config = config

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
        if agent.init_cont is not None:
            self.init_dict[agent.id] = copy.deepcopy(agent.init_cont) 
        if agent.init_disc is not None:
            self.init_mode_dict[agent.id] = copy.deepcopy(agent.init_disc)

        if agent.static_parameters is not None:
            self.static_dict[agent.id] = copy.deepcopy(agent.static_parameters)
        else:
            self.static_dict[agent.id] = []
        if agent.uncertain_parameters is not None:
            self.uncertain_param_dict[agent.id] = copy.deepcopy(agent.uncertain_parameters)
        else:
            self.uncertain_param_dict[agent.id] = []

    # TODO-PARSER: update this function
    def update_agent_lane_mode(self, agent: BaseAgent, lane_map: LaneMap):
        for lane_id in lane_map.lane_dict:
            if 'LaneMode' in agent.controller.mode_defs and lane_id not in agent.controller.mode_defs['LaneMode'].modes:
                agent.controller.mode_defs['LaneMode'].modes.append(lane_id)
        # mode_vals = list(agent.controller.modes.values())
        # agent.controller.vertices = list(itertools.product(*mode_vals))
        # agent.controller.vertexStrings = [','.join(elem) for elem in agent.controller.vertices]

    def set_init_single(self, agent_id, init: list, init_mode: tuple, static=[], uncertain_param=[]):
        assert agent_id in self.agent_dict, 'agent_id not found'
        agent = self.agent_dict[agent_id]
        assert len(init) == 1 or len(
            init) == 2, 'the length of init should be 1 or 2'
        # print(agent.controller.state_defs.values())
        if agent.controller != agent.controller.empty():
            for i in init:
                assert len(i) == len(
                    list(agent.controller.state_defs.values())[0].cont),  'the length of element in init not fit the number of continuous variables'
            # print(agent.controller.mode_defs)
            assert len(init_mode) == len(
                list(agent.controller.state_defs.values())[0].disc),  'the length of element in init_mode not fit the number of discrete variables'
        if len(init) == 1:
            init = init+init
        self.init_dict[agent_id] = copy.deepcopy(init)
        self.init_mode_dict[agent_id] = copy.deepcopy(init_mode)
        self.agent_dict[agent_id].set_initial(init, init_mode)
        if static:
            self.static_dict[agent_id] = copy.deepcopy(static)
            self.agent_dict[agent_id].set_static_parameter(static)
        else:
            self.static_dict[agent_id] = []
        if uncertain_param:
            self.uncertain_param_dict[agent_id] = copy.deepcopy(
                uncertain_param)
            self.agent_dict[agent_id].set_uncertain_parameter(uncertain_param)
        else:
            self.uncertain_param_dict[agent_id] = []
        return

    def set_init(self, init_list, init_mode_list, static_list=[], uncertain_param_list=[]):
        assert len(init_list) == len(
            self.agent_dict), 'the length of init_list not fit the number of agents'
        assert len(init_mode_list) == len(
            self.agent_dict), 'the length of init_mode_list not fit the number of agents'
        assert len(static_list) == len(
            self.agent_dict) or len(static_list) == 0, 'the length of static_list not fit the number of agents or equal to 0'
        assert len(uncertain_param_list) == len(self.agent_dict)\
            or len(uncertain_param_list) == 0, 'the length of uncertain_param_list not fit the number of agents or equal to 0'
        print(init_mode_list)
        print(type(init_mode_list))
        if not static_list:
            static_list = [[] for i in range(0, len(self.agent_dict))]
            # print(static_list)
        if not uncertain_param_list:
            uncertain_param_list = [[] for i in range(0, len(self.agent_dict))]
            # print(uncertain_param_list)
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.set_init_single(agent_id, init_list[i],
                                 init_mode_list[i], static_list[i], uncertain_param_list[i])

    def check_init(self):
        for agent_id in self.agent_dict.keys():
            assert agent_id in self.init_dict, 'init of {} not initialized'.format(
                agent_id)
            assert agent_id in self.init_mode_dict, 'init_mode of {} not initialized'.format(
                agent_id)
            assert agent_id in self.static_dict, 'static of {} not initialized'.format(
                agent_id)
            assert agent_id in self.uncertain_param_dict, 'uncertain_param of {} not initialized'.format(
                agent_id)
        return

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id]))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        tree = self.simulator.simulate(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, self, time_horizon, time_step, self.map, len(self.past_runs), self.past_runs)
        self.past_runs.append(tree)
        return tree

    def verify(self, time_horizon, time_step, params={}) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init = self.init_dict[agent_id]
            tmp = np.array(init)
            if tmp.ndim < 2:
                init = [init, init]
            init_list.append(init)
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        tree = self.verifier.compute_full_reachtube(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, self, time_horizon,
                                                    time_step, self.map, self.config.init_seg_length, self.config.reachability_method, len(self.past_runs), self.past_runs, params)
        self.past_runs.append(tree)
        return tree

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
            reset, disc_var_dict, cont_var_dict, _, _p = reset_tuple
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
                    raise ValueError(
                        f'Reset continuous variable {cts_variable} not found')
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

    def get_transition_simulate_new(self, cache: Dict[str, CachedSegment], paths: PathDiffs, node: AnalysisTreeNode) -> Tuple[Optional[Dict[str, List[str]]], Optional[Dict[str, List[Tuple[str, List[str], List[float]]]]], int]:
        lane_map = self.map
        trace_length = len(list(node.trace.values())[0])

        # For each agent
        agent_guard_dict = defaultdict(list)
        cached_guards = defaultdict(list)
        min_trans_ind = None
        cached_trans = defaultdict(list)

        if not cache:
            paths = [(agent, p) for agent in node.agent.values() for p in agent.controller.paths]
        else:
            _transitions = [trans.transition for seg in cache.values() for trans in seg.transitions if sim_trans_suit(trans.inits, node.init)]
            # pp(("cached trans", _transitions))
            if len(_transitions) == 0:
                return None, None, 0
            min_trans_ind = min(_transitions)
            for agent_id, seg in cache.items():
                # TODO: check for asserts
                for tran in seg.transitions:
                    if tran.transition == min_trans_ind:
                        # pp(("chosen tran", agent_id, tran))
                        cached_trans[agent_id].append((agent_id, tran.disc, tran.cont, tran.paths))
            for agent_id in cached_trans:
                cached_trans[agent_id] = dedup(cached_trans[agent_id], lambda p: p[:3])
            if len(paths) == 0:
                # print(red("full cache"))
                return None, dict(cached_trans), min_trans_ind

            path_transitions = defaultdict(int)
            for seg in cache.values():
                for tran in seg.transitions:
                    for p in tran.paths:
                        path_transitions[p.cond] = max(path_transitions[p.cond], tran.transition)
            for agent_id, segment in cache.items():
                agent = node.agent[agent_id]
                if len(agent.controller.args) == 0:
                    continue
                state_dict = {aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent}
                agent_paths = dedup([p for tran in segment.transitions for p in tran.paths], lambda i: (i.var, i.cond, i.val))
                cont_var_dict_template, discrete_variable_dict, len_dict = self.sensor.sense(self, agent, state_dict, self.map)
                for path in agent_paths:
                    cached_guards[agent_id].append((path, discrete_variable_dict, path_transitions[path.cond]))

        for agent, path in paths:
            # Get guard
            if len(agent.controller.args) == 0:
                continue
            agent_id = agent.id
            agent_mode = node.mode[agent_id]
            state_dict = {aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent}
            cont_var_dict_template, discrete_variable_dict, len_dict = self.sensor.sense(self, agent, state_dict, self.map)
            agent_guard_dict[agent_id].append((path, discrete_variable_dict))

        transitions = defaultdict(list)
        # TODO: We can probably rewrite how guard hit are detected and resets are handled for simulation
        for idx in range(trace_length):
            if min_trans_ind != None and idx >= min_trans_ind:
                return None, dict(cached_trans), min_trans_ind
            satisfied_guard = []
            all_asserts = defaultdict(list)
            for agent_id in agent_guard_dict:
                agent: BaseAgent = self.agent_dict[agent_id]
                state_dict = {aid: (node.trace[aid][idx], node.mode[aid], node.static[aid]) for aid in node.agent}
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                agent_state = agent_state[1:]
                continuous_variable_dict, orig_disc_vars, _ = self.sensor.sense(self, agent, state_dict, self.map)
                unchecked_cache_guards = [g[:2] for g in cached_guards[agent_id] if g[2] < idx]     # FIXME: off by 1?
                asserts, satisfied = check_sim_transitions(agent, agent_guard_dict[agent_id] + unchecked_cache_guards, continuous_variable_dict, orig_disc_vars, self.map, agent_state, agent_mode)
                if asserts != None:
                    all_asserts[agent_id] = asserts
                    continue
                if len(satisfied) != 0:
                    satisfied_guard.extend(satisfied)
                    assert all(len(s[2]) == 4 for s in satisfied)
            if len(all_asserts) > 0:
                return all_asserts, dict(transitions), idx
            if len(satisfied_guard) > 0:
                for agent_idx, dest_mode, next_init, paths in satisfied_guard:
                    assert isinstance(paths, list)
                    transitions[agent_idx].append((agent_idx, dest_mode, next_init, paths))
                # print("transitions", transitions)
                break
        return None, dict(transitions), idx

    def get_transition_verify_new(self, cache: Dict[str, CachedRTTrans], paths: PathDiffs, node: AnalysisTreeNode) -> Tuple[Optional[Dict[str, List[str]]], Optional[Dict[str, List[Tuple[str, List[str], List[float]]]]]]:
        lane_map = self.map

        # For each agent
        agent_guard_dict = defaultdict(list)
        cached_guards = defaultdict(list)
        min_trans_ind = None
        cached_trans = defaultdict(list)

        if not cache:
            paths = [(agent, p) for agent in node.agent.values() for p in agent.controller.paths]
        else:

            # _transitions = [trans.transition for seg in cache.values() for trans in seg.transitions]
            _transitions = [trans.transition for seg in cache.values() for trans in seg.transitions if reach_trans_suit(trans.inits, node.init)]
            # pp(("cached trans", len(_transitions)))
            if len(_transitions) == 0:
                return None, []
            min_trans_ind = min(_transitions)
            cached_trans = [(agent_id, tran) for agent_id, seg in cache.items() for tran in seg.transitions if tran.transition == min_trans_ind]
            # TODO: check for asserts
            cached_trans = [(aid, tran.mode, tran.dest, tran.reset, tran.reset_idx, tran.paths) for aid, tran in dedup(cached_trans, lambda p: (p[0], p[1].mode, p[1].dest))]
            if len(paths) == 0:
                # print(red("full cache"))
                return None, cached_trans

            path_transitions = defaultdict(int)
            for seg in cache.values():
                for tran in seg.transitions:
                    for p in tran.paths:
                        path_transitions[p.cond] = max(path_transitions[p.cond], tran.transition)
            for agent_id, segment in cache.items():
                agent = node.agent[agent_id]
                if len(agent.controller.args) == 0:
                    continue
                state_dict = {aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent}

                agent_paths = dedup([p for tran in segment.transitions for p in tran.paths], lambda i: (i.var, i.cond, i.val))
                for path in agent_paths:
                    cont_var_dict_template, discrete_variable_dict, length_dict = self.sensor.sense(
                        self, agent, state_dict, self.map)
                    reset = (path.var, path.val_veri)
                    guard_expression = GuardExpressionAst([path.cond_veri])

                    cont_var_updater = guard_expression.parse_any_all_new(
                        cont_var_dict_template, discrete_variable_dict, length_dict)
                    self.apply_cont_var_updater(
                        cont_var_dict_template, cont_var_updater)
                    guard_can_satisfied = guard_expression.evaluate_guard_disc(
                        agent, discrete_variable_dict, cont_var_dict_template, self.map)
                    if not guard_can_satisfied:
                        continue
                    cached_guards[agent_id].append((path, guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), reset, path_transitions[path.cond]))

        # for aid, trace in node.trace.items():
        #     if len(trace) < 2:
        #         pp(("weird state", aid, trace))
        for agent, path in paths:
            if len(agent.controller.args) == 0:
                continue
            agent_id = agent.id
            state_dict = {aid: (node.trace[aid][0:2], node.mode[aid], node.static[aid]) for aid in node.agent}
            cont_var_dict_template, discrete_variable_dict, length_dict = self.sensor.sense(
                self, agent, state_dict, self.map)
            # TODO-PARSER: Get equivalent for this function
            # Construct the guard expression
            guard_expression = GuardExpressionAst([path.cond_veri])

            cont_var_updater = guard_expression.parse_any_all_new(
                cont_var_dict_template, discrete_variable_dict, length_dict)
            self.apply_cont_var_updater(
                cont_var_dict_template, cont_var_updater)
            guard_can_satisfied = guard_expression.evaluate_guard_disc(
                agent, discrete_variable_dict, cont_var_dict_template, self.map)
            if not guard_can_satisfied:
                continue
            agent_guard_dict[agent_id].append(
                (guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), path))

        trace_length = int(min(len(v) for v in node.trace.values()) // 2)
        # pp(("trace len", trace_length, {a: len(t) for a, t in node.trace.items()}))
        guard_hits = []
        guard_hit = False
        for idx in range(trace_length):
            if min_trans_ind != None and idx >= min_trans_ind:
                return None, cached_trans
            any_contained = False
            hits = []
            state_dict = {aid: (node.trace[aid][idx*2:idx*2+2], node.mode[aid], node.static[aid]) for aid in node.agent}

            asserts = defaultdict(list)
            for agent_id in self.agent_dict.keys():
                agent: BaseAgent = self.agent_dict[agent_id]
                if len(agent.controller.args) == 0:
                    continue
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                # if np.array(agent_state).ndim != 2:
                #     pp(("weird state", agent_id, agent_state))
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

                unchecked_cache_guards = [g[:-1] for g in cached_guards[agent_id] if g[-1] < idx]     # FIXME: off by 1?
                for guard_expression, continuous_variable_updater, discrete_variable_dict, path in agent_guard_dict[agent_id] + unchecked_cache_guards:
                    assert isinstance(path, ModePath)
                    new_cont_var_dict = copy.deepcopy(cont_vars)
                    one_step_guard: GuardExpressionAst = copy.deepcopy(guard_expression)

                    self.apply_cont_var_updater(new_cont_var_dict, continuous_variable_updater)
                    guard_can_satisfied = one_step_guard.evaluate_guard_hybrid(
                        agent, discrete_variable_dict, new_cont_var_dict, self.map)
                    if not guard_can_satisfied:
                        continue
                    guard_satisfied, is_contained = one_step_guard.evaluate_guard_cont(
                        agent, new_cont_var_dict, self.map)
                    any_contained = any_contained or is_contained
                    # TODO: Can we also store the cont and disc var dict so we don't have to call sensor again?
                    if guard_satisfied:
                        reset_expr = ResetExpression((path.var, path.val_veri))
                        resets[reset_expr.var].append(
                            (reset_expr, discrete_variable_dict,
                             new_cont_var_dict, guard_expression.guard_idx, path)
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
                    hits.append((agent_id, tuple(reset_idx), combined_reset_list[i]))
            if len(asserts) > 0:
                return (asserts, idx), None
            if hits != []:
                guard_hits.append((hits, state_dict, idx))
                guard_hit = True
            elif guard_hit:
                break
            if any_contained:
                break

        reset_dict = {}  # defaultdict(lambda: defaultdict(list))
        for hits, all_agent_state, hit_idx in guard_hits:
            for agent_id, reset_idx, reset_list in hits:
                # TODO: Need to change this function to handle the new reset expression and then I am done
                dest_list, reset_rect = self.apply_reset(node.agent[agent_id], reset_list, all_agent_state)
                # pp(("dests", dest_list, *[astunparser.unparse(reset[-1].val_veri) for reset in reset_list]))
                if agent_id not in reset_dict:
                    reset_dict[agent_id] = {}
                if not dest_list:
                    warnings.warn(
                        f"Guard hit for mode {node.mode[agent_id]} for agent {agent_id} without available next mode")
                    dest_list.append(None)
                if reset_idx not in reset_dict[agent_id]:
                    reset_dict[agent_id][reset_idx] = {}
                for dest in dest_list:
                    if dest not in reset_dict[agent_id][reset_idx]:
                        reset_dict[agent_id][reset_idx][dest] = []
                    reset_dict[agent_id][reset_idx][dest].append((reset_rect, hit_idx, reset_list[-1]))

        possible_transitions = []
        # Combine reset rects and construct transitions
        for agent in reset_dict:
            for reset_idx in reset_dict[agent]:
                for dest in reset_dict[agent][reset_idx]:
                    # reset_data = reset_dict[agent][reset_idx][dest]
                    reset_data = tuple(map(list, zip(*reset_dict[agent][reset_idx][dest])))
                    paths = [r[-1] for r in reset_data[-1]]
                    transition = (agent, node.mode[agent],dest, *reset_data[:-1], paths)
                    possible_transitions.append(transition)
        # Return result
        return None, possible_transitions
