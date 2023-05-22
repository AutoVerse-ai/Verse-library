from dataclasses import dataclass
import pickle
import timeit
from typing import Dict, List, Optional, Tuple
import copy, itertools, functools, pprint
from pympler.asizeof import asizeof
from collections import defaultdict
from typing import DefaultDict, Optional, Tuple, List, Dict, Any
from types import SimpleNamespace
import warnings
import types
import sys
from enum import Enum

from verse.agents.base_agent import BaseAgent
from verse.analysis.incremental import CachedSegment, SimTraceCache, convert_sim_trans, to_simulate
from verse.analysis.utils import dedup
from verse.map.lane_map import LaneMap
from verse.parser.parser import ModePath, find, unparse
from verse.analysis.incremental import (
    CachedRTTrans,
    CachedSegment,
    combine_all,
    reach_trans_suit,
    sim_trans_suit,
)

pp = functools.partial(pprint.pprint, compact=True, width=130)

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree, TraceType

PathDiffs = List[Tuple[BaseAgent, ModePath]]

EGO, OTHERS = "ego", "others"


def red(s):
    return "\x1b[31m" + s + "\x1b[0m"  # ]]


def pack_env(
    agent: BaseAgent, ego_ty_name: str, cont: Dict[str, float], disc: Dict[str, str], track_map
) -> Dict[str, Any]:
    packed: DefaultDict[str, Any] = defaultdict(dict)
    # packed = {}
    for e in [cont, disc]:
        for k, v in e.items():
            k1, k2 = k.split(".")
            packed[k1][k2] = v
    env: Dict[str, Any] = {EGO: SimpleNamespace(**packed[EGO])}
    for arg in agent.decision_logic.args:
        if "map" in arg.name:
            env[arg.name] = track_map
        elif arg.name != EGO:
            other = arg.name
            if other in packed:
                other_keys, other_vals = tuple(map(list, zip(*packed[other].items())))
                env[other] = list(
                    map(
                        lambda v: SimpleNamespace(**{k: v for k, v in zip(other_keys, v)}),
                        zip(*other_vals),
                    )
                )
                if not arg.is_list:
                    env[other] = packed[other][0]
            else:
                if arg.is_list:
                    env[other] = []
                else:
                    raise ValueError(f"Expected one {ego_ty_name} for {other}, got none")

    return env


def check_sim_transitions(agent: BaseAgent, guards: List[Tuple], cont, disc, map, state, mode):
    asserts = []
    satisfied_guard = []
    agent_id = agent.id
    # Unsafety checking
    ego_ty_name = find(agent.decision_logic.args, lambda a: a.name == EGO).typ
    packed_env = pack_env(agent, ego_ty_name, cont, disc, map)

    # Check safety conditions
    for assertion in agent.decision_logic.asserts:
        if eval(assertion.pre, packed_env):
            if not eval(assertion.cond, packed_env):
                del packed_env["__builtins__"]
                print(f'assert hit for {agent_id}: "{assertion.label}" @ {packed_env}')
                asserts.append(assertion.label)
    if len(asserts) != 0:
        return asserts, satisfied_guard

    all_resets = defaultdict(list)
    env = pack_env(agent, ego_ty_name, cont, disc, map)  # TODO: diff disc -> disc_vars?
    for path, disc_vars in guards:
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
            ego_type = agent.decision_logic.state_defs[ego_ty_name]
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
                f"Guard hit for mode {mode} for agent {agent_id} without available next mode"
            )
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


def convertStrToEnum(inp, agent: BaseAgent, dl):
    res = inp
    for field in res.__dict__:
        for state_def_name in agent.decision_logic.state_defs:
            if field in agent.decision_logic.state_defs[state_def_name].disc:
                idx = agent.decision_logic.state_defs[state_def_name].disc.index(field)
                field_type = agent.decision_logic.state_defs[state_def_name].disc_type[idx]
                enum_class = getattr(dl, field_type)
                setattr(res, field, enum_class[getattr(res, field)])
    return res


def convertEnumToStr(inp, agent: BaseAgent, dl):
    res = inp
    for field in res.__dict__:
        val = getattr(res, field)
        if isinstance(val, Enum):
            setattr(res, field, val.name)
    return res


def disc_field(field: str, agent: BaseAgent):
    for state_def_name in agent.decision_logic.state_defs:
        state_def = agent.decision_logic.state_defs[state_def_name]
        if field in state_def.disc:
            return True
    return False


@dataclass
class SimConsts:
    time_step: float
    lane_map: LaneMap
    run_num: int
    past_runs: List[AnalysisTree]
    sensor: "BaseSensor"
    agent_dict: Dict


class Simulator:
    def __init__(self, config):
        self.simulation_tree = None
        self.cache = SimTraceCache()
        self.config = config
        self.cache_hits = (0, 0)
        if self.config.parallel:
            import ray

            self.simulate_one_remote = ray.remote(Simulator.simulate_one)

    @staticmethod
    def simulate_one(
        config: "ScenarioConfig",
        cached_segments: Dict[str, CachedSegment],
        node: AnalysisTreeNode,
        old_node_id: Optional[Tuple[int, int]],
        later: int,
        remain_time: float,
        consts: SimConsts,
    ) -> Tuple[int, int, List[AnalysisTreeNode], Dict[str, TraceType], list]:
        print(f"node {node.id} start: {node.start_time}")
        # print(f"node id: {node.id}")
        cache_updates = []
        for agent_id in node.agent:
            if agent_id not in node.trace:
                if agent_id in cached_segments:
                    node.trace[agent_id] = cached_segments[agent_id].trace
                else:
                    # pp(("sim", agent_id, *mode, *init))
                    # Simulate the trace starting from initial condition
                    mode = node.mode[agent_id]
                    init = node.init[agent_id]
                    trace = node.agent[agent_id].TC_simulate(
                        mode, init, remain_time, consts.time_step, consts.lane_map
                    )
                    trace[:, 0] += node.start_time
                    node.trace[agent_id] = trace
        # pp(("cached_segments", cached_segments.keys()))
        # TODO: for now, make sure all the segments comes from the same node; maybe we can do
        # something to combine results from different nodes in the future
        new_cache, paths_to_sim = {}, []
        if old_node_id != None:
            if old_node_id[0] != consts.run_num:
                old_node = find(
                    consts.past_runs[old_node_id[0]].nodes, lambda n: n.id == old_node_id[1]
                )
                assert old_node != None
                new_cache, paths_to_sim = to_simulate(old_node.agent, node.agent, cached_segments)

        asserts, transitions, transition_idx = Simulator.get_transition_simulate(
            new_cache, paths_to_sim, node, consts.lane_map, consts.sensor, consts.agent_dict
        )
        # pp(("transitions:", transition_idx, transitions))

        node.assert_hits = asserts
        # pp(("next init:", {a: trace[transition_idx] for a, trace in node.trace.items()}))

        # truncate the computed trajectories from idx and store the content after truncate
        truncated_trace: Dict[str, TraceType] = {}
        full_traces: Dict[str, TraceType] = {}
        for agent_idx in node.agent:
            full_traces[agent_idx] = node.trace[agent_idx]
            if transitions:
                truncated_trace[agent_idx] = node.trace[agent_idx][transition_idx:]
                node.trace[agent_idx] = node.trace[agent_idx][: transition_idx + 1]

        if asserts != None:  # FIXME
            return (node.id, later, [], node.trace, cache_updates)
            # print(transition_idx)
            # pp({a: len(t) for a, t in node.trace.items()})
        else:
            # If there's no transitions (returned transitions is empty), continue
            if not transitions:
                if config.incremental:
                    for agent_id in node.agent:
                        cache_updates.append(
                            (
                                agent_id not in cached_segments,
                                agent_id,
                                [],
                                full_traces[agent_id],
                                [],
                                transition_idx,
                                consts.run_num,
                            )
                        )
                # print(red("no trans"))
                # print(f"node {node.id} dur {timeit.default_timer() - t}")
                return (node.id, later, [], node.trace, cache_updates)

            transit_agents = transitions.keys()
            # pp(("transit agents", transit_agents))
            if config.incremental:
                for agent_id in node.agent:
                    transition = transitions[agent_id] if agent_id in transit_agents else []
                    # TODO: update current transitions
                    cache_updates.append(
                        (
                            agent_id not in cached_segments,
                            agent_id,
                            transit_agents,
                            full_traces[agent_id],
                            transition,
                            transition_idx,
                            consts.run_num,
                        )
                    )
            # pp(("cached inits", self.cache.get_cached_inits(3)))
            # Generate the transition combinations if multiple agents can transit at the same time step
            transition_list = list(transitions.values())
            all_transition_combinations = itertools.product(*transition_list)

            # For each possible transition, construct the new node.
            # Obtain the new initial condition for agent having transition
            # copy the traces that are not under transition
            all_transition_paths = []
            next_nodes = []
            for transition_combination in all_transition_combinations:
                transition_paths = []
                next_node_mode = copy.deepcopy(node.mode)
                next_node_agent = node.agent
                next_node_start_time = list(truncated_trace.values())[0][0][0]
                next_node_init = {}
                next_node_trace = {}
                for transition in transition_combination:
                    transit_agent_idx, dest_mode, next_init, paths = transition
                    if dest_mode is None:
                        continue
                    transition_paths.extend(paths)
                    next_node_mode[transit_agent_idx] = dest_mode
                    next_node_init[transit_agent_idx] = next_init
                for agent_idx in next_node_agent:
                    if agent_idx not in next_node_init:
                        next_node_trace[agent_idx] = truncated_trace[agent_idx]
                        next_node_init[agent_idx] = truncated_trace[agent_idx][0][1:].tolist()

                all_transition_paths.append(transition_paths)
                tmp = node.new_child(
                    trace=next_node_trace,
                    init=next_node_init,
                    mode=next_node_mode,
                    start_time=next_node_start_time,
                    id=-1,
                )
                next_nodes.append(tmp)
            # print(len(next_nodes))
            # print(f"node {node.id} dur {timeit.default_timer() - t}")
            return (node.id, later, next_nodes, node.trace, cache_updates)

    def proc_result(self, id, later, next_nodes, traces, cache_updates):
        t = timeit.default_timer()
        # print("got id:", id)
        done_node = self.nodes[id]
        done_node.child = next_nodes
        done_node.trace = traces
        last_id = self.nodes[-1].id
        # assert max(n.id for n in self.nodes) == last_id
        for i, node in enumerate(next_nodes):
            node.id = i + 1 + last_id
            later = 0 if i == 0 else 1
            self.simulation_queue.append((node, later))
        self.simulation_queue.sort(key=lambda p: p[1:])
        self.nodes.extend(next_nodes)
        for (
            new,
            aid,
            transit_agents,
            full_trace,
            transition,
            transition_idx,
            run_num,
        ) in cache_updates:
            cached = self.cache.check_hit(
                aid, done_node.mode[aid], done_node.init[aid], done_node.init
            )
            if new and not cached:
                self.cache.add_segment(
                    aid, done_node, transit_agents, full_trace, transition, transition_idx, run_num
                )
                self.num_cached += 1
            else:
                assert cached != None
                cached.transitions.extend(
                    convert_sim_trans(
                        aid, transit_agents, done_node.init, transition, transition_idx
                    )
                )
                cached.transitions = dedup(cached.transitions, lambda i: (i.disc, i.cont, i.inits))
                cached.node_ids.add((run_num, done_node.id))
            # pre_len = len(cached_segments[aid].transitions)
            # pp(("dedup!", pre_len, len(cached_segments[aid].transitions)))
        # print(f"proc dur {timeit.default_timer() - t}")

    def simulate(
        self,
        root: AnalysisTreeNode,
        sensor,
        time_horizon,
        time_step,
        max_height,
        lane_map,
        run_num,
        past_runs,
    ):
        # Setup the root of the simulation tree
        if max_height == None:
            max_height = float("inf")

        self.simulation_queue: List[Tuple[AnalysisTreeNode, int]] = [(root, 0)]
        self.result_refs = []
        self.nodes = [root]
        self.num_cached = 0
        # Perform BFS through the simulation tree to loop through all possible transitions
        consts = SimConsts(time_step, lane_map, run_num, past_runs, sensor, root.agent)
        if self.config.parallel:
            import ray

            consts_ref = ray.put(consts)
        while True:
            wait = False
            start = timeit.default_timer()
            if len(self.simulation_queue) > 0:
                node, later = self.simulation_queue.pop(0)
                # pp(("start sim", node.start_time, {a: (*node.mode[a], *node.init[a]) for a in node.mode}))
                remain_time = round(time_horizon - node.start_time, 10)
                if remain_time <= 0:
                    continue
                # For trace not already simulated
                cached_segments = {}
                for agent_id in node.agent:
                    mode = node.mode[agent_id]
                    init = node.init[agent_id]
                    if self.config.incremental:
                        # pp(("check hit", agent_id, mode, init))
                        cached = self.cache.check_hit(agent_id, mode, init, node.init)
                        if cached != None:
                            self.cache_hits = self.cache_hits[0] + 1, self.cache_hits[1]
                        else:
                            self.cache_hits = self.cache_hits[0], self.cache_hits[1] + 1
                        # pp(("check hit res", agent_id, len(cached.transitions) if cached != None else None))
                        if cached != None:
                            cached_segments[agent_id] = cached
                old_node_id = None
                if len(cached_segments) == len(node.agent):
                    all_node_ids = [s.node_ids for s in cached_segments.values()]
                    node_ids = list(functools.reduce(lambda a, b: a.intersection(b), all_node_ids))
                    if len(node_ids) > 0:
                        old_node_id = node_ids[0]
                    # else:
                    #     print(f"not full {node.id}: {node_ids}, {len(cached_segments) == len(node.agent)} | {all_node_ids}")
                if not self.config.parallel or old_node_id != None:
                    # print(f"local {node.id}")
                    t = timeit.default_timer()
                    self.proc_result(
                        *self.simulate_one(
                            self.config,
                            cached_segments,
                            node,
                            old_node_id,
                            later,
                            remain_time,
                            consts,
                        )
                    )
                    # print(f"node {node.id} dur {timeit.default_timer() - t}")
                else:
                    self.result_refs.append(
                        self.simulate_one_remote.remote(
                            self.config,
                            cached_segments,
                            node,
                            old_node_id,
                            later,
                            remain_time,
                            consts_ref,
                        )
                    )
                if len(self.result_refs) >= self.config.parallel_sim_ahead:
                    wait = True
            elif len(self.result_refs) > 0:
                wait = True
            else:
                break
            if wait:
                [res], remaining = ray.wait(self.result_refs)
                id, later, next_nodes, traces, cache_updates = ray.get(res)
                self.proc_result(id, later, next_nodes, traces, cache_updates)
                self.result_refs = remaining
        # print("cached", self.num_cached)
        # pp(self.cache.get_cached_inits(3))
        self.simulation_tree = AnalysisTree(root)
        return self.simulation_tree

    def simulate_simple(
        self,
        root: AnalysisTreeNode,
        time_horizon,
        time_step,
        max_height,
        lane_map,
        sensor,
        run_num,
        past_runs,
    ):
        # Setup the root of the simulation tree
        if max_height == None:
            max_height = float("inf")

        simulation_queue = []
        simulation_queue.append(root)
        # Perform BFS through the simulation tree to loop through all possible transitions
        while simulation_queue != []:
            node: AnalysisTreeNode = simulation_queue.pop(0)
            if node.height >= max_height:
                print("max depth reached")
                continue
            # continue if we are at the depth limit

            pp(
                (
                    "start sim",
                    node.start_time,
                    {a: (*node.mode[a], *node.init[a]) for a in node.mode},
                )
            )
            remain_time = round(time_horizon - node.start_time, 10)
            if remain_time <= 0:
                continue
            # For trace not already simulated
            for agent_id in node.agent:
                mode = node.mode[agent_id]
                init = node.init[agent_id]
                # pp(("sim", agent_id, *mode, *init))
                # Simulate the trace starting from initial condition
                trace = node.agent[agent_id].TC_simulate(
                    mode, init, remain_time, time_step, lane_map
                )
                trace[:, 0] += node.start_time
                node.trace[agent_id] = trace
            # pp(("cached_segments", cached_segments.keys()))
            # TODO: for now, make sure all the segments comes from the same node; maybe we can do
            # something to combine results from different nodes in the future

            asserts, transitions, transition_idx = Simulator.get_transition_simulate_simple(
                node, lane_map, sensor
            )
            # pp(("transitions:", transition_idx, transitions))

            node.assert_hits = asserts
            # pp(("next init:", {a: trace[transition_idx] for a, trace in node.trace.items()}))

            # truncate the computed trajectories from idx and store the content after truncate
            truncated_trace, full_traces = {}, {}
            for agent_idx in node.agent:
                full_traces[agent_idx] = node.trace[agent_idx]
                if transitions or asserts:
                    truncated_trace[agent_idx] = node.trace[agent_idx][transition_idx:]
                    node.trace[agent_idx] = node.trace[agent_idx][: transition_idx + 1]

            if asserts != None:
                pass
                # print(transition_idx)
                # pp({a: len(t) for a, t in node.trace.items()})
            else:
                # If there's no transitions (returned transitions is empty), continue
                if not transitions:
                    continue

                # pp(("transit agents", transit_agents))

                transition_list = list(transitions.values())
                all_transition_combinations = itertools.product(*transition_list)

                # For each possible transition, construct the new node.
                # Obtain the new initial condition for agent having transition
                # copy the traces that are not under transition
                all_transition_paths = []
                for transition_combination in all_transition_combinations:
                    transition_paths = []
                    next_node_mode = copy.deepcopy(node.mode)
                    next_node_agent = node.agent
                    next_node_start_time = list(truncated_trace.values())[0][0][0]
                    next_node_init = {}
                    next_node_trace = {}
                    for transition in transition_combination:
                        transit_agent_idx, dest_mode, next_init, paths = transition
                        if dest_mode is None:
                            continue
                        transition_paths.extend(paths)
                        next_node_mode[transit_agent_idx] = dest_mode
                        next_node_init[transit_agent_idx] = next_init
                    for agent_idx in next_node_agent:
                        if agent_idx not in next_node_init:
                            next_node_trace[agent_idx] = truncated_trace[agent_idx]
                            next_node_init[agent_idx] = truncated_trace[agent_idx][0][1:]

                    all_transition_paths.append(transition_paths)

                    tmp = node.new_child(
                        trace=next_node_trace,
                        init=next_node_init,
                        mode=next_node_mode,
                        start_time=next_node_start_time,
                        id=-1,
                    )
                    node.child.append(tmp)
                    simulation_queue.append(tmp)
                # print(red("end sim"))
                # Put the node in the child of current node. Put the new node in the queue
        # checkHeight(root, max_height)

        self.simulation_tree = AnalysisTree(root)

        return self.simulation_tree

    @staticmethod
    def get_transition_simulate(
        cache: Dict[str, CachedSegment],
        paths: PathDiffs,
        node: AnalysisTreeNode,
        track_map: LaneMap,
        sensor,
        agent_dict,
    ) -> Tuple[
        Optional[Dict[str, List[str]]],
        Optional[Dict[str, List[Tuple[str, List[str], List[float]]]]],
        int,
    ]:
        trace_length = len(list(node.trace.values())[0])

        # For each agent
        agent_guard_dict = defaultdict(list)
        cached_guards = defaultdict(list)
        min_trans_ind = None
        cached_trans = defaultdict(list)

        if not cache:
            paths = [
                (agent, p) for agent in node.agent.values() for p in agent.decision_logic.paths
            ]
        else:
            _transitions = [
                (aid, trans)
                for aid, seg in cache.items()
                for trans in seg.transitions
                if sim_trans_suit(trans.inits, node.init)
            ]
            # pp(("cached trans", _transitions))
            if len(_transitions) > 0:
                min_trans_ind = min([t.transition for _, t in _transitions])
                # pp(("min", min_trans_ind))
                for aid, trans in _transitions:
                    # TODO: check for asserts
                    if trans.transition == min_trans_ind:
                        # pp(("chosen tran", aid, trans))
                        cached_trans[aid].append((aid, trans.disc, trans.cont, trans.paths))
                for agent_id in cached_trans:
                    cached_trans[agent_id] = dedup(cached_trans[agent_id], lambda p: p[:3])
                if len(paths) == 0:
                    # print(red("full cache"))
                    return None, dict(cached_trans), min_trans_ind

                path_transitions = defaultdict(int)
                for seg in cache.values():
                    for tran in seg.transitions:
                        for p in tran.paths:
                            path_transitions[p.cond] = max(
                                path_transitions[p.cond], tran.transition
                            )
                for agent_id, segment in cache.items():
                    agent = node.agent[agent_id]
                    if len(agent.decision_logic.args) == 0:
                        continue
                    state_dict = {
                        aid: (node.trace[aid][0], node.mode[aid], node.static[aid])
                        for aid in node.agent
                    }
                    agent_paths = dedup(
                        [p for tran in segment.transitions for p in tran.paths],
                        lambda i: (i.var, i.cond, i.val),
                    )
                    cont_var_dict_template, discrete_variable_dict, len_dict = sensor.sense(
                        agent, state_dict, track_map
                    )
                    for path in agent_paths:
                        cached_guards[agent_id].append(
                            (path, discrete_variable_dict, path_transitions[path.cond])
                        )

        for agent, path in paths:
            # Get guard
            if len(agent.decision_logic.args) == 0:
                continue
            agent_id = agent.id
            agent_mode = node.mode[agent_id]
            state_dict = {
                aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent
            }
            cont_var_dict_template, discrete_variable_dict, len_dict = sensor.sense(
                agent, state_dict, track_map
            )
            agent_guard_dict[agent_id].append((path, discrete_variable_dict))

        transitions = defaultdict(list)
        # TODO: We can probably rewrite how guard hit are detected and resets are handled for simulation
        for idx in range(trace_length):
            if min_trans_ind != None and idx >= min_trans_ind:
                return None, dict(cached_trans), min_trans_ind
            satisfied_guard = []
            all_asserts = defaultdict(list)
            for agent_id in agent_guard_dict:
                agent: BaseAgent = agent_dict[agent_id]
                state_dict = {
                    aid: (node.trace[aid][idx], node.mode[aid], node.static[aid])
                    for aid in node.agent
                }
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                agent_state = agent_state[1:]
                continuous_variable_dict, orig_disc_vars, _ = sensor.sense(
                    agent, state_dict, track_map
                )
                unchecked_cache_guards = [
                    g[:2] for g in cached_guards[agent_id] if g[2] < idx
                ]  # FIXME: off by 1?
                asserts, satisfied = check_sim_transitions(
                    agent,
                    agent_guard_dict[agent_id] + unchecked_cache_guards,
                    continuous_variable_dict,
                    orig_disc_vars,
                    track_map,
                    agent_state,
                    agent_mode,
                )
                if asserts != None:
                    all_asserts[agent_id] = asserts
                    continue
                if len(satisfied) != 0:
                    satisfied_guard.extend(satisfied)
                    # assert all(len(s[2]) == 4 for s in satisfied)
            if len(all_asserts) > 0:
                return all_asserts, dict(transitions), idx
            if len(satisfied_guard) > 0:
                print(len(satisfied_guard))
                for agent_idx, dest, next_init, paths in satisfied_guard:
                    assert isinstance(paths, list)
                    dest = tuple(dest)
                    src_mode = node.get_mode(agent_idx, node.mode[agent_idx])
                    src_track = node.get_track(agent_idx, node.mode[agent_idx])
                    dest_mode = node.get_mode(agent_idx, dest)
                    dest_track = node.get_track(agent_idx, dest)
                    if dest_track == track_map.h(src_track, src_mode, dest_mode):
                        transitions[agent_idx].append((agent_idx, dest, next_init, paths))
                break
        transitions = {aid: dedup(v, lambda p: p[1]) for aid, v in transitions.items()}
        return None, transitions, idx

    @staticmethod
    def get_transition_simulate_simple(
        node: AnalysisTreeNode, track_map, sensor
    ) -> Tuple[
        Optional[Dict[str, List[str]]],
        Optional[Dict[str, List[Tuple[str, List[str], List[float]]]]],
        int,
    ]:
        trace_length = len(list(node.trace.values())[0])

        # For each agent
        agent_guard_dict = defaultdict(list)

        paths = [(agent, p) for agent in node.agent.values() for p in agent.decision_logic.paths]

        for agent, path in paths:
            # Get guard
            if len(agent.decision_logic.args) == 0:
                continue
            agent_id = agent.id
            state_dict = {
                aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent
            }
            cont_var_dict_template, discrete_variable_dict, len_dict = sensor.sense(
                agent, state_dict, track_map
            )
            agent_guard_dict[agent_id].append((path, discrete_variable_dict))

        transitions = defaultdict(list)
        # TODO: We can probably rewrite how guard hit are detected and resets are handled for simulation
        for idx in range(trace_length):
            state_dict = {
                aid: (node.trace[aid][idx], node.mode[aid], node.static[aid]) for aid in node.agent
            }
            satisfied_guard = []
            all_asserts = defaultdict(list)
            for agent_id in agent_guard_dict:
                # Get agent controller
                # Reference: https://stackoverflow.com/questions/55905240/python-dynamically-import-modules-code-from-string-with-importlib
                agent: BaseAgent = node.agent[agent_id]
                dl = types.ModuleType("dl")
                exec(agent.decision_logic.controller_code, dl.__dict__)

                # Get the input arguments for the controller function
                # Pack the environment (create ego and others list)
                continuous_variable_dict, orig_disc_vars, _ = sensor.sense(
                    agent, state_dict, track_map
                )
                arg_list = []
                env = pack_env(agent, EGO, continuous_variable_dict, orig_disc_vars, track_map)
                for arg in agent.decision_logic.args:
                    if arg.name == EGO:
                        ego = env[EGO]
                        ego = convertStrToEnum(ego, agent, dl)
                        arg_list.append(ego)
                    elif arg.name == "track_map":
                        arg_list.append(track_map)
                    else:
                        if isinstance(env[arg.name], list):
                            tmp_list = []
                            for item in env[arg.name]:
                                tmp = convertStrToEnum(item, agent, dl)
                                tmp_list.append(tmp)
                            arg_list.append(tmp_list)
                        else:
                            tmp = convertStrToEnum(env[arg.name], agent, dl)
                            arg_list.append(tmp)

                try:
                    # Input the environment into the actual controller
                    output = dl.decisionLogic(*arg_list)
                    # output = convertEnumToStr(output, agent, dl)
                    # Check if output is the same as ego
                    if env[EGO] != output:
                        # If not, a transition happen, get source and destination, break
                        next_init = []
                        pure_dest = []
                        output = convertEnumToStr(output, agent, dl)
                        for field in env[EGO].__dict__:
                            if disc_field(field, agent):
                                pure_dest.append(getattr(output, field))
                            else:
                                next_init.append(getattr(output, field))
                        satisfied_guard.append((agent_id, pure_dest, next_init, []))
                except AssertionError:
                    # If assertion error happen, means assert hit happen
                    # Get the which assert hit happen and return
                    _, error, _ = sys.exc_info()
                    assertion_label = error.args[0]
                    all_asserts[agent_id] = assertion_label
                    continue

            if len(all_asserts) > 0:
                for agent_idx, dest, next_init, paths in satisfied_guard:
                    transitions[agent_idx].append((agent_idx, dest, next_init, paths))
                return all_asserts, dict(transitions), idx
            if len(satisfied_guard) > 0:
                break
            # Convert output to asserts, transition and idx
        for agent_idx, dest, next_init, paths in satisfied_guard:
            transitions[agent_idx].append((agent_idx, dest, next_init, paths))
        return None, dict(transitions), idx


# print all height of leaves
def checkHeight(root, max_height):
    if root:
        # First recur on left child
        # then print the data of node
        if root.child == []:
            print("HEIGHT", root.height)
            if root.height > max_height:
                print("Exceeds max height")
        for c in root.child:
            checkHeight(c, max_height)
