from dataclasses import dataclass
import pickle
import timeit
from typing import Dict, List, Optional, Tuple
import copy, itertools, functools, pprint, ray
from pympler.asizeof import asizeof

from verse.agents.base_agent import BaseAgent
from verse.analysis.incremental import CachedSegment, SimTraceCache, convert_sim_trans, to_simulate
from verse.analysis.utils import dedup
from verse.map.lane_map import LaneMap
from verse.parser.parser import ModePath, find
# from verse.scenario.scenario import ScenarioConfig
# from verse.scenario.scenario import Scenario
pp = functools.partial(pprint.pprint, compact=True, width=130)

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree, TraceType

PathDiffs = List[Tuple[BaseAgent, ModePath]]


def red(s):
    return "\x1b[31m" + s + "\x1b[0m" #]]

@dataclass
class SimConsts:
    time_step: float
    lane_map: LaneMap
    run_num: int
    past_runs: List[AnalysisTree]
    transition_graph: "Scenario"

class Simulator:
    def __init__(self, config):
        self.simulation_tree = None
        self.cache = SimTraceCache()
        self.config = config
        self.cache_hits = (0, 0)

    @staticmethod
    def simulate_one(config: "ScenarioConfig", cached_segments: Dict[str, CachedSegment], node: AnalysisTreeNode, old_node_id: Optional[Tuple[int, int]], later: int, remain_time: float, consts: SimConsts) -> Tuple[int, int, List[AnalysisTreeNode], Dict[str, TraceType], list]:
        t = timeit.default_timer()
        print(f"node {node.id} start: {t}")
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
                    trace = node.agent[agent_id].TC_simulate(mode, init, remain_time, consts.time_step, consts.lane_map)
                    trace[:, 0] += node.start_time
                    node.trace[agent_id] = trace
        # pp(("cached_segments", cached_segments.keys()))
        # TODO: for now, make sure all the segments comes from the same node; maybe we can do
        # something to combine results from different nodes in the future
        new_cache, paths_to_sim = {}, []
        if old_node_id != None:
            if old_node_id[0] != consts.run_num:
                old_node = find(consts.past_runs[old_node_id[0]].nodes, lambda n: n.id == old_node_id[1])
                assert old_node != None
                new_cache, paths_to_sim = to_simulate(old_node.agent, node.agent, cached_segments)

        asserts, transitions, transition_idx = consts.transition_graph.get_transition_simulate(new_cache, paths_to_sim, node)
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
                node.trace[agent_idx] = node.trace[agent_idx][:transition_idx+1]

        if asserts != None:     # FIXME
            return (node.id, later, [], node.trace, cache_updates)
            # print(transition_idx)
            # pp({a: len(t) for a, t in node.trace.items()})
        else:
            # If there's no transitions (returned transitions is empty), continue
            if not transitions:
                if config.incremental:
                    for agent_id in node.agent:
                        cache_updates.append((agent_id not in cached_segments, agent_id, [], full_traces[agent_id], [], transition_idx, consts.run_num))
                # print(red("no trans"))
                # print(f"node {node.id} dur {timeit.default_timer() - t}")
                return (node.id, later, [], node.trace, cache_updates)

            transit_agents = transitions.keys()
            # pp(("transit agents", transit_agents))
            if config.incremental:
                for agent_id in node.agent:
                    transition = transitions[agent_id] if agent_id in transit_agents else []
                    # TODO: update current transitions
                    cache_updates.append((agent_id not in cached_segments, agent_id, transit_agents, full_traces[agent_id], transition, transition_idx, consts.run_num))
            # pp(("cached inits", self.cache.get_cached_inits(3)))
            # Generate the transition combinations if multiple agents can transit at the same time step
            transition_list = list(transitions.values())
            all_transition_combinations = itertools.product(
                *transition_list)

            # For each possible transition, construct the new node.
            # Obtain the new initial condition for agent having transition
            # copy the traces that are not under transition
            all_transition_paths = []
            next_nodes = []
            for transition_combination in all_transition_combinations:
                transition_paths = []
                next_node_mode = copy.deepcopy(node.mode)
                next_node_static = copy.deepcopy(node.static)
                next_node_uncertain_param = copy.deepcopy(node.uncertain_param)
                next_node_agent = node.agent
                next_node_start_time = list(
                    truncated_trace.values())[0][0][0]
                next_node_init = {}
                next_node_trace = {}
                for transition in transition_combination:
                    transit_agent_idx, dest_mode, next_init, paths = transition
                    if dest_mode is None:
                        continue
                    transition_paths.extend(paths)
                    # next_node = AnalysisTreeNode(trace = {},init={},mode={},agent={}, child = [], start_time = 0)
                    next_node_mode[transit_agent_idx] = dest_mode
                    next_node_init[transit_agent_idx] = next_init
                for agent_idx in next_node_agent:
                    if agent_idx not in next_node_init:
                        next_node_trace[agent_idx] = truncated_trace[agent_idx]
                        next_node_init[agent_idx] = truncated_trace[agent_idx][0][1:].tolist()

                all_transition_paths.append(transition_paths)
                tmp = AnalysisTreeNode(
                    trace=next_node_trace,
                    init=next_node_init,
                    mode=next_node_mode,
                    static=next_node_static,
                    uncertain_param=next_node_uncertain_param,
                    agent=next_node_agent,
                    child=[],
                    start_time=next_node_start_time,
                    type='simtrace'
                )
                next_nodes.append(tmp)
            print(len(next_nodes))
            print(f"node {node.id} dur {timeit.default_timer() - t}")
            return (node.id, later, next_nodes, node.trace, cache_updates)

    @ray.remote
    def simulate_one_remote(config: "ScenarioConfig", cached_segments: Dict[str, CachedSegment], node: AnalysisTreeNode, old_node_id: Optional[Tuple[int, int]], later: int, remain_time: float, consts: SimConsts) -> Tuple[int, int, List[AnalysisTreeNode], Dict[str, TraceType], list]:
        return Simulator.simulate_one(config, cached_segments, node, old_node_id, later, remain_time, consts)

    def proc_result(self, id, later, next_nodes, traces, cache_updates):
        t = timeit.default_timer()
        print("got id:", id)
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
        for new, aid, transit_agents, full_trace, transition, transition_idx, run_num in cache_updates:
            cached = self.cache.check_hit(aid, done_node.mode[aid], done_node.init[aid], done_node.init)
            if new and not cached:
                self.cache.add_segment(aid, done_node, transit_agents, full_trace, transition, transition_idx, run_num)
                self.num_cached += 1
            else:
                assert cached != None
                cached.transitions.extend(convert_sim_trans(aid, transit_agents, done_node.init, transition, transition_idx))
                cached.transitions = dedup(cached.transitions, lambda i: (i.disc, i.cont, i.inits))
                cached.node_ids.add((run_num, done_node.id))
            # pre_len = len(cached_segments[aid].transitions)
            # pp(("dedup!", pre_len, len(cached_segments[aid].transitions)))
        print(f"proc dur {timeit.default_timer() - t}")

    def simulate(self, init_list, init_mode_list, static_list, uncertain_param_list, agent_list,
                 transition_graph, time_horizon, time_step, max_height, lane_map, run_num, past_runs):
        # Setup the root of the simulation tree
        if(max_height == None):
            max_height = float('inf')
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static={},
            uncertain_param={},
            height = 0,
            agent={},
            child=[],
            start_time=0,
        )
        for i, agent in enumerate(agent_list):
            root.init[agent.id] = init_list[i]
            init_mode = [elem.name for elem in init_mode_list[i]]
            root.mode[agent.id] = init_mode
            init_static = [elem.name for elem in static_list[i]]
            root.static[agent.id] = init_static
            root.uncertain_param[agent.id] = uncertain_param_list[i]
            root.agent[agent.id] = agent
            root.type = 'simtrace'

        root.id = 0     # FIXME
        self.simulation_queue: List[Tuple[AnalysisTreeNode, int]] = [(root, 0)]
        self.result_refs = []
        self.nodes = [root]
        self.num_cached = 0
        # Perform BFS through the simulation tree to loop through all possible transitions
        consts = SimConsts(time_step, lane_map, run_num, past_runs, transition_graph)
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
                        # if cached != None:
                        #     self.cache_hits = self.cache_hits[0] + 1, self.cache_hits[1]
                        # else:
                        #     self.cache_hits = self.cache_hits[0], self.cache_hits[1] + 1
                        # pp(("check hit res", agent_id, len(cached.transitions) if cached != None else None))
                        if cached != None:
                            cached_segments[agent_id] = cached
                old_node_id = None
                if len(cached_segments) == len(node.agent):
                    all_node_ids = [s.node_ids for s in cached_segments.values()]
                    node_ids = list(functools.reduce(lambda a, b: a.intersection(b), all_node_ids))
                    if len(node_ids) > 0:
                        old_node_id = node_ids[0]
                    else:
                        print(f"not full {node.id}: {node_ids}, {len(cached_segments) == len(node.agent)} | {all_node_ids}")
                if not self.config.parallel or old_node_id != None:
                    print(f"local {node.id}")
                    t = timeit.default_timer()
                    self.proc_result(*self.simulate_one(self.config, cached_segments, node, old_node_id, later, remain_time, consts))
                    print(f"node {node.id} dur {timeit.default_timer() - t}")
                else:
                    self.result_refs.append(self.simulate_one_remote.remote(self.config, cached_segments, node, old_node_id, later, remain_time, consts_ref))
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
        print("cached", self.num_cached)
        pp(self.cache.get_cached_inits(3))
        self.simulation_tree = AnalysisTree(root)
        return self.simulation_tree

    def simulate_simple(self, init_list, init_mode_list, static_list, uncertain_param_list, agent_list,
                 transition_graph, time_horizon, time_step, max_height, lane_map, run_num, past_runs):
        # Setup the root of the simulation tree
        if(max_height == None):
            max_height = float('inf')
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static={},
            uncertain_param={},
            agent={},
            height =0,
            child=[],
            start_time=0,
        )
        for i, agent in enumerate(agent_list):
            root.init[agent.id] = init_list[i]
            init_mode = [elem.name for elem in init_mode_list[i]]
            root.mode[agent.id] = init_mode
            init_static = [elem.name for elem in static_list[i]]
            root.static[agent.id] = init_static
            root.uncertain_param[agent.id] = uncertain_param_list[i]
            root.agent[agent.id] = agent
            root.type = 'simtrace'

        simulation_queue = []
        simulation_queue.append(root)
        # Perform BFS through the simulation tree to loop through all possible transitions
        while simulation_queue != []:
            node: AnalysisTreeNode = simulation_queue.pop(0)
            if (node.height >= max_height):
                print("max depth reached")
                continue
            #continue if we are at the depth limit

            pp(("start sim", node.start_time, {a: (*node.mode[a], *node.init[a]) for a in node.mode}))
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
                    mode, init, remain_time, time_step, lane_map)
                trace[:, 0] += node.start_time
                node.trace[agent_id] = trace
            # pp(("cached_segments", cached_segments.keys()))
            # TODO: for now, make sure all the segments comes from the same node; maybe we can do
            # something to combine results from different nodes in the future
            
            asserts, transitions, transition_idx = transition_graph.get_transition_simulate_simple(node)
            # pp(("transitions:", transition_idx, transitions))

            node.assert_hits = asserts
            # pp(("next init:", {a: trace[transition_idx] for a, trace in node.trace.items()}))

            # truncate the computed trajectories from idx and store the content after truncate
            truncated_trace, full_traces = {}, {}
            for agent_idx in node.agent:
                full_traces[agent_idx] = node.trace[agent_idx]
                if transitions or asserts:
                    truncated_trace[agent_idx] = node.trace[agent_idx][transition_idx:]
                    node.trace[agent_idx] = node.trace[agent_idx][:transition_idx+1]

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
                all_transition_combinations = itertools.product(
                    *transition_list)

                # For each possible transition, construct the new node.
                # Obtain the new initial condition for agent having transition
                # copy the traces that are not under transition
                all_transition_paths = []
                for transition_combination in all_transition_combinations:
                    transition_paths = []
                    next_node_mode = copy.deepcopy(node.mode)
                    next_node_static = copy.deepcopy(node.static)
                    next_node_uncertain_param = copy.deepcopy(node.uncertain_param)
                    next_node_agent = node.agent
                    next_node_start_time = list(
                        truncated_trace.values())[0][0][0]
                    next_node_init = {}
                    next_node_trace = {}
                    for transition in transition_combination:
                        transit_agent_idx, dest_mode, next_init, paths = transition
                        if dest_mode is None:
                            continue
                        transition_paths.extend(paths)
                        # next_node = AnalysisTreeNode(trace = {},init={},mode={},agent={}, child = [], start_time = 0)
                        next_node_mode[transit_agent_idx] = dest_mode
                        next_node_init[transit_agent_idx] = next_init
                    for agent_idx in next_node_agent:
                        if agent_idx not in next_node_init:
                            next_node_trace[agent_idx] = truncated_trace[agent_idx]
                            next_node_init[agent_idx] = truncated_trace[agent_idx][0][1:]

                    all_transition_paths.append(transition_paths)


                    tmp = AnalysisTreeNode(
                        trace=next_node_trace,
                        init=next_node_init,
                        mode=next_node_mode,
                        static=next_node_static,
                        uncertain_param=next_node_uncertain_param,
                        agent=next_node_agent,
                        height=node.height + 1,
                        child=[],
                        start_time=next_node_start_time,
                        type='simtrace'
                    )
                    node.child.append(tmp)
                    simulation_queue.append(tmp)
                # print(red("end sim"))
                # Put the node in the child of current node. Put the new node in the queue
            #     node.child.append(AnalysisTreeNode(
            #         trace = next_node_trace,
            #         init = next_node_init,
            #         mode = next_node_mode,
            #         agent = next_node_agent,
            #         child = [],
            #         start_time = next_node_start_time
            #     ))
            # simulation_queue += node.child
        #checkHeight(root, max_height)

        self.simulation_tree = AnalysisTree(root)


        return self.simulation_tree

#print all height of leaves
def checkHeight(root, max_height):
    if root:
        # First recur on left child
        # then print the data of node
        if(root.child == []):
            print("HEIGHT", root.height)
            if(root.height > max_height):
                print("Exceeds max height")
        for c in root.child:
            checkHeight(c, max_height)


