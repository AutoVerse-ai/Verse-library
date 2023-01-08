from typing import Dict, List, Optional, Tuple
import copy, itertools, functools, pprint, ray

from verse.agents.base_agent import BaseAgent
from verse.analysis.incremental import SimTraceCache, convert_sim_trans, to_simulate
from verse.analysis.utils import dedup
from verse.map.lane_map import LaneMap
from verse.parser.parser import ModePath, find
# from verse.scenario.scenario import Scenario
pp = functools.partial(pprint.pprint, compact=True, width=130)

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree

PathDiffs = List[Tuple[BaseAgent, ModePath]]

def red(s):
    return "\x1b[31m" + s + "\x1b[0m" #]]

class Simulator:
    def __init__(self, config):
        self.simulation_tree = None
        self.cache = SimTraceCache()
        self.config = config
        self.cache_hits = (0, 0)

    @ray.remote
    def simulate_one(self, node: AnalysisTreeNode, remain_time: float, time_step: float, lane_map: LaneMap, run_num: int, past_runs: List[AnalysisTree], transition_graph: "Scenario") -> Tuple[int, List[AnalysisTreeNode], Dict[str, list]]:
        print(f"node id: {node.id}")
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
            else:
                cached = None
            if agent_id in node.trace:
                if cached != None:
                    cached_segments[agent_id] = cached
            else:
                if cached != None:
                    node.trace[agent_id] = cached.trace
                    if len(cached.trace) < remain_time / time_step:
                        node.trace[agent_id] += node.agent[agent_id].TC_simulate(mode, cached.trace[-1], remain_time - time_step * len(cached.trace), lane_map)
                    cached_segments[agent_id] = cached
                else:
                    # pp(("sim", agent_id, *mode, *init))
                    # Simulate the trace starting from initial condition
                    trace = node.agent[agent_id].TC_simulate(
                        mode, init, remain_time, time_step, lane_map)
                    trace[:, 0] += node.start_time
                    trace = trace.tolist()
                    node.trace[agent_id] = trace
        # pp(("cached_segments", cached_segments.keys()))
        # TODO: for now, make sure all the segments comes from the same node; maybe we can do
        # something to combine results from different nodes in the future
        node_ids = list(set((s.run_num, s.node_id) for s in cached_segments.values()))
        # assert len(node_ids) <= 1, f"{node_ids}"
        new_cache, paths_to_sim = {}, []
        if len(node_ids) == 1 and len(cached_segments.keys()) == len(node.agent):
            old_run_num, old_node_id = node_ids[0]
            if old_run_num != run_num:
                old_node = find(past_runs[old_run_num].nodes, lambda n: n.id == old_node_id)
                assert old_node != None
                new_cache, paths_to_sim = to_simulate(old_node.agent, node.agent, cached_segments)
                # pp(("to sim", new_cache.keys(), len(paths_to_sim)))
            # else:
            #     print("!!!")

        asserts, transitions, transition_idx = transition_graph.get_transition_simulate(new_cache, paths_to_sim, node)
        # pp(("transitions:", transition_idx, transitions))

        node.assert_hits = asserts
        # pp(("next init:", {a: trace[transition_idx] for a, trace in node.trace.items()}))

        # truncate the computed trajectories from idx and store the content after truncate
        truncated_trace, full_traces = {}, {}
        for agent_idx in node.agent:
            full_traces[agent_idx] = node.trace[agent_idx]
            if transitions:
                truncated_trace[agent_idx] = node.trace[agent_idx][transition_idx:]
                node.trace[agent_idx] = node.trace[agent_idx][:transition_idx+1]

        if asserts != None:     # FIXME
            return (node.id, [], node.trace)
            # print(transition_idx)
            # pp({a: len(t) for a, t in node.trace.items()})
        else:
            # If there's no transitions (returned transitions is empty), continue
            if not transitions:
                if self.config.incremental:
                    for agent_id in node.agent:
                        if agent_id not in cached_segments:
                            self.cache.add_segment(agent_id, node, [], full_traces[agent_id], [], transition_idx, run_num)
                # print(red("no trans"))
                return (node.id, [], node.trace)

            transit_agents = transitions.keys()
            # pp(("transit agents", transit_agents))
            if self.config.incremental:
                for agent_id in node.agent:
                    transition = transitions[agent_id] if agent_id in transit_agents else []
                    if agent_id in cached_segments:
                        cached_segments[agent_id].transitions.extend(convert_sim_trans(agent_id, transit_agents, node.init, transition, transition_idx))
                        cached_segments[agent_id].transitions = dedup(cached_segments[agent_id].transitions, lambda i: (i.disc, i.cont, i.inits))
                        # pre_len = len(cached_segments[agent_id].transitions)
                        # pp(("dedup!", pre_len, len(cached_segments[agent_id].transitions)))
                    else:
                        self.cache.add_segment(agent_id, node, transit_agents, full_traces[agent_id], transition, transition_idx, run_num)
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
                        next_node_init[agent_idx] = truncated_trace[agent_idx][0][1:]

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
            return (node.id, next_nodes, node.trace)

    def simulate(self, init_list, init_mode_list, static_list, uncertain_param_list, agent_list,
                 transition_graph, time_horizon, time_step, lane_map, run_num, past_runs):
        # Setup the root of the simulation tree
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static={},
            uncertain_param={},
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
        simulation_queue = [root]
        result_refs = []
        nodes = [root]
        # Perform BFS through the simulation tree to loop through all possible transitions
        while True:
            wait = False
            if len(simulation_queue) > 0:
                node: AnalysisTreeNode = simulation_queue.pop(0)
                # pp(("start sim", node.start_time, {a: (*node.mode[a], *node.init[a]) for a in node.mode}))
                remain_time = round(time_horizon - node.start_time, 10)
                if remain_time <= 0:
                    continue
                # For trace not already simulated
                result_refs.append(self.simulate_one.remote(self, node, remain_time, time_step, lane_map, run_num, past_runs, transition_graph))
                if len(result_refs) >= self.config.parallel_sim_ahead:
                    wait = True
            elif len(result_refs) > 0:
                wait = True
            else:
                break
            print(len(simulation_queue), len(result_refs))
            if wait:
                [res], remaining = ray.wait(result_refs)
                id, next_nodes, traces = ray.get(res)
                print("got id:", id)
                nodes[id].child = next_nodes
                nodes[id].trace = traces
                last_id = nodes[-1].id
                for i, node in enumerate(next_nodes):
                    node.id = i + 1 + last_id
                simulation_queue.extend(next_nodes)
                nodes.extend(next_nodes)
                result_refs = remaining
        
        self.simulation_tree = AnalysisTree(root)
        return self.simulation_tree

    def simulate_simple(self, init_list, init_mode_list, static_list, uncertain_param_list, agent_list,
                 transition_graph, time_horizon, time_step, lane_map, run_num, past_runs):
        # Setup the root of the simulation tree
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static={},
            uncertain_param={},
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

        simulation_queue = []
        simulation_queue.append(root)
        # Perform BFS through the simulation tree to loop through all possible transitions
        while simulation_queue != []:
            node: AnalysisTreeNode = simulation_queue.pop(0)
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
                trace = trace.tolist()
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
        
        self.simulation_tree = AnalysisTree(root)
        return self.simulation_tree

