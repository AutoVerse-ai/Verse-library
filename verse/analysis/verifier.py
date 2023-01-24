import copy, itertools, functools, pprint, ray
from typing import List

import numpy as np

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree
from verse.analysis.dryvr import calc_bloated_tube, SIMTRACENUM
from verse.analysis.mixmonotone import calculate_bloated_tube_mixmono_cont, calculate_bloated_tube_mixmono_disc
from verse.analysis.incremental import ReachTubeCache, TubeCache, convert_reach_trans, to_simulate, combine_all
from verse.analysis.utils import dedup
from verse.parser.parser import find
pp = functools.partial(pprint.pprint, compact=True, width=130)

class Verifier:
    def __init__(self, config):
        self.reachtube_tree = None
        self.cache = TubeCache()
        self.trans_cache = ReachTubeCache()
        self.tube_cache_hits = (0, 0)
        self.trans_cache_hits = (0, 0)
        self.config = config

    def calculate_full_bloated_tube(
        self,
        agent_id,
        mode_label,
        initial_set,
        time_horizon,
        time_step,
        sim_func,
        params,
        kvalue,
        sim_trace_num,
        combine_seg_length = 1000,
        guard_checker=None,
        guard_str="",
        lane_map = None
    ):
        # Handle Parameters
        bloating_method = 'PW'
        if 'bloating_method' in params:
            bloating_method = params['bloating_method']
        
        res_tube = None
        tube_length = 0
        for combine_seg_idx in range(0, len(initial_set), combine_seg_length):
            rect_seg = initial_set[combine_seg_idx:combine_seg_idx+combine_seg_length]
            combined_rect = None
            for rect in rect_seg:
                rect = np.array(rect)
                if combined_rect is None:
                    combined_rect = rect
                else:
                    combined_rect[0, :] = np.minimum(
                        combined_rect[0, :], rect[0, :])
                    combined_rect[1, :] = np.maximum(
                        combined_rect[1, :], rect[1, :])
            combined_rect = combined_rect.tolist()
            if self.config.incremental:
                cached = self.cache.check_hit(agent_id, mode_label, combined_rect)
                if cached != None:
                    self.tube_cache_hits = self.tube_cache_hits[0] + 1, self.tube_cache_hits[1]
                else:
                    self.tube_cache_hits = self.tube_cache_hits[0], self.tube_cache_hits[1] + 1
            else:
                cached = None
            if cached != None:
                cur_bloated_tube = cached.tube
            else:
                cur_bloated_tube = calc_bloated_tube(mode_label,
                                            combined_rect,
                                            time_horizon,
                                            time_step, 
                                            sim_func,
                                            bloating_method,
                                            kvalue,
                                            sim_trace_num,
                                            lane_map = lane_map
                                            )
                if self.config.incremental:
                    self.cache.add_tube(agent_id, mode_label, combined_rect, cur_bloated_tube)
            if combine_seg_idx == 0:
                res_tube = cur_bloated_tube
                tube_length = cur_bloated_tube.shape[0]
            else:
                cur_bloated_tube = cur_bloated_tube[:tube_length - combine_seg_idx*2,:]
                # Handle Lower Bound
                res_tube[combine_seg_idx*2::2,1:] = np.minimum(
                    res_tube[combine_seg_idx*2::2,1:],
                    cur_bloated_tube[::2,1:]
                )
                # Handle Upper Bound
                res_tube[combine_seg_idx*2+1::2,1:] = np.maximum(
                    res_tube[combine_seg_idx*2+1::2,1:],
                    cur_bloated_tube[1::2,1:]
                )
        return res_tube.tolist()

    @ray.remote
    def compute_full_reachtube_step(
        self,
        init_list: List[float],
        init_mode_list: List[str],
        static_list: List[str],
        uncertain_param_list: List[float],
        agent_list,
        transition_graph,
        time_horizon,
        time_step,
        lane_map,
        init_seg_length,
        reachability_method,
        run_num,
        past_runs,
        params = {},
    ):
        return

    def compute_full_reachtube(
        self,
        init_list: List[float],
        init_mode_list: List[str],
        static_list: List[str],
        uncertain_param_list: List[float],
        agent_list,
        transition_graph,
        time_horizon,
        time_step,
        lane_map,
        init_seg_length,
        reachability_method,
        run_num,
        past_runs,
        params = {},
    ):
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static = {},
            uncertain_param={},
            agent={},
            assert_hits={},
            child=[],
            start_time = 0,
            ndigits = 10,
            type = 'simtrace',
            id = 0
        )
        # root = AnalysisTreeNode()
        for i, agent in enumerate(agent_list):
            root.init[agent.id] = [init_list[i]]
            init_mode = [elem.name for elem in init_mode_list[i]]
            root.mode[agent.id] = init_mode
            init_static = [elem.name for elem in static_list[i]]
            root.static[agent.id] = init_static
            root.uncertain_param[agent.id] = uncertain_param_list[i]
            root.agent[agent.id] = agent
            root.type = 'reachtube'
        verification_queue = []
        verification_queue.append(root)
        num_calls = 0
        num_transitions = 0
        while verification_queue != []:
            node: AnalysisTreeNode = verification_queue.pop(0)
            combined_inits = {a: combine_all(inits) for a, inits in node.init.items()}
            print(node.init)
            print(node.mode)
            print("###############")
            # pp(("start sim", node.start_time, {a: (*node.mode[a], *combined_inits[a]) for a in node.mode}))
            remain_time = round(time_horizon - node.start_time, 10)
            if remain_time <= 0:
                continue
            num_transitions += 1
            cached_tubes = {}
            # For reachtubes not already computed
            # TODO: can add parallalization for this loop
            for agent_id in node.agent:
                mode = node.mode[agent_id]
                inits = node.init[agent_id]
                combined = combine_all(inits)
                if self.config.incremental:
                    cached = self.trans_cache.check_hit(agent_id, mode, combined, node.init)
                    if cached != None:
                        self.trans_cache_hits = self.trans_cache_hits[0] + 1, self.trans_cache_hits[1]
                    else:
                        self.trans_cache_hits = self.trans_cache_hits[0], self.trans_cache_hits[1] + 1
                    # pp(("check hit", agent_id, mode, combined))
                    if cached != None:
                        cached_tubes[agent_id] = cached
                if agent_id not in node.trace:
                    # Compute the trace starting from initial condition
                    uncertain_param = node.uncertain_param[agent_id]
                    # trace = node.agent[agent_id].TC_simulate(mode, init, remain_time,lane_map)
                    # trace[:,0] += node.start_time
                    # node.trace[agent_id] = trace.tolist()
                    if reachability_method == "DRYVR":
                        # pp(('tube', agent_id, mode, inits))
                        cur_bloated_tube = self.calculate_full_bloated_tube(agent_id,
                                            mode,
                                            inits,
                                            remain_time,
                                            time_step, 
                                            node.agent[agent_id].TC_simulate,
                                            params,
                                            100,
                                            SIMTRACENUM,
                                            combine_seg_length=init_seg_length,
                                            lane_map = lane_map
                                            )
                    elif reachability_method == "NeuReach":
                        from verse.analysis.NeuReach.NeuReach_onestep_rect import postCont
                        cur_bloated_tube = postCont(
                            mode, 
                            inits[0], 
                            remain_time, 
                            time_step, 
                            node.agent[agent_id].TC_simulate, 
                            lane_map,
                            params, 
                        )
                    elif reachability_method == "MIXMONO_CONT":
                        cur_bloated_tube = calculate_bloated_tube_mixmono_cont(
                            mode, 
                            inits, 
                            uncertain_param, 
                            remain_time,
                            time_step, 
                            node.agent[agent_id],
                            lane_map
                        )
                    elif reachability_method == "MIXMONO_DISC":
                        cur_bloated_tube = calculate_bloated_tube_mixmono_disc(
                            mode, 
                            inits, 
                            uncertain_param,
                            remain_time,
                            time_step,
                            node.agent[agent_id],
                            lane_map
                        ) 
                    else:
                        raise ValueError(f"Reachability computation method {reachability_method} not available.")
                    num_calls += 1
                    trace = np.array(cur_bloated_tube)
                    trace[:, 0] += node.start_time
                    node.trace[agent_id] = trace.tolist()
            # pp(("cached tubes", cached_tubes.keys()))
            node_ids = list(set((s.run_num, s.node_id) for s in cached_tubes.values()))
            # assert len(node_ids) <= 1, f"{node_ids}"
            new_cache, paths_to_sim = {}, []
            if len(node_ids) == 1 and len(cached_tubes.keys()) == len(node.agent):
                old_run_num, old_node_id = node_ids[0]
                if old_run_num != run_num:
                    old_node = find(past_runs[old_run_num].nodes, lambda n: n.id == old_node_id)
                    assert old_node != None
                    new_cache, paths_to_sim = to_simulate(old_node.agent, node.agent, cached_tubes)
                    # pp(("to sim", new_cache.keys(), len(paths_to_sim)))

            # Get all possible transitions to next mode
            asserts, all_possible_transitions = transition_graph.get_transition_verify(new_cache, paths_to_sim, node)
            # pp(("transitions:", [(t[0], t[2]) for t in all_possible_transitions]))
            node.assert_hits = asserts
            if asserts != None:
                asserts, idx = asserts
                for agent in node.agent:
                    node.trace[agent] = node.trace[agent][:(idx + 1) * 2]
                continue

            transit_map = {k: list(l) for k, l in itertools.groupby(all_possible_transitions, key=lambda p:p[0])}
            transit_agents = transit_map.keys()
            # pp(("transit agents", transit_agents))
            if self.config.incremental:
                transit_ind = max(l[-2][-1] for l in all_possible_transitions) if len(all_possible_transitions) > 0 else len(list(node.trace.values())[0])
                for agent_id in node.agent:
                    transition = transit_map[agent_id] if agent_id in transit_agents else []
                    if agent_id in cached_tubes:
                        cached_tubes[agent_id].transitions.extend(convert_reach_trans(agent_id, transit_agents, node.init, transition, transit_ind))
                        pre_len = len(cached_tubes[agent_id].transitions)
                        cached_tubes[agent_id].transitions = dedup(cached_tubes[agent_id].transitions, lambda i: (i.mode, i.dest, i.inits))
                        # pp(("dedup!", pre_len, len(cached_tubes[agent_id].transitions)))
                    else:
                        self.trans_cache.add_tube(agent_id, combined_inits, node, transit_agents, transition, transit_ind, run_num)

            # Check if multiple agents can transit at the same time
            max_end_idx = 0
            transition_dict = {}
            for transition in all_possible_transitions:
                if transition[0] not in transition_dict:
                    transition_dict[transition[0]] = [transition]
                else:
                    transition_dict[transition[0]].append(transition)
            transition_list = list(transition_dict.values())
            combined_transitions = list(itertools.product(*transition_list))
            aligned_transitions = self.align_transitions(combined_transitions)

            for all_agent_transition in aligned_transitions:
                # Each transition will contain a list of rectangles and their corresponding indexes in the original list
                # if len(transition) != 6:
                #     pp(("weird trans", transition))
                if not all_agent_transition:
                    continue
                next_node_mode = copy.deepcopy(node.mode)
                next_node_static = node.static
                next_node_uncertain_param = node.uncertain_param
                start_idx, end_idx = all_agent_transition[0][4][0], all_agent_transition[0][4][-1]
                truncated_trace = {}
                for agent_idx in node.agent:
                    truncated_trace[agent_idx] = node.trace[agent_idx][start_idx*2:]
                if end_idx > max_end_idx:
                    max_end_idx = end_idx
                next_node_start_time = list(truncated_trace.values())[0][0][0]
                next_node_agent = node.agent
                next_node_init = {}
                next_node_trace = {}
                
                transit_agent = []
                for transition in all_agent_transition:
                    transit_agent_idx, src_mode, dest_mode, next_init, idx, path = transition
                    # start_idx, end_idx = idx[0], idx[-1]

                    if dest_mode is None:
                        continue

                    next_node_mode[transit_agent_idx] = dest_mode
                    # for agent_idx in next_node_agent:
                        # if agent_idx == transit_agent_idx:
                    next_node_init[transit_agent_idx] = next_init
                    transit_agent.append(transit_agent_idx)
                        # else:
                        #     # pp(("infer init", agent_idx, next_node_init[agent_idx]))
                for agent_idx in next_node_agent :
                    if agent_idx not in transit_agent:
                        next_node_init[agent_idx] = [[truncated_trace[agent_idx][0][1:], truncated_trace[agent_idx][1][1:]]]
                        next_node_trace[agent_idx] = truncated_trace[agent_idx]

                tmp = AnalysisTreeNode(
                    trace=next_node_trace,
                    init=next_node_init,
                    mode=next_node_mode,
                    static = next_node_static,
                    uncertain_param = next_node_uncertain_param,
                    agent=next_node_agent,
                    assert_hits = {},
                    child=[],
                    start_time=round(next_node_start_time, 10),
                    type='reachtube'
                )
                node.child.append(tmp)
                verification_queue.append(tmp)

            """Truncate trace of current node based on max_end_idx"""
            """Only truncate when there's transitions"""
            if all_possible_transitions:
                for agent_idx in node.agent:
                    node.trace[agent_idx] = node.trace[agent_idx][:(
                        max_end_idx+1)*2]

        self.reachtube_tree = AnalysisTree(root)
        # print(f">>>>>>>> Number of calls to reachability engine: {num_calls}")
        # print(f">>>>>>>> Number of transitions happening: {num_transitions}")
        self.num_transitions = num_transitions

        return self.reachtube_tree

    def align_transitions(self, all_transition_list):
        return all_transition_list
