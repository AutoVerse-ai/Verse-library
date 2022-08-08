from typing import List
import copy

import numpy as np

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree
from verse.analysis.dryvr import calc_bloated_tube, SIMTRACENUM


class Verifier:
    def __init__(self):
        self.reachtube_tree = None
        self.unsafe_set = None
        self.verification_result = None

    def caculate_full_bloated_tube(
        self,
        mode_label,
        initial_set,
        time_horizon,
        time_step,
        sim_func,
        bloating_method,
        kvalue,
        sim_trace_num,
        combine_seg_length = 1000,
        guard_checker=None,
        guard_str="",
        lane_map = None
    ):
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


    def compute_full_reachtube(
        self,
        init_list: List[float],
        init_mode_list: List[str],
        static_list: List[str],
        agent_list,
        transition_graph,
        time_horizon,
        time_step,
        lane_map,
        init_seg_length,
        verify_method
    ):
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static = {},
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
            root.agent[agent.id] = agent
            root.type = 'reachtube'
        verification_queue = []
        verification_queue.append(root)
        while verification_queue != []:
            node: AnalysisTreeNode = verification_queue.pop(0)
            print(node.start_time, node.mode)
            remain_time = round(time_horizon - node.start_time, 10)
            if remain_time <= 0:
                continue
            # For reachtubes not already computed
            # TODO: can add parallalization for this loop
            for agent_id in node.agent:
                if agent_id not in node.trace:
                    # Compute the trace starting from initial condition
                    mode = node.mode[agent_id]
                    init = node.init[agent_id]
                    # trace = node.agent[agent_id].TC_simulate(mode, init, remain_time,lane_map)
                    # trace[:,0] += node.start_time
                    # node.trace[agent_id] = trace.tolist()

                    cur_bloated_tube = self.caculate_full_bloated_tube(mode,
                                        init,
                                        remain_time,
                                        time_step, 
                                        node.agent[agent_id].TC_simulate,
                                        verify_method,
                                        100,
                                        SIMTRACENUM,
                                        combine_seg_length=init_seg_length,
                                        lane_map = lane_map
                                        )
                    trace = np.array(cur_bloated_tube)
                    trace[:, 0] += node.start_time
                    node.trace[agent_id] = trace.tolist()
                    # print("here")
            
            # Get all possible transitions to next mode
            asserts, all_possible_transitions = transition_graph.get_transition_verify_new(node)
            if asserts != None:
                asserts, idx = asserts
                for agent in node.agent:
                    node.trace[agent] = node.trace[agent][:(idx + 1) * 2]
                node.assert_hits = asserts
                continue

            max_end_idx = 0
            for transition in all_possible_transitions:
                # Each transition will contain a list of rectangles and their corresponding indexes in the original list
                transit_agent_idx, src_mode, dest_mode, next_init, idx = transition
                start_idx, end_idx = idx[0], idx[-1]

                truncated_trace = {}
                for agent_idx in node.agent:
                    truncated_trace[agent_idx] = node.trace[agent_idx][start_idx*2:]
                if end_idx > max_end_idx:
                    max_end_idx = end_idx

                if dest_mode is None:
                    continue

                next_node_mode = copy.deepcopy(node.mode)
                next_node_static = node.static
                next_node_mode[transit_agent_idx] = dest_mode
                next_node_agent = node.agent
                next_node_start_time = list(truncated_trace.values())[0][0][0]
                next_node_init = {}
                next_node_trace = {}
                for agent_idx in next_node_agent:
                    if agent_idx == transit_agent_idx:
                        next_node_init[agent_idx] = next_init
                    else:
                        next_node_trace[agent_idx] = truncated_trace[agent_idx]

                tmp = AnalysisTreeNode(
                    trace=next_node_trace,
                    init=next_node_init,
                    mode=next_node_mode,
                    static = next_node_static,
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
        return self.reachtube_tree
