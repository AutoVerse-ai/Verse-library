from typing import List, Dict
import copy

import numpy as np

from ourtool.agents.base_agent import BaseAgent
from ourtool.analysis.analysis_tree_node import AnalysisTreeNode
from ourtool.dryvr.core.dryvrcore import calc_bloated_tube
import ourtool.dryvr.common.config as userConfig

class Verifier:
    def __init__(self):
        self.reachtube_tree_root = None
        self.unsafe_set = None
        self.verification_result = None 

    def compute_full_reachtube(
        self, 
        init_list, 
        init_mode_list, 
        agent_list:List[BaseAgent], 
        transition_graph, 
        time_horizon, 
        lane_map
    ):
        root = AnalysisTreeNode()
        for i, agent in enumerate(agent_list):
            root.init[agent.id] = init_list[i]
            init_mode = [elem.name for elem in init_mode_list[i]]
            init_mode =','.join(init_mode)
            root.mode[agent.id] = init_mode 
            root.agent[agent.id] = agent 
        self.reachtube_tree_root = root 
        verification_queue = []
        verification_queue.append(root)
        while verification_queue != []:
            node:AnalysisTreeNode = verification_queue.pop(0)
            print(node.mode)
            remain_time = time_horizon - node.start_time 
            if remain_time <= 0:
                continue 
            # For reachtubes not already computed
            for agent_id in node.agent:
                if agent_id not in node.trace:
                    # Compute the trace starting from initial condition
                    mode = node.mode[agent_id]
                    init = node.init[agent_id]
                    # trace = node.agent[agent_id].TC_simulate(mode, init, remain_time,lane_map)
                    # trace[:,0] += node.start_time
                    # node.trace[agent_id] = trace.tolist()

                    cur_bloated_tube = calc_bloated_tube(mode,
                                        init,
                                        remain_time,
                                        node.agent[agent_id].TC_simulate,
                                        'GLOBAL',
                                        None,
                                        userConfig.SIMTRACENUM,
                                        lane_map = lane_map
                                        )
                    trace = np.array(cur_bloated_tube)
                    trace[:,0] += node.start_time
                    node.trace[agent_id] = trace.tolist()
                    print("here")

            trace_length = int(len(list(node.trace.values())[0])/2)
            guard_hits = []
            for idx in range(0,trace_length):
                # For each trace, check with the guard to see if there's any possible transition
                # Store all possible transition in a list
                # A transition is defined by (agent, src_mode, dest_mode, corresponding reset, transit idx)
                # Here we enforce that only one agent transit at a time
                all_agent_state = {}
                for agent_id in node.agent:
                    all_agent_state[agent_id] = (node.trace[agent_id][idx*2:idx*2+2], node.mode[agent_id])
                guards, resets, is_contain = transition_graph.check_guard_hit(all_agent_state)
                if possible_transitions != []:
                    for agent_idx, src_mode, dest_mode, next_init, contained in possible_transitions:
                        transitions.append((agent_idx, src_mode, dest_mode, next_init, idx))
                        any_contained = any_contained or contained
                    if any_contained:
                        break
        pass

