# this file contains helper functions for mp

import numpy as np
import portion
from typing import Any, Tuple, Dict, Callable, List
from verse.analysis import AnalysisTreeNode, AnalysisTree

def sample(init_dict):
    """
    TODO:   given the initial set,
            generate multiple initial points located in the initial set
            as the input of multiple simulation.
            note that output should be formatted correctly and every point should be in inital set.
            refer the following sample code to write your code. 
    """
    print(init_dict)
    ############## Your Code Start Here ##############
    sample_dict_list = []
    num_sample = 50

    np.random.seed(2023)
    for i in range(num_sample):
        sample_dict={}
        for agent in init_dict:
            point = np.random.uniform(init_dict[agent][0], init_dict[agent][1]).tolist()
            sample_dict[agent] = point
        sample_dict_list.append(sample_dict)
    ############## Your Code End Here ##############
    print(sample_dict_list)

    return sample_dict_list

def combine_tree(tree_list: List[AnalysisTree]):
    combined_trace={}
    for tree in tree_list:
        for node in tree.nodes:
            for agent_id in node.agent:
                traces = node.trace
                trace = np.array(traces[agent_id])
                if agent_id not in combined_trace:
                    combined_trace[agent_id]={}
                for i in range (0, len(trace), 2):
                    step = round(trace[i][0], 3)
                    if step not in combined_trace[agent_id]:
                        combined_trace[agent_id][step]=[trace[i], trace[i+1]]
                    else:
                        lower = np.min([combined_trace[agent_id][step][0],trace[i]], 0)
                        upper = np.max([combined_trace[agent_id][step][1],trace[i+1]], 0)
                        combined_trace[agent_id][step]=[lower, upper]

    final_trace = {agent_id:np.array(list(combined_trace[agent_id].values())).flatten().reshape((-1, trace[i].size)).tolist() for agent_id in combined_trace}
    print(final_trace)
