from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict
import copy, itertools, functools, pprint
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings
import ast
import ray, time
from verse.parser import unparse
import json

from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree, TraceType
from verse.analysis.dryvr import calc_bloated_tube, SIMTRACENUM
from verse.analysis.incremental import (
    ReachTubeCache,
    TubeCache,
    convert_reach_trans,
    to_simulate,
    combine_all,
)
from verse.analysis.incremental import CachedRTTrans, combine_all, reach_trans_suit
from verse.utils.utils import dedup
from verse.map.lane_map import LaneMap
from verse.parser.parser import find, ModePath, unparse
from verse.agents.base_agent import BaseAgent
from verse.automaton import GuardExpressionAst, ResetExpression
import json
import os
import pyvista as pv
from verse.plotter.plotter3D import *
pp = functools.partial(pprint.pprint, compact=True, width=130)

PathDiffs = List[Tuple[BaseAgent, ModePath]]
EGO, OTHERS = "ego", "others"


class ReachabilityMethod(Enum):
    DRYVR = auto()
    NEU_REACH = auto()
    MIXMONO_CONT = auto()
    MIXMONO_DISC = auto()
    DRYVR_DISC = auto()
    STAR_SETS = auto()


@dataclass
class ReachConsts:
    time_step: float
    lane_map: LaneMap
    init_seg_length: int
    reachability_method: ReachabilityMethod
    run_num: int
    past_runs: List[AnalysisTree]
    sensor: "BaseSensor"
    agent_dict: Dict


def save_node(node):
    filename = "nodes.json"
    
    # Load existing data if the file exists
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # Start fresh if file is corrupted
    else:
        data = []

    # Append new node data
    data.append({ "id": node.id, "trace": node.trace })

    # Write back the updated data
    with open(filename, "w") as file:
        print("dumping")
        json.dump(data, file, indent=4)

color_map = {}

int_to_color = {1:'b', 2:'r', 3:'g'}


class Verifier:
    def __init__(self, config):
        self.reachtube_tree = None
        self.cache = TubeCache()
        self.trans_cache = ReachTubeCache()
        self.tube_cache_hits = (0, 0)
        self.trans_cache_hits = (0, 0)
        self.config = config
        self.compute_full_reachtube_step_remote = ray.remote(Verifier.compute_full_reachtube_step)

    def check_cache_bloated_tube_stars(
        self,
        agent_id,
        mode_label,
        initial_set,
        combine_seg_length=1000,
    ):
        """
        Check the bloated tubes cached already

        :param TBA
        :return:    the combined bloated tube with all cached tube segs
                    a list of indexs of missing segs
        """
        print("in stars check cache")
        missing_seg_idx_list = []
        res_tube = None
        tube_length = 0
        for combine_seg_idx in range(0, len(initial_set), combine_seg_length):
            rect_seg = initial_set[combine_seg_idx : combine_seg_idx + combine_seg_length]
            combined_rect = None
            for rect in rect_seg:
                rect = np.array(rect)
                if combined_rect is None:
                    combined_rect = rect
                else:
                    combined_rect[0, :] = np.minimum(combined_rect[0, :], rect[0, :])
                    combined_rect[1, :] = np.maximum(combined_rect[1, :], rect[1, :])
            combined_rect = combined_rect.tolist()
            if self.config.incremental:
                cached = self.cache.check_hit(agent_id, mode_label, combined_rect)
                if cached != None:
                    self.tube_cache_hits = self.tube_cache_hits[0] + 1, self.tube_cache_hits[1]
                    # print('cache', agent_id, time_horizon, self.tube_cache_hits )
                else:
                    self.tube_cache_hits = self.tube_cache_hits[0], self.tube_cache_hits[1] + 1
                    # print('noncache', agent_id, time_horizon, self.tube_cache_hits )
            else:
                cached = None
            if cached != None:
                cur_bloated_tube = cached.tube
            else:
                missing_seg_idx_list.append(combine_seg_idx)
                continue
            # FIXME
            if res_tube is None:
                res_tube = cur_bloated_tube
                tube_length = cur_bloated_tube.shape[0]
            else:
                cur_bloated_tube = cur_bloated_tube[: tube_length - combine_seg_idx * 2, :]
                # Handle Lower Bound
                res_tube[combine_seg_idx * 2 :: 2, 1:] = np.minimum(
                    res_tube[combine_seg_idx * 2 :: 2, 1:], cur_bloated_tube[::2, 1:]
                )
                # Handle Upper Bound
                res_tube[combine_seg_idx * 2 + 1 :: 2, 1:] = np.maximum(
                    res_tube[combine_seg_idx * 2 + 1 :: 2, 1:], cur_bloated_tube[1::2, 1:]
                )
        return res_tube, missing_seg_idx_list

    @staticmethod
    def calculate_full_bloated_tube_stars(
        agent_id,
        cached_tube_info,
        incremental,
        mode_label,
        initial_set,
        time_horizon,
        time_step,
        sim_func,
        params,
        kvalue,
        sim_trace_num,
        combine_seg_length=1000,
        guard_checker=None,
        guard_str="",
        lane_map=None,
        pca=True,
    ):
        #this should return a list of stars for one time step along the horizon
        """
        Get the full bloated tube. use cached tubes, calculate noncached tubes

        :param TBA
        :return:    the full bloated tube
                    cache to be updated
        """
        bloating_method = "PW"
        if "bloating_method" in params:
            bloating_method = params["bloating_method"]
        cache_tube_updates = []
        if incremental:
            cached_tube, missing_seg_idx_list = cached_tube_info
        else:
            cached_tube, missing_seg_idx_list = None, range(0, len(initial_set), combine_seg_length)
        res_tube = cached_tube
        if res_tube is None:
            tube_length = 0
        else:
            tube_length = res_tube.shape[0]
        #What is this and why does the same star set appear twice?    
        print(missing_seg_idx_list)

        for combine_seg_idx in missing_seg_idx_list:
            rect_seg = initial_set[combine_seg_idx : combine_seg_idx + combine_seg_length]
            # print("len of rect_sec:")
            # print(len(rect_seg))
            #combined_rect = None
            #for rect in rect_seg:
            #    rect = np.array(rect)
            #    if combined_rect is None:
            #        combined_rect = rect
            #    else:
            #        combined_rect[0, :] = np.minimum(combined_rect[0, :], rect[0, :])
            #        combined_rect[1, :] = np.maximum(combined_rect[1, :], rect[1, :])
            from verse.stars.starset import StarSet
            combined_star = initial_set[0]
            if len(rect_seg) > 1:
                combined_star = StarSet.combine_stars(rect_seg)
            #print(combined_rect)
            #print("done with combined rec")
        
        #TODO: what to do with the list of initial set? Some sort of combining?

        #fix this!
        #print("TODO: is the initial set correct?")
        #print(initial_set)
        #breakpoint()
        #KB HERE: combine the stars into a rectangle
            #print(initial_set)
            #inital_star = initial_set[0]

            ### add a parameter here, either bool or create an enum
            #To use Alex's version, change to "calc_reach_tube", and uncomment bloating method, kvalue, sim_trace_num
            #To use Katherine's version, change to "calc_reach_tube_linear", and comment bloating method, kvalue, sim_trace_num
            # reach_tube = combined_star.calc_reach_tube_linear(
            reach_tube = combined_star.calc_reach_tube(
            mode_label,
            time_horizon,
            time_step,
            sim_func,
            bloating_method,
            kvalue,
            sim_trace_num,
            lane_map=lane_map,
            pca=pca,
            )

           
            if incremental:
                cache_tube_updates.append((agent_id, mode_label, combined_rect, cur_bloated_tube))
                #EXISTING CODE, NOT SURE WHAT TO DO:
            old_code = False
            if old_code:
                if res_tube is None:
                    res_tube = cur_bloated_tube
                    tube_length = cur_bloated_tube.shape[0]
                else:
                    if not tube_length <= 2 * combine_seg_idx:
                        
                        cur_bloated_tube = cur_bloated_tube[: tube_length - combine_seg_idx * 2, :]
                        # Handle Lower Bound
                        res_tube[combine_seg_idx * 2 :: 2, 1:] = np.minimum(
                            res_tube[combine_seg_idx * 2 :: 2, 1:], cur_bloated_tube[::2, 1:]
                        )
                        # Handle Upper Bound
                        res_tube[combine_seg_idx * 2 + 1 :: 2, 1:] = np.maximum(
                            res_tube[combine_seg_idx * 2 + 1 :: 2, 1:], cur_bloated_tube[1::2, 1:]
                        )
        
        return reach_tube, cache_tube_updates

    def check_cache_bloated_tube(
        self,
        agent_id,
        mode_label,
        initial_set,
        combine_seg_length=1000,
    ):
        """
        Check the bloated tubes cached already

        :param TBA
        :return:    the combined bloated tube with all cached tube segs
                    a list of indexs of missing segs
        """
        missing_seg_idx_list = []
        res_tube = None
        tube_length = 0
        for combine_seg_idx in range(0, len(initial_set), combine_seg_length):
            rect_seg = initial_set[combine_seg_idx : combine_seg_idx + combine_seg_length]
            combined_rect = None
            for rect in rect_seg:
                rect = np.array(rect)
                if combined_rect is None:
                    combined_rect = rect
                else:
                    combined_rect[0, :] = np.minimum(combined_rect[0, :], rect[0, :])
                    combined_rect[1, :] = np.maximum(combined_rect[1, :], rect[1, :])
            combined_rect = combined_rect.tolist()
            if self.config.incremental:
                cached = self.cache.check_hit(agent_id, mode_label, combined_rect)
                if cached != None:
                    self.tube_cache_hits = self.tube_cache_hits[0] + 1, self.tube_cache_hits[1]
                    # print('cache', agent_id, time_horizon, self.tube_cache_hits )
                else:
                    self.tube_cache_hits = self.tube_cache_hits[0], self.tube_cache_hits[1] + 1
                    # print('noncache', agent_id, time_horizon, self.tube_cache_hits )
            else:
                cached = None
            if cached != None:
                cur_bloated_tube = cached.tube
            else:
                missing_seg_idx_list.append(combine_seg_idx)
                continue
            # FIXME
            if res_tube is None:
                res_tube = cur_bloated_tube
                tube_length = cur_bloated_tube.shape[0]
            else:
                cur_bloated_tube = cur_bloated_tube[: tube_length - combine_seg_idx * 2, :]
                # Handle Lower Bound
                res_tube[combine_seg_idx * 2 :: 2, 1:] = np.minimum(
                    res_tube[combine_seg_idx * 2 :: 2, 1:], cur_bloated_tube[::2, 1:]
                )
                # Handle Upper Bound
                res_tube[combine_seg_idx * 2 + 1 :: 2, 1:] = np.maximum(
                    res_tube[combine_seg_idx * 2 + 1 :: 2, 1:], cur_bloated_tube[1::2, 1:]
                )
        return res_tube, missing_seg_idx_list

    @staticmethod
    def calculate_full_bloated_tube_simple(
        agent_id,
        cached_tube_info,
        incremental,
        mode_label,
        initial_set,
        time_horizon,
        time_step,
        sim_func,
        params,
        kvalue,
        sim_trace_num,
        combine_seg_length=1000,
        guard_checker=None,
        guard_str="",
        lane_map=None,
    ):
        """
        Get the full bloated tube. use cached tubes, calculate noncached tubes

        :param TBA
        :return:    the full bloated tube
                    cache to be updated
        """
        # Handle Parameters
        bloating_method = "PW"
        if "bloating_method" in params:
            bloating_method = params["bloating_method"]
        cache_tube_updates = []
        if incremental:
            cached_tube, missing_seg_idx_list = cached_tube_info
        else:
            cached_tube, missing_seg_idx_list = None, range(0, len(initial_set), combine_seg_length)
        res_tube = cached_tube
        if res_tube is None:
            tube_length = 0
        else:
            tube_length = res_tube.shape[0]
        for combine_seg_idx in missing_seg_idx_list:
            rect_seg = initial_set[combine_seg_idx : combine_seg_idx + combine_seg_length]
            combined_rect = None
            for rect in rect_seg:
                rect = np.array(rect)
                if combined_rect is None:
                    combined_rect = rect
                else:
                    combined_rect[0, :] = np.minimum(combined_rect[0, :], rect[0, :])
                    combined_rect[1, :] = np.maximum(combined_rect[1, :], rect[1, :])
            combined_rect = combined_rect.tolist()
            cur_bloated_tube = calc_bloated_tube(
                mode_label,
                combined_rect,
                time_horizon,
                time_step,
                sim_func,
                bloating_method,
                kvalue,
                sim_trace_num,
                lane_map=lane_map,
            )
            if incremental:
                cache_tube_updates.append((agent_id, mode_label, combined_rect, cur_bloated_tube))
            if res_tube is None:
                res_tube = cur_bloated_tube
                tube_length = cur_bloated_tube.shape[0]
            else:
                if tube_length <= 2 * combine_seg_idx:
                    break
                cur_bloated_tube = cur_bloated_tube[: tube_length - combine_seg_idx * 2, :]
                # Handle Lower Bound
                res_tube[combine_seg_idx * 2 :: 2, 1:] = np.minimum(
                    res_tube[combine_seg_idx * 2 :: 2, 1:], cur_bloated_tube[::2, 1:]
                )
                # Handle Upper Bound
                res_tube[combine_seg_idx * 2 + 1 :: 2, 1:] = np.maximum(
                    res_tube[combine_seg_idx * 2 + 1 :: 2, 1:], cur_bloated_tube[1::2, 1:]
                )
        return res_tube.tolist(), cache_tube_updates

    @staticmethod
    def compute_full_reachtube_step(
        config: "ScenarioConfig",
        cached_trans_tubes: Dict[str, CachedRTTrans],
        cached_tubes: Dict[str, Tuple],
        node: AnalysisTreeNode,
        old_node_id: Optional[Tuple[int, int]],
        later: int,
        remain_time: float,
        consts: ReachConsts,
        max_height: int,
        ax: pv.Plotter,

        params={},
    ) -> Tuple[int, int, List[AnalysisTreeNode], Dict[str, TraceType], list]:
        # t = timeit.default_timer()
        if config.print_level >= 1:
            print("=============================================================")
            print(f"node {node.id} start: {node.start_time}")
            # print(f"node id: {node.id}")
            print(node.mode)
        cache_trans_tube_updates = []
        cache_tube_updates = []
        if max_height == None:
            max_height = float("inf")
        # combined_inits = {a: combine_all(inits) for a, inits in node.init.items()}
        next_nodes = []
        for agent_id in node.agent:
            mode = node.mode[agent_id]
            inits = node.init[agent_id]
            # combined = combine_all(inits)
            if agent_id not in node.trace:
                # Compute the trace starting from initial condition
                uncertain_param = node.uncertain_param[agent_id]
                if consts.reachability_method == ReachabilityMethod.DRYVR:
                    # pp(('tube', agent_id, mode, inits))
                    (
                        cur_bloated_tube,
                        cache_tube_update,
                    ) = Verifier.calculate_full_bloated_tube_simple(
                        agent_id,
                        cached_tubes[agent_id] if config.incremental else None,
                        config.incremental,
                        mode,
                        inits,
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id].TC_simulate,
                        params,
                        100,
                        SIMTRACENUM,
                        combine_seg_length=consts.init_seg_length,
                        lane_map=consts.lane_map,
                    )
                    if config.incremental:
                        cache_tube_updates.extend(cache_tube_update)
                elif consts.reachability_method == ReachabilityMethod.STAR_SETS:
                    # pp(('tube', agent_id, mode, inits))
                    (
                        cur_bloated_tube,
                        cache_tube_update,
                    ) = Verifier.calculate_full_bloated_tube_stars(
                        agent_id,
                        cached_tubes[agent_id] if config.incremental else None,
                        config.incremental,
                        mode,
                        inits,
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id].TC_simulate,
                        params,
                        100,
                        SIMTRACENUM,
                        combine_seg_length=consts.init_seg_length,
                        lane_map=consts.lane_map,
                        pca=config.pca
                    )
                    if config.incremental:
                        cache_tube_updates.extend(cache_tube_update)
                
                elif consts.reachability_method == ReachabilityMethod.DRYVR_DISC:
                    from verse.analysis.dryvr_disc import calc_bloated_tube_dryvr
                    bloating_method = 'PW'
                    if 'bloating_method' in params:
                        bloating_method = params['bloating_method']
                    traces = None
                    if 'traces' in params:
                        traces = params['traces']     
                    if 'sim_trace_num' in params:
                        sim_trace_num = params['sim_trace_num']
                    else:
                        sim_trace_num = SIMTRACENUM              
                    init = inits[0]
                    if isinstance(init, np.ndarray):
                        init = init.tolist()
                    cur_bloated_tube = calc_bloated_tube_dryvr(
                        mode,
                        init,
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id].TC_simulate,
                        bloating_method, 
                        100,
                        sim_trace_num,
                        lane_map = consts.lane_map,
                        traces = traces
                    )
                    if isinstance(cur_bloated_tube, np.ndarray):
                        cur_bloated_tube = cur_bloated_tube.tolist()
                elif consts.reachability_method == ReachabilityMethod.NEU_REACH:
                    # pylint: disable=E0401
                    from verse.analysis.NeuReach.NeuReach_onestep_rect import postCont
                    # pylint: enable=E0401
                    cur_bloated_tube = postCont(
                        mode,
                        inits[0],
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id].TC_simulate,
                        consts.lane_map,
                        params,
                    )
                elif consts.reachability_method == ReachabilityMethod.MIXMONO_CONT:
                    from verse.analysis.mixmonotone import (
                        calculate_bloated_tube_mixmono_cont,
                    )
                    cur_bloated_tube = calculate_bloated_tube_mixmono_cont(
                        mode,
                        inits,
                        uncertain_param,
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id],
                        consts.lane_map,
                    )
                elif consts.reachability_method == ReachabilityMethod.MIXMONO_DISC:
                    from verse.analysis.mixmonotone import (
                        calculate_bloated_tube_mixmono_disc,
                    )
                    cur_bloated_tube = calculate_bloated_tube_mixmono_disc(
                        mode,
                        inits,
                        uncertain_param,
                        remain_time,
                        consts.time_step,
                        node.agent[agent_id],
                        consts.lane_map,
                    )
                # num_calls += 1
                trace = np.array(cur_bloated_tube)
                trace[:, 0] += node.start_time
                node.trace[agent_id] = trace.tolist()
        # pp(("cached tubes", cached_tubes.keys()))
        new_cache, paths_to_sim = {}, []
        if old_node_id != None:
            if old_node_id[0] != consts.run_num:
                old_node = find(
                    consts.past_runs[old_node_id[0]].nodes, lambda n: n.id == old_node_id[1]
                )
                assert old_node != None
                new_cache, paths_to_sim = to_simulate(
                    old_node.agent, node.agent, cached_trans_tubes
                )
                # pp(("to sim", new_cache.keys(), len(paths_to_sim)))

        # Get all possible transitions to next mode
        asserts, all_possible_transitions = Verifier.get_transition_verify_opt(
            config, new_cache, paths_to_sim, node, consts.lane_map, consts.sensor
        )
        node.assert_hits = asserts

        if not config.unsafe_continue and asserts != None:
            asserts, idx = asserts
            if not consts.reachability_method == ReachabilityMethod.STAR_SETS:
                for agent in node.agent:
                    node.trace[agent] = node.trace[agent][: (idx + 1) * 2]
            else:
                for agent in node.agent:
                    node.trace[agent] = node.trace[agent][: (idx + 1)]

            #print("writing data")
            data = []
            ids = []
            colors = []
            for agent_id in node.trace:
                ids.append(agent_id)
                if agent_id not in color_map:
                    color_map[agent_id] = len(color_map) +1

            l  = len(node.trace[ids[0]])
            for i in range(0, l, 2):
                for agent_id in ids:
                    trace = node.trace[agent_id]
                    data.append([trace[i], trace[i + 1]])
                    colors.append(int_to_color[color_map[agent_id]])
            print("COLORS", colors)


            plot3dReachtubeSingle(data, 0, 1, 2, ax, colors,  edge=True)
            #ax.render()
            return (
                node.id,
                later,
                next_nodes,
                node.trace,
                asserts,
                cache_tube_updates,
                cache_trans_tube_updates,
            )

        # pp(("transitions:", [(t[0], t[2]) for t in all_possible_transitions]))
        transit_map = {
            k: list(l) for k, l in itertools.groupby(all_possible_transitions, key=lambda p: p[0])
        }
        transit_agents = transit_map.keys()
        # pp(("transit agents", transit_agents))
        if config.incremental:
            transit_ind = (
                min(l[-2][-1] for l in all_possible_transitions)
                if len(all_possible_transitions) > 0
                else len(list(node.trace.values())[0])
            )
            for agent_id in node.agent:
                transition = transit_map[agent_id] if agent_id in transit_agents else []
                cache_trans_tube_updates.append(
                    (
                        agent_id not in cached_trans_tubes,
                        agent_id,
                        transit_agents,
                        transition,
                        transit_ind,
                        consts.run_num,
                    )
                )

        if node.height >= max_height:
            print("max depth reached")
            return (
                node.id,
                later,
                next_nodes,
                node.trace,
                asserts,
                cache_tube_updates,
                cache_trans_tube_updates,
            )
        max_end_idx = 0
        for transition in all_possible_transitions:
            # Each transition will contain a list of rectangles and their corresponding indexes in the original list
            # if len(transition) != 6:
            #     pp(("weird trans", transition))
            transit_agent_idx, src_mode, dest_mode, next_init, idx, path = transition
            start_idx, end_idx = idx[0], idx[-1]

            truncated_trace = {}
            if not consts.reachability_method == ReachabilityMethod.STAR_SETS:
                for agent_idx in node.agent:
                    truncated_trace[agent_idx] = node.trace[agent_idx][start_idx * 2 :]
            else:
               for agent_idx in node.agent:
                    truncated_trace[agent_idx] = node.trace[agent_idx][start_idx :]                
            if end_idx > max_end_idx:
                max_end_idx = end_idx

            if dest_mode is None:
                continue

            next_node_mode = copy.deepcopy(node.mode)
            next_node_static = node.static
            next_node_uncertain_param = node.uncertain_param
            next_node_mode[transit_agent_idx] = dest_mode
            next_node_agent = node.agent
            next_node_start_time = list(truncated_trace.values())[0][0][0]
            next_node_init = {}
            next_node_trace = {}
            for agent_idx in next_node_agent:
                if agent_idx == transit_agent_idx:
                    next_node_init[agent_idx] = next_init
                else:
                    next_node_init[agent_idx] = [
                        [truncated_trace[agent_idx][0][1:], truncated_trace[agent_idx][1][1:]]
                    ]
                    # pp(("infer init", agent_idx, next_node_init[agent_idx]))
                    next_node_trace[agent_idx] = truncated_trace[agent_idx]

            tmp = node.new_child(
                trace=next_node_trace,
                init=next_node_init,
                mode=next_node_mode,
                start_time=round(next_node_start_time, 10),
                id=-1,
            )
            next_nodes.append(tmp)

        """Truncate trace of current node based on max_end_idx"""
        """Only truncate when there's transitions"""
        if not consts.reachability_method == ReachabilityMethod.STAR_SETS:
            if all_possible_transitions:
                for agent_idx in node.agent:
                    node.trace[agent_idx] = node.trace[agent_idx][: (max_end_idx + 1) * 2]
        else:
            if all_possible_transitions:
                for agent_idx in node.agent:
                    node.trace[agent_idx] = node.trace[agent_idx][: (max_end_idx + 1)]
       

        # with open("nodes.jsonl", "a") as file:
        #     print("dumping")
        #     try:
        #         file.write(json.dumps({ "id": node.id, "trace": node.trace }) + "\n")
        #     except json.JSONDecodeError:
        #         print("Error")
        # if(node.start_time == 29.7):
        #     print(len(node.trace['car']))
        #     print(len(node.trace['pedestrian']))
        #     with open("nodes.jsonl", "a") as file:
        #         print("dumping")
        #         try:
        #             file.write(json.dumps({ "id": node.id, "trace": node.trace['car'] }) + "\n")
        #         except json.JSONDecodeError:
        #             print("Error")
        colors = []

        data = []
        ids = []
        for agent_id in node.trace:
            ids.append(agent_id)
            if agent_id not in color_map:
                color_map[agent_id] = len(color_map) +1


        l  = len(node.trace[ids[0]])
        for i in range(0, l, 2):
            for agent_id in ids:
                trace = node.trace[agent_id]
                data.append([trace[i], trace[i + 1]])
                colors.append(int_to_color[color_map[agent_id]])

        plot3dReachtubeSingle(data, 0, 1, 2, ax, colors, edge=True)
        ax.render()
            
            #ax.show()



       

        return (
            node.id,
            later,
            next_nodes,
            node.trace,
            asserts,
            cache_tube_updates,
            cache_trans_tube_updates,
        )

    def proc_result(
        self,
        id,
        later,
        next_nodes,
        traces,
        assert_hits,
        cache_tube_updates,
        cache_trans_tube_updates,
        max_height
        ):
        # t = timeit.default_timer()
        # print('get id: ', id, self.nodes[id].start_time)
        done_node: AnalysisTreeNode = self.nodes[id]
        done_node.child = next_nodes
        done_node.trace = traces
        done_node.assert_hits = assert_hits
        last_id = self.nodes[-1].id
        # print([x.id for x in self.nodes])
        for i, next_node in enumerate(next_nodes):
            next_node.id = i + 1 + last_id
            later = 0 if i == 0 else 1
            if done_node.height <= max_height:
                self.verification_queue.append((next_node, later))
        self.verification_queue.sort(key=lambda p: p[1:])
        if done_node.height <= max_height:
            self.nodes.extend(next_nodes)
        combined_inits = {a: combine_all(inits,self.config.reachability_method == ReachabilityMethod.STAR_SETS) for a, inits in done_node.init.items()}
        for (
            new,
            aid,
            transit_agents,
            transition,
            transition_idx,
            run_num,
        ) in cache_trans_tube_updates:
            cached = self.trans_cache.check_hit(
                aid, done_node.mode[aid], combine_all(done_node.init[aid]), done_node.init
            )
            if new and not cached:
                self.trans_cache.add_tube(
                    aid,
                    combined_inits,
                    done_node,
                    transit_agents,
                    transition,
                    transition_idx,
                    run_num,
                )
                self.num_cached += 1
            else:
                assert cached != None
                cached.transitions.extend(
                    convert_reach_trans(
                        aid, transit_agents, done_node.init, transition, transition_idx
                    )
                )
                cached.transitions = dedup(cached.transitions, lambda i: (i.mode, i.dest, i.inits))
                cached.node_ids.add((run_num, done_node.id))
        for agent_id, mode_label, combined_rect, cur_bloated_tube in cache_tube_updates:
            self.cache.add_tube(agent_id, mode_label, combined_rect, cur_bloated_tube)
        # print(f"proc dur {timeit.default_timer() - t}")

    def compute_full_reachtube(
        self,
        root: AnalysisTreeNode,
        sensor,
        time_horizon,
        time_step,
        max_height,
        lane_map,
        init_seg_length,
        reachability_method,
        run_num,
        past_runs,
        ax,
        params={},
    ):
        if max_height == None:
            max_height = float("inf")

        self.verification_queue: List[Tuple[AnalysisTreeNode, int]] = [(root, 0)]
        self.result_refs = []
        self.nodes = [root]
        self.num_cached = 0
        num_calls = 0
        num_transitions = 0
        consts = ReachConsts(
            time_step,
            lane_map,
            init_seg_length,
            reachability_method,
            run_num,
            past_runs,
            sensor,
            root.agent,
        )
        if self.config.parallel:
            # import ray
            consts_ref = ray.put(consts)
        while True:
            wait = False
            if len(self.verification_queue) > 0:
                # print([node.id for node in verification_queue])
                node, later = self.verification_queue.pop(0)
                # check height
                if node.height >= max_height-1:
                    print("max depth reached")
                    continue
                num_transitions += 1
                # pp(("start ver", node.start_time, {a: (*node.mode[a], *node.init[a]) for a in node.mode}))
                remain_time = round(time_horizon - node.start_time, 10)
                if remain_time <= 0:
                    continue
                cached_trans_tubes = {}
                cached_tubes = {}
                for agent_id in node.agent:
                    mode = node.mode[agent_id]
                    inits = node.init[agent_id]
                    combined = combine_all(inits, reachability_method == ReachabilityMethod.STAR_SETS)
                    if self.config.incremental:
                        # CachedRTTrans
                        cached = self.trans_cache.check_hit(agent_id, mode, combined, node.init)
                        if cached != None:
                            self.trans_cache_hits = (
                                self.trans_cache_hits[0] + 1,
                                self.trans_cache_hits[1],
                            )
                        else:
                            self.trans_cache_hits = (
                                self.trans_cache_hits[0],
                                self.trans_cache_hits[1] + 1,
                            )
                        # pp(("check hit", agent_id, mode, combined))
                        if cached != None:
                            cached_trans_tubes[agent_id] = cached
                        if agent_id not in node.trace and reachability_method == ReachabilityMethod.STAR_SETS:
                            print("in check bloat tube")
                            # uncertain_param = node.uncertain_param[agent_id]
                            # CachedTube.tube
                            cur_bloated_tube, miss_seg_idx_list = self.check_cache_bloated_tube_stars(
                                agent_id, mode, inits, combine_seg_length=init_seg_length
                            )
                            cached_tubes[agent_id] = (cur_bloated_tube, miss_seg_idx_list)
                            print("done")

						# if incremental and DRYVR, check cache tube first
                        elif agent_id not in node.trace and reachability_method == ReachabilityMethod.DRYVR:
                            # uncertain_param = node.uncertain_param[agent_id]
                            # CachedTube.tube
                            cur_bloated_tube, miss_seg_idx_list = self.check_cache_bloated_tube(
                                agent_id, mode, inits, combine_seg_length=init_seg_length
                            )
                            cached_tubes[agent_id] = (cur_bloated_tube, miss_seg_idx_list)
                # FIXME
                old_node_id = None
                if len(cached_trans_tubes) == len(node.agent):
                    all_node_ids = [s.node_ids for s in cached_trans_tubes.values()]
                    # print('all_node_ids', all_node_ids)
                    node_ids = list(functools.reduce(lambda a, b: a.intersection(b), all_node_ids))
                    # print('node_ids', node_ids)
                    if len(node_ids) > 0:
                        old_node_id = node_ids[0]
                    # else:
                    #     print(f"not full {node.id}: {node_ids}, {len(cached_trans_tubes) == len(node.agent)} | {all_node_ids}")
                if not self.config.parallel or (old_node_id != None and self.config.try_local):
                    self.proc_result(
                        *self.compute_full_reachtube_step(
                            self.config,
                            cached_trans_tubes,
                            cached_tubes,
                            node,
                            old_node_id,
                            later,
                            remain_time,
                            consts,
                            max_height,
                            ax,
                            params,
                        ),
                        max_height,
                    )
                else:
                    self.result_refs.append(
                        self.compute_full_reachtube_step_remote.remote(
                            self.config,
                            cached_trans_tubes,
                            cached_tubes,
                            node,
                            old_node_id,
                            later,
                            remain_time,
                            consts_ref,
                            max_height,
                            params,
                        )
                    )
                if len(self.result_refs) >= self.config.parallel_ver_ahead:
                    wait = True
            elif len(self.result_refs) > 0:
                wait = True
            else:
                break
            # print(len(verification_queue), len(result_refs))
            if wait:
                [res], self.result_refs = ray.wait(self.result_refs)
                (
                    id,
                    later,
                    next_nodes,
                    traces,
                    assert_hits,
                    cache_tube_updates,
                    cache_trans_tube_updates,
                ) = ray.get(res)
                # TODO: may add pipelining
                self.proc_result(
                    id,
                    later,
                    next_nodes,
                    traces,
                    assert_hits,
                    cache_tube_updates,
                    cache_trans_tube_updates,
                    max_height,
                )
        self.reachtube_tree = AnalysisTree(root)
        # print(f">>>>>>>> Number of calls to reachability engine: {num_calls}")
        # print(f">>>>>>>> Number of transitions happening: {num_transitions}")
        self.num_transitions = num_transitions

        return self.reachtube_tree


    @staticmethod
    def get_transition_verify_opt(
        config: "ScenarioConfig", cache: Dict[str, CachedRTTrans], paths: PathDiffs, node: AnalysisTreeNode, track_map, sensor
    ) -> Tuple[
        Optional[Dict[str, List[str]]],
        Optional[Dict[str, List[Tuple[str, List[str], List[float]]]]],
    ]:
        # For each agent
        agent_guard_dict = defaultdict(list)
        cached_guards = defaultdict(list)
        min_trans_ind = None
        cached_trans = []
        agent_dict = node.agent

        if not cache:
            paths = [
                (agent, p) for agent in node.agent.values() for p in agent.decision_logic.paths
            ]
        else:
            # _transitions = [trans.transition for seg in cache.values() for trans in seg.transitions]
            _transitions = [
                (aid, trans)
                for aid, seg in cache.items()
                for trans in seg.transitions
                if reach_trans_suit(trans.inits, node.init)
            ]
            # pp(("cached trans", len(_transitions)))
            if len(_transitions) > 0:
                min_trans_ind = min([t.transition for _, t in _transitions])
                # TODO: check for asserts
                cached_trans = [
                    (aid, tran.mode, tran.dest, tran.reset, tran.reset_idx, tran.paths)
                    for aid, tran in dedup(_transitions, lambda p: (p[0], p[1].mode, p[1].dest))
                    if tran.transition == min_trans_ind
                ]
                if len(paths) == 0:
                    # print(red("full cache"))
                    return None, cached_trans

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
                    for path in agent_paths:
                        cont_var_dict_template, discrete_variable_dict, length_dict = sensor.sense(
                            agent, state_dict, track_map, False
                        )
                        reset = (path.var, path.val_veri)
                        guard_expression = GuardExpressionAst([path.cond_veri])

                        cont_var_updater = guard_expression.parse_any_all_new(
                            cont_var_dict_template, discrete_variable_dict, length_dict
                        )
                        Verifier.apply_cont_var_updater(cont_var_dict_template, cont_var_updater)
                        guard_can_satisfied = guard_expression.evaluate_guard_disc(
                            agent, discrete_variable_dict, cont_var_dict_template, track_map
                        )
                        if not guard_can_satisfied:
                            continue
                        cached_guards[agent_id].append(
                            (
                                path,
                                guard_expression,
                                cont_var_updater,
                                copy.deepcopy(discrete_variable_dict),
                                reset,
                                path_transitions[path.cond],
                            )
                        )

        # for aid, trace in node.trace.items():
        #     if len(trace) < 2:
        #         pp(("weird state", aid, trace))
        for agent, path in paths:
            if len(agent.decision_logic.args) == 0:
                continue
            agent_id = agent.id
            state_dict = {}
            if config.reachability_method == ReachabilityMethod.STAR_SETS:
                state_dict = {
                    aid: (node.trace[aid][0:1][0], node.mode[aid], node.static[aid]) for aid in node.agent
                }
            else:
                state_dict = {
                    aid: (node.trace[aid][0:2], node.mode[aid], node.static[aid]) for aid in node.agent
                }
            cont_var_dict_template, discrete_variable_dict, length_dict = sensor.sense(agent, state_dict, track_map, False)
            # TODO-PARSER: Get equivalent for this function
            # Construct the guard expression
            guard_expression = GuardExpressionAst([path.cond_veri])

            cont_var_updater = guard_expression.parse_any_all_new(
                cont_var_dict_template, discrete_variable_dict, length_dict
            )
            Verifier.apply_cont_var_updater(cont_var_dict_template, cont_var_updater)
            guard_can_satisfied = guard_expression.evaluate_guard_disc(
                agent, discrete_variable_dict, cont_var_dict_template, track_map
            )
            if not guard_can_satisfied:
                continue
            agent_guard_dict[agent_id].append(
                (guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), path)
            )


        # pp(("trace len", trace_length, {a: len(t) for a, t in node.trace.items()}))
        guard_hits = []
        guard_hit = False
        reduction_rate = 10
        reduction_queue = []
        if config.reachability_method == ReachabilityMethod.STAR_SETS:
            trace_length = int(min(len(v) for v in node.trace.values()))
            reduction_queue = [(i, i+1, 1) for i in range(0, trace_length)]
            reduction_queue.reverse()
        else:
            trace_length = int(min(len(v) for v in node.trace.values()) // 2)
            reduction_queue = [(0, trace_length, trace_length)]
        # for idx, end_idx,combine_len in reduction_queue:
        hits = []
        while reduction_queue:
            idx, end_idx, combine_len = reduction_queue.pop()
            reduction_needed = False
            # print((idx, combine_len))
            any_contained = False
            # end_idx = min(idx+combine_len, trace_length)
            state_dict = {}
            if config.reachability_method == ReachabilityMethod.STAR_SETS:
                state_dict = {
					aid: (
						node.trace[aid][idx],
						node.mode[aid],
						node.static[aid],
					)
					for aid in node.agent
				}                
            else:
                state_dict = {
					aid: (
						combine_rect(node.trace[aid][idx * 2 : end_idx * 2]),
						node.mode[aid],
						node.static[aid],
					)
					for aid in node.agent
				}
            if min_trans_ind != None and idx > min_trans_ind:
                if hits:
                    guard_hits.append((hits, state_dict, idx))
                    guard_hit = True
                    break
                else:
                    return None, cached_trans
            hits = []

            asserts = defaultdict(list)
            for agent_id in agent_dict.keys():
                agent: BaseAgent = agent_dict[agent_id]
                if len(agent.decision_logic.args) == 0:
                    continue
                # if np.array(agent_state).ndim != 2:
                #     pp(("weird state", agent_id, agent_state))
                cont_vars, disc_vars, len_dict = sensor.sense(agent, state_dict, track_map, False)
                resets = defaultdict(list)
                # Check safety conditions
                for i, a in enumerate(agent.decision_logic.asserts_veri):
                    pre_expr = a.pre

                    def eval_expr(expr):
                        ge = GuardExpressionAst([copy.deepcopy(expr)])
                        cont_var_updater = ge.parse_any_all_new(cont_vars, disc_vars, len_dict)
                        Verifier.apply_cont_var_updater(cont_vars, cont_var_updater)
                        sat = ge.evaluate_guard_disc(agent, disc_vars, cont_vars, track_map)
                        if sat:
                            sat = ge.evaluate_guard_hybrid(agent, disc_vars, cont_vars, track_map, config.reachability_method == ReachabilityMethod.STAR_SETS)
                            if sat:
                                sat, contained = ge.evaluate_guard_cont(agent, cont_vars, track_map, config.reachability_method == ReachabilityMethod.STAR_SETS)
                                sat = sat and contained
                        return sat

                    if eval_expr(pre_expr):
                        if not eval_expr(a.cond):
                            if combine_len == 1:
                                label = a.label if a.label != None else f"<assert {i}>"
                                if config.print_level >= 1:
                                    print(f'assert hit for {agent_id}: "{label}"')
                                    print("index", idx)
                                    print("start_time", node.start_time)
                                asserts[agent_id].append(label)
                            else:
                                new_len = int(np.ceil(combine_len / reduction_rate))
                                next_list = [
                                    (i, min(i + new_len, end_idx), new_len)
                                    for i in range(idx, end_idx, new_len)
                                ]
                                reduction_queue.extend(next_list[::-1])
                                reduction_needed = True
                                break
                if reduction_needed:
                    break
                if agent_id in asserts:
                    continue
                if agent_id not in agent_guard_dict:
                    continue

                unchecked_cache_guards = [
                    g[:-1] for g in cached_guards[agent_id] if g[-1] < idx
                ]  # FIXME: off by 1?
                for guard_expression, continuous_variable_updater, discrete_variable_dict, path in (
                    agent_guard_dict[agent_id] + unchecked_cache_guards
                ):
                    assert isinstance(path, ModePath)
                    new_cont_var_dict = copy.deepcopy(cont_vars)
                    one_step_guard: GuardExpressionAst = copy.deepcopy(guard_expression)

                    Verifier.apply_cont_var_updater(new_cont_var_dict, continuous_variable_updater)
                    guard_can_satisfied = one_step_guard.evaluate_guard_hybrid(
                        agent, discrete_variable_dict, new_cont_var_dict, track_map,  config.reachability_method == ReachabilityMethod.STAR_SETS
                    )
                    if not guard_can_satisfied:
                        continue
                    guard_satisfied, is_contained = one_step_guard.evaluate_guard_cont(
                        agent, new_cont_var_dict, track_map, config.reachability_method == ReachabilityMethod.STAR_SETS
                    )
                    if combine_len == 1:
                        any_contained = any_contained or is_contained
                    # TODO: Can we also store the cont and disc var dict so we don't have to call sensor again?
                    if guard_satisfied and combine_len == 1:
                        reset_expr = ResetExpression((path.var, path.val_veri))
                        resets[reset_expr.var].append(
                            (
                                reset_expr,
                                discrete_variable_dict,
                                new_cont_var_dict,
                                guard_expression.guard_idx,
                                path,
                            )
                        )
                    elif guard_satisfied and combine_len > 1:
                        new_len = int(np.ceil(combine_len / reduction_rate))
                        next_list = [
                            (i, min(i + new_len, end_idx), new_len)
                            for i in range(idx, end_idx, new_len)
                        ]
                        reduction_queue.extend(next_list[::-1])
                        reduction_needed = True
                        break
                if reduction_needed:
                    break
                if combine_len == 1:
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

            if reduction_needed or combine_len > 1:
                continue
            if len(asserts) > 0:
                return (asserts, idx), None
            if hits != []:
                guard_hits.append((hits, state_dict, idx))
                guard_hit = True
            elif guard_hit:
                break
            if any_contained:
                break
        reset_func = Verifier.apply_reset
        if config.reachability_method == ReachabilityMethod.STAR_SETS:
            reset_func = Verifier.apply_reset_stars
        reset_dict = {}  # defaultdict(lambda: defaultdict(list))
        for hits, all_agent_state, hit_idx in guard_hits:
            for agent_id, reset_idx, reset_list in hits:
                # TODO: Need to change this function to handle the new reset expression and then I am done
                dest_list, reset_rect = reset_func(
                    node.agent[agent_id], reset_list, all_agent_state, track_map
                )
                # pp(("dests", dest_list, *[astunparser.unparse(reset[-1].val_veri) for reset in reset_list]))
                if agent_id not in reset_dict:
                    reset_dict[agent_id] = {}
                if not dest_list:
                    warnings.warn(
                        f"Guard hit for mode {node.mode[agent_id]} for agent {agent_id} without available next mode"
                    )
                    dest_list.append(None)
                if reset_idx not in reset_dict[agent_id]:
                    reset_dict[agent_id][reset_idx] = {}
                for dest in dest_list:
                    if dest not in reset_dict[agent_id][reset_idx]:
                        reset_dict[agent_id][reset_idx][dest] = []
                    reset_dict[agent_id][reset_idx][dest].append(
                        (reset_rect, hit_idx, reset_list[-1])
                    )

        possible_transitions = []
        # Combine reset rects and construct transitions

        count = 0
        for agent in reset_dict:
            for reset_idx in reset_dict[agent]:
                for dest in reset_dict[agent][reset_idx]:
                    #output.x output.vx
                    reset_data = tuple(map(list, zip(*reset_dict[agent][reset_idx][dest])))
                    paths = [r[-1] for r in reset_data[-1]]
                    transition = (agent, node.mode[agent], dest, *reset_data[:-1], paths)
                    src_mode = node.get_mode(agent, node.mode[agent])
                    src_track = node.get_track(agent, node.mode[agent])
                    dest_mode = node.get_mode(agent, dest)
                    dest_track = node.get_track(agent, dest)
                    if dest_track == track_map.h(src_track, src_mode, dest_mode):
                        if config.print_level >= 2:
                            print(count)
                        count += 1
                        if config.print_level >= 2:
                            print(agent, src_mode, src_track, "->", dest_mode, dest_track)
                            print("start_time: ", node.start_time)
                            print("cond_veri", unparse(paths[0].cond_veri))
                            print("val_veri", unparse(paths[0].val_veri))
                        possible_transitions.append(transition)
                        # print(transition[4])
        # Return result
        return None, possible_transitions + cached_trans

    @staticmethod
    def apply_cont_var_updater(cont_var_dict, updater):
        for variable in updater:
            for unrolled_variable, unrolled_variable_index in updater[variable]:
                cont_var_dict[unrolled_variable] = cont_var_dict[variable][unrolled_variable_index]

    @staticmethod
    def apply_reset(
        agent: BaseAgent, reset_list, all_agent_state, track_map
    ) -> Tuple[str, np.ndarray]:
        dest = []
        rect = []

        agent_state, agent_mode, agent_static = all_agent_state[agent.id]
        dest = copy.deepcopy(agent_mode)
        possible_dest = [[elem] for elem in dest]
        ego_type = find(agent.decision_logic.args, lambda a: a.name == EGO).typ
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
                for var_loc, discrete_variable_ego in enumerate(
                    agent.decision_logic.state_defs[ego_type].disc
                ):
                    if discrete_variable_ego == reset_variable:
                        found = True
                        break
                if not found:
                    raise ValueError(f"Reset discrete variable {discrete_variable_ego} not found")
                if isinstance(reset.val_ast, ast.Constant):
                    val = eval(expr)
                    possible_dest[var_loc] = [val]
                else:
                    tmp = expr.split(".")
                    if "map" in tmp[0]:
                        for var in disc_var_dict:
                            expr = expr.replace(var, f"'{disc_var_dict[var]}'")
                        res = eval(expr)
                        if not isinstance(res, list):
                            res = [res]
                        possible_dest[var_loc] = res
                    else:
                        expr = tmp
                        if expr[0].strip(" ") in agent.decision_logic.mode_defs:
                            possible_dest[var_loc] = [expr[1]]

            # Assume linear function for continuous variables
            else:
                lhs = reset_variable
                rhs = expr
                found = False
                for lhs_idx, cts_variable in enumerate(
                    agent.decision_logic.state_defs[ego_type].cont
                ):
                    if cts_variable == lhs:
                        found = True
                        break
                if not found:
                    raise ValueError(f"Reset continuous variable {cts_variable} not found")
                # substituting low variables

                symbols = []
                for var in cont_var_dict:
                    if var in expr:
                        symbols.append(var)

                # TODO: Implement this function
                # The input to this function is a list of used symbols and the cont_var_dict
                # The ouput of this function is a list of tuple of values for each variable in the symbols list
                # The function will explor all possible combinations of low bound and upper bound for the variables in the symbols list
                comb_list = Verifier._get_combinations(symbols, cont_var_dict)

                lb = float("inf")
                ub = -float("inf")

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

    @staticmethod
    def apply_reset_stars(
        agent: BaseAgent, reset_list, all_agent_state, track_map
    ) -> Tuple[str, np.ndarray]:
        dest = []
        rect = []

        agent_state, agent_mode, agent_static = all_agent_state[agent.id]
        dest = copy.deepcopy(agent_mode)
        possible_dest = [[elem] for elem in dest]
        ego_type = find(agent.decision_logic.args, lambda a: a.name == EGO).typ


        #Modified
        old_state =  agent_state[1].starcopy() #copy.deepcopy([agent_state[0][1:], agent_state[1][1:]])
        reset_vars = {}
        expr_list = {}

        # The reset_list here are all the resets for a single transition. Need to evaluate each of them
        # and then combine them together
        for reset_tuple in reset_list:
            reset, disc_var_dict, cont_var_dict, _, _p = reset_tuple
            reset_variable = reset.var
            expr = reset.expr
            #breakpoint()

            # First get the transition destinations
            if "mode" in reset_variable:
                found = False
                for var_loc, discrete_variable_ego in enumerate(
                    agent.decision_logic.state_defs[ego_type].disc
                ):
                    if discrete_variable_ego == reset_variable:
                        found = True
                        break
                if not found:
                    raise ValueError(f"Reset discrete variable {discrete_variable_ego} not found")
                if isinstance(reset.val_ast, ast.Constant):
                    val = eval(expr)
                    possible_dest[var_loc] = [val]
                else:
                    tmp = expr.split(".")
                    if "map" in tmp[0]:
                        for var in disc_var_dict:
                            expr = expr.replace(var, f"'{disc_var_dict[var]}'")
                        res = eval(expr)
                        if not isinstance(res, list):
                            res = [res]
                        possible_dest[var_loc] = res
                    else:
                        expr = tmp
                        if expr[0].strip(" ") in agent.decision_logic.mode_defs:
                            possible_dest[var_loc] = [expr[1]]

            # Assume linear function for continuous variables
            else:
                #agent_state.continuous_reset(reset_variable, expr, agent, ego_type,cont_var_dict, rect)
                #breakpoint()
                lhs = reset_variable
                rhs = expr
                found = False
                for lhs_idx, cts_variable in enumerate(
                    agent.decision_logic.state_defs[ego_type].cont
                ):
                    if cts_variable == lhs:
                        found = True
                        expr_list[lhs_idx] = rhs
                        reset_vars[lhs_idx] = lhs
                        break
                if not found:
                    raise ValueError(f"Reset continuous variable {cts_variable} not found")
                # substituting low variables
                statevec = []
                for var in cont_var_dict:
                    #TODO: check that this only gets run on ego?
                    if 'ego' in var:
                        statevec.append(var)
                



                #print(statevec)

                #concern: how to handle the case where you need other agents state. for now: assume you do not
        def reset_func(state, expr_list, reset_vars): #[ego.x, ego.y, ...]
            #breakpoint()
            output = np.copy(state)
            idxs = list(reset_vars.keys())
            for idx in idxs:
                val_dict = {}
                tmp_exp = copy.deepcopy(expr_list[idx])
                for i in range(0, len(state)):
                    if statevec[i] in tmp_exp:
                        tmp_exp = tmp_exp.replace(statevec[i], str(state[i]))
                #print(tmp_exp)
                result = eval(tmp_exp, {}, val_dict)
                for i in range(0, len(state)):
                    if reset_vars[idx] == statevec[i].split('.',1)[1]:
                        output[i] = result 
            return output
            #print("TODO: find where/when this gets set elsewhere")
            #breakpoint()
            #print("foo")
            #breakpoint()
        new_state = old_state.apply_reset(reset_func, expr_list, reset_vars)
            #print("bar")
            #breakpoint()


        all_dest = itertools.product(*possible_dest)
        dest = []
        for tmp in all_dest:
            dest.append(tmp)
        #breakpoint()

        # print("apply_reset")
        # print(dest)
        # print(new_state.overapprox_rectangle())
        return dest, new_state


    @staticmethod
    def _get_combinations(symbols, cont_var_dict):
        data_list = []
        for symbol in symbols:
            data_list.append(cont_var_dict[symbol])
        comb_list = list(itertools.product(*data_list))
        return comb_list


def combine_rect(trace):
    """
    Combine a reachtube into one rect

    :param trace: the reachtube (2d list) to be combined
    :return: the combined rect (2d list)
    """
    trace = np.array(trace)
    # assert trace.shape[0] % 2 == 0
    combined_trace = np.ndarray(shape=(2, trace.shape[-1]))
    combined_trace[0] = np.min(trace[::2], 0)
    combined_trace[1] = np.max(trace[1::2], 0)
    return combined_trace.tolist()


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
