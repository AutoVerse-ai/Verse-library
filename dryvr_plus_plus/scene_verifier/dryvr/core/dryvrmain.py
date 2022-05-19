"""
This file contains a single function that verifies model
"""
from __future__ import print_function
import time

import dryvrpy.common.config as userConfig
from dryvrpy.common.io import parseVerificationInputFile, parseRrtInputFile, writeRrtResultFile
from dryvrpy.common.utils import buildModeStr, isIpynb, overloadConfig
from dryvrpy.core.distance import DistChecker
from dryvrpy.core.dryvrcore import *
from dryvrpy.core.goalchecker import GoalChecker
from dryvrpy.core.graph import Graph
from dryvrpy.core.guard import Guard
from dryvrpy.core.initialset import InitialSet
from dryvrpy.core.initialsetstack import InitialSetStack, GraphSearchNode
from dryvrpy.core.reachtube import ReachTube
from dryvrpy.core.reset import Reset
from dryvrpy.core.uniformchecker import UniformChecker
# from dryvr_plus_plus.tube_computer.backend.reachabilityengine import ReachabilityEngine
# from dryvr_plus_plus.tube_computer.backend.initialset import InitialSet

def verify(data, sim_function, param_config=None):
    """
    DryVR verification algorithm.
    It does the verification and print out the verify result.
    
    Args:
        data (dict): dictionary that contains params for the input file
        sim_function (function): black-box simulation function
        param_config (dict or None): example-specified configuration

    Returns:
        Safety (str): safety of the system
        Reach (obj): reach tube object

    """
    if param_config is None:
        param_config = {}
    # There are some fields can be config by example,
    # If example specified these fields in paramConfig,
    # overload these parameters to userConfig
    overloadConfig(userConfig, param_config)

    refine_counter = 0

    params = parseVerificationInputFile(data)
    # Build the graph object
    graph = build_graph(
        params.vertex,
        params.edge,
        params.guards,
        params.resets
    )

    # Build the progress graph for jupyter notebook
    # isIpynb is used to detect if the code is running
    # on notebook or terminal, the graph will only be shown
    # in notebook mode
    progress_graph = Graph(params, isIpynb())

    # Make sure the initial mode is specified if the graph is dag
    # FIXME should move this part to input check
    # Bolun 02/12/2018
    assert graph.is_dag() or params.initialVertex != -1, "Graph is not DAG and you do not have initial mode!"

    checker = UniformChecker(params.unsafeSet, params.variables)
    guard = Guard(params.variables)
    reset = Reset(params.variables)
    t_start = time.time()

    # Step 1) Simulation Test
    # Random generate points, then simulate and check the result
    for _ in range(userConfig.SIMUTESTNUM):
        rand_init = randomPoint(params.initialSet[0], params.initialSet[1])

        if DEBUG:
            print('Random checking round ', _, 'at point ', rand_init)

        # Do a full hybrid simulation
        sim_result = simulate(
            graph,
            rand_init,
            params.timeHorizon,
            guard,
            sim_function,
            reset,
            params.initialVertex,
            params.deterministic
        )

        # Check the traces for each mode
        for mode in sim_result:
            safety = checker.check_sim_trace(sim_result[mode], mode)
            if safety == -1:
                print('Current simulation is not safe. Program halt')
                print('simulation time', time.time() - t_start)                    
                return "UNSAFE", None
    sim_end_time = time.time()

    # Step 2) Check Reach Tube
    # Calculate the over approximation of the reach tube and check the result
    print("Verification Begin")

    # Get the initial mode
    if params.initialVertex == -1:
        compute_order = graph.topological_sorting(mode=igraph.OUT)
        initial_vertex = compute_order[0]
    else:
        initial_vertex = params.initialVertex

    # Build the initial set stack
    cur_mode_stack = InitialSetStack(initial_vertex, userConfig.REFINETHRES, params.timeHorizon,0)
    cur_mode_stack.stack.append(InitialSet(params.initialSet[0], params.initialSet[1]))
    cur_mode_stack.bloated_tube.append(buildModeStr(graph, initial_vertex))
    while True:
        # backward_flag can be SAFE, UNSAFE or UNKNOWN
        # If the backward_flag is SAFE/UNSAFE, means that the children nodes
        # of current nodes are all SAFE/UNSAFE. If one of the child node is
        # UNKNOWN, then the backward_flag is UNKNOWN.
        backward_flag = SAFE

        while cur_mode_stack.stack:
            print(str(cur_mode_stack))
            print(cur_mode_stack.stack[-1])

            if not cur_mode_stack.is_valid():
                # A stack will be invalid if number of initial sets 
                # is more than refine threshold we set for each stack.
                # Thus we declare this stack is UNKNOWN
                print(cur_mode_stack.mode, "is not valid anymore")
                backward_flag = UNKNOWN
                break

            # This is condition check to make sure the reach tube output file 
            # will be readable. Let me try to explain this.
            # A reachtube output will be something like following
            # MODEA->MODEB
            # [0.0, 1.0, 1.1]
            # [0.1, 1.1, 1.2]
            # .....
            # Once we have refinement, we will add multiple reach tube to
            # this cur_mode_stack.bloatedTube
            # However, we want to copy MODEA->MODEB so we know that two different
            # reach tube from two different refined initial set
            # The result will be look like following
            # MODEA->MODEB
            # [0.0, 1.0, 1.1]
            # [0.1, 1.1, 1.2]
            # .....
            # MODEA->MODEB (this one gets copied!)
            # [0.0, 1.5, 1.6]
            # [0.1, 1.6, 1.7]
            # .....
            if isinstance(cur_mode_stack.bloated_tube[-1], list):
                cur_mode_stack.bloated_tube.append(cur_mode_stack.bloated_tube[0])

            cur_stack = cur_mode_stack.stack
            cur_vertex = cur_mode_stack.mode
            cur_remain_time = cur_mode_stack.remain_time
            cur_label = graph.vs[cur_vertex]['label']
            cur_successors = graph.successors(cur_vertex)
            cur_initial = [cur_stack[-1].lower_bound, cur_stack[-1].upper_bound]
            # Update the progress graph
            progress_graph.update(buildModeStr(graph, cur_vertex), cur_mode_stack.bloated_tube[0],
                                  cur_mode_stack.remain_time)

            if len(cur_successors) == 0:
                # If there is not successor
                # Calculate the current bloated tube without considering the guard
                cur_bloated_tube = calc_bloated_tube(cur_label,
                                                     cur_initial,
                                                     cur_remain_time,
                                                     sim_function,
                                                     params.bloatingMethod,
                                                     params.kvalue,
                                                     userConfig.SIMTRACENUM,
                                                     )

            candidate_tube = []
            shortest_time = float("inf")
            shortest_tube = None

            for cur_successor in cur_successors:
                edge_id = graph.get_eid(cur_vertex, cur_successor)
                cur_guard_str = graph.es[edge_id]['guards']
                cur_reset_str = graph.es[edge_id]['resets']
                # Calculate the current bloated tube with guard involved
                # Pre-check the simulation trace so we can get better bloated result
                cur_bloated_tube = calc_bloated_tube(cur_label,
                                                     cur_initial,
                                                     cur_remain_time,
                                                     sim_function,
                                                     params.bloatingMethod,
                                                     params.kvalue,
                                                     userConfig.SIMTRACENUM,
                                                     guard_checker=guard,
                                                     guard_str=cur_guard_str,
                                                     )

                # Use the guard to calculate the next initial set
                # TODO: Made this function return multiple next_init
                next_init, truncated_result, transit_time = guard.guard_reachtube(
                    cur_bloated_tube,
                    cur_guard_str,
                )

                if next_init is None:
                    continue

                # Reset the next initial set
                # TODO: Made this function handle multiple next_init
                next_init = reset.reset_set(cur_reset_str, next_init[0], next_init[1])

                # Build next mode stack
                next_mode_stack = InitialSetStack(
                    cur_successor,
                    userConfig.CHILDREFINETHRES,
                    cur_remain_time - transit_time,
                    start_time=transit_time+cur_mode_stack.start_time
                )
                next_mode_stack.parent = cur_mode_stack
                # TODO: Append all next_init into the next_mode_stack
                next_mode_stack.stack.append(InitialSet(next_init[0], next_init[1]))
                next_mode_stack.bloated_tube.append(
                    cur_mode_stack.bloated_tube[0] + '->' + buildModeStr(graph, cur_successor))
                cur_stack[-1].child[cur_successor] = next_mode_stack
                if len(truncated_result) > len(candidate_tube):
                    candidate_tube = truncated_result

                # In case of must transition
                # We need to record shortest tube
                # As shortest tube is the tube invoke transition
                if truncated_result[-1][0] < shortest_time:
                    shortest_time = truncated_result[-1][0]
                    shortest_tube = truncated_result

            # Handle must transition
            if params.deterministic and len(cur_stack[-1].child) > 0:
                next_modes_info = []
                for next_mode in cur_stack[-1].child:
                    next_modes_info.append((cur_stack[-1].child[next_mode].remain_time, next_mode))
                # This mode gets transit first, only keep this mode
                max_remain_time, max_time_mode = max(next_modes_info)
                # Pop other modes because of deterministic system
                for _, next_mode in next_modes_info:
                    if next_mode == max_time_mode:
                        continue
                    cur_stack[-1].child.pop(next_mode)
                candidate_tube = shortest_tube
                print("Handle deterministic system, next mode", graph.vs[list(cur_stack[-1].child.keys())[0]]['label'])

            if not candidate_tube:
                candidate_tube = cur_bloated_tube

            for i in range(len(candidate_tube)):
                candidate_tube[i][0] += cur_mode_stack.start_time

            # Check the safety for current bloated tube
            safety = checker.check_reachtube(candidate_tube, cur_label)
            if safety == UNSAFE:
                print("System is not safe in Mode ", cur_label)
                # Start back Tracking from this point and print tube to a file
                # push current unsafe_tube to unsafe tube holder
                unsafe_tube = [cur_mode_stack.bloated_tube[0]] + candidate_tube
                while cur_mode_stack.parent is not None:
                    prev_mode_stack = cur_mode_stack.parent
                    unsafe_tube = [prev_mode_stack.bloated_tube[0]] + prev_mode_stack.stack[-1].bloated_tube \
                        + unsafe_tube
                    cur_mode_stack = prev_mode_stack
                print('simulation time', sim_end_time - t_start)
                print('verification time', time.time() - sim_end_time)
                print('refine time', refine_counter)
                writeReachTubeFile(unsafe_tube, UNSAFEFILENAME)
                ret_reach = ReachTube(cur_mode_stack.bloated_tube, params.variables, params.vertex)
                return "UNSAFE", ret_reach

            elif safety == UNKNOWN:
                # Refine the current initial set
                print(cur_mode_stack.mode, "check bloated tube unknown")
                discard_initial = cur_mode_stack.stack.pop()
                init_one, init_two = discard_initial.refine()
                cur_mode_stack.stack.append(init_one)
                cur_mode_stack.stack.append(init_two)
                refine_counter += 1

            elif safety == SAFE:
                print("Mode", cur_mode_stack.mode, "check bloated tube safe")
                if cur_mode_stack.stack[-1].child:
                    cur_mode_stack.stack[-1].bloated_tube += candidate_tube
                    next_mode, next_mode_stack = cur_mode_stack.stack[-1].child.popitem()
                    cur_mode_stack = next_mode_stack
                    print("Child exist in cur mode inital", cur_mode_stack.mode, "is cur_mode_stack Now")
                else:
                    cur_mode_stack.bloated_tube += candidate_tube
                    cur_mode_stack.stack.pop()
                    print("No child exist in current initial, pop")

        if cur_mode_stack.parent is None:
            # We are at head now
            if backward_flag == SAFE:
                # All the nodes are safe
                print("System is Safe!")
                print("refine time", refine_counter)
                writeReachTubeFile(cur_mode_stack.bloated_tube, REACHTUBEOUTPUT)
                ret_reach = ReachTube(cur_mode_stack.bloated_tube, params.variables, params.vertex)
                print('simulation time', sim_end_time - t_start)
                print('verification time', time.time() - sim_end_time)
                return "SAFE", ret_reach
            elif backward_flag == UNKNOWN:
                print("Hit refine threshold, system halt, result unknown")
                print('simulation time', sim_end_time - t_start)
                print('verification time', time.time() - sim_end_time)
                return "UNKNOWN", None
        else:
            if backward_flag == SAFE:
                prev_mode_stack = cur_mode_stack.parent
                prev_mode_stack.stack[-1].bloated_tube += cur_mode_stack.bloated_tube
                print('back flag safe from', cur_mode_stack.mode, 'to', prev_mode_stack.mode)
                if len(prev_mode_stack.stack[-1].child) == 0:
                    # There is no next mode from this initial set
                    prev_mode_stack.bloated_tube += prev_mode_stack.stack[-1].bloated_tube
                    prev_mode_stack.stack.pop()
                    cur_mode_stack = prev_mode_stack
                    print("No child in prev mode initial, pop,", prev_mode_stack.mode, "is cur_mode_stack Now")
                else:
                    # There is another mode transition from this initial set
                    next_mode, next_mode_stack = prev_mode_stack.stack[-1].child.popitem()
                    cur_mode_stack = next_mode_stack
                    print("Child exist in prev mode inital", next_mode_stack.mode, "is cur_mode_stack Now")
            elif backward_flag == UNKNOWN:
                prev_mode_stack = cur_mode_stack.parent
                print('back flag unknown from', cur_mode_stack.mode, 'to', prev_mode_stack.mode)
                discard_initial = prev_mode_stack.stack.pop()
                init_one, init_two = discard_initial.refine()
                prev_mode_stack.stack.append(init_one)
                prev_mode_stack.stack.append(init_two)
                cur_mode_stack = prev_mode_stack
                refine_counter += 1


def graph_search(data, sim_function, param_config=None):
    """
    DryVR controller synthesis algorithm.
    It does the controller synthesis and print out the search result.
    tube and transition graph will be stored in output folder if algorithm finds one
    
    Args:
        data (dict): dictionary that contains params for the input file
        sim_function (function): black-box simulation function
        param_config (dict or None): example-specified configuration

    Returns:
        None

    """
    if param_config is None:
        param_config = {}
    # There are some fields can be config by example,
    # If example specified these fields in paramConfig,
    # overload these parameters to userConfig
    overloadConfig(userConfig, param_config)
    # Parse the input json file and read out the parameters
    params = parseRrtInputFile(data)
    # Construct objects
    checker = UniformChecker(params.unsafeSet, params.variables)
    goal_set_checker = GoalChecker(params.goalSet, params.variables)
    distance_checker = DistChecker(params.goal, params.variables)
    # Read the important param
    available_modes = params.modes
    start_modes = params.modes
    remain_time = params.timeHorizon
    min_time_thres = params.minTimeThres

    # Set goal reach flag to False
    # Once the flag is set to True, It means we find a transition Graph
    goal_reached = False

    # Build the initial mode stack
    # Current Method is ugly, we need to get rid of the initial Mode for GraphSearch
    # It helps us to achieve the full automate search
    # TODO Get rid of the initial Mode thing
    random.shuffle(start_modes)
    dummy_node = GraphSearchNode("start", remain_time, min_time_thres, 0)
    for mode in start_modes:
        dummy_node.children[mode] = GraphSearchNode(mode, remain_time, min_time_thres, dummy_node.level + 1)
        dummy_node.children[mode].parent = dummy_node
        dummy_node.children[mode].initial = (params.initialSet[0], params.initialSet[1])

    cur_mode_stack = dummy_node.children[start_modes[0]]
    dummy_node.visited.add(start_modes[0])

    t_start = time.time()
    while True:

        if not cur_mode_stack:
            break

        if cur_mode_stack == dummy_node:
            start_modes.pop(0)
            if len(start_modes) == 0:
                break

            cur_mode_stack = dummy_node.children[start_modes[0]]
            dummy_node.visited.add(start_modes[0])
            continue

        print(str(cur_mode_stack))

        # Keep check the remain time, if the remain time is less than minTime
        # It means it is impossible to stay in one mode more than minTime
        # Therefore, we have to go back to parents
        if cur_mode_stack.remain_time < min_time_thres:
            print("Back to previous mode because we cannot stay longer than the min time thres")
            cur_mode_stack = cur_mode_stack.parent
            continue

        # If we have visited all available modes
        # We should select a new candidate point to proceed
        # If there is no candidates available,
        # Then we can say current node is not valid and go back to parent
        if len(cur_mode_stack.visited) == len(available_modes):
            if len(cur_mode_stack.candidates) < 2:
                print("Back to previous mode because we do not have any other modes to pick")
                cur_mode_stack = cur_mode_stack.parent
                # If the tried all possible cases with no luck to find path
                if not cur_mode_stack:
                    break
                continue
            else:
                print("Pick a new point from candidates")
                cur_mode_stack.candidates.pop(0)
                cur_mode_stack.visited = set()
                cur_mode_stack.children = {}
                continue

        # Generate bloated tube if we haven't done so
        if not cur_mode_stack.bloated_tube:
            print("no bloated tube find in this mode, generate one")
            cur_bloated_tube = calc_bloated_tube(
                cur_mode_stack.mode,
                cur_mode_stack.initial,
                cur_mode_stack.remain_time,
                sim_function,
                params.bloatingMethod,
                params.kvalue,
                userConfig.SIMTRACENUM
            )

            # Cut the bloated tube once it intersect with the unsafe set
            cur_bloated_tube = checker.cut_tube_till_unsafe(cur_bloated_tube)

            # If the tube time horizon is less than minTime, it means
            # we cannot stay in this mode for min thres time, back to the parent node
            if not cur_bloated_tube or cur_bloated_tube[-1][0] < min_time_thres:
                print("bloated tube is not long enough, discard the mode")
                cur_mode_stack = cur_mode_stack.parent
                continue
            cur_mode_stack.bloated_tube = cur_bloated_tube

            # Generate candidates points for next node
            random_sections = cur_mode_stack.random_picker(userConfig.RANDSECTIONNUM)

            if not random_sections:
                print("bloated tube is not long enough, discard the mode")
                cur_mode_stack = cur_mode_stack.parent
                continue

            # Sort random points based on the distance to the goal set
            random_sections.sort(key=lambda x: distance_checker.calc_distance(x[0], x[1]))
            cur_mode_stack.candidates = random_sections
            print("Generate new bloated tube and candidate, with candidates length", len(cur_mode_stack.candidates))

            # Check if the current tube reaches goal
            result, tube = goal_set_checker.goal_reachtube(cur_bloated_tube)
            if result:
                cur_mode_stack.bloated_tube = tube
                goal_reached = True
                break

        # We have visited all next mode we have, generate some thing new
        # This is actually not necessary, just shuffle all modes would be enough
        # There should not be RANDMODENUM things since it does not make any difference
        # Anyway, for each candidate point, we will try to visit all modes eventually
        # Therefore, using RANDMODENUM to get some random modes visit first is useless
        # TODO, fix this part
        if len(cur_mode_stack.visited) == len(cur_mode_stack.children):
            # leftMode = set(available_modes) - set(cur_mode_stack.children.keys())
            # random_modes = random.sample(leftMode, min(len(leftMode), RANDMODENUM))
            random_modes = available_modes
            random.shuffle(random_modes)

            random_sections = cur_mode_stack.random_picker(userConfig.RANDSECTIONNUM)
            for mode in random_modes:
                candidate = cur_mode_stack.candidates[0]
                cur_mode_stack.children[mode] = GraphSearchNode(mode, cur_mode_stack.remain_time - candidate[1][0],
                                                                min_time_thres, cur_mode_stack.level + 1)
                cur_mode_stack.children[mode].initial = (candidate[0][1:], candidate[1][1:])
                cur_mode_stack.children[mode].parent = cur_mode_stack

        # Random visit a candidate that is not visited before
        for key in cur_mode_stack.children:
            if key not in cur_mode_stack.visited:
                break

        print("transit point is", cur_mode_stack.candidates[0])
        cur_mode_stack.visited.add(key)
        cur_mode_stack = cur_mode_stack.children[key]

    # Back track to print out trace
    print("RRT run time", time.time() - t_start)
    if goal_reached:
        print("goal reached")
        traces = []
        modes = []
        while cur_mode_stack:
            modes.append(cur_mode_stack.mode)
            if not cur_mode_stack.candidates:
                traces.append([t for t in cur_mode_stack.bloated_tube])
            else:
                # Cut the trace till candidate
                temp = []
                for t in cur_mode_stack.bloated_tube:
                    if t == cur_mode_stack.candidates[0][0]:
                        temp.append(cur_mode_stack.candidates[0][0])
                        temp.append(cur_mode_stack.candidates[0][1])
                        break
                    else:
                        temp.append(t)
                traces.append(temp)
            if cur_mode_stack.parent != dummy_node:
                cur_mode_stack = cur_mode_stack.parent
            else:
                break
        # Reorganize the content in modes list for plotter use
        modes = modes[::-1]
        traces = traces[::-1]
        build_rrt_graph(modes, traces, isIpynb())
        for i in range(1, len(modes)):
            modes[i] = modes[i - 1] + '->' + modes[i]

        writeRrtResultFile(modes, traces, RRTOUTPUT)
    else:
        print("could not find graph")
