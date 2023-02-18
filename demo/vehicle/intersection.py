import functools, pprint, random, timeit
import math
from enum import Enum, auto
from pympler import asizeof
from verse.agents.example_agent import CarAgent, NPCAgent
from verse.analysis.utils import wrap_to_pi
from verse.parser.parser import ast_dump
from verse.map.example_map.intersection import Intersection
from verse.scenario.scenario import ScenarioConfig, Scenario
from typing import List
pp = functools.partial(pprint.pprint, compact=True, width=130)

from controller.intersection_car import AgentMode, TrackMode

CAR_NUM = 3
CAR_ACCEL_RANGE = (0.7, 3)
CAR_SPEED_RANGE = (1, 3)
CAR_THETA_RANGE = (-0.1, 0.1)

def rand(start: float, end: float) -> float:
    return random.random() * (end - start) + start

import sys
arg = sys.argv[1]

if 'p' in arg:
    import plotly.graph_objects as go
    from verse.plotter.plotter2D import reachtube_tree, simulation_tree

def run(sim, meas=False):
    time = timeit.default_timer()
    if sim:
        traces = scenario.simulate(60, 0.05)
    else:
        traces = scenario.verify(60, 0.05)
    dur = timeit.default_timer() - time

    if 'd' in arg:
        traces.dump_tree()
        traces.dump("main.json") 
        traces.dump("tree2.json" if meas else "tree1.json") 

    if 'p' in arg and meas:
        fig = go.Figure()
        if sim:
            fig = simulation_tree(traces, map, fig, 1, 2, print_dim_list=[1, 2])
        else:
            fig = reachtube_tree(traces, map, fig, 1, 2, [1, 2], 'lines',combine_rect=5)
        fig.show()

    if sim:
        cache_size = asizeof.asizeof(scenario.simulator.cache)
    else:
        cache_size = asizeof.asizeof(scenario.verifier.cache) + asizeof.asizeof(scenario.verifier.trans_cache)
    if meas:
        pp({
            "dur": dur,
            "cache_size": cache_size,
            "node_count": ((0 if sim else scenario.verifier.num_transitions), len(traces.nodes)),
            "hits": scenario.simulator.cache_hits if sim else (scenario.verifier.tube_cache_hits, scenario.verifier.trans_cache_hits),
        })

if __name__ == "__main__":
    ctlr_src = "demo/vehicle/controller/intersection_car.py"
    config = ScenarioConfig(incremental='i' in arg, parallel='l' in arg)
    scenario = Scenario(config)
    random.seed(0)

    sim = "v" not in arg
    dirs = "WSEN"
    map = Intersection()
    scenario.set_map(map)
    for i in range(CAR_NUM):
        car = CarAgent(f"car{i}", file_name=ctlr_src, speed=rand(*CAR_SPEED_RANGE), accel=rand(*CAR_ACCEL_RANGE))
        scenario.add_agent(car)
        dir = random.randint(0, 3)
        src = dirs[dir]
        dst_dirs = list(dirs)
        dst_dirs.remove(src)
        dst = dst_dirs[random.randint(0, 2)]
        start, off = map.size + rand(0, map.length * 0.3), rand(0, map.width)
        pos = { "N": (-off, start), "S": (off, -start), "W": (-start, -off), "E": (start, off) }[src]
        init = [*pos, wrap_to_pi(dir * math.pi / 2 + rand(*CAR_THETA_RANGE)), rand(*CAR_SPEED_RANGE)]
        modes = (AgentMode.Accel, getattr(TrackMode, src + dst))
        scenario.set_init_single(car.id, (init,), modes)

    if 'b' in arg:
        run(sim, True)
    elif 'r' in arg:
        run(sim)
        run(sim, True)
