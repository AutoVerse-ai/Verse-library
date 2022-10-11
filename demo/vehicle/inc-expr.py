# SM: Norng some things about the example

import timeit
from pympler import asizeof
from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap4
from verse import Scenario

from enum import Enum, auto
from verse.plotter.plotter2D import reachtube_tree
from verse.scenario.scenario import ScenarioConfig
import functools, pprint
pp = functools.partial(pprint.pprint, compact=True, width=130)
from typing import List

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()
    Stop = auto()

class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()
    Lane3 = auto()

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles

import sys
arg = sys.argv[1]

def jerk(l: List[List[float]], x=0, y=0):
    return [[l[0][0] - x, l[0][1] - y, *l[0][2:]], [l[1][0] + x, l[1][1] + y, *l[1][2:]]]

def jerks(ls: List[List[List[float]]], js: List[List[float]]):
    return [jerk(l, *j) for l, j in zip(ls, js)]

def dupi(l: List[List[float]]):
    return [[i, i] for i in l]

if 'p' in arg:
    import plotly.graph_objects as go
    from verse.plotter.plotter2D import simulation_tree

def run(sim, meas=False):
    time = timeit.default_timer()
    if sim:
        traces = scenario.simulate(60, 0.05)
    else:
        traces = scenario.verify(60, 0.1)

    if 'd' in arg:
        traces.dump_tree()
        traces.dump("main.json") 
        traces.dump("tree2.json" if meas else "tree1.json") 

    if 'p' in arg:
        fig = go.Figure()
        if sim:
            fig = simulation_tree(traces, tmp_map, fig, 1, 2, print_dim_list=[1, 2])
        else:
            fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], 'lines')
        fig.show()

    if sim:
        cache_size = asizeof.asizeof(scenario.simulator.cache)
    else:
        cache_size = asizeof.asizeof(scenario.verifier.cache) + asizeof.asizeof(scenario.verifier.trans_cache)
    if meas:
        pp({
            "dur": timeit.default_timer() - time,
            "cache_size": cache_size,
            "node_count": len(traces.nodes),
        })

if __name__ == "__main__":
    input_code_name = './demo/vehicle/controller/inc-expr.py'
    config = ScenarioConfig()
    config.incremental = 'i' in arg
    scenario = Scenario(config)

    scenario.add_agent(CarAgent('car1', file_name=input_code_name))
    scenario.add_agent(NPCAgent('car2'))
    scenario.add_agent(CarAgent('car3', file_name=input_code_name))
    scenario.add_agent(NPCAgent('car4'))
    scenario.add_agent(NPCAgent('car5'))
    scenario.add_agent(NPCAgent('car6'))
    scenario.add_agent(NPCAgent('car7'))
    scenario.add_agent(CarAgent('car8', file_name=input_code_name))
    tmp_map = SimpleMap4()
    scenario.set_map(tmp_map)
    sim = "v" not in arg
    mode_inits = ([
            (VehicleMode.Normal, LaneMode.Lane1), (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane0), (VehicleMode.Normal, LaneMode.Lane0),
            (VehicleMode.Normal, LaneMode.Lane1), (VehicleMode.Normal, LaneMode.Lane2),
            (VehicleMode.Normal, LaneMode.Lane2), (VehicleMode.Normal, LaneMode.Lane2),
        ],
        [
            (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
        ])
    poses = [
        [0, 0, 0, 1.0], [10, 0, 0, 0.5],
        [14, 3, 0, 0.6], [20, 3, 0, 0.5],
        [30, 0, 0, 0.5], [28.5, -3, 0, 0.5],
        [39.5, -3, 0, 0.5], [30, -3, 0, 0.6],
    ]
    _jerks = [
        [0.05, 0.1], [],
        [0.05, 0.1], [],
        [], [],
        [], [0.05, 0.1],
    ]
    cont_inits = dupi(poses)
    if not sim:
        cont_inits = jerks(cont_inits, _jerks)
    scenario.set_init(cont_inits, *mode_inits)

    if 'b' in arg:
        run(sim, True)
    elif 'r' in arg:
        run(sim)
        run(sim, True)
    elif 'n' in arg:
        run(sim)
        poses[0][0] = 1
        cont_inits = dupi(poses)
        if not sim:
            cont_inits = jerks(cont_inits, _jerks)
        scenario.set_init(cont_inits, *mode_inits)
        run(sim, True)
    elif '3' in arg:
        run(sim)
        scenario.agent_dict["car3"] = CarAgent('car3', file_name=input_code_name.replace(".py", "-fsw7.py"))
        run(sim, True)
    elif '8' in arg:
        run(sim)
        scenario.agent_dict["car8"] = CarAgent('car8', file_name=input_code_name.replace(".py", "-fsw4.py"))
        run(sim, True)
