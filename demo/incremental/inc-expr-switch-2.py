from verse.agents.example_agent import CarAgentSwitch2, NPCAgent
from verse.map.example_map import SimpleMap4Switch2

from enum import Enum, auto
from verse.plotter.plotter2D import reachtube_tree
from verse.scenario.scenario import Benchmark
import functools, pprint

pp = functools.partial(pprint.pprint, compact=True, width=130)
from typing import List


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchLeft2 = auto()
    SwitchRight = auto()
    SwitchRight2 = auto()
    Brake = auto()
    Stop = auto()


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()


def jerk(l: List[List[float]], x=0, y=0):
    return [[l[0][0] - x, l[0][1] - y, *l[0][2:]], [l[1][0] + x, l[1][1] + y, *l[1][2:]]]


def jerks(ls: List[List[List[float]]], js: List[List[float]]):
    return [jerk(l, *j) for l, j in zip(ls, js)]


def dupi(l: List[List[float]]):
    return [[i, i] for i in l]


def run(meas=False):
    if bench.config.sim:
        bench.scenario.simulator.cache_hits = (0, 0)
    else:
        bench.scenario.verifier.tube_cache_hits = (0, 0)
        bench.scenario.verifier.trans_cache_hits = (0, 0)
    traces = bench.run(60, 0.1)

    if bench.config.dump:
        traces.dump("main.json")
        traces.dump("tree2.json" if meas else "tree1.json")

    if bench.config.plot and meas:
        fig = go.Figure()
        if bench.config.sim:
            fig = simulation_tree(traces, bench.scenario.map, fig, 1, 2, print_dim_list=[1, 2])
        else:
            fig = reachtube_tree(
                traces, bench.scenario.map, fig, 1, 2, [1, 2], "lines", combine_rect=5
            )
        fig.show()

    if meas:
        bench.report()


if __name__ == "__main__":
    import sys
    import os

    bench = Benchmark(sys.argv)
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "decision_logic/inc-expr-switch-2.py")

    if bench.config.plot:
        import plotly.graph_objects as go
        from verse.plotter.plotter2D import simulation_tree

    smarts = [1, 3, 8]
    for i in range(8):
        id = i + 1
        if id in smarts:
            bench.scenario.add_agent(CarAgentSwitch2(f"car{id}", file_name=input_code_name))
        else:
            bench.scenario.add_agent(NPCAgent(f"car{id}"))
    tmp_map = SimpleMap4Switch2()
    bench.scenario.set_map(tmp_map)
    mode_inits = (
        [
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T0),
            (AgentMode.Normal, TrackMode.T0),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T2),
            (AgentMode.Normal, TrackMode.T2),
            (AgentMode.Normal, TrackMode.T2),
        ],
        [(LaneObjectMode.Vehicle,) for _ in range(8)],
    )

    poses = [
        [0, 0, 0, 1.0, 0],
        [10, 0, 0, 0.5, 0],
        [14, 3, 0, 0.6, 0],
        [20, 3, 0, 0.5, 0],
        [30, 0, 0, 0.5, 0],
        [28.5, -3, 0, 0.5, 0],
        [39.5, -3, 0, 0.5, 0],
        [30, -3, 0, 0.6, 0],
    ]
    _jerks = [[0, 0.05] if i + 1 in smarts else [] for i in range(8)]
    cont_inits = dupi(poses)
    if not bench.config.sim:
        cont_inits = jerks(cont_inits, _jerks)
    bench.scenario.set_init(cont_inits, *mode_inits)

    args = bench.config.args

    if "b" in args:
        run(True)
    elif "r" in args:
        run()
        run(True)
    elif "n" in args:
        run()
        poses[6][0] = 50
        cont_inits = dupi(poses)
        if not bench.config.sim:
            cont_inits = jerks(cont_inits, _jerks)
        bench.scenario.set_init(cont_inits, *mode_inits)
        run(True)
    elif "3" in args:
        run()
        bench.scenario.agent_dict["car3"] = CarAgentSwitch2(
            "car3", file_name=input_code_name.replace(".py", "-fsw7.py")
        )
        run(True)
    elif "8" in args:
        run()
        bench.scenario.agent_dict["car8"] = CarAgentSwitch2(
            "car8", file_name=input_code_name.replace(".py", "-fsw4.py")
        )
        run(True)
