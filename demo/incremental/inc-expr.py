# SM: Norng some things about the example

from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap4

from enum import Enum, auto
from verse.scenario.scenario import Benchmark
import functools, pprint

pp = functools.partial(pprint.pprint, compact=True, width=130)
from typing import List
import copy


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()
    Stop = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M01 = auto()
    M12 = auto()
    M23 = auto()
    M40 = auto()
    M04 = auto()
    M32 = auto()
    M21 = auto()
    M10 = auto()


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


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

    if bench.config.dump and meas:
        traces.dump("main.json")
        traces.dump("tree2.json" if meas else "tree1.json")

    if bench.config.plot and meas:
        traces.visualize_dot("/home/haoqing/inc-2-twopi", font="Iosevka Term")
        traces.visualize_dot("/home/haoqing/inc-2-dot", engine="dot", font="Iosevka Term")
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
    return traces


if __name__ == "__main__":
    import sys
    import os

    bench = Benchmark(sys.argv)
    # input_code_name = os.path.join(script_dir, 'decision_logic/inc-expr6.py') if "6" in bench.config.args else os.path.join(script_dir, 'decision_logic/inc-expr.py')
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "decision_logic/inc-expr.py")
    bench.agent_type = "D"
    bench.noisy_s = "No"
    if bench.config.plot:
        import plotly.graph_objects as go
        from verse.plotter.plotter2D import simulation_tree, reachtube_tree

    bench.scenario.add_agent(CarAgent("car1", file_name=input_code_name))
    bench.scenario.add_agent(NPCAgent("car2"))
    bench.scenario.add_agent(CarAgent("car3", file_name=input_code_name))
    bench.scenario.add_agent(NPCAgent("car4"))
    bench.scenario.add_agent(NPCAgent("car5"))
    bench.scenario.add_agent(NPCAgent("car6"))
    bench.scenario.add_agent(NPCAgent("car7"))
    # if "6" not in bench.config.args:
    bench.scenario.add_agent(CarAgent("car8", file_name=input_code_name))
    tmp_map = SimpleMap4()
    bench.scenario.set_map(tmp_map)
    # if "6" in bench.config.args:
    #     mode_inits = ([
    #             (AgentMode.Normal, TrackMode.T1),
    #             (AgentMode.Normal, TrackMode.T1),
    #             (AgentMode.Normal, TrackMode.T0),
    #             (AgentMode.Normal, TrackMode.T0),
    #             (AgentMode.Normal, TrackMode.T1),
    #             (AgentMode.Normal, TrackMode.T2),
    #             (AgentMode.Normal, TrackMode.T3),
    #         ],
    #         [
    #             (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
    #             (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
    #             (LaneObjectMode.Vehicle,), (LaneObjectMode.Vehicle,),
    #             (LaneObjectMode.Vehicle,),
    #         ])
    #     poses = [
    #         [0, 0, 0, 1],
    #         [10, 0, 0, 0.5],
    #         [14.5, 3, 0, 0.6],
    #         [25, 3, 0, 0.5],
    #         [30, 0, 0, 0.5],
    #         [23, -3, 0, 0.5],
    #         [40, -6, 0, 0.5],
    #     ]
    # else:
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
        [
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
        ],
    )
    poses = [
        [0, 0, 0, 1.0],
        [10, 0, 0, 0.5],
        [14, 3, 0, 0.6],
        [20, 3, 0, 0.5],
        [30, 0, 0, 0.5],
        [28.5, -3, 0, 0.5],
        [39.5, -3, 0, 0.5],
        [30, -3, 0, 0.6],
    ]
    _jerks = [
        [0, 0.05],
        [],
        [0, 0.05],
        [],
        [],
        [],
        [],
        [0, 0.05],
    ]
    cont_inits = dupi(poses)
    if not bench.config.sim:
        cont_inits = jerks(cont_inits, _jerks)
    bench.scenario.set_init(cont_inits, *mode_inits)
    backup_scenario = copy.deepcopy(bench.scenario)

    args = bench.config.args

    def inc_test():
        if "b" in args:
            return run(True)
        elif "r" in args:
            run()
            return run(True)
        elif "n" in args:
            run()
            poses[6][0] = 50
            cont_inits = dupi(poses)
            if not bench.config.sim:
                cont_inits = jerks(cont_inits, _jerks)
            bench.scenario.set_init(cont_inits, *mode_inits)
            return run(True)
        elif "1" in args:
            run()
            bench.swap_dl("car3", input_code_name.replace(".py", "-fsw7.py"))
            return run(True)
        elif "2" in args:
            run()
            bench.swap_dl("car8", input_code_name.replace(".py", "-fsw4.py"))
            return run(True)

    if not bench.config.compare:
        inc_test()
    else:
        trace1 = inc_test()
        bench.replace_scenario(backup_scenario)
        trace2 = inc_test()
        print("trace1 contains trace2?", trace1.contains(trace2))
        print("trace2 contains trace1?", trace2.contains(trace1))
