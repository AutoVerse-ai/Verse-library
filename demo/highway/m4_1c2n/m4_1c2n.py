from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map import opendrive_map
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *

import time
from enum import Enum, auto
import sys
import plotly.graph_objects as go


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    # ctlr_src = "demo/vehicle/controller/intersection_car.py"
    input_code_name = os.path.join(script_dir, "example_controller5.py")

    bench = Benchmark(sys.argv, init_seg_length=1)
    bench.agent_type = "C"
    bench.noisy_s = "No"
    bench.scenario.add_agent(CarAgent("car1", file_name=input_code_name))
    bench.scenario.add_agent(NPCAgent("car2"))
    bench.scenario.add_agent(NPCAgent("car3"))
    tmp_map = opendrive_map(os.path.join(script_dir, "t1_triple.xodr"))

    bench.scenario.set_map(tmp_map)
    bench.scenario.set_init(
        [
            [[134, 11.5, 0, 5.0], [136, 14.5, 0, 5.0]],
            [[179.5, 59.5, np.pi / 2, 2.5], [180.5, 62.5, np.pi / 2, 2.5]],
            [[124.5, 87.5, np.pi, 2.0], [125.5, 90.5, np.pi, 2.0]],
            # [[-65, -57.5, 0, 5.0], [-65, -57.5, 0, 5.0]],
            # [[15, -67.0, 0, 2.5], [15, -67.0, 0, 2.5]],
            # [[106, 18.0, 0, 2.0], [106, 18.0, 0, 2.0]],
        ],
        [
            (
                AgentMode.Normal,
                TrackMode.T1,
            ),
            (
                AgentMode.Normal,
                TrackMode.T1,
            ),
            (
                AgentMode.Normal,
                TrackMode.T2,
            ),
        ],
    )
    time_step = 0.1
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(60, time_step)
        exit(0)
    traces = bench.run(60, time_step)  # traces.dump('./output1.json')
    if bench.config.dump:
        traces.dump(os.path.join(script_dir, "output5.json"))
    # traces = AnalysisTree.load('./output5.json')
    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
        fig.show()
    bench.report()
