from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from noisy_sensor import NoisyVehicleSensor
from verse.plotter.plotter2D import *

from enum import Enum, auto
import time
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from verse.scenario.scenario import ScenarioConfig


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
    input_code_name = os.path.join(script_dir, "example_controller5.py")

    bench = Benchmark(sys.argv, init_seg_length=5)
    bench.agent_type = "C"
    bench.noisy_s = "Yes"
    car = CarAgent("car1", file_name=input_code_name)
    bench.scenario.add_agent(car)
    car = NPCAgent("car2")
    bench.scenario.add_agent(car)
    car = NPCAgent("car3")
    bench.scenario.add_agent(car)
    tmp_map = M1()
    bench.scenario.set_map(tmp_map)
    bench.scenario.set_init(
        [
            [[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]],
            [[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]],
            [[4 - 2.5, 2.8, 0, 1.0], [4.5 - 2.5, 3.2, 0, 1.0]],
        ],
        [
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T0),
        ],
    )
    bench.scenario.set_sensor(NoisyVehicleSensor((0.5, 0.5), (0.0, 0.0)))
    time_step = 0.1
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(40, time_step, params={"bloating_method": "GLOBAL"})
        exit(0)
    traces = bench.run(40, time_step, params={"bloating_method": "GLOBAL"})
    if bench.config.dump:
        traces.dump(os.path.join(script_dir, "output4_noisy.json"))
    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    bench.report()
