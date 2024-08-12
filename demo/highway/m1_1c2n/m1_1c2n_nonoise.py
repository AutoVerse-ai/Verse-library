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
    bench.noisy_s = "No"
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
    time_step = 0.1
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(40, time_step, params={"bloating_method": "GLOBAL"})
        exit(0)
    traces = bench.run(40, time_step, params={"bloating_method": "GLOBAL"})
    if bench.config.dump:
        traces.dump(os.path.join(script_dir, "output4_nonoise.json"))
    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(
            traces,
            tmp_map,
            fig,
            1,
            2,
            [1, 2],
            "lines",
            "trace",
            plot_color=[
                ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],  # red
                ["#0000CC", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF"],  # blue
                ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#CCFFCC"],  # green
                ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],  # yellow
                #   ['#66CC00', '#80FF00', '#99FF33', '#B2FF66', '#CCFF99', '#FFFFCC'], # yellowgreen
                ["#CC00CC", "#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF", "#FFCCFF"],  # magenta
                #   ['#00CC66', '#00FF80', '#33FF99', '#66FFB2', '#99FFCC', '#CCFFCC'], # springgreen
                ["#00CCCC", "#00FFFF", "#33FFFF", "#66FFFF", "#99FFFF", "#CCFFE5"],  # cyan
                #   ['#0066CC', '#0080FF', '#3399FF', '#66B2FF', '#99CCFF', '#CCE5FF'], # cyanblue
                ["#CC6600", "#FF8000", "#FF9933", "#FFB266", "#FFCC99", "#FFE5CC"],  # orange
                #   ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'], # purple
                ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],  # lime
                ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],  # pink
            ],
        )
        fig.show()
    bench.report()
