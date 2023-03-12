from origin_agent import craft_agent
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor
import time 
import sys 

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    input_code_name = './demo/tacas2023/exp12/rendezvous_controller.py'
    bench = Benchmark(sys.argv)
    bench.agent_type = "S"
    bench.noisy_s = "No"
    car = craft_agent('test', file_name=input_code_name)
    bench.scenario.add_agent(car)
    bench.scenario.set_sensor(CraftSensor())
    # modify mode list input
    bench.scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.ProxA]),
        ]
    )
    time_step = 1
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(200, time_step)
        exit(0)
    traces = bench.run(200, time_step)

    if bench.config.plot:       
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                                'lines', 'trace')
        fig.show()
    bench.report()