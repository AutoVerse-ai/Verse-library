from origin_agent import vanderpol_agent
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
import time
import sys

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = "./demo/tacas2023/exp12/vanderpol_controller.py"
    bench = Benchmark(sys.argv)
    bench.agent_type = "V"
    bench.noisy_s = "N/A"
    car = vanderpol_agent("car1", file_name=input_code_name)
    bench.scenario.add_agent(car)
    # modify mode list input
    bench.scenario.set_init(
        [
            [[1.25, 2.25], [1.55, 2.35]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
    )
    time_step = 0.01
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(7, time_step)
        exit(0)
    traces = bench.run(7, time_step)

    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
        fig.show()
    bench.report()
