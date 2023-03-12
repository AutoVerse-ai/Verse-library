from sleeve_agent import sleeve_agent
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
import plotly.graph_objects as go
from enum import Enum, auto
import sys
import time 

class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = './demo/tacas2023/exp12/sleeve_controller.py'
    bench = Benchmark(sys.argv, init_seg_length=1)
    bench.agent_type = "G"
    bench.noisy_s = "No"

    car = sleeve_agent('sleeve', file_name=input_code_name)
    bench.scenario.add_agent(car)

    bench.scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0], [-0.0166, 0.0031, 0, 0, 0]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    time_step = 0.0001
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(0.1, time_step)
        exit(0)
    traces = bench.run(0.1, time_step)
    if bench.config.dump:
        traces.dump('./demo/gearbox/output.json')
    if bench.config.plot:        
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
        fig.show()
    bench.report()
