from uncertain_agents import Agent6
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *

import time
import sys
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))

    bench = Benchmark(sys.argv, reachability_method="MIXMONO_CONT")
    bench.agent_type = "C"
    bench.noisy_s = "No"
    car = Agent6("car1")
    bench.scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    bench.scenario.set_init(
        [
            [[0.3, 0.3], [2, 2]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
        uncertain_param_list=[
            [[-0.1, -0.1], [0.1, 0.1]],
        ],
    )
    time_step = 0.01
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(10, time_step)
        exit(0)
    traces = bench.run(10, time_step)
    # fig = plt.figure(0)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [1],fig=fig)
    # fig = plt.figure(1)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [2],fig=fig)
    # fig = plt.figure(4)
    # fig = plot_reachtube_tree(traces.root, 'car1', 1, [2],fig=fig)
    # plt.show()
    if bench.config.dump:
        traces.dump(os.path.join(script_dir, "output7.json"))
    # traces = AnalysisTree.load('./output7.json')
    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1], "lines", "trace")
        fig.show()
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 0, 2, [0, 2], "lines", "trace")
        fig.show()
    bench.report()
