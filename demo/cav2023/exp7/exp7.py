
from uncertain_agents import Agent6
from verse import Scenario
from verse.scenario import ScenarioConfig
from verse.plotter.plotter2D import *

import time
import sys
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(reachability_method='MIXMONO_CONT',parallel=False))

    car = Agent6('car1')
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    scenario.set_init(
        [
            [[0.3, 0.3], [2, 2]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
        uncertain_param_list=[
            [[-0.1, -0.1], [0.1, 0.1]],
        ]
    )
    start_time = time.time()
    traces = scenario.verify(10, 0.01)
    run_time = time.time() - start_time
    # fig = plt.figure(0)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [1],fig=fig)
    # fig = plt.figure(1)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [2],fig=fig)
    # fig = plt.figure(4)
    # fig = plot_reachtube_tree(traces.root, 'car1', 1, [2],fig=fig)
    # plt.show()
    traces.dump("./demo/cav2023/exp7/output7.json")
    # traces = AnalysisTree.load('./output7.json')
    if len(sys.argv)>1 and sys.argv[1]=='p':
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1],
                            'lines', 'trace')
        fig.show()
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 0, 2, [0, 2],
                            'lines', 'trace')
        fig.show()
