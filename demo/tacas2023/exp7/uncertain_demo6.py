
from uncertain_agents import Agent6
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    scenario = Scenario()

    car = Agent6('car1')
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    scenario.set_init(
        [
             [[0.3,0.3], [2,2]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
        uncertain_param_list=[
            [[-0.1,-0.1],[0.1,0.1]],
        ]
    )
    traces = scenario.verify(10, 0.01, reachability_method='MIXMONO_CONT')
    # fig = plt.figure(0)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [1],fig=fig)
    # fig = plt.figure(1)
    # fig = plot_reachtube_tree(traces.root, 'car1', 0, [2],fig=fig)
    # fig = plt.figure(4)
    # fig = plot_reachtube_tree(traces.root, 'car1', 1, [2],fig=fig)
    # plt.show()
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1],
                          'lines', 'trace')
    fig.show()
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0, 2, [0, 2],
                          'lines', 'trace')
    fig.show()
