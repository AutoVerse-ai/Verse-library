from origin_agent import robertson_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()
import os

if __name__ == "__main__":
    print(os.getcwd())
    input_code_name = './demo/dryvr_demo/robertson_controller.py'
    fig = go.Figure()

    scenario1 = Scenario()
    car1 = robertson_agent('car1', file_name=input_code_name)
    scenario1.add_agent(car1)
    scenario1.set_init(
        [
            [[1, 0, 0, 10 ** 3, 10 ** 7], [1, 0, 0, 10 ** 3, 10 ** 7]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    traces1 = scenario1.verify(40, .1)
    fig = reachtube_tree(traces1, None, fig, 0, 1, [0, 1], 'lines', 'trace', combine_rect=3)
    scenario = Scenario()
    car = robertson_agent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.set_init(
        [
            [[1, 0, 0, 10**2, 10**3], [1, 0, 0, 10**2, 10**3]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    traces = scenario.verify(40, .1)
    print(traces)
    fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1], 'lines', 'trace', combine_rect=3)

    scenario2 = Scenario()
    car2 = robertson_agent('car1', file_name=input_code_name)
    scenario2.add_agent(car2)
    scenario2.set_init(
        [
            [[1, 0, 0, 10 ** 3, 10 ** 5], [1, 0, 0, 10 ** 3, 10 ** 5]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    traces2 = scenario2.verify(40, .1)
    fig = reachtube_tree(traces2, None, fig, 0, 1, [0, 1], 'lines', 'trace', combine_rect=3)
    fig.update_layout(
        xaxis_title="t", yaxis_title="s"
    )


    fig.show()
