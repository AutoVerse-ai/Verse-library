from origin_agent import laub_loomis_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()
import os

if __name__ == "__main__":
    print(os.getcwd())
    input_code_name = './laub_loomis_controller.py'
    scenario = Scenario()

    car = laub_loomis_agent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    W  = .1
    scenario.set_init(
        [
            [[1.2-W, 1.05-W, 1.5-W, 2.4-W, 1-W,.1 -W,.45-W], [1.2+W, 1.05+W, 1.5+W, 2.4+W, 1+W,.1 +W,.45+W]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    traces = scenario.verify(20, 0.02,params={"bloating_method":"GLOBAL"})
    print(traces)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0, 4, [1, 2], 'lines', 'trace', combine_rect=3)
    fig.show()
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 1, 2, [1, 2],
    #                       'lines', 'trace')
    # fig.show()
