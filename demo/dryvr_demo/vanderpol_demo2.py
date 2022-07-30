from origin_agent import vanderpol_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    input_code_name = './vanderpol_controller.py'
    scenario = Scenario()

    car = vanderpol_agent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    # modify mode list input
    scenario.set_init(
        [
            [[1.25, 2.25], [1.55, 2.35]],
        ],
        [
            tuple([AgentMode.Default]),
        ]
    )
    traces = scenario.verify(7, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2,
                         'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
