from origin_agent import thermo_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto

class ThermoMode(Enum):
    ON = auto()
    OFF = auto()

if __name__ == "__main__":
    input_code_name = './thermo_controller.py'
    scenario = Scenario()

    car = thermo_agent('test', file_name=input_code_name)
    scenario.add_agent(car)
    # modify mode list input
    scenario.set_init(
        [
            [[75.0, 0.0, 0.0], [76, 0.0, 0.0]],
        ],
        [
            tuple([ThermoMode.ON]),
        ]
    )
    traces = scenario.verify(3.5, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 2, 1,
                         'lines', 'trace', print_dim_list=[2, 1])
    fig.show()
