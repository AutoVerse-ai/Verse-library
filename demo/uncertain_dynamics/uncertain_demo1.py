
from uncertain_agents import Agent1
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.plotter.plotter2D_old import plot_reachtube_tree

import matplotlib.pyplot as plt  
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = './uncertain_dynamics/controller1.py'
    scenario = Scenario()

    car = Agent1('car1')
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    scenario.set_init(
        [
            [[1, 1], [1, 1]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
        uncertain_param_list=[
            [[-0.1, -0.1], [0.1, 0.1]]
        ]
    )
    traces = scenario.verify(10, 0.01, reachability_method='MIXMONO_DISC')
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [1])
    plt.show()
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 1, 2,
    #                       'lines', 'trace', print_dim_list=[1, 2])
    # fig.show()
