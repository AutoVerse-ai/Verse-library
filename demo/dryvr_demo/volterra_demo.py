from origin_agent import volterra_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    inside = auto()
    outside = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/volterra_controller.py'
    scenario = Scenario()

    car = volterra_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    e = .012
    scenario.set_init(
        [
            [[1.3 - e, 1], [1.3 + e, 1]],
        ],
        [
            tuple([CraftMode.outside]),
        ]
    )

    # traces = scenario.simulate(200, 1)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()



    traces = scenario.verify(3.64, .01)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    fig.show()
