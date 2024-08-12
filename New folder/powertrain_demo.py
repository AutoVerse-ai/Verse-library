from origin_agent import powertrain_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    negAngle = auto()
    deadzone = auto()
    posAngle = auto()
    negAngleInit = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/powertrain_controller.py'
    scenario = Scenario()

    car = powertrain_agent('test', file_name=input_code_name)
    scenario.add_agent(car)
    x0 = np.array([-0.0432, -11, 0, 30, 0, 30, 360, -0.0013, 30,0])
    g = np.array([0.0056, 4.67, 0, 10, 0, 10, 120, 0.0006, 10,0])
    # modify mode list input
    scenario.set_init(
        [
            [list(x0 -g), list(x0+g)],
        ],
        [
            tuple([CraftMode.negAngleInit]),
        ]
    )

    # traces = scenario.simulate(200, 1)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()



    traces = scenario.verify(3, .1)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')


    fig.show()
