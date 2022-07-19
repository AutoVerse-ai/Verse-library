from dryvr_plus_plus.example.example_agent.origin_agent import craft_agent
from dryvr_plus_plus.scene_verifier.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap2, SimpleMap3, SimpleMap5, SimpleMap6
from dryvr_plus_plus.plotter.plotter2D import *
from dryvr_plus_plus.example.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/rendezvous_controller.py'
    scenario = Scenario()

    car = craft_agent('test', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.set_sensor(CraftSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.ProxA]),
        ]
    )
    traces = scenario.verify(200, 1)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2,
                         'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
