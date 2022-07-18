from dryvr_plus_plus.example.example_agent.car_agent import CarAgent, NPCAgent
from dryvr_plus_plus.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap2, SimpleMap3, SimpleMap5, SimpleMap6
from dryvr_plus_plus.example.example_sensor.fake_sensor import FakeSensor2
from enum import Enum, auto
import plotly.graph_objects as go
from dryvr_plus_plus.plotter.plotter2D import *


import numpy as np
from enum import Enum, auto


class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()


class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: VehicleMode = VehicleMode.Normal
    lane_mode: LaneMode = LaneMode.Lane0

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode):
        self.data = []


if __name__ == "__main__":
    input_code_name = './demo/example_controller2.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[0, -0.2, 0, 1.0], [0.1, 0.2, 0, 1.0]],
            [[10, 0, 0, 0.5], [10, 0, 0, 0.5]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
        ]
    )

    traces = scenario.simulate(70, 0.05)
    # traces = scenario.verify(70, 0.05)
    fig = go.Figure()
    fig = simulation_anime_trail(traces, tmp_map, fig, 1, 2, 'lines')
    # fig = reachtube_anime(traces, tmp_map, fig, 1,
    #                       2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
