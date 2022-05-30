from dryvr_plus_plus.example.example_agent.car_agent import CarAgent, NPCAgent
from dryvr_plus_plus.scene_verifier.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap2, SimpleMap3, SimpleMap5, SimpleMap6
from dryvr_plus_plus.plotter.plotter2D import *
from dryvr_plus_plus.example.example_sensor.fake_sensor import FakeSensor2

import matplotlib.pyplot as plt
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
    input_code_name = 'example_controller1.py'
    scenario = Scenario()

    car = NPCAgent('car1')
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    tmp_map = SimpleMap2()
    scenario.set_map(tmp_map)
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[10.0, 0, 0, 0.5], [10.0, 0, 0, 0.5]],
            [[5.0, -0.2, 0, 2.0], [6.0, 0.2, 0, 3.0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
        ]
    )
    res_list = scenario.simulate_multi(10, 1)
    # traces = scenario.verify(10)

    # fig = plt.figure(2)
    # fig = plot_map(tmp_map, 'g', fig)
    # plt.show()
    # # fig = plot_reachtube_tree(traces, 'car1', 0, [1], 'b', fig, (1000,-1000), (1000,-1000))
    # # fig = plot_reachtube_tree(traces, 'car2', 0, [1], 'r', fig)
    # # AnalysisTreeNode
    # for traces in res_list:
    #     # fig = plot_simulation_tree(
    #     #     traces, 'car1', 0, [1], 'b', fig, (1000, -1000), (1000, -1000))
    #     fig = plot_simulation_tree(traces, 'car2', 1, [2], 'r', fig)
    #     # generate_simulation_anime(traces, tmp_map, fig)

    # plt.show()

    fig = go.Figure()
    for traces in res_list:
        # plotly_map(tmp_map, 'g', fig)
        fig = plotly_simulation_anime(traces, tmp_map, fig)
    fig.show()
