from dryvr_plus_plus.example.example_agent.car_agent import CarAgent, NPCAgent
from dryvr_plus_plus.example.example_agent.sign_agent import SignAgent
from dryvr_plus_plus.scene_verifier.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap2, SimpleMap3, SimpleMap5, SimpleMap6
from dryvr_plus_plus.plotter.plotter2D import *
from dryvr_plus_plus.example.example_sensor.fake_sensor import FakeSensor3

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from enum import Enum, auto

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles

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
    type_mode: LaneObjectMode = LaneObjectMode.Vehicle

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode, type_mode: LaneObjectMode):
        pass


if __name__ == "__main__":
    input_code_name = './example_controller9.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    sign = SignAgent('stop_sign')
    scenario.add_agent(sign)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[0, 0, 0, 1.0],[0, 0, 0, 1.0]],
            [[15, 0, 0, 0.5],[15, 0, 0, 0.5]], 
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Vehicle),
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Sign),
        ]
    )
    traces = scenario.simulate(70)
    # traces = scenario.verify(70)

    # fig = plt.figure(2)
    # fig = plot_map(tmp_map, 'g', fig)
    # fig = plot_reachtube_tree(traces, 'car1', 1, [2], 'b', fig)
    # fig = plot_reachtube_tree(traces, 'car2', 1, [2], 'r', fig)
    # fig = plot_reachtube_tree(traces, 'car3', 1, [2], 'r', fig)
    # fig = plot_reachtube_tree(traces, 'car4', 1, [2], 'r', fig)
    # plt.show()    

    fig = go.Figure()
    fig = plotly_simulation_anime(traces, tmp_map, fig)
    fig.show()

