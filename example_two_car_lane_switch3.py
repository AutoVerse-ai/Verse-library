from enum import Enum, auto
from src.scene_verifier.map.lane_map import LaneMap

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


def controller(ego: State, other: State, lane_map:LaneMap):
    output = ego
    if ego.vehicle_mode == VehicleMode.Normal:
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_right(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if  lane_map.get_lateral(ego.lane_mode, [ego.x, ego.y]) >= 2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lane_map.get_lateral(ego.lane_mode, [ego.x, ego.y]) <= -2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)

    return output


from src.example.example_agent.car_agent2 import CarAgent2
from src.scene_verifier.scenario.scenario import Scenario
from src.example.example_map.simple_map2 import SimpleMap3, SimpleMap4
from src.plotter.plotter2D import *
from src.example.example_sensor.fake_sensor import FakeSensor2

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    input_code_name = 'example_two_car_lane_switch2.py'
    scenario = Scenario()

    car = CarAgent2('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent2('car2', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_map(SimpleMap4())
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[10, 0, 0, 0.5],[10, 0, 0, 0.5]], 
            [[0, -0.2, 0, 1.0],[0.1, 0.2, 0, 1.0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1)
        ]
    )
    # res_list = scenario.simulate_multi(40,10)
    traces = scenario.verify(33)

    fig = plt.figure(2)
    fig = plot_reachtube_tree(traces, 'car1', 1, [2], 'b', fig)
    fig = plot_reachtube_tree(traces, 'car2', 1, [2], 'r', fig)
    # for traces in res_list:
    #     fig = plot_simulation_tree(traces, 'car1', 1, [2], 'b', fig)
    #     fig = plot_simulation_tree(traces, 'car2', 1, [2], 'r', fig)

    plt.show()
