from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap3
from verse import Scenario
from verse.plotter.plotter2D import *
# from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_map
from noisy_sensor import NoisyVehicleSensor

import plotly.graph_objects as go
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
    input_code_name = './demo/vehicle/controller/example_controller11.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    # scenario.set_sensor(FakeSensor1())
    scenario.set_init(
        [
            [[10.0, 0, 0, 1], [10.0, 0, 0, 1]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
        ]
    )

    traces = scenario.simulate(20, 0.05)
    fig = go.Figure()
    fig = simulation_anime(traces, tmp_map, fig, 1, 2,
                           'lines', 'trace', print_dim_list=[1, 2, 4])
    fig.show()
