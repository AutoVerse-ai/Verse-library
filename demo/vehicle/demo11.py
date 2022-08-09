from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap3, SimpleMap6
from verse import Scenario
from noisy_sensor import NoisyVehicleSensor
from verse.plotter.plotter2D import *
from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_map

from enum import Enum, auto
import plotly.graph_objects as go
import matplotlib.pyplot as plt 

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
    x: float
    y: float
    theta: float
    v: float
    vehicle_mode: VehicleMode
    lane_mode: LaneMode
    type_mode: LaneObjectMode

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode, type_mode: LaneObjectMode):
        pass


if __name__ == "__main__":
    input_code_name = './demo/vehicle/controller/example_controller5.py'

    scenario = Scenario()
    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent('car2')
    scenario.add_agent(car)
    car = NPCAgent('car3')
    scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]],
            [[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]],
            [[4-2.5, 2.8, 0, 1.0], [4.5-2.5, 3.2, 0, 1.0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane0),
        ]
    )
    scenario.set_sensor(NoisyVehicleSensor((0.5,0.5), (0,0)))

    scenario.init_seg_length = 5
    traces = scenario.verify(40, 0.1)

    fig = plt.figure(2)
    fig = plot_reachtube_tree(traces.root, 'car1', 1, [2], 'b', fig)
    fig = plot_reachtube_tree(traces.root, 'car2', 1, [2], 'r', fig)
    fig = plot_reachtube_tree(traces.root, 'car3', 1, [2], 'r', fig)
    fig = plot_map(tmp_map, 'g', fig)

    scenario1 = Scenario()
    car = CarAgent('car1', file_name=input_code_name)
    scenario1.add_agent(car)
    car = NPCAgent('car2')
    scenario1.add_agent(car)
    car = NPCAgent('car3')
    scenario1.add_agent(car)
    tmp_map = SimpleMap3()
    scenario1.set_map(tmp_map)
    scenario1.set_init(
        [
            [[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]],
            [[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]],
            [[4-2.5, 2.8, 0, 1.0], [4.5-2.5, 3.2, 0, 1.0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1,),
            (VehicleMode.Normal, LaneMode.Lane1,),
            (VehicleMode.Normal, LaneMode.Lane0,),
        ]
    )

    scenario1.init_seg_length = 5
    traces = scenario1.verify(40, 0.1)

    fig = plot_reachtube_tree(traces.root, 'car1', 1, [2], 'g', fig)
    plt.show()