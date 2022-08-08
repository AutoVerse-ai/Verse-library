from verse.agents.example_agent import CarAgent, SignAgent
from verse.map.example_map import SimpleMap3
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.plotter.plotter2D_old import *

from enum import Enum, auto
import plotly.graph_objects as go

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
    input_code_name = './demo/vehicle/controller/example_controller9.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    sign = SignAgent('stop_sign')
    scenario.add_agent(sign)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[0, 0, 0, 1], [0.5, 0, 0, 1]],
            [[15, 0, 0, 0.5], [15, 0, 0, 0.5]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Vehicle),
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Sign),
        ]
    )
    # traces = scenario.simulate(13, 0.05)
    scenario.init_seg_length = 5
    scenario.verify_method = 'GLOBAL'
    traces = scenario.verify(13, 0.05)

    # fig = plt.figure(2)
    # fig = plot_map(tmp_map, 'g', fig)
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [1], 'b')
    fig = plot_reachtube_tree(traces.root, 'stop_sign', 0, [1], 'r', fig)
    # fig = plot_reachtube_tree(traces, 'car3', 1, [2], 'r', fig)
    # fig = plot_reachtube_tree(traces, 'car4', 1, [2], 'r', fig)
    plt.show()

    # fig = go.Figure()
    # fig = simulation_anime(traces, tmp_map, fig, 1,
    #                        2, 'lines', 'trace', print_dim_list=[1, 2])
    # fig.show()
