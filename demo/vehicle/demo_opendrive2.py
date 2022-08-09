from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map import opendrive_map
from verse import Scenario
from enum import Enum, auto
from verse.plotter.plotter2D import *

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
    input_code_name = './demo/vehicle/controller/example_controller8.py'
    scenario = Scenario()
    scenario.add_agent(CarAgent('car1', file_name=input_code_name))
    scenario.add_agent(NPCAgent('car2'))
    tmp_map = opendrive_map('./demo/vehicle/t1_triple.xodr')

    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[-65, -57.5, 0, 1.0],[-65, -57.5, 0, 1.0]],  
            # [[-37, -63.0, 0, 0.5],[-37, -63.0, 0, 0.5]],
            # [[18, -67.0, 0, 1.0],[18, -67.0, 0, 1.0]], 
            # [[32, -68.0, 0, 1.0],[32, -68.0, 0, 1.0]], 
            [[46, -69.0, 0, 0.5],[46, -69.0, 0, 0.5]], 
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane2, ),
            (VehicleMode.Normal, LaneMode.Lane2, ),
        ],
        [
            (LaneObjectMode.Vehicle, ),
            (LaneObjectMode.Vehicle, ),
        ]
    )
    traces = scenario.simulate(400, 0.1)# traces.dump('./output1.json')
    fig = go.Figure()
    fig = simulation_tree(traces, tmp_map, fig, 1,
                          2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()

