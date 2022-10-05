from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap4
from verse import Scenario
from verse.plotter.plotter2D import *

from enum import Enum, auto
import plotly.graph_objects as go

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M01 = auto()
    M12 = auto() 
    M23 = auto() 
    M40 = auto() 
    M04 = auto() 
    M32 = auto()
    M21 = auto()
    M10 = auto()

if __name__ == "__main__":
    input_code_name = './demo/tacas2023/exp3/example_controller7.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent('car2')
    scenario.add_agent(car)
    car = CarAgent('car3', file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent('car4')
    scenario.add_agent(car)
    car = NPCAgent('car5')
    scenario.add_agent(car)
    car = NPCAgent('car6')
    scenario.add_agent(car)
    car = NPCAgent('car7')
    scenario.add_agent(car)
    tmp_map = SimpleMap4()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[0, -0.1, 0, 1.0], [0.0, 0.1, 0, 1.0]],
            [[10, -0.1, 0, 0.5], [10, 0.1, 0, 0.5]],
            [[14.5, 2.9, 0, 0.6], [14.5, 3.1, 0, 0.6]],
            [[20, 2.9, 0, 0.5], [20, 3.1, 0, 0.5]],
            [[30, -0.1, 0, 0.5], [30, 0.1, 0, 0.5]],
            [[23, -3.1, 0, 0.5], [23, -2.9, 0, 0.5]],
            [[40, -6.1, 0, 0.5], [40, -5.9, 0, 0.5]],
        ],
        [
            (VehicleMode.Normal, TrackMode.T1),
            (VehicleMode.Normal, TrackMode.T1),
            (VehicleMode.Normal, TrackMode.T0),
            (VehicleMode.Normal, TrackMode.T0),
            (VehicleMode.Normal, TrackMode.T1),
            (VehicleMode.Normal, TrackMode.T2),
            (VehicleMode.Normal, TrackMode.T3),
        ]
    )
 
    # traces = scenario.simulate(80, 0.05)
    # fig = go.Figure()
    # fig = simulation_anime(traces, tmp_map, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()

    traces = scenario.verify(60, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], 'lines', 'trace')
    fig.show()
