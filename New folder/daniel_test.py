from verse.agents.example_agent.car_agent import CarAgent
from verse.scenario import Scenario
from verse.map.example_map.simple_map2 import SimpleMap4
from verse.map.example_map.map_tacas import M1

#from verse.map.example_map.simple_map import SimpleMap

from verse.plotter.plotter2D import *
from enum import Enum, auto

from verse.sensor.base_sensor import BaseSensor


class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()
class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()

if __name__ == "__main__":

    scenario = Scenario()
    controller1 = './demo/dryvr_demo/daniel_controller.py'

    car = CarAgent('ego', file_name=controller1)
    car2 = CarAgent('other', file_name=controller1)
    scenario.add_agent(car)
    scenario.add_agent(car2)
    scenario.set_map(M1())

    scenario.set_sensor(BaseSensor())
    scenario.set_init(
        [
            [[-1, 3, 1, 0.5],[1, 4, 3, 0.5]],
            [[2, 2, 1, 0.5], [5, 4, 15, 0.5]]

        ],
        [
            (VehicleMode.Normal, TrackMode.T0),
            (VehicleMode.Normal, TrackMode.T1)
        ]
    )
    traces = scenario.verify(50,.1)

    fig = go.Figure()
    fig = reachtube_tree(traces, M1(), fig, 1, 2, [1, 2],
                         'lines', 'trace')
    fig.show()
    #traces_simu = scenario.simulate(10, 0.01)