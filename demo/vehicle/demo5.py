from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap3
from verse import Scenario
from verse.plotter.plotter2D import *
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


if __name__ == "__main__":
    input_code_name = './demo/vehicle/controller/example_controller7.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent('car2')
    scenario.add_agent(car)
    car = NPCAgent('car3')
    scenario.add_agent(car)
    car = NPCAgent('car4')
    scenario.add_agent(car)
    # car = NPCAgent('car5')
    # scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[0, -0.0, 0, 1.0], [0.0, 0.0, 0, 1.0]],
            [[10, 0, 0, 0.5], [10, 0, 0, 0.5]],
            [[30, 0, 0, 0.5], [30, 0, 0, 0.5]],
            [[10, 3, 0, 0.5], [10, 3, 0, 0.5]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane0),
        ],
        [
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
            (LaneObjectMode.Vehicle,),
        ]
    )
    traces = scenario.simulate(70, 0.05)
    fig = go.Figure()
    fig = simulation_tree(traces, tmp_map, fig, 1,
                                 2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()

    traces = scenario.verify(70, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1,
                                 2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
