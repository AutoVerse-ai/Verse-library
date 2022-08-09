from verse.agents.example_agent import CarAgent
from verse.map.example_map import SimpleMap3
from verse.sensor.example_sensor import FakeSensor2
from verse import Scenario
from verse.plotter.plotter2D import *
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
    input_code_name = './demo/vehicle/controller/example_controller2.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[0, -0.2, 0, 1.0], [0.1, 0.2, 0, 1.0]],
            [[10, 0, 0, 0.5], [10, 0, 0, 0.5]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
        ]
    )

    # traces = scenario.simulate(30, 1)
    # path = os.path.abspath('.')
    # if os.path.exists(path+'/demo'):
    #     path += '/demo'
    # path += '/output'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # file = path+"/output.json"
    # write_json(traces, file)

    # root = read_json(file)
    # fig = go.Figure()
    # fig = simulation_tree(root, tmp_map, fig, 1, 2, 'lines')
    # fig.show()
    # # traces = scenario.verify(70, 0.05)
    # fig = go.Figure()
    # fig = simulation_tree(traces, tmp_map, fig, 1, 2,
    #                       'lines', 'trace', print_dim_list=[1, 2])
    # # # fig = reachtube_anime(traces, tmp_map, fig, 1,
    # # #                       2, 'lines', 'trace', print_dim_list=[1, 2])
    # fig.show()
    fig = go.Figure()
    traces = scenario.verify(30, 1)
    fig = reachtube_anime(traces, tmp_map, fig, 1,
                          2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
