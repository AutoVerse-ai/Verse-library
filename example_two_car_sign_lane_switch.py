from enum import Enum, auto
import copy

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
    obj_mode: LaneObjectMode

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, v: float = 0, vehicle_mode: VehicleMode = VehicleMode.Normal, lane_mode: LaneMode = LaneMode.Lane0, obj_mode: LaneObjectMode = LaneObjectMode.Vehicle):
        self.data = []
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.vehicle_mode = vehicle_mode
        self.lane_mode = lane_mode
        self.obj_mode = obj_mode

def controller(ego: State, other: State, sign: State, lane_map):
    output = copy.deepcopy(ego)
    if sign.y - ego.y < 3 and sign.lane_mode == ego.lane_mode:
        output.vehicle_mode = VehicleMode.SwitchLeft
        return output
    if ego.vehicle_mode == VehicleMode.Normal:
        # A simple example to demonstrate how our tool can handle change in controller
        # if ego.x > 30 and ego.lane_mode == LaneMode.Lane0:
        #     output.vehicle_mode = VehicleMode.SwitchRight
        
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_right(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if  lane_map.lane_geometry(ego.lane_mode) - ego.y <= -2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lane_map.lane_geometry(ego.lane_mode)-ego.y >= 2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)

    return output


from src.example.example_agent.car_agent import CarAgent
from src.example.example_agent.sign_agent import SignAgent
from src.scene_verifier.scenario.scenario import Scenario
from src.example.example_map.simple_map import SimpleMap2
from src.plotter.plotter2D import plot_tree
from src.example.example_sensor.fake_sensor import FakeSensor2

import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_code_name = 'example_two_car_lane_switch.py'
    scenario = Scenario()

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_agent(SignAgent("sign", file_name=input_code_name))
    scenario.add_map(SimpleMap2())
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[10, 0, 0, 0.5],[10, 0, 0, 0.5]], 
            [[-0.2, -0.2, 0, 1.0],[0.2, 0.2, 0, 1.0]],
            [[20, 3, 0, 0], [20, 3, 0, 0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane2),
        ]
    )
    # simulator = Simulator()
    # traces = scenario.simulate(40)
    traces = scenario.verify(40)

    fig = plt.figure()
    fig = plot_tree(traces, 'car1', 1, [2], 'b', fig)
    fig = plot_tree(traces, 'car2', 1, [2], 'r', fig)

    plt.show()

    # plt.plot([0, 40], [3, 3], 'g')
    # plt.plot([0, 40], [0, 0], 'g')
    # plt.plot([0, 40], [-3, -3], 'g')

    # queue = [traces]
    # while queue != []:
    #     node = queue.pop(0)
    #     traces = node.trace
    #     # for agent_id in traces:
    #     agent_id = 'car2'
    #     trace = np.array(traces[agent_id])
    #     plt.plot(trace[:, 1], trace[:, 2], 'r')

    #     agent_id = 'car1'
    #     trace = np.array(traces[agent_id])
    #     plt.plot(trace[:, 1], trace[:, 2], 'b')

    #     # if node.child != []:
    #     queue += node.child
    # plt.show()
