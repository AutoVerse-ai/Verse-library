from verse.plotter2D import *

from enum import Enum, auto
import copy
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
    type: LaneObjectMode

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, v: float = 0, vehicle_mode: VehicleMode = VehicleMode.Normal, lane_mode: LaneMode = LaneMode.Lane0, type: LaneObjectMode = LaneObjectMode.Vehicle):
        pass
        # self.data = []
        # self.x = x
        # self.y = y
        # self.theta = theta
        # self.v = v
        # self.vehicle_mode = vehicle_mode
        # self.lane_mode = lane_mode
        # self.obj_mode = obj_mode

def controller(ego: State, other: State, sign: State, lane_map):
    output = copy.deepcopy(ego)
    if ego.vehicle_mode == VehicleMode.Normal:
        if sign.type == LaneObjectMode.Obstacle and sign.x - ego.x < 3 and sign.x - ego.x > 0 and ego.lane_mode == sign.lane_mode:
            output.vehicle_mode = VehicleMode.SwitchLeft
            return output
        if lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 3 \
        and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 5 \
        and ego.lane_mode == other.lane_mode:
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 3 \
        and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 5 \
        and ego.lane_mode == other.lane_mode:
            if lane_map.has_right(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if  lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) >= 2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) <= -2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)

    return output


from verse.example.example_agent.car_agent import CarAgent
from verse.example.example_agent.sign_agent import SignAgent
from verse.scene_verifier.scenario.scenario import Scenario
from verse.map.example_map.simple_map2 import SimpleMap3
from verse.sensor.example_sensor.fake_sensor import FakeSensor2

if __name__ == "__main__":
    import sys
    input_code_name = sys.argv[0]
    scenario = Scenario()

    car = CarAgent('car1', file_name="example_controller3.py")
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_agent(SignAgent("sign"))
    scenario.set_map(SimpleMap3())
    scenario.set_sensor(FakeSensor2())
    scenario.set_init(
        [
            [[0, -0.2, 0, 1.0],[0.2, 0.2, 0, 1.0]],
            [[10, 0, 0, 0.5],[10, 0, 0, 0.5]], 
            [[20, 0, 0, 0],[20, 0, 0, 0]],
        ],
        [
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Vehicle),
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Vehicle),
            (VehicleMode.Normal, LaneMode.Lane1, LaneObjectMode.Obstacle),
        ]
    )
    # simulator = Simulator()
    traces = scenario.simulate(40,0.05)
    # traces = scenario.verify(40)

    fig = go.Figure()
    fig = simulation_anime(traces, None, fig, 1,
                           2, 'lines', 'trace', print_dim_list=[1, 2])
    fig.show()
