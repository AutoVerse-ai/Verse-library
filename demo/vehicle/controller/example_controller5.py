from enum import Enum, auto
import copy
from typing import List

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

def vehicle_front(ego, others, lane_map):
    res = any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 3 \
            and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 5 \
            and ego.lane_mode == other.lane_mode) for other in others)
    return res

def vehicle_close(ego, others):
    res = any(ego.x-other.x<1.0 and ego.x-other.x>-1.0 and ego.y-other.y<1.0 and ego.y-other.y>-1.0 for other in others)
    return res

def controller(ego:State, others:List[State], lane_map):
    output = copy.deepcopy(ego)
    if ego.vehicle_mode == VehicleMode.Normal:
        if vehicle_front(ego, others, lane_map):
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if vehicle_front(ego, others, lane_map):
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

    assert not vehicle_close(ego, others)

    return output

