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

def controller(ego:State, others:State, lane_map):
    output = copy.deepcopy(ego)
    test = lambda other: other.x-ego.x > 3 and other.x-ego.x < 5 and ego.lane_mode == other.lane_mode
    if ego.vehicle_mode == VehicleMode.Normal:
        if any((test(other) and other.type_mode==LaneObjectMode.Vehicle) for other in others):
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if any(test(other) for other in others):
            if lane_map.has_right(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchRight
    lat_dist = lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y])
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if lat_dist >= 2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lat_dist <= -2.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)
    def abs_diff(a, b):
        if a < b:
            r = b - a
        else:
            r = a - b
        return r
    assert all(abs_diff(ego.x, o.x) > 5 for o in others)
    return output

