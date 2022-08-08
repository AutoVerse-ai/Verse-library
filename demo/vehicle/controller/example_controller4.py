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

def controller(ego:State, others:List[State], lane_map):
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
        if lat_dist >= (lane_map.get_lane_width(ego.lane_mode)-0.2):
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
            output.x = ego.x
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lat_dist <= -(lane_map.get_lane_width(ego.lane_mode)-0.2):
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)
    def abs_diff(a, b):
        # if a < b:
        #     r = b - a
        # else:
        #     r = a - b
        return a - b
    def test(o):
        # if ego.lane_mode == o.lane_mode:
        #     r = abs_diff(ego.x, o.x) > 6
        # else:
        #     r = True
        return abs_diff(o.x, ego.x) > 5.1
    # assert all(test(o) for o in others)
    # assert ego.lane_mode != LaneMode.Lane0, "lane 0"
    # assert ego.x < 40, "x"
    # assert not (ego.lane_mode == LaneMode.Lane2 and ego.x > 30 and ego.x<50), "lane 2"
    assert not (ego.x>30 and ego.x<50 and ego.y>-4 and ego.y<-2), "Danger Zone"
    return output

