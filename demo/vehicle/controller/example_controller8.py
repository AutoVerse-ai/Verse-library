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
    Stop = auto()

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
    type: LaneObjectMode = LaneObjectMode.Vehicle

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode, type: LaneObjectMode):
        pass

def car_front(ego, others, lane_map, thresh_far, thresh_close):
    return any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > thresh_close \
            and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < thresh_far \
            and ego.lane_mode == other.lane_mode) for other in others)

def car_left(ego, others, lane_map):
    return any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 8 and \
                 lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) >-3 and \
                 other.lane_mode==lane_map.left_lane(ego.lane_mode)) for other in others)

def car_right(ego, others, lane_map):
    return any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 8 and \
                 lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) >-3 and \
                 other.lane_mode==lane_map.right_lane(ego.lane_mode)) for other in others)

def controller(ego:State, others:List[State], lane_map):
    output = copy.deepcopy(ego)
    if ego.vehicle_mode == VehicleMode.Normal:
        # Switch left if left lane is empty
        if car_front(ego, others, lane_map, 5, 3):
            if lane_map.has_left(ego.lane_mode) and \
             not car_left(ego, others, lane_map):
                output.vehicle_mode = VehicleMode.SwitchLeft
        
        # Switch right if right lane is empty
        if car_front(ego, others, lane_map, 5, 3):
            if lane_map.has_right(ego.lane_mode) and \
             not car_right(ego, others, lane_map):
                output.vehicle_mode = VehicleMode.SwitchRight
        
        # If really close just brake
        if car_front(ego, others, lane_map, 3, -0.5):
                output.vehicle_mode = VehicleMode.Stop 
                output.v = 0.1

    # If switched left enough, return to normal mode
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if  lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) >= (lane_map.get_lane_width(ego.lane_mode)-0.2):
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)

    # If switched right enough,return to normal mode
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) <= -(lane_map.get_lane_width(ego.lane_mode)-0.2):
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)

    if ego.vehicle_mode == VehicleMode.Brake:
        if all((\
            (lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) -\
            lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 5 or \
            lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) -\
            lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < -0.5) and\
            other.lane_mode==ego.lane_mode) for other in others):
            output.vehicle_mode = VehicleMode.Normal 
            output.v = 1.0

    return output

