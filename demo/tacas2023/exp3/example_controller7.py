from enum import Enum, auto
import copy
from typing import List

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
    M01 = auto()
    M12 = auto() 
    M23 = auto() 
    M32 = auto()
    M21 = auto()
    M10 = auto()

class State:
    x:float
    y:float
    theta:float
    v:float
    agent_mode:VehicleMode
    lane_mode:TrackMode

    def __init__(self, x, y, theta, v, agent_mode: VehicleMode, lane_mode: TrackMode):
        pass

def car_front(ego, others, lane_map):
    return any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 3 \
            and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 5 \
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
    if ego.agent_mode == VehicleMode.Normal:
        if car_front(ego, others, lane_map):
            if lane_map.h_exist(ego.lane_mode, ego.agent_mode, 'SwitchLeft') and \
             not car_left(ego, others, lane_map):
                output.agent_mode = VehicleMode.SwitchLeft
                output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, 'SwitchLeft')
        if car_front(ego, others, lane_map):
            if lane_map.h_exist(ego.lane_mode, ego.agent_mode, 'SwitchRight') and \
             not car_right(ego, others, lane_map):
                output.agent_mode = VehicleMode.SwitchRight
                output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, 'SwitchRight')
    if ego.agent_mode == VehicleMode.SwitchLeft:
        if  lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) >= 2.5:
            output.agent_mode = VehicleMode.Normal
            output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, 'Normal')
    if ego.agent_mode == VehicleMode.SwitchRight:
        if lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y]) <= -2.5:
            output.agent_mode = VehicleMode.Normal
            output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, 'Normal')

    return output

