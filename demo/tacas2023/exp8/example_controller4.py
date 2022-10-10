from enum import Enum, auto
import copy
from typing import List

from verse.map.lane_map import LaneMap

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

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
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

def vehicle_front(ego, others, lane_map):
    res = any((lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 3 \
            and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 5 \
            and ego.lane_mode == other.lane_mode) for other in others)
    return res

def vehicle_close(ego, others):
    res = any(ego.x-other.x<1.0 and ego.x-other.x>-1.0 and ego.y-other.y<1.0 and ego.y-other.y>-1.0 for other in others)
    return res

def enter_unsafe_region(ego):
    res = (ego.x > 30 and ego.x<40 and ego.lane_mode == TrackMode.T2)
    return res

def controller(ego:State, others:List[State], lane_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == VehicleMode.Normal:
        if vehicle_front(ego, others, lane_map):
            if lane_map.h_exist(ego.lane_mode, ego.agent_mode, VehicleMode.SwitchLeft):
                output.agent_mode = VehicleMode.SwitchLeft
                output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, VehicleMode.SwitchLeft)
        if vehicle_front(ego, others, lane_map):
            if lane_map.h_exist(ego.lane_mode, ego.agent_mode, VehicleMode.SwitchRight):
                output.agent_mode = VehicleMode.SwitchRight
                output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, VehicleMode.SwitchRight)
    lat_dist = lane_map.get_lateral_distance(ego.lane_mode, [ego.x, ego.y])
    if ego.agent_mode == VehicleMode.SwitchLeft:
        if lat_dist >= 2.5:
            output.agent_mode = VehicleMode.Normal
            output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, VehicleMode.Normal)
    if ego.agent_mode == VehicleMode.SwitchRight:
        if lat_dist <= -2.5:
            output.agent_mode = VehicleMode.Normal
            output.lane_mode = lane_map.h(ego.lane_mode, ego.agent_mode, VehicleMode.Normal)

    assert not enter_unsafe_region(ego), 'Unsafe Region'
    assert not vehicle_close(ego, others), 'Safe Seperation'
    return output

