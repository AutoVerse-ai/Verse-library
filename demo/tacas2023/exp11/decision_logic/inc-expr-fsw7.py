from enum import Enum, auto
import copy
from typing import List

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles

class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()
    Stop = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M40 = auto()
    M01 = auto()
    M12 = auto()
    M23 = auto()
    M04 = auto()
    M10 = auto()
    M21 = auto()
    M32 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    agent_mode: AgentMode = AgentMode.Normal
    track_mode: TrackMode = TrackMode.T0
    type: LaneObjectMode = LaneObjectMode.Vehicle

def car_front(ego, others, track_map, thresh_far, thresh_close):
    return any((thresh_far > track_map.get_longitudinal_position(other.track_mode, [other.x,other.y]) - track_map.get_longitudinal_position(ego.track_mode, [ego.x,ego.y]) > thresh_close \
            and ego.track_mode == other.track_mode) for other in others)

def car_left(ego, others, track_map):
    return any((-3 < track_map.get_longitudinal_position(other.track_mode, [other.x,other.y]) - track_map.get_longitudinal_position(ego.track_mode, [ego.x,ego.y]) < 8 and \
                 other.track_mode==track_map.left_lane(ego.track_mode)) for other in others)

def car_right(ego, others, track_map):
    return any((-3 < track_map.get_longitudinal_position(other.track_mode, [other.x,other.y]) - track_map.get_longitudinal_position(ego.track_mode, [ego.x,ego.y]) < 8 and \
                 other.track_mode==track_map.right_lane(ego.track_mode)) for other in others)

def decisionLogic(ego:State, others:List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Normal:
        if car_front(ego, others, track_map, 7, 3):
            # Switch left if left lane is empty
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft) and \
             not car_left(ego, others, track_map):
                output.agent_mode = AgentMode.SwitchLeft
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)

            # Switch right if right lane is empty
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight) and \
             not car_right(ego, others, track_map):
                output.agent_mode = AgentMode.SwitchRight
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)

    # If switched left enough, return to normal mode
    if ego.agent_mode == AgentMode.SwitchLeft:
        if  track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y]) >= (track_map.get_lane_width(ego.track_mode)-0.2):
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    # If switched right enough,return to normal mode
    if ego.agent_mode == AgentMode.SwitchRight:
        if track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y]) <= -(track_map.get_lane_width(ego.track_mode)-0.2):
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    return output

