from enum import Enum, auto
import copy
from typing import List


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M01 = auto()
    M12 = auto()
    M23 = auto()
    M40 = auto()
    M04 = auto()
    M32 = auto()
    M21 = auto()
    M10 = auto()

class GPSMode(Enum):
    Passive = auto()
    Active = auto()

class State:
    x: float
    y: float
    theta: float
    v: float
    hx: float; hy: float; htheta: float; hv: float
    ex: float; ey: float; etheta: float; ev: float
    timer: float
    agent_mode: AgentMode
    track_mode: TrackMode
    gps_mode: GPSMode

    def __init__(self, x, y, theta, v, 
                 hx, hy, htheta, hv, 
                 ex, ey, etheta, ev, 
                 timer,
        agent_mode: AgentMode, track_mode: TrackMode, gps_mode: GPSMode):
        pass

class OtherState:
    x: float; y: float; theta: float; v: float # make these slightly noisy, constant linear for now -- in the future, can be a function of how close other car is
    agent_mode: AgentMode; track_mode: TrackMode

    def __init__(self, x, y, theta, v, 
        agent_mode: AgentMode, track_mode: TrackMode):
        pass

def car_front(ego, others, track_map):
    return any(
        (
            5
            > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.hx, ego.hy])
            > 3
            and ego.track_mode == other.track_mode
        )
        for other in others
    )


def car_left(ego, others, track_map):
    return any(
        (
            -3
            < track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.hx, ego.hy])
            < 8
            and other.track_mode == track_map.left_lane(ego.track_mode)
        )
        for other in others
    )


def car_right(ego, others, track_map):
    return any(
        (
            -3
            < track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.hx, ego.hy])
            < 8
            and other.track_mode == track_map.right_lane(ego.track_mode)
        )
        for other in others
    )


def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    # to disable sensor updates, just comment out these lines
    if ego.gps_mode == GPSMode.Passive and ego.timer > 15:
        output.gps_mode = GPSMode.Active
        output.timer = 0 
    if ego.gps_mode == GPSMode.Active:
        output.ex = ego.ex * 1
        output.ey = ego.ey * 1
        output.hx = ego.hx * 1
        output.hy = ego.hy * 1 # need to register that a transition is happening
        output.gps_mode = GPSMode.Passive


    if ego.agent_mode == AgentMode.Normal:
        if car_front(ego, others, track_map):
            if track_map.h_exist(
                ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft
            ) and not car_left(ego, others, track_map):
                output.agent_mode = AgentMode.SwitchLeft
                output.track_mode = track_map.h(
                    ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft
                )
        if car_front(ego, others, track_map):
            if track_map.h_exist(
                ego.track_mode, ego.agent_mode, AgentMode.SwitchRight
            ) and not car_right(ego, others, track_map):
                output.agent_mode = AgentMode.SwitchRight
                output.track_mode = track_map.h(
                    ego.track_mode, ego.agent_mode, AgentMode.SwitchRight
                )
    if ego.agent_mode == AgentMode.SwitchLeft:
        if track_map.get_lateral_distance(ego.track_mode, [ego.hx, ego.hy]) >= 2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
    if ego.agent_mode == AgentMode.SwitchRight:
        if track_map.get_lateral_distance(ego.track_mode, [ego.hx, ego.hy]) <= -2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    return output
