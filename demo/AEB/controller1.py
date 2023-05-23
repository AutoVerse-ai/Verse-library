from enum import Enum, auto
import copy
from typing import List


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
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
    x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass


def vehicle_front(ego, others, track_map):
    res = any(
        (
            track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
            > 3
            and track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
            < 5
            and ego.track_mode == other.track_mode
        )
        for other in others
    )
    return res


def close(ego, others, dsafe):
    res = any(
        ego.x - other.x < dsafe
        and ego.x - other.x > -dsafe
        and ego.y - other.y < dsafe
        and ego.y - other.y > -dsafe
        for other in others
    )
    return res


def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Normal:
        if vehicle_front(ego, others, track_map):
            output.agent_mode = AgentMode.Brake
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Brake)
    #         if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft):
    #             output.agent_mode = AgentMode.SwitchLeft
    #             output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
    #     if vehicle_front(ego, others, track_map):
    #         if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight):
    #             output.agent_mode = AgentMode.SwitchRight
    #             output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)
    # lat_dist = track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y])
    # if ego.agent_mode == AgentMode.SwitchLeft:
    #     if lat_dist >= 2.5:
    #         output.agent_mode = AgentMode.Normal
    #         output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
    # if ego.agent_mode == AgentMode.SwitchRight:
    #     if lat_dist <= -2.5:
    #         output.agent_mode = AgentMode.Normal
    #         output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    # Check for safety with dsafe = 1.0
    assert not close(ego, others, 1.0), "Seperation"
    return output
