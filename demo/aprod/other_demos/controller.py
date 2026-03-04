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

        # agent_mode_src = agent_mode_src[0] if isinstance(agent_mode_src, tuple) else agent_mode_src
        # agent_mode_dest = agent_mode_dest[0] if isinstance(agent_mode_dest, tuple) else agent_mode_dest

def vehicle_front(ego, others, track_map):
    res = any(
        (
            5
            > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y]) # either just make other.x, other.y noisy or change this to be other.hx, other.hy
            - track_map.get_longitudinal_position(ego.track_mode, [ego.hx, ego.hy])
            > 3
            and ego.track_mode == other.track_mode
        )
        for other in others
    )
    return res


def vehicle_close(ego, others):
    return any(-1 < ego.x - other.x < 1 and -1 < ego.y - other.y < 1 for other in others) 


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
        if vehicle_front(ego, others, track_map):
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft):
                output.agent_mode = AgentMode.SwitchLeft
                output.track_mode = track_map.h(
                    ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft
                )
        if vehicle_front(ego, others, track_map):
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight):
                output.agent_mode = AgentMode.SwitchRight
                output.track_mode = track_map.h(
                    ego.track_mode, ego.agent_mode, AgentMode.SwitchRight
                )
    lat_dist = track_map.get_lateral_distance(ego.track_mode, [ego.hx, ego.hx]) # this is a slight overapproximation, could also just keep copy of hx,hy
    if ego.agent_mode == AgentMode.SwitchLeft:
        if lat_dist >= 2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
    if ego.agent_mode == AgentMode.SwitchRight:
        if lat_dist <= -2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    # assert not vehicle_close(ego, others)
    return output
