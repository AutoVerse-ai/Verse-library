from enum import Enum, auto
import copy
from typing import List


class TacticalMode(Enum):
    Normal = auto()
    MoveUp = auto()
    MoveDown = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M10 = auto()
    M12 = auto()
    M21 = auto()


class State:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    tactical_mode: TacticalMode
    track_mode: TrackMode

    def __init__(self, x, y, z, vx, vy, vz, tactical_mode, track_mode):
        pass


def safe_seperation(ego, other):
    res = ego.x-other.x < 1 and ego.x-other.x >-1 and \
        ego.y-other.y < 1 and ego.y-other.y > -1 and \
        ego.z-other.z < 1 and ego.z-other.z > -1
    return res

def is_close(ego, other):
    res = (((ego.vx <= 0) and (ego.x-other.x > 2 and ego.x-other.x < 5)) or ((ego.vx > 0) and (ego.x-other.x <= -2 and ego.x-other.x > - 5))) \
        and (((ego.vy <= 0) and (ego.y-other.y > 2 and ego.y-other.y < 5)) or ((ego.vy > 0) and (ego.y-other.y <= -2 and ego.y-other.y > - 5)))
    return res


def decisionLogic(ego: State, others: List[State], track_map):
    next = copy.deepcopy(ego)

    if ego.tactical_mode == TacticalMode.Normal:
        if any((is_close(ego, other) and ego.track_mode == other.track_mode) for other in others):
            if track_map.h_exist(ego.track_mode, ego.tactical_mode, TacticalMode.MoveUp):
                next.tactical_mode = TacticalMode.MoveUp
                next.track_mode = track_map.h(
                    ego.track_mode, ego.tactical_mode, TacticalMode.MoveUp)
            if track_map.h_exist(ego.track_mode, ego.tactical_mode, TacticalMode.MoveDown):
                next.tactical_mode = TacticalMode.MoveDown
                next.track_mode = track_map.h(
                    ego.track_mode, ego.tactical_mode, TacticalMode.MoveDown)

    if ego.tactical_mode == TacticalMode.MoveUp:
        if track_map.altitude(ego.track_mode)-ego.z > -1 and track_map.altitude(ego.track_mode)-ego.z < 1:
            next.tactical_mode = TacticalMode.Normal
            if track_map.h_exist(ego.track_mode, ego.tactical_mode, TacticalMode.Normal):
                next.track_mode = track_map.h(
                    ego.track_mode, ego.tactical_mode, TacticalMode.Normal)

    if ego.tactical_mode == TacticalMode.MoveDown:
        if track_map.altitude(ego.track_mode)-ego.z > -1 and track_map.altitude(ego.track_mode)-ego.z < 1:
            next.tactical_mode = TacticalMode.Normal
            if track_map.h_exist(ego.track_mode, ego.tactical_mode, TacticalMode.Normal):
                next.track_mode = track_map.h(
                    ego.track_mode, ego.tactical_mode, TacticalMode.Normal)
   
    assert not any(safe_seperation(ego, other) for other in others), "Safe Seperation"

    return next
