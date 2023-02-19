from enum import Enum, auto
import copy
from typing import List


class CraftMode(Enum):
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
    craft_mode: CraftMode
    track_mode: TrackMode

    def __init__(self, x, y, z, vx, vy, vz, craft_mode, track_mode):
        pass


def safe_seperation(ego, other):
    return -1 < ego.x - other.x < 1 and -1 < ego.y - other.y < 1 and -1 < ego.z - other.z < 1

def is_close(ego, other):
    return (8 < other.x - ego.x < 10
        or 8 < other.y-ego.y < 10
        or 8 < other.z-ego.z < 10)


def decisionLogic(ego: State, others: List[State], track_map):
    next = copy.deepcopy(ego)

    if ego.craft_mode == CraftMode.Normal:
        if any((is_close(ego, other) and ego.track_mode == other.track_mode) for other in others):
            if track_map.h_exist(ego.track_mode, ego.craft_mode, CraftMode.MoveUp):
                next.craft_mode = CraftMode.MoveUp
                next.track_mode = track_map.h(
                    ego.track_mode, ego.craft_mode, CraftMode.MoveUp)
            if track_map.h_exist(ego.track_mode, ego.craft_mode, CraftMode.MoveDown):
                next.craft_mode = CraftMode.MoveDown
                next.track_mode = track_map.h(
                    ego.track_mode, ego.craft_mode, CraftMode.MoveDown)

    if ego.craft_mode == CraftMode.MoveUp:
        if 1 > track_map.altitude(ego.track_mode)-ego.z > -1:
            next.craft_mode = CraftMode.Normal
            if track_map.h_exist(ego.track_mode, ego.craft_mode, CraftMode.Normal):
                next.track_mode = track_map.h(
                    ego.track_mode, ego.craft_mode, CraftMode.Normal)

    if ego.craft_mode == CraftMode.MoveDown:
        if 1 > track_map.altitude(ego.track_mode)-ego.z > -1:
            next.craft_mode = CraftMode.Normal
            if track_map.h_exist(ego.track_mode, ego.craft_mode, CraftMode.Normal):
                next.track_mode = track_map.h(
                    ego.track_mode, ego.craft_mode, CraftMode.Normal)

    assert not any(-1 < ego.x-other.x < 1
        and -1 < ego.y-other.y < 1
        and -1 < ego.z-other.z < 1
        for other in others),\
        "Safe Seperation"

    assert not (50 > ego.x > 40
        and 5 > ego.y>-5
        and -6 > ego.z > -10),\
        "Unsafe Region"
    
    return next
