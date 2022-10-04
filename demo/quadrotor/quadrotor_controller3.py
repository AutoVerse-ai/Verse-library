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


# def is_close(ego, other):
#     res = ego.x-other.x > -5 and ego.x-other.x < 5 and \
#         ego.x-other.x < -4 and ego.x-other.x > 4 and \
#         ego.y-other.y > -5 and ego.y-other.y < 5 and \
#         ego.y-other.y < -4 and ego.y-other.y > 4
#     return res

def is_close(ego, other):
    res = (((ego.vx <= 0) and (ego.x-other.x > 2 and ego.x-other.x < 5)) or ((ego.vx > 0) and (ego.x-other.x <= -2 and ego.x-other.x > - 5))) \
        and (((ego.vy <= 0) and (ego.y-other.y > 2 and ego.y-other.y < 5)) or ((ego.vy > 0) and (ego.y-other.y <= -2 and ego.y-other.y > - 5)))
    return res


def controller(ego: State, others: List[State], lane_map):
    next = copy.deepcopy(ego)

    if ego.craft_mode == CraftMode.Normal:
        if any((is_close(ego, other) and ego.track_mode == other.track_mode) for other in others):
            if lane_map.h_exist(ego.track_mode, ego.craft_mode, 'MoveUp'):
                next.craft_mode = CraftMode.MoveUp
                next.track_mode = lane_map.h(
                    ego.track_mode, ego.craft_mode, 'MoveUp')
            if lane_map.h_exist(ego.track_mode, ego.craft_mode, 'MoveDown'):
                next.craft_mode = CraftMode.MoveDown
                next.track_mode = lane_map.h(
                    ego.track_mode, ego.craft_mode, 'MoveDown')

    if ego.craft_mode == CraftMode.MoveUp:
        if lane_map.altitude(ego.track_mode)-ego.z > -1 and lane_map.altitude(ego.track_mode)-ego.z < 1:
            next.craft_mode = CraftMode.Normal
            if lane_map.h_exist(ego.track_mode, ego.craft_mode, 'Normal'):
                next.track_mode = lane_map.h(
                    ego.track_mode, ego.craft_mode, 'Normal')

    if ego.craft_mode == CraftMode.MoveDown:
        if lane_map.altitude(ego.track_mode)-ego.z > -1 and lane_map.altitude(ego.track_mode)-ego.z < 1:
            next.craft_mode = CraftMode.Normal
            if lane_map.h_exist(ego.track_mode, ego.craft_mode, 'Normal'):
                next.track_mode = lane_map.h(
                    ego.track_mode, ego.craft_mode, 'Normal')

    return next
