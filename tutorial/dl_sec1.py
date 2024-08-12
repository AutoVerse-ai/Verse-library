from enum import Enum, auto


class CraftMode(Enum):
    Normal = auto()
    MoveUp = auto()
    MoveDown = auto()
    AvoidUp = auto()
    AvoidDown = auto()


class TrackMode(Enum):
    T0 = auto()
    TAvoidUp = auto()


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


import copy


def decisionLogic(ego: State, track_map):
    next = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Normal:
        if ego.x > 20:
            next.craft_mode = CraftMode.AvoidUp
            next.track_mode = track_map.h(ego.track_mode, ego.craft_mode, CraftMode.AvoidUp)
    return next
