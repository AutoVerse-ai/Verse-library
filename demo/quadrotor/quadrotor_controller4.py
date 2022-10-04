from enum import Enum, auto
import copy
from typing import List


class CraftMode(Enum):
    Normal = auto()
    MoveUp = auto()
    MoveDown = auto()


class LaneMode(Enum):
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
    lane_mode: LaneMode

    def __init__(self, x, y, z, vx, vy, vz, craft_mode, lane_mode):
        pass


def controller(ego: State, others: List[State], lane_map):
    output = copy.deepcopy(ego)
    return output
