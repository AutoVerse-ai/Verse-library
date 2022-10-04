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
    thetax: float 
    omegax: float 
    thetay: float 
    omegay: float 
    craft_mode: CraftMode
    lane_mode: TrackMode

    def __init__(self, x, y, z, vx, vy, vz, thetax, omegax, thetay, omegay, craft_mode, lane_mode):
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
    output = copy.deepcopy(ego)

    return output
