from enum import Enum, auto
import copy
from typing import List

class SatelliteMode(Enum):
    Passive = auto()

class State:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    hx: float
    hy: float
    hz: float
    x: float
    y: float
    z: float
    satellite_mode: SatelliteMode = SatelliteMode.Passive

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, satellite_mode: SatelliteMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    return output
