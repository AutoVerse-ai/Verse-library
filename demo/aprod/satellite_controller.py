from enum import Enum, auto
import copy
from typing import List

class SatelliteMode(Enum):
    Passive = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    q0: float; q1: float; q2: float; q3: float
    hq0: float; hq1: float; hq2: float; hq3: float
    om_x: float; om_y: float; om_z: float 
    satellite_mode: SatelliteMode = SatelliteMode.Passive

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z, satellite_mode: SatelliteMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    return output
