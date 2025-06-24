from enum import Enum, auto
import copy
from typing import List

class OrbitalMode(Enum):
    Passive = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    orbital_mode: OrbitalMode = OrbitalMode.Passive

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, orbital_mode: OrbitalMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    return output
