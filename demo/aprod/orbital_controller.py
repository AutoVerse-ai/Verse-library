from enum import Enum, auto
import copy
from typing import List

class OrbitalMode(Enum):
    Passive = auto()
    GroundSensor = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    timer: float; time: float 
    orbital_mode: OrbitalMode = OrbitalMode.Passive

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, timer, time, orbital_mode: OrbitalMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900:
        output.orbital_mode = OrbitalMode.GroundSensor
    if ego.orbital_mode == OrbitalMode.GroundSensor:
        output.hx = ego.hx * 1
        output.hy = ego.hy * 1
        output.hz = ego.hz * 1
        output.timer = 0
        output.orbital_mode = OrbitalMode.Passive

    print(output)
    return output
