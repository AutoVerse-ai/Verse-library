from enum import Enum, auto
import copy
from typing import List

class OrbitalMode(Enum):
    Passive = auto()
    GroundSensor = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; time: float 
    orbital_mode: OrbitalMode = OrbitalMode.Passive

    def __init__(self, x, y, z, vx, vy, vz, ex, ey, ez, evx, evy, evz, timer, time, orbital_mode: OrbitalMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900:
        output.orbital_mode = OrbitalMode.GroundSensor
    if ego.orbital_mode == OrbitalMode.GroundSensor:
        output.ex = ego.ex * 1
        output.ey = ego.ey * 1
        output.ez = ego.ez * 1

        output.evx = ego.evx * 1
        output.evy = ego.evy * 1
        output.evz = ego.evz * 1

        output.timer = 0
        output.orbital_mode = OrbitalMode.Passive

    print(output)
    return output
