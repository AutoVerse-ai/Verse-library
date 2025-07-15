from enum import Enum, auto
import copy
from typing import List

rad_col = 50

class OrbitalMode(Enum):
    Passive = auto()
    GroundSensor = auto()
    ProximitySensor = auto()

class TrajMode(Enum):
    Normal = auto()
    Avoid = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float # don't need time variable here
    dist: float 
    orbital_mode: OrbitalMode = OrbitalMode.Passive
    traj_mode: TrajMode = TrajMode.Normal

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, timer, time, dist, orbital_mode: OrbitalMode, traj_mode: TrajMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900: # presumably going to add other conditions as well 
        # output.orbital_mode = OrbitalMode.GroundSensor
        output.orbital_mode = OrbitalMode.ProximitySensor 
        output.timer = 0
    
    if ego.orbital_mode == OrbitalMode.GroundSensor:
        output.ex = ego.ex * 1
        output.ey = ego.ey * 1
        output.ez = ego.ez * 1

        output.evx = ego.evx * 1
        output.evy = ego.evy * 1
        output.evz = ego.evz * 1

        output.hx = ego.hx * 1
        output.hy = ego.hy * 1
        output.hz = ego.hz * 1

        output.hvx = ego.hvx * 1
        output.hvy = ego.hvy * 1
        output.hvz = ego.hvz * 1

        output.orbital_mode = OrbitalMode.Passive

    if ego.orbital_mode == OrbitalMode.ProximitySensor: 
        output.ex = ego.ex * 1
        output.ey = ego.ey * 1
        output.ez = ego.ez * 1

        output.evx = ego.evx * 1
        output.evy = ego.evy * 1
        output.evz = ego.evz * 1

        output.hx = ego.hx * 1
        output.hy = ego.hy * 1
        output.hz = ego.hz * 1

        output.hvx = ego.hvx * 1
        output.hvy = ego.hvy * 1
        output.hvz = ego.hvz * 1

        output.orbital_mode = OrbitalMode.Passive

        if ego.dist < rad_col and ego.traj_mode != TrajMode.Avoid:
            output.traj_mode = TrajMode.Avoid

        if ego.dist > rad_col and ego.traj_mode != TrajMode.Normal:
            output.traj_mode = TrajMode.Normal
    return output
