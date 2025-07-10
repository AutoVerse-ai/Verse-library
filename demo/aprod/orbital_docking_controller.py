from enum import Enum, auto
import copy
from typing import List

epsilon = 0.1

class OrbitalMode(Enum):
    Passive = auto()
    GroundSensor = auto()
    ProximitySensor = auto()

class MoveMode(Enum):
    NMT = auto()
    Docking = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; time: float 
    orbital_mode: OrbitalMode = OrbitalMode.Passive
    move_mode: MoveMode = MoveMode.NMT

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, timer, time, orbital_mode: OrbitalMode, move_mode: MoveMode):
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
    
    # for now, just consider the estimated state from any sensor
    # could also compute this just using ex and x by doing hx = x - ex
    # if ego.move_mode != MoveMode.Docking: 
    #     output.move_mode = MoveMode.Docking
    
    if ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= -75-epsilon and ego.hy <= -75+epsilon and ego.move_mode != MoveMode.Docking:
        output.move_mode = MoveMode.Docking
        # output.timer = ego.timer * 1
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez

        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz

    return output
