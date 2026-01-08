from enum import Enum, auto
import copy
from typing import List
import numpy as np

# gpa for ground, proxy, and angles-only
# for now, model the sensor as being always active in the sensor (read online that sensor can be active for hundreds of km)

epsilon = 0.1
# prox_dist = 500
prox_dist = 5
T_prox = 150
angle_bound = 0.1
half_pi = np.pi/2

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto() # in this instance, there is no obstacle so only activity should be from being close to chief

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
    timer: float; po_timer: float; 
    angle_minus: float; angle_plus: float; # need to verify both are within whatever bounds 
    go_mode: GOMode = GOMode.Passive; po_mode: POMode = POMode.Passive
    move_mode: MoveMode = MoveMode.NMT

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, timer, po_timer, angle_minus, angle_plus,
                 go_mode: GOMode, po_mode: POMode, move_mode: MoveMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900 and ego.go_mode != GOMode.Active: # presumably going to add other conditions as well 
        output.go_mode = GOMode.Active 
        # output.po_mode = POMode.Active
        output.timer = 0
    
    if ego.x**2+ego.y**2+ego.z**2 < prox_dist**2 and ego.po_mode != POMode.Active and output.po_timer >= T_prox:
        output.po_mode = POMode.Active
        output.po_timer = 0

    if ego.go_mode == GOMode.Active:
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

        output.go_mode = GOMode.Passive 

    if ego.po_mode == POMode.Active:
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

        output.po_mode = POMode.Passive 
    
    # for now, just consider the estimated state from any sensor
    # could also compute this just using ex and x by doing hx = x - ex
    # if ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= -75-epsilon and ego.hy <= -75+epsilon and ego.move_mode != MoveMode.Docking:
    #     output.move_mode = MoveMode.Docking
    #     # output.timer = ego.timer * 1
    #     output.hx = ego.x - ego.ex
    #     output.hy = ego.y - ego.ey
    #     output.hz = ego.z - ego.ez

    #     output.hvx = ego.vx - ego.evx
    #     output.hvy = ego.vy - ego.evy
    #     output.hvz = ego.vz - ego.evz

    if half_pi-angle_bound< ego.angle_minus < half_pi+angle_bound and half_pi-angle_bound < ego.angle_plus < half_pi+angle_bound and ego.move_mode != MoveMode.Docking:
        output.move_mode = MoveMode.Docking
        # output.timer = ego.timer * 1
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez

        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz        

    # if half_pi*2-angle_bound< ego.angle_minus < half_pi*2 and -half_pi*2< ego.angle_plus < -half_pi*2+angle_bound and ego.move_mode != MoveMode.Docking:
    # # if (half_pi*2-angle_bound< ego.angle_minus < half_pi*2 or -half_pi*2< ego.angle_plus < -half_pi*2+angle_bound) and ego.move_mode != MoveMode.Docking:        
    #     output.move_mode = MoveMode.Docking
    #     # output.timer = ego.timer * 1
    #     output.hx = ego.x - ego.ex
    #     output.hy = ego.y - ego.ey
    #     output.hz = ego.z - ego.ez

    #     output.hvx = ego.vx - ego.evx
    #     output.hvy = ego.vy - ego.evy
    #     output.hvz = ego.vz - ego.evz        

    return output
