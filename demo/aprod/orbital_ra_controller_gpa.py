from enum import Enum, auto
import copy
from typing import List
import numpy as np
# rad_col = 5 # unsafe radius where we should begin to transition
rad_col = 4.5
dist_prox = 5 # 5 km for proximity sensor to be active; fairly long range
# dist_prox = 3 # testing
T_prox = 10 # 10 s period, exists to make sure some time passes before next sensor update, should be >= time step
pi = np.pi
angle_bound = 0.001 # in radians, the amount of tolerance for whatever angle is being looked at 

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    OActive = auto()
    CActive = auto()
    OCActive = auto()

class TrajMode(Enum):
    Normal = auto()
    Avoid = auto()


class State:
    """
    Note that currently in Verse, the number of variables in this state class need to be the same as the number in the init (i.e., that can be handled by the agent file)
    TODO: fix this limition, instead just make sure the variables here are a subset of the variables from sensor 
    """
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; po_timer: float 
    dist: float; hdist: float # dist to obstacle, dist to chief can be computed easily
    angle_minus: float; angle_plus: float
    go_mode: GOMode = GOMode.Passive
    po_mode: POMode = POMode.Passive
    traj_mode: TrajMode = TrajMode.Normal

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, 
                 timer, po_timer, dist, h_dist, angle_minus, angle_plus,
                 go_mode: GOMode, po_mode: POMode, traj_mode: TrajMode):
        pass

def decisionLogic(ego: State, others: List[State]) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900: # presumably going to add other conditions as well 
        output.go_mode = GOMode.Active
        output.timer = 0
        # output.po_timer = 0
        # output.dist = 0
        # output.hdist = 0

    #TODO: add way to switch into O, C, and OC modes for prox observer
    # may want to add a buffer between active/passive distances if branching behavior not desired
    if ego.po_timer >= T_prox and ego.po_mode != POMode.OCActive and ego.dist < dist_prox and ego.x**2+ego.y**2+ego.z**2 < dist_prox**2:
        output.po_mode = POMode.OCActive
        output.po_timer = 0

    if ego.po_mode == POMode.Passive and ego.dist < dist_prox and ego.x**2+ego.y**2+ego.z**2 > dist_prox**2 and ego.traj_mode != TrajMode.Avoid: 
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez

        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz 
        output.po_mode = POMode.OActive

    if ego.po_timer >= T_prox and ego.po_mode != POMode.CActive and ego.dist > dist_prox and ego.x**2+ego.y**2+ego.z**2 < dist_prox**2: 
        output.po_mode = POMode.CActive
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

    # think I need to invent period for error/estimated state updating so that I don't have infinite updates; should be >= time step length
    # if ego.po_mode == POMode.OCActive or ego.po_mode == POMode.CActive: 
    if ego.po_mode == POMode.OCActive: 
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

        output.po_mode = POMode.CActive 

    if ego.po_mode == POMode.CActive: 
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

    # if (ego.po_mode == POMode.OCActive or ego.po_mode == POMode.OActive) and ego.hdist < rad_col and ego.traj_mode != TrajMode.Avoid: 
    #     output.traj_mode = TrajMode.Avoid
        
    #     output.hx = ego.x - ego.ex
    #     output.hy = ego.y - ego.ey
    #     output.hz = ego.z - ego.ez

    #     output.hvx = ego.vx - ego.evx
    #     output.hvy = ego.vy - ego.evy
    #     output.hvz = ego.vz - ego.evz   

    # if (ego.po_mode == POMode.OCActive or ego.po_mode == POMode.OActive) and ego.hdist > rad_col + 100 and ego.traj_mode == TrajMode.Avoid:
    #     output.traj_mode = TrajMode.Normal

    #     output.hx = ego.x - ego.ex
    #     output.hy = ego.y - ego.ey
    #     output.hz = ego.z - ego.ez

    #     output.hvx = ego.vx - ego.evx
    #     output.hvy = ego.vy - ego.evy
    #     output.hvz = ego.vz - ego.evz   

    if -angle_bound< ego.angle_minus < angle_bound and -angle_bound < ego.angle_plus < angle_bound and ego.hdist<rad_col and ego.traj_mode != TrajMode.Avoid:
    # if pi/2-angle_bound< ego.angle_minus < pi/2+angle_bound and pi/2-angle_bound < ego.angle_plus < pi/2+angle_bound and ego.traj_mode != TrajMode.Avoid:
        output.traj_mode = TrajMode.Avoid
        # output.timer = ego.timer * 1
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez

        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz   
    # TODO: add way to switch out of oactive
    return output
