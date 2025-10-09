from enum import Enum, auto
import copy
from typing import List

epsilon = 0.1
prox_dist = 2.5
T_prox = 100
buffer = 30
unsafe_dist = 20
angle_bound = 1
M = 1000 # proxy for not infinite -- make it agrees with sensor

class MoveMode(Enum):
    NMT = auto()
    Inner = auto()

class State:
    """
    Throw all relative states into others
    """
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; po_timer: float; time: float 
    prox_m: float
    # dist: float; hdist: float # time is never needed, cut it right now so that num sensor variables match num agent variables
    # angle_minus: float; angle_plus: float
    # go_mode: GOMode = GOMode.Passive; po_mode: POMode = POMode.Passive; 
    # priority_mode: PriorityMode; 
    move_mode: MoveMode = MoveMode.NMT; 

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, 
                 timer, po_timer, time, prox_m,
                 move_mode: MoveMode,):
        pass

class OtherState:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    dist: float; hdist: float # time is never needed, cut it right now so that num sensor variables match num agent variables
    angle_minus: float; angle_plus: float
    sensor_emit: float
    move_mode: MoveMode; 

    def __init__(self, x, y, z, vx, vy, vz,
                 dist, hdist, angle_minus, angle_plus, 
                 move_mode: MoveMode):
        pass 

class SensorStates:
    index: float; prox_m: float
    
    def __init__(self, index, prox_m):
        pass 
# def vehicle_front(ego, others, track_map):
#     res = any(
#         (
#             5
#             > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y]) # either just make other.x, other.y noisy or change this to be other.hx, other.hy
#             - track_map.get_longitudinal_position(ego.track_mode, [ego.hx, ego.hy])
#             > 3
#             and ego.track_mode == other.track_mode
#         )
#         for other in others
#     )
#     return res

# def has_priority(ego: State, others: List[State], desired_mode: POMode):
#     '''
#     returns whether ego can transition to desired state based on priority
#     '''
#     res = all(
#     ((ego.id < other.id) or (ego.id>=other.id and other.po_mode == desired_mode)) for other in others
#     )
#     return res

def seems_unsafe(ego: State, others: List[OtherState]) -> bool:
    """
    returns whether ego thinks it is unsafe (True if unsafe) to transition to inner orbit
    currently using if prox sensor is only (other.hdist<M) and 
    """
    res = any(
        (
            other.hdist < M and other.move_mode == MoveMode.Inner and -angle_bound< other.angle_minus < angle_bound and -angle_bound < other.angle_plus < angle_bound
        ) for other in others
    )
    return res

# def decisionLogic(ego: State, other: State, obs: MiniState) -> State:
def decisionLogic(ego: State, others: List[OtherState], prox: SensorStates) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900:
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
        output.timer = 0
        # output.time = any((other.x < 10) for other in others)

    if any((other.sensor_emit > 0) for other in others):
        output.prox_m = prox.prox_m
        output.ex = ego.ex * 1
        output.ey = ego.ey * 1
        output.ez = ego.ez * 1

        output.evx = ego.evx * 1
        output.evy = ego.evy * 1
        output.evz = ego.evz * 1
    # for now, just consider the estimated state from any sensor
    # could also compute this just using ex and x by doing hx = x - ex
    # note this should be using hdist instead of dist
    if (ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= -75-epsilon and ego.hy <= -75+epsilon and ego.move_mode != MoveMode.Inner) or (ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= 75-epsilon and ego.hy <= 75+epsilon and ego.move_mode != MoveMode.Inner):
        # if not (ego.po_mode == POMode.Active and other.move_mode == MoveMode.Inner and ego.dist < unsafe_dist):
        if not seems_unsafe(ego, others):
            output.move_mode = MoveMode.Inner
            # output.timer = ego.timer * 1
            output.hx = ego.x - ego.ex
            output.hy = ego.y - ego.ey
            output.hz = ego.z - ego.ez

            output.hvx = ego.vx - ego.evx
            output.hvy = ego.vy - ego.evy
            output.hvz = ego.vz - ego.evz

    return output
