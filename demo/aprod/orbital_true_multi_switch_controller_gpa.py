from enum import Enum, auto
import copy
from typing import List

epsilon = 0.1
prox_dist = 2.5
T_prox = 100
buffer = 30
unsafe_dist = 20
angle_bound = 1

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto() # in this instance, there is no obstacle so only activity should be from being close to chief

class MoveMode(Enum):
    NMT = auto()
    Inner = auto()

class PriorityMode(Enum):
    First = auto()
    Second = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; po_timer: float # time: float 
    dist: float; hdist: float # time is never needed, cut it right now so that num sensor variables match num agent variables
    angle_minus: float; angle_plus: float
    go_mode: GOMode = GOMode.Passive; po_mode: POMode = POMode.Passive; 
    priority_mode: PriorityMode; move_mode: MoveMode = MoveMode.NMT; 

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, 
                 timer, po_timer, 
                 dist, hdist, angle_minus, angle_plus,
                 go_mode: GOMode, po_mode: POMode, priority_mode: PriorityMode, move_mode: MoveMode,):
        pass

class MiniState:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    move_mode: MoveMode = MoveMode.Inner

    def __init__(self, x, y, z, vx, vy, vz, move_mode: MoveMode):
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

def has_priority(ego: State, others: List[State], desired_mode: POMode):
    '''
    returns whether ego can transition to desired state based on priority
    '''
    res = all(
    ((ego.id < other.id) or (ego.id>=other.id and other.po_mode == desired_mode)) for other in others
    )
    return res

# def decisionLogic(ego: State, other: State, obs: MiniState) -> State:
def decisionLogic(ego: State, other: State, obs) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900 and ego.go_mode != GOMode.Active:
        output.go_mode = GOMode.Active 
        # output.po_mode = POMode.Active
        output.timer = 0
    
    # if (ego.x-other.x)**2+(ego.y-other.y)**2+(ego.z-other.z)**2 < prox_dist**2 and ego.po_mode != POMode.Active and (ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other.priority_mode == PriorityMode.First and other.po_mode == POMode.Active)):
    # if ego.dist < prox_dist and ego.po_mode != POMode.Active and (ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other.priority_mode == PriorityMode.First and other.po_mode == POMode.Active)):
    if ego.dist < prox_dist and ego.po_mode != POMode.Active and (ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other.priority_mode == PriorityMode.First and other.po_mode == POMode.Active)):
        '''
        Indexing into others right now -- pretty sure it won't work, but want to check
        '''
        # if has_priority(ego, others, POMode.Active): # this won't work because of lack of symmetry b/t agents
        # if able to index into other agents, then will be able to do this since the other agent transitioning out is known
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez
        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz
    
        output.po_mode = POMode.Active

    # if (ego.x-other.x)**2+(ego.y-other.y)**2+(ego.z-other.z)**2 > (prox_dist+buffer)**2 and ego.po_mode == POMode.Active:
    if ego.dist > prox_dist+buffer and ego.po_mode == POMode.Active:
        if ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other.priority_mode == PriorityMode.First and other.po_mode == POMode.Passive):
            output.hx = ego.x - ego.ex
            output.hy = ego.y - ego.ey
            output.hz = ego.z - ego.ez
            output.hvx = ego.vx - ego.evx
            output.hvy = ego.vy - ego.evy
            output.hvz = ego.vz - ego.evz
            
            output.po_mode = POMode.Passive

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

    # for now, just consider the estimated state from any sensor
    # could also compute this just using ex and x by doing hx = x - ex
    # note this should be using hdist instead of dist
    if (ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= -75-epsilon and ego.hy <= -75+epsilon and ego.move_mode != MoveMode.Inner) or (ego.hx >= -epsilon and ego.hx <= epsilon and ego.hy >= 75-epsilon and ego.hy <= 75+epsilon and ego.move_mode != MoveMode.Inner):
        # if not (ego.po_mode == POMode.Active and other.move_mode == MoveMode.Inner and ego.dist < unsafe_dist):
        if not (ego.po_mode == POMode.Active and other.move_mode == MoveMode.Inner and -angle_bound< ego.angle_minus < angle_bound and -angle_bound < ego.angle_plus < angle_bound):
            if not (obs.move_mode == MoveMode.Inner and (ego.x-obs.x)**2+(ego.y-obs.y)**2+(ego.z-obs.z)**2<unsafe_dist**2):
                output.move_mode = MoveMode.Inner
                # output.timer = ego.timer * 1
                output.hx = ego.x - ego.ex
                output.hy = ego.y - ego.ey
                output.hz = ego.z - ego.ez

                output.hvx = ego.vx - ego.evx
                output.hvy = ego.vy - ego.evy
                output.hvz = ego.vz - ego.evz

    return output
