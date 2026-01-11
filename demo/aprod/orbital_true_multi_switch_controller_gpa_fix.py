from enum import Enum, auto
import copy
from typing import List

epsilon = 1 # originally 0.1
prox_dist = 10 # 2.5 is standard for 2.5 init est state error -- revised to 10
T_prox = 100
buffer = 30
unsafe_dist = 20
angle_bound = 1.5 # used to be 1, increasing slightly 

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto() # in this instance, there is no obstacle so only activity should be from being close to chief

class POTwoMode(Enum):
    Passive = auto()
    Active = auto() 

class MoveMode(Enum):
    NMT = auto()
    Inner = auto()

class PriorityMode(Enum):
    First = auto()
    Second = auto()
    Third = auto()

class State:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    hx: float; hy: float; hz: float
    hvx: float; hvy: float; hvz: float
    ex: float; ey: float; ez: float
    evx: float; evy: float; evz: float
    timer: float; po_timer: float; time: float 
    prox_m: float; priority: float

    go_mode: GOMode = GOMode.Passive; po_mode: POMode = POMode.Passive; po_two_mode: POTwoMode = POTwoMode.Passive
    priority_mode: PriorityMode; move_mode: MoveMode = MoveMode.NMT; 

    def __init__(self, x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, ex, ey, ez, evx, evy, evz, 
                 timer, po_timer, 
                 time, prox_m, priority, # these are essentially just filler values 
                 go_mode: GOMode, po_mode: POMode, po_two_mode: POTwoMode, priority_mode: PriorityMode, move_mode: MoveMode,):
        pass

class MiniState:
    dist: float; hdist: float 
    angle_minus: float; angle_plus: float
    priority_mode: PriorityMode; po_mode: POMode; po_two_mode: POTwoMode # only used for sequential updating
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

def decisionLogic(ego: State, other_one: MiniState, other_two: MiniState) -> State:
    output = copy.deepcopy(ego)
    if ego.timer >= 900 and ego.go_mode != GOMode.Active:
        output.go_mode = GOMode.Active 
        # output.po_mode = POMode.Active
        output.timer = 0
    
    # for reference, dep calls dep_ahead other_one and dep_aheader other_two; dep_ahead calls dep other_one and dep_aheader other_two; and dep_aheader calls dep other_one and dep_ahead other_two
    # there still might be some synching updates between whether pomode or pomode2 gets updated first -- may have to adjust to account for this
    if other_one.dist < prox_dist and ego.po_mode != POMode.Active and (ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other_one.po_mode == POMode.Active) or (ego.priority_mode == PriorityMode.Third and other_one.po_two_mode == POTwoMode.Active)):
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez
        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz
    
        output.po_mode = POMode.Active

    if other_two.dist < prox_dist and ego.po_two_mode != POTwoMode.Active and (ego.priority_mode == PriorityMode.First or ego.priority_mode == PriorityMode.Second or (ego.priority_mode == PriorityMode.Third and other_two.po_two_mode == POTwoMode.Active)):
        output.hx = ego.x - ego.ex
        output.hy = ego.y - ego.ey
        output.hz = ego.z - ego.ez
        output.hvx = ego.vx - ego.evx
        output.hvy = ego.vy - ego.evy
        output.hvz = ego.vz - ego.evz
    
        output.po_two_mode = POTwoMode.Active

    # if (ego.x-other.x)**2+(ego.y-other.y)**2+(ego.z-other.z)**2 > (prox_dist+buffer)**2 and ego.po_mode == POMode.Active:
    if other_one.dist > prox_dist+buffer and ego.po_mode == POMode.Active:
        if ego.priority_mode == PriorityMode.First or (ego.priority_mode == PriorityMode.Second and other_one.po_mode == POMode.Passive) or (ego.priority_mode == PriorityMode.Third and other_one.po_two_mode == POTwoMode.Passive):
            output.hx = ego.x - ego.ex
            output.hy = ego.y - ego.ey
            output.hz = ego.z - ego.ez
            output.hvx = ego.vx - ego.evx
            output.hvy = ego.vy - ego.evy
            output.hvz = ego.vz - ego.evz
            
            output.po_mode = POMode.Passive

    if other_two.dist > prox_dist+buffer and ego.po_two_mode == POMode.Active:
        if ego.priority_mode == PriorityMode.First or ego.priority_mode == PriorityMode.Second or (ego.priority_mode == PriorityMode.Third and other_two.po_two_mode == POTwoMode.Passive):
            output.hx = ego.x - ego.ex
            output.hy = ego.y - ego.ey
            output.hz = ego.z - ego.ez
            output.hvx = ego.vx - ego.evx
            output.hvy = ego.vy - ego.evy
            output.hvz = ego.vz - ego.evz
            
            output.po_two_mode = POTwoMode.Passive

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
        if not (ego.po_mode == POMode.Active and other_one.move_mode == MoveMode.Inner and (-angle_bound< other_one.angle_minus < angle_bound or -angle_bound < other_one.angle_plus < angle_bound)):
            if not (ego.po_two_mode == POTwoMode.Active and other_two.move_mode == MoveMode.Inner and (-angle_bound< other_two.angle_minus < angle_bound or -angle_bound < other_two.angle_plus < angle_bound)):
                output.move_mode = MoveMode.Inner
                # output.timer = ego.timer * 1
                output.hx = ego.x - ego.ex
                output.hy = ego.y - ego.ey
                output.hz = ego.z - ego.ez

                output.hvx = ego.vx - ego.evx
                output.hvy = ego.vy - ego.evy
                output.hvz = ego.vz - ego.evz

    return output
