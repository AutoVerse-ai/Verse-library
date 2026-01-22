from enum import Enum, auto
import copy
from typing import List

merge_dist = 2 # should be > 2/omega

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
    Normal = auto()
    Left = auto()
    Right = auto()

class AssignMode(Enum):
    Assigned = auto()
    Waiting = auto()
    Complete = auto()

class State:
    x: float
    y: float
    theta: float
    v: float
    hx: float; hy: float; htheta: float; hv: float
    ex: float; ey: float; etheta: float; ev: float
    timer: float; _id: float; connected_ids: float; assigned_id: float
    dist: float; prev_sense: float; cur_sense: float
    agent_mode: AgentMode
    assign_mode: AssignMode

    def __init__(self, x, y, theta, v, 
                 hx, hy, htheta, hv, 
                 ex, ey, etheta, ev, 
                 timer, _id, connected_ids, assigned_id,
                 dist, prev_sense, cur_sense,
        agent_mode: AgentMode, assign_mode: AssignMode):
        pass

        # agent_mode_src = agent_mode_src[0] if isinstance(agent_mode_src, tuple) else agent_mode_src
        # agent_mode_dest = agent_mode_dest[0] if isinstance(agent_mode_dest, tuple) else agent_mode_dest

class OtherState:
    has_priority: float; next_mode: float; next_assign: float; assign_wait_priority: float; lower_priority: float; allowed_switch: float
    x: float
    y: float
    theta: float
    v: float
    hx: float; hy: float; htheta: float; hv: float
    ex: float; ey: float; etheta: float; ev: float
    connected_ids: float
    cur_sense: float
    agent_mode: AgentMode

    def __init__(self, has_priority, next_mode, next_assign, assign_wait_priority, lower_priority, allowed_switch,
                 x, y, theta, v, 
                 hx, hy, htheta, hv, 
                 ex, ey, etheta, ev,
                 connected_ids,
                 cur_sense, agent_mode):
        pass

def decisionLogic(ego: State, other: OtherState):
    output = copy.deepcopy(ego)
    if ego.dist < merge_dist and ego.assign_mode == AssignMode.Assigned and other.assign_wait_priority >= 0.1:
    # if ego.dist < merge_dist and ego.assign_mode == AssignMode.Assigned: # need to have some synching here
        output.assign_mode = AssignMode.Waiting
        # output.connected_ids = ego.connected_ids + 2**ego.assigned_id # this isn't correct
        output.connected_ids = ego.connected_ids + other.connected_ids
        output.v = 0

        # assume that sensor will handle new mode with Waiting mode trigger

    if ego.prev_sense == -3 and other.has_priority == 1: # sentinel value of prev_sense, just update it to an appropriate reading based on initial sensed info 
        # this is a bit of a hack, things won't happen if cur_sense = 0 
        output.prev_sense = ego.cur_sense

    if ego.prev_sense == -1 and (ego.cur_sense <= -1.9 or ego.cur_sense == 2) and ego.assign_mode == AssignMode.Assigned and ego.agent_mode != AgentMode.Left and ego.timer >= 0.1 and other.has_priority == 1:
        output.agent_mode = AgentMode.Left
        # output.prev_sense = -2 # update our previous sensed value since y_past!=y_current -- need to update to singleton instead of set
        output.prev_sense = ego.cur_sense
        output.timer = 0

    # if (ego.prev_sense == -1 and ego.cur_sense == 1) or (ego.prev_sense == 1 and ego.cur_sense == -1) and ego.timer >= 0.1 and other.allowed_switch > 0.9:
    #     output.prev_sense = ego.cur_sense
    #     output.timer = 0 # just updating prev_sense

    if (ego.prev_sense <= -1.9 or ego.prev_sense == 2) and (ego.cur_sense == -1 or ego.cur_sense==-1) and ego.assign_mode == AssignMode.Assigned and ego.agent_mode != AgentMode.Normal and ego.timer >= 0.1 and other.has_priority == 1:
        output.agent_mode = AgentMode.Normal
        # if ego.cur_sense == -1:
        #     output.prev_sense = -1
        # if ego.cur_sense == 1:
        #     output.prev_sense = 1
        output.prev_sense = ego.cur_sense
        output.timer = 0


    if ego.prev_sense == 1 and (ego.cur_sense <= -1.9 or ego.cur_sense == 2) and ego.assign_mode == AssignMode.Assigned and ego.agent_mode != AgentMode.Right and ego.timer >= 0.1 and other.has_priority == 1:
        output.agent_mode = AgentMode.Right
        output.prev_sense = ego.cur_sense
        # output.prev_sense = 2
        output.timer = 0

    if ego.assign_mode == AssignMode.Waiting and other.has_priority == 1 and other.next_mode == 0:
        output.assign_mode = AssignMode.Assigned
        output.agent_mode = AgentMode.Normal
        output.assigned_id = other.next_assign
        output.prev_sense = ego.cur_sense
        output.v = 1
        # output.agent_mode = AgentMode.Normal
        # output.agent_mode = AgentMode.Right
        # this should also update prev_sense properly


    if ego.assign_mode == AssignMode.Waiting and other.has_priority == 1 and other.next_mode == 1:
        output.assign_mode = AssignMode.Complete
        # don't know if the following is necessary, but keep it for now 
        output.v = 0
        output.ev = 0
        output.hv = 0 

    return output