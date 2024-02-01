from enum import Enum, auto
from typing import List
import copy
import math


class AgentMode(Enum):
    Left = auto()
    Right = auto()


class TrackMode(Enum):
    T0 = auto()

#modified T
class State:
    x: float
    y: float
    theta: float
    v: float
    t: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v,t, agent_mode: AgentMode, track_mode: TrackMode):
        pass



def decisionLogic(ego: State, track_map):
    output = copy.deepcopy(ego)

    if ego.s == 0 and ego.t >= 2: #we're on the left side
        output.agent_mode = AgentMode.Right #go right
        output.t = 0
    elif ego.s == 1 and ego.t >= 2: #we're on the right side
        output.agent_mode = AgentMode.Left #go left
        output.t = 0
    '''
    if ego.s == -1 and ego.agent_mode == AgentMode.Left and ego.t >= 5:
        output.t = 0
        output.agent_mode = AgentMode.Right
    elif ego.s == 1 and ego.agent_mode == AgentMode.Right and ego.t >= 5:
        output.agent_mode = AgentMode.Left
        output.t = 0
    '''
    assert ego.x >= -2 and ego.x <= 17
        
    return output


