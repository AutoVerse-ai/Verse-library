from enum import Enum, auto

class AgentMode(Enum):
    Normal = auto()
    Brake = auto()

class TrackMode(Enum):
    T0 = auto()

class State:
    x:float
    y:float
    theta:float
    v:float
    agent_mode:AgentMode 
    track_mode:TrackMode 

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass

from typing import List
import copy
def decisionLogic(ego:State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Normal:
        if any(0 < other.x-ego.x < 8 for other in others):
            output.agent_mode = AgentMode.Brake
    return output
