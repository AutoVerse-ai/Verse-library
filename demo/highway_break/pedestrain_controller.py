from enum import Enum, auto
import copy
from typing import List

class PedestrainMode(Enum):
    Normal = auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    dist: float 
    agent_mode: PedestrainMode 

    def __init__(self, x, y, theta, v, dist, agent_mode: PedestrainMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output 