from enum import Enum, auto
import copy
from typing import List

class VehicleMode(Enum):
    Normal = auto()
    Brake = auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    dist: float 
    agent_mode: VehicleMode 

    def __init__(self, x, y, theta, v, dist, agent_mode: VehicleMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.agent_mode == VehicleMode.Normal and ego.dist < 10:
        output.agent_mode = VehicleMode.Brake 

    return output 