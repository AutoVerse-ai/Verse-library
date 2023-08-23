from enum import Enum, auto
import copy
from typing import List

class PedestrianMode(Enum):
    Normal=auto()

class VehicleMode(Enum):
    Normal = auto()
    Brake = auto()
    Accel = auto()
    HardBrake = auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    agent_mode: VehicleMode 

    def __init__(self, x, y, theta, v, agent_mode: VehicleMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)
    if ego.agent_mode == VehicleMode.Normal and other.x - ego.x < 40 and (other.y - ego.y < 20 and other.y - ego.y>-20):
        output.agent_mode = VehicleMode.Brake 

    assert ego.dist > 2.0

    return output 