from enum import Enum, auto
import copy
from typing import List

class TLMode(Enum):
    GREEN=auto()
    YELLOW=auto()
    RED=auto()

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
    if ego.agent_mode == VehicleMode.Normal and other.signal_mode == TLMode.RED and other.dist<20:
        output.agent_mode = VehicleMode.Brake 
    if ego.agent_mode == VehicleMode.Brake and other.signal_mode != TLMode.RED:
        output.agent_mode = VehicleMode.Accel
    # if (ego.agent_mode == VehicleMode.Brake or ego.agent_mode == VehicleMode.HardBrake) and other.y>5:
    #     output.agent_mode = VehicleMode.Accel

    # assert not (other.signal_mode == TLMode.RED and (ego.x>190 and ego.x<210))

    return output 