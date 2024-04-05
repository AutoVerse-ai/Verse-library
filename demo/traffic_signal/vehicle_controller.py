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

    # TODO: Edit this part of decision logic
    
    if ego.agent_mode == VehicleMode.Normal and other.signal_mode == TLMode.RED:
        output.agent_mode = VehicleMode.HardBrake

    ###########################################

    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-20 and ego.x<other.x-15)), "run red light"  
    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-15 and ego.x<other.x) and ego.v<1), "stop at intersection"

    return output 