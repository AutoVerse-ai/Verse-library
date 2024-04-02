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
    # if ego.x > 100000:
    #     output.agent_mode = VehicleMode.Brake
    # if ego.agent_mode == VehicleMode.Normal:
    #     output.agent_mode = VehicleMode.Accel
    # print(ego)
    # print(ego.agent_mode == VehicleMode.Normal)
    # print(other.signal_mode == TLMode.RED)
    # print(other.dist<60)
    if ego.agent_mode == VehicleMode.Normal and other.signal_mode == TLMode.RED and other.x-ego.x<60 and other.x-ego.x>0:
        output.agent_mode = VehicleMode.Brake 
    elif ego.agent_mode == VehicleMode.Normal and other.signal_mode == TLMode.YELLOW and other.x-ego.x<60 and other.x-ego.x>0:
        output.agent_mode = VehicleMode.Brake
    if ego.agent_mode == VehicleMode.Brake and other.signal_mode == TLMode.GREEN:
        output.agent_mode = VehicleMode.Accel
    # if (ego.agent_mode == VehicleMode.Brake or ego.agent_mode == VehicleMode.HardBrake) and other.y>5:
    #     output.agent_mode = VehicleMode.Accel

    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-20 and ego.x<other.x-15)), "run red light"  
    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-15 and ego.x<other.x) and ego.v<1), "stop at intersection"

    return output 