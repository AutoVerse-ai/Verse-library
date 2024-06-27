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
    # if ego.agent_mode == VehicleMode.Normal and other.signal_mode == TLMode.RED:
    #     output.agent_mode = VehicleMode.Brake

    if (ego.agent_mode == VehicleMode.Normal or ego.agent_mode == VehicleMode.HardBrake) and other.signal_mode == TLMode.GREEN:
        output.agent_mode = VehicleMode.Accel #accelerate no matter what if green
    if ego.agent_mode == VehicleMode.Accel and other.signal_mode == TLMode.YELLOW and ego.x>other.x-80 and ego.x<other.x-65:
        output.agent_mode = VehicleMode.HardBrake

    ## hypothesis: elif treated as if, else _always_ runs because verse evaluates conditions in parallel
    # if ego.agent_mode != VehicleMode.Accel and other.signal_mode == TLMode.GREEN:
    #     output.agent_mode = VehicleMode.Accel #accelerate no matter what if green 
    # elif ego.agent_mode != VehicleMode.Accel and other.signal_mode == TLMode.GREEN: 
    #     output.agent_mode = VehicleMode.HardBrake #checking if elif works
    # elif ego.agent_mode != VehicleMode.Accel and other.signal_mode == TLMode.GREEN:
    #     output.agent_mode = VehicleMode.Normal
    # else:
    #     output.agent_mode = VehicleMode.Accel

    # for now, adding an error message would be good, see if one could transform elif to if (condition) and not (prev if conditions)
    # goal: make elif and else working by converting them to ifs 
    # TO-DO check out parser
    ###########################################

    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-20 and ego.x<other.x-15)), "run red light"  
    assert not (other.signal_mode == TLMode.RED and (ego.x>other.x-15 and ego.x<other.x) and ego.v<1), "stop at intersection"

    return output 