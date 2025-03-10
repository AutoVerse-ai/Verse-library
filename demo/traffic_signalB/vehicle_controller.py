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

    if ego.agent_mode != VehicleMode.Accel and other.dist > 100:
        output.agent_mode = VehicleMode.Accel

    # TODO: Edit this part of decision logic
    if ego.agent_mode == VehicleMode.Accel and other.dist < 30:
        output.agent_mode = VehicleMode.Brake

    # if ego.agent_mode != VehicleMode.Accel and other.dist < 25:
    #     output.agent_mode = VehicleMode.Accel

    # if ego.agent_mode != VehicleMode.Brake and other.dist < 10:
    #     output.agent_mode = VehicleMode.Brake





    ###########################################

    # DO NOT CHANGE THIS
    #assert other.dist > 1.0, "Too Close!"

    return output 