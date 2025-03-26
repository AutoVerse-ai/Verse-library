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
    
    # TODO: Edit this part of decision logic
    ''' This controller is part of the scenario where stars are safe but rectangles aren't'''
    if ego.agent_mode == VehicleMode.Normal:
        output.agent_mode = VehicleMode.Accel


    if (ego.agent_mode == VehicleMode.Accel) and other.dist <= (ego.v*10)/(6):
        output.agent_mode = VehicleMode.HardBrake
    if (ego.agent_mode == VehicleMode.HardBrake) and other.dist > (ego.v*10)/(6)+5:
        output.agent_mode = VehicleMode.Accel

    # if other.dist >= 60:
    #         if ego.agent_mode != VehicleMode.Accel:
    #             output.agent_mode = VehicleMode.Accel
    # if other.dist < 40:
    #         if ego.agent_mode != VehicleMode.HardBrake:
    #             output.agent_mode = VehicleMode.HardBrake

    ###########################################

    assert other.dist > 2.0, "Unsafe distance"

    return output 