from enum import Enum, auto
import copy
from typing import List

class BallMode(Enum):
    Normal = auto()

class State:
    y: float 
    vy: float 
    agent_mode: BallMode 

    def __init__(self, y, vy, agent_mode: BallMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)

    # TODO: Edit this part of decision logic
    output = copy.deepcopy(ego)
    if ego.y < 0:
        output.vy = -ego.vy # arbitrary value to simulate the loss of energy from hitting the ground
        output.y = 0
    # if ego.vy!=0 and ((ego.vy<=0.01 and ego.vy>0) or (ego.vy>=-0.01 and ego.vy<0)):
    #     output.vy = 0

    return output 