from enum import Enum, auto
import copy
from typing import List

class CellMode(Enum):
    On=auto()
    Off=auto()

class State:
    u: float 
    v: float 
    agent_mode: CellMode 

    def __init__(self, u, v, agent_mode: CellMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)

    if ego.agent_mode == CellMode.On and ego.u>=0.5:
        output.agent_mode = CellMode.Off
    if ego.agent_mode==CellMode.Off and ego.u<=0:
        output.agent_mode = CellMode.On

    return output 