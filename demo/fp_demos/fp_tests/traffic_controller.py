from enum import Enum, auto
import copy
from typing import List

class TLMode(Enum):
    GREEN=auto()
    YELLOW=auto()
    RED=auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    timer: float 
    signal_mode: TLMode 

    def __init__(self, x, y, theta, v, timer, signal_mode: TLMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.signal_mode == TLMode.GREEN and ego.timer > 20:
        output.signal_mode = TLMode.YELLOW
        output.timer = 0
    if ego.signal_mode == TLMode.YELLOW and ego.timer > 5:
        output.signal_mode = TLMode.RED 
        output.timer = 0 
    if ego.signal_mode == TLMode.RED and ego.timer > 20:
        output.signal_mode = TLMode.GREEN
        output.timer = 0  

    # assert True
    return output 