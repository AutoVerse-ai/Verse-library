from enum import Enum, auto
import copy

class NullMode(Enum):
    Null = auto()

class LaneMode(Enum):
    Normal = auto()

class State:
    def __init__(self):
        self.mode = NullMode.Null
        self.lane_mode = LaneMode.Normal

def controller():
    return State()
