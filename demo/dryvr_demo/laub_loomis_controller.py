from enum import Enum, auto
import copy

class AgentMode(Enum):
    Default = auto()

class State:
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0
    x7 = 0.0
    W= 0.0
    agent_mode: AgentMode = AgentMode.Default

    def __init__(self, x, y, agent_mode: AgentMode):
        pass

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    assert (ego.W != .1 or \
            ego.x4 < 5), "unsafe set"
    assert (ego.W == .1 or \
            (ego.x4 < 4.5)), "unsafe"

    return output
