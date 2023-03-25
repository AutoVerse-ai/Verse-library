from enum import Enum, auto
import copy

class AgentMode(Enum):
    Default = auto()

class State:
    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0
    b = 0.0
    agent_mode: AgentMode = AgentMode.Default

    def __init__(self, x, y, agent_mode: AgentMode):
        pass

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    assert (ego.y1 < 2.75  and ego.y2 < 2.75 ), "unsafe set"
    return output
