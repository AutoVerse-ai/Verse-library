from enum import Enum, auto
import copy


class AgentMode(Enum):
    Default = auto()


class State:
    x = 0.0
    y = 0.0
    agent_mode: AgentMode = AgentMode.Default

    def __init__(self, x, y, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)

    return output
