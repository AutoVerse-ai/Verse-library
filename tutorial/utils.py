dl1 = "from enum import Enum, auto\n\
\n\
class AgentMode(Enum):\n\
    Normal = auto()\n\
    Brake = auto()\n\
\n\
class TrackMode(Enum):\n\
    T0 = auto()\n\
\n\
class State:\n\
    x:float\n\
    y:float\n\
    theta:float\n\
    v:float\n\
    agent_mode:AgentMode\n\
    track_mode:TrackMode \n\
\n\
    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):\n\
        pass\n\
\n\
import copy\n\
def decisionLogic(ego:State, track_map):\n\
    output = copy.deepcopy(ego)\n\
    if ego.agent_mode == AgentMode.Normal:\n\
        if ego.x > 10:\n\
            output.agent_mode = 'Brake'\n\
    return output\n\
"