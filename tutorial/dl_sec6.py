from enum import Enum, auto


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    M01 = auto()
    M10 = auto()


class State:
    x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass


from typing import List
import copy

#DL 1
# def decisionLogic(ego: State, other: State, track_map):
#     output = copy.deepcopy(ego)
#     if ego.agent_mode == AgentMode.Normal:
#         if ego.track_mode == other.track_mode and 6 < (other.x - ego.x) and (other.x - ego.x) < 8:
#             if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft):
#                 output.agent_mode = AgentMode.SwitchLeft
#                 output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
        
#         if ego.track_mode == other.track_mode and 6 < (other.x - ego.x) and (other.x - ego.x) < 8:
#             if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight):
#                 output.agent_mode = AgentMode.SwitchRight
#                 output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)

#     if ego.agent_mode == AgentMode.SwitchLeft:
#         if ego.y >= 2.5:
#             output.agent_mode = AgentMode.Normal
#             output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
#     if ego.agent_mode == AgentMode.SwitchRight:
#         if ego.y <= -2.5:
#             output.agent_mode = AgentMode.Normal
#             output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
#     return output

#DL 2 and 3
def decisionLogic(ego: State, other: State, track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Normal:
        if ego.track_mode == other.track_mode and 6 < other.dist and other.dist < 8:
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft):
                output.agent_mode = AgentMode.SwitchLeft
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
        
        #drop
        #if ego.track_mode == other.track_mode and 6 < other.dist and other.dist < 8:
            if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight):
                output.agent_mode = AgentMode.SwitchRight
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)

    if ego.agent_mode == AgentMode.SwitchLeft:
        if ego.y >= 2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
    if ego.agent_mode == AgentMode.SwitchRight:
        if ego.y <= -2.5:
            output.agent_mode = AgentMode.Normal
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
    return output