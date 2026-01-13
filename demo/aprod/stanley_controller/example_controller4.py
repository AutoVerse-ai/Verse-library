from enum import Enum, auto
import copy
from typing import List


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

class SensorMode(Enum):
    Ready = auto()
    Updated = auto()

class State:
    x: float
    y: float
    theta: float
    v: float
    d: float; psi: float; timer: float; priority: int
    agent_mode: AgentMode
    track_mode: TrackMode
    sensor_mode: SensorMode

    def __init__(self, x, y, theta, v, d, psi, timer, priority,
                 agent_mode: AgentMode, track_mode: TrackMode, sensor_mode: SensorMode):
        pass

class MiniState:
    long: float # longitudinal position
    x: float
    track_mode: TrackMode

    def __init__(self, long, x, track_mode: TrackMode):
        pass

class ExtraState:
    long: float; sensor_update: int

# def vehicle_front(ego, others, track_map):
#     res = any(
#         (
#             5
#             > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
#             - track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
#             > 3
#             and ego.track_mode == other.track_mode
#         )
#         for other in others
#     )
#     return res

# def vehicle_front(ego: State, others: List[MiniState]):
def vehicle_front(ego: State, extra: ExtraState, others: List[MiniState]):
    res = any(
        (
            5 > other.long - extra.long > 3 # originally 5>diff>3
            # 5 > other.x - ego.x > 3 # originally 5>diff>3
            and ego.track_mode == other.track_mode
        ) for other in others
    )
    return res

# def vehicle_close(ego, others):
#     return any(-1 < ego.x - other.x < 1 and -1 < ego.y - other.y < 1 for other in others)

ts = 1

def decisionLogic(ego: State, others: List[MiniState], extra: ExtraState, track_map):
    output = copy.deepcopy(ego)

    # ready for update 
    if extra.sensor_update >= 1 and ego.sensor_mode == SensorMode.Ready:
        output.d = ego.d*1
        output.psi = ego.psi*1        
        output.sensor_mode = SensorMode.Updated

        # only allow track/mode transitions in update
        if ego.agent_mode == AgentMode.Normal:
            if vehicle_front(ego, extra, others): # NOTE: due to weirdness established above, does it make sense to include  
                if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft):
                    output.agent_mode = AgentMode.SwitchLeft
                    output.track_mode = track_map.h(
                        ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft
                    )
            if vehicle_front(ego, extra, others):
                if track_map.h_exist(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight):
                    output.agent_mode = AgentMode.SwitchRight
                    output.track_mode = track_map.h(
                        ego.track_mode, ego.agent_mode, AgentMode.SwitchRight
                    )

        if ego.agent_mode == AgentMode.SwitchLeft:
            if ego.d >= 2.5: # may need to inc
                output.agent_mode = AgentMode.Normal
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
        
        if ego.agent_mode == AgentMode.SwitchRight:
            if ego.d <= -2.5:
                output.agent_mode = AgentMode.Normal
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)

    # done with continuous post, need to get update
    if extra.sensor_update >= 1 and ego.sensor_mode == SensorMode.Updated and ego.timer >= 0.9*ts:
        output.timer = 0
        output.sensor_mode = SensorMode.Ready
    # assert not vehicle_close(ego, others)
    return output
