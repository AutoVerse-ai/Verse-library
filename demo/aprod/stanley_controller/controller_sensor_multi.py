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


def vehicle_front(ego, others, track_map):
    res = any(
        (
            5
            > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
            > 3
            and ego.track_mode == other.track_mode
        )
        for other in others
    )
    return res


def vehicle_close(ego, others):
    return any(-1 < ego.x - other.x < 1 and -1 < ego.y - other.y < 1 for other in others)

ts = 0.1 # time step of the scenario

def decisionLogic(ego: State, other, track_map):
    output = copy.deepcopy(ego)
    # if other.sensor_update >= 1:
    #     output.d = ego.d*1
    #     output.psi = ego.psi*1
    #     output.timer = 0 

    # ready for update 
    if other.sensor_update >= 1 and ego.sensor_mode == SensorMode.Ready:
        output.d = ego.d*1
        output.psi = ego.psi*1        
        output.sensor_mode = SensorMode.Updated

    # done with continuous post, need to get update
    if other.sensor_update >= 1 and ego.sensor_mode == SensorMode.Updated and ego.timer >= 0.9*ts:
        output.timer = 0
        output.sensor_mode = SensorMode.Ready

    # NOTE: in the future, only permit agentmode updates if sensormode is updated to further prevent extra updates
    return output
