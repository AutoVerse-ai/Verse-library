from enum import Enum, auto
import copy
from typing import List

class AgentMode(Enum):
    Accel = auto()
    Brake = auto()

class TrackMode(Enum):
    NS = auto()
    NE = auto()
    NW = auto()
    SN = auto()
    SE = auto()
    SW = auto()
    EN = auto()
    ES = auto()
    EW = auto()
    WN = auto()
    WS = auto()
    WE = auto()

# class LaneObjectMode(Enum):
#     Vehicle = auto()
#     Ped = auto()        # Pedestrians

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: AgentMode = AgentMode.Accel
    track_mode: TrackMode = TrackMode.NS
    # type: LaneObjectMode = LaneObjectMode.Vehicle

    def __init__(
        self,
        x,
        y,
        theta,
        v,
        vehicle_mode: AgentMode,
        track_mode: TrackMode,
        # type: LaneObjectMode,
    ):
       pass

lane_width = 3
def cars_front(ego, others, track_map):
    def car_front(other):
        ego_long = track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
        ego_lat = track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y])
        car_long = track_map.get_longitudinal_position(ego.track_mode, [other.x, other.y])
        car_lat = track_map.get_lateral_distance(ego.track_mode, [other.x, other.y])
        return 0 < car_long - ego_long < 7 and -lane_width / 2 < car_lat - ego_lat < lane_width / 2
    return any(car_front(other) for other in others)

def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.vehicle_mode == AgentMode.Accel and cars_front(ego, others, track_map):
        output.vehicle_mode = AgentMode.Brake
    if ego.vehicle_mode == AgentMode.Brake and not cars_front(ego, others, track_map):
        output.vehicle_mode = AgentMode.Accel
    return output
