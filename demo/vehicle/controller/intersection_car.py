from enum import Enum, auto
import copy
from typing import List

class AgentMode(Enum):
    Accel = auto()
    Brake = auto()
    SwitchLeft = auto()
    SwitchRight = auto()

class TrackMode(Enum):
    none = auto()

# class LaneObjectMode(Enum):
#     Vehicle = auto()
#     Ped = auto()        # Pedestrians

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    agent_mode: AgentMode = AgentMode.Accel
    track_mode: TrackMode = TrackMode.none

lane_width = 3

def car_left(ego, others, track_map):
    return any((-7 < track_map.get_longitudinal_position(other.track_mode, [other.x,other.y]) - track_map.get_longitudinal_position(ego.track_mode, [ego.x,ego.y]) < 7 and \
                 other.track_mode==track_map.left_lane(ego.track_mode)) for other in others)

def car_right(ego, others, track_map):
    return any((-7 < track_map.get_longitudinal_position(other.track_mode, [other.x,other.y]) - track_map.get_longitudinal_position(ego.track_mode, [ego.x,ego.y]) < 7 and \
                 other.track_mode==track_map.right_lane(ego.track_mode)) for other in others)

def cars_ahead(track, ego, others, track_map):
    def car_front(car):
        ego_long = track_map.get_longitudinal_position(track, [ego.x, ego.y])
        ego_lat = track_map.get_lateral_distance(track, [ego.x, ego.y])
        car_long = track_map.get_longitudinal_position(track, [car.x, car.y])
        car_lat = track_map.get_lateral_distance(track, [car.x, car.y])
        return 0 < car_long - ego_long < 7 and -lane_width / 2 < car_lat - ego_lat < lane_width / 2
    return any(car_front(other) for other in others)

def cars_front(ego, others, track_map):
    return cars_ahead(ego.track_mode, ego, others, track_map)

def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Accel and cars_front(ego, others, track_map):
        alledged_left_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
        alledged_right_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)
        if alledged_left_lane != 'none' and not car_left(ego, others, track_map):
            output.agent_mode = AgentMode.SwitchLeft
            output.track_mode = alledged_left_lane
        if alledged_right_lane != 'none' and not car_right(ego, others, track_map):
            output.agent_mode = AgentMode.SwitchRight
            output.track_mode = alledged_right_lane
        # else:
        #     output.agent_mode = AgentMode.Brake
    if ego.agent_mode == AgentMode.Brake and not cars_front(ego, others, track_map):
        output.agent_mode = AgentMode.Accel
    lat_dist = track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y])
    lat = 2
    if ego.agent_mode == AgentMode.SwitchLeft and lat_dist >= lat:
        output.agent_mode = AgentMode.Accel
        output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)
    if ego.agent_mode == AgentMode.SwitchRight and lat_dist <= -lat:
        output.agent_mode = AgentMode.Accel
        output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)

    # if ego.agent_mode == AgentMode.SwitchLeft:
    #     if lat_dist >= lat:
    #         output.agent_mode = AgentMode.Accel
    #         output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)
    #     if cars_front(ego, others, track_map):
    #         output.agent_mode = AgentMode.Brake

    # if ego.agent_mode == AgentMode.SwitchRight:
    #     if lat_dist <= -lat:
    #         output.agent_mode = AgentMode.Accel
    #         output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)
    #     if cars_front(ego, others, track_map):
    #         output.agent_mode = AgentMode.Brake

    return output
