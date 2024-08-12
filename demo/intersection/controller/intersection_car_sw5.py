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
    sw_time = 0.0
    agent_mode: AgentMode = AgentMode.Accel
    track_mode: TrackMode = TrackMode.none


lane_width = 3


def car_ahead(ego, others, track, track_map, thresh_far, thresh_close):
    def car_front(car):
        ego_long = track_map.get_longitudinal_position(track, [ego.x, ego.y])
        ego_lat = track_map.get_lateral_distance(track, [ego.x, ego.y])
        car_long = track_map.get_longitudinal_position(track, [car.x, car.y])
        car_lat = track_map.get_lateral_distance(track, [car.x, car.y])
        return (
            thresh_close < car_long - ego_long < thresh_far
            and -lane_width / 2 < car_lat - ego_lat < lane_width / 2
        )

    return any(car_front(other) and other.track_mode == track for other in others)


def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.sw_time >= 1:
        car_front = car_ahead(ego, others, ego.track_mode, track_map, 5, 0)
        if ego.agent_mode == AgentMode.Accel and car_front:
            left_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
            right_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)
            if left_lane != None and not car_ahead(ego, others, left_lane, track_map, 8, -3):
                output.agent_mode = AgentMode.SwitchLeft
                output.track_mode = left_lane
                output.sw_time = 0
            if right_lane != None and not car_ahead(ego, others, right_lane, track_map, 8, -3):
                output.agent_mode = AgentMode.SwitchRight
                output.track_mode = right_lane
                output.sw_time = 0
            # else:
            #     output.agent_mode = AgentMode.Brake
        if ego.agent_mode == AgentMode.Brake and not car_front:
            output.agent_mode = AgentMode.Accel
            output.sw_time = 0
        lat_dist = track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y])
        lat = 2
        if ego.agent_mode == AgentMode.SwitchLeft and lat_dist >= lat:
            output.agent_mode = AgentMode.Accel
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)
            output.sw_time = 0
        if ego.agent_mode == AgentMode.SwitchRight and lat_dist <= -lat:
            output.agent_mode = AgentMode.Accel
            output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Accel)
            output.sw_time = 0

    return output
