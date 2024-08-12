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
    SwitchLeft2 = auto()
    SwitchRight = auto()
    SwitchRight2 = auto()
    Brake = auto()
    Stop = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()


class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    sw_time = 0.0
    agent_mode: AgentMode = AgentMode.Normal
    track_mode: TrackMode = TrackMode.T0
    type: LaneObjectMode = LaneObjectMode.Vehicle


def car_ahead(ego, others, track, track_map, thresh_far, thresh_close):
    return any(
        (
            thresh_far
            > track_map.get_longitudinal_position(other.track_mode, [other.x, other.y])
            - track_map.get_longitudinal_position(ego.track_mode, [ego.x, ego.y])
            > thresh_close
            and other.track_mode == track
        )
        for other in others
    )


def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.sw_time >= 1:
        if ego.agent_mode == AgentMode.Normal:
            left_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft)
            left2_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchLeft2)
            right_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight)
            right2_lane = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.SwitchRight2)
            if car_ahead(ego, others, ego.track_mode, track_map, 5, 3):
                # Switch left if left lane is empty
                if left_lane != None and not car_ahead(ego, others, left_lane, track_map, 8, -3):
                    output.agent_mode = AgentMode.SwitchLeft
                    output.track_mode = left_lane
                    output.sw_time = 0
                if left2_lane != None and not car_ahead(ego, others, left2_lane, track_map, 8, -3):
                    output.agent_mode = AgentMode.SwitchLeft2
                    output.track_mode = left2_lane
                    output.sw_time = 0
                if right_lane != None and not car_ahead(ego, others, right_lane, track_map, 8, -3):
                    output.agent_mode = AgentMode.SwitchRight
                    output.track_mode = right_lane
                    output.sw_time = 0
                if right2_lane != None and not car_ahead(
                    ego, others, right2_lane, track_map, 8, -3
                ):
                    output.agent_mode = AgentMode.SwitchRight2
                    output.track_mode = right2_lane
                    output.sw_time = 0
        else:  # If switched enough, return to normal mode
            lat = track_map.get_lateral_distance(ego.track_mode, [ego.x, ego.y])
            lane_width = track_map.get_lane_width(ego.track_mode)
            if (
                ego.agent_mode == AgentMode.SwitchLeft
                and lat >= (lane_width - 0.2)
                or ego.agent_mode == AgentMode.SwitchLeft2
                and lat >= (lane_width * 2 - 0.2)
                or ego.agent_mode == AgentMode.SwitchRight
                and lat <= -(lane_width - 0.2)
                or ego.agent_mode == AgentMode.SwitchRight2
                and lat <= -(lane_width * 2 - 0.2)
            ):
                output.agent_mode = AgentMode.Normal
                output.track_mode = track_map.h(ego.track_mode, ego.agent_mode, AgentMode.Normal)
                output.sw_time = 0
    return output
