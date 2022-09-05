from audioop import ratecv
from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from verse.map.lane_segment_3d import AbstractLane_3d
from verse.map.lane_3d import Lane_3d


class LaneMap_3d:

    def __init__(self, lane_seg_list: List[Lane_3d] = [], waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        self.lane_dict: Dict[str, Lane_3d] = {}
        self.left_lane_dict: Dict[str, List[str]] = {}
        self.right_lane_dict: Dict[str, List[str]] = {}
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []
        # these are for the Follow_Waypoint mode of qurdrotor
        self.waypoints = waypoints
        self.guard_boxes = guard_boxes
        self.time_limits = time_limits
        # these are for the Follow_Lane mode of qurdrotor
        self.box_side = box_side
        self.t_v_pair = t_v_pair

    def add_lanes(self, lane_seg_list: List[AbstractLane_3d]):
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def has_left(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        left_lane_list = self.left_lane_dict[lane_idx]
        return len(left_lane_list) > 0

    def left_lane(self, lane_idx):
        assert all((elem in self.left_lane_dict) for elem in self.lane_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.left_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_dict")
        left_lane_list = self.left_lane_dict[lane_idx]
        return copy.deepcopy(left_lane_list[0])

    def has_right(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        right_lane_list = self.right_lane_dict[lane_idx]
        return len(right_lane_list) > 0

    def right_lane(self, lane_idx):
        assert all((elem in self.right_lane_dict) for elem in self.lane_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.right_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_dict")
        right_lane_list = self.right_lane_dict[lane_idx]
        return copy.deepcopy(right_lane_list[0])

    def lane_geometry(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        return self.lane_dict[lane_idx].get_geometry()

    def get_longitudinal_position(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        # print(self.lane_dict)
        lane = self.lane_dict[lane_idx]
        return lane.get_longitudinal_position(position)

    def get_lateral_distance(self, lane_idx: str, position: np.ndarray) -> float:
        if position[0] > 138 and position[0] < 140 and position[1] > -9 and position[1] < -8:
            print("stop")
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        return lane.get_lateral_distance(position)

    def get_theta_angle(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        # print(self.lane_dict)
        lane = self.lane_dict[lane_idx]
        return lane.get_theta_angle(position)

    # def get_lane_heading(self, lane_idx: str, position: np.ndarray) -> float:
    #     if not isinstance(position, np.ndarray):
    #         position = np.array(position)
    #     lane = self.lane_dict[lane_idx]
    #     return lane.get_heading(position)

    def get_lane_segment(self, lane_idx: str, position: np.ndarray) -> AbstractLane_3d:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        seg_idx, segment = lane.get_lane_segment(position)
        return segment

    # def get_speed_limit(self, lane_idx: str) -> float:
    #     lane = self.lane_dict[lane_idx]
    #     # print(lane.get_speed_limit())
    #     return lane.get_speed_limit()

    # def get_all_speed_limit(self) -> Dict[str, float]:
    #     ret_dict = {}
    #     for lane_idx, lane in self.lane_dict.items():
    #         ret_dict[lane_idx] = lane.get_speed_limit()
    #     # print(ret_dict)
    #     return ret_dict

    def get_lane_width(self, lane_idx: str) -> float:
        lane: Lane_3d = self.lane_dict[lane_idx]
        return lane.get_lane_width()

# waypoints related
    def get_waypoint_by_id(self, agent_id, waypoint_id):
        return self.waypoints[agent_id][waypoint_id]

    def check_guard_box(self, agent_id, state, waypoint_id):
        # print("check_guard_box", state)
        waypoint_id = int(waypoint_id)
        if agent_id not in self.guard_boxes or waypoint_id >= len(self.guard_boxes[agent_id]):
            dest = self.waypoints[agent_id][waypoint_id][3:]
            box_side = self.box_side[agent_id]
            for i in range(len(dest)):
                if state[i] < dest[i]-box_side[i]/2 or state[i] > dest[i]+box_side[i]/2:
                    return False
        else:
            box = self.guard_boxes[agent_id][int(waypoint_id)]
            for i in range(len(box[0])):
                if state[i] < box[0][i] or state[i] > box[1][i]:
                    return False
        print("check_guard_box", state)
        # print(self.waypoints[agent_id][waypoint_id])
        return True

    def get_timelimit_by_id(self, agent_id, waypoint_id):
        return self.time_limits[agent_id][waypoint_id]

    def get_next_point(self, lane, agent_id, waypoint_id):
        curr_waypoint = self.waypoints[agent_id][int(waypoint_id)]
        curr_point = np.array(curr_waypoint[3:])
        longitudinal = self.get_longitudinal_position(lane, curr_point)
        lateral = self.get_lateral_distance(lane, curr_point)
        theta = self.get_theta_angle(lane, curr_point)
        print('get_next_point', theta)
        seg = self.get_lane_segment(lane, curr_point)
        est_len = self.t_v_pair[agent_id][0]*self.t_v_pair[agent_id][1]
        rate = 0.02
        if est_len >= lateral/rate:
            next_point = seg.position(
                longitudinal+(est_len**2-lateral**2)**0.5, 0, theta)
        else:
            next_point = seg.position(
                longitudinal+est_len*(1-rate**2)**0.5, lateral-rate*est_len, theta)

        next_seg = self.get_lane_segment(lane, next_point)
        if seg == next_seg:
            pass
        else:
            next_point = next_seg.position(0, lateral, theta)

        next_waypoint = curr_point.tolist() + next_point.tolist()
        if len(curr_waypoint) == 3:
            self.waypoints[agent_id][waypoint_id] = curr_point.tolist() + \
                next_point.tolist()
        else:
            self.waypoints[agent_id].append(
                curr_point.tolist()+next_point.tolist())
        print('next', next_waypoint)
        return next_waypoint
