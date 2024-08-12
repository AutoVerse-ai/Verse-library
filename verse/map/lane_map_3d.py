from audioop import ratecv
from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from verse.map.lane_segment_3d import AbstractLane_3d
from verse.map.lane_3d import Lane_3d


class LaneMap_3d:
    def __init__(self, lane_seg_list: List[Lane_3d] = []):
        self.lane_dict: Dict[str, Lane_3d] = {}
        self.left_lane_dict: Dict[str, List[str]] = {}
        self.right_lane_dict: Dict[str, List[str]] = {}
        self.up_lane_dict: Dict[str, List[str]] = {}
        self.down_lane_dict: Dict[str, List[str]] = {}
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []
            self.up_lane_dict[lane_seg.id] = []
            self.down_lane_dict[lane_seg.id] = []

        # self.box_side = box_side
        # self.t_v_pair = t_v_pair
        self.curr_wp = {}
        self.curr_seg = {}
        self.wps = {}

    def trans_func(self, lane_idx: str) -> str:
        lane = "T" + lane_idx[-1]
        return lane

    def add_lanes(self, lane_seg_list: List[AbstractLane_3d]):
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []
            self.up_lane_dict[lane_seg.id] = []
            self.down_lane_dict[lane_seg.id] = []

    def h(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if isinstance(agent_mode_src, Enum):
            agent_mode_src = agent_mode_src.name
        if isinstance(agent_mode_dest, Enum):
            agent_mode_dest = agent_mode_dest.name
        return self.h_func(lane_idx, agent_mode_src, agent_mode_dest)

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if isinstance(agent_mode_src, Enum):
            agent_mode_src = agent_mode_src.name
        if isinstance(agent_mode_dest, Enum):
            agent_mode_dest = agent_mode_dest.name
        return self.h_exist_func(lane_idx, agent_mode_src, agent_mode_dest)

    def h_func(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        raise NotImplementedError

    def h_exist_func(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        raise NotImplementedError

    def g_func(self, agent_state, lane_idx) -> np.ndarray:
        raise NotImplementedError

    # def trans_func(self, lane_idx: str) -> str:
    #     raise NotImplementedError

    def pair_lanes(self, lane_idx_src, lane_idx_dest, relation):
        if isinstance(lane_idx_src, Enum):
            lane_idx_src = lane_idx_src.name
        if isinstance(lane_idx_dest, Enum):
            lane_idx_dest = lane_idx_dest.name
        if relation == "left":
            self.left_lane_dict[lane_idx_src].append(lane_idx_dest)
            self.right_lane_dict[lane_idx_dest].append(lane_idx_src)
        elif relation == "right":
            self.right_lane_dict[lane_idx_src].append(lane_idx_dest)
            self.left_lane_dict[lane_idx_dest].append(lane_idx_src)
        elif relation == "up":
            self.up_lane_dict[lane_idx_src].append(lane_idx_dest)
            self.down_lane_dict[lane_idx_dest].append(lane_idx_src)
        elif relation == "down":
            self.down_lane_dict[lane_idx_src].append(lane_idx_dest)
            self.up_lane_dict[lane_idx_dest].append(lane_idx_src)
        else:
            raise ValueError

    def has_left(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_dict:
            Warning(f"lane {lane_idx} not available")
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
            Warning(f"lane {lane_idx} not available")
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

    def has_up(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_dict:
            Warning(f"lane {lane_idx} not available")
            return False
        up_lane_list = self.up_lane_dict[lane_idx]
        return len(up_lane_list) > 0

    def up_lane(self, lane_idx):
        assert all((elem in self.up_lane_dict) for elem in self.lane_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.left_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_dict")
        up_lane_list = self.up_lane_dict[lane_idx]
        return copy.deepcopy(up_lane_list[0])

    def has_down(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_dict:
            Warning(f"lane {lane_idx} not available")
            return False
        down_lane_list = self.down_lane_dict[lane_idx]
        return len(down_lane_list) > 0

    def down_lane(self, lane_idx):
        assert all((elem in self.down_lane_dict) for elem in self.lane_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.left_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_dict")
        down_lane_list = self.down_lane_dict[lane_idx]
        return copy.deepcopy(down_lane_list[0])

    def lane_geometry(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        return self.lane_dict[lane_idx].get_geometry()

    def altitude(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        return self.lane_dict[self.trans_func(lane_idx)].get_altitude()

    def get_longitudinal_position(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        # print(self.lane_dict)
        lane = self.lane_dict[lane_idx]
        return lane.get_longitudinal_position(position)

    def get_lateral_distance(self, lane_idx: str, position: np.ndarray) -> float:
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

    def get_l_r_theta(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        # print(self.lane_dict)
        lane = self.lane_dict[lane_idx]
        return lane.get_l_r_theta(position)

    # def get_lane_heading(self, lane_idx: str, position: np.ndarray) -> float:
    #     if not isinstance(position, np.ndarray):
    #         position = np.array(position)
    #     lane = self.lane_dict[lane_idx]
    #     return lane.get_heading(position)

    def get_lane_segment(self, lane_idx: str, position: np.ndarray) -> AbstractLane_3d:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        seg_idx, segment, possible_seg = lane.get_lane_segment(position)
        return segment, possible_seg

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

    def get_curr_waypoint(self, agent_id):
        return self.wps[agent_id]

    def check_guard_box(self, agent_id, state, box_side):
        dest = self.curr_wp[agent_id][3:]
        for i in range(len(dest)):
            if state[i] < dest[i] - box_side[i] / 2 or state[i] > dest[i] + box_side[i] / 2:
                return False
        return True

    def get_next_point(self, lane, agent_id, pos, velocity, t_v_pair):
        est_len = t_v_pair[0] * t_v_pair[1]
        if isinstance(pos, np.ndarray):
            curr_point = pos[:3]
        elif agent_id in self.curr_wp:
            curr_point = self.curr_wp[agent_id][3:]
        else:
            raise ValueError
        seg, possible_seg = self.get_lane_segment(lane, curr_point)
        if agent_id not in self.curr_seg:
            self.curr_seg[agent_id] = seg
        else:
            if self.curr_seg[agent_id] in possible_seg:
                seg = self.curr_seg[agent_id]
        longitudinal, lateral, theta = seg.local_coordinates(curr_point)

        rate = 0.7
        if est_len >= lateral / rate:
            next_longitudinal = longitudinal + (est_len**2 - lateral**2) ** 0.5
            next_lateral = 0
            next_theta = theta
        else:
            next_longitudinal = longitudinal + est_len * (1 - rate**2) ** 0.5
            next_lateral = lateral - rate * est_len
            next_theta = theta
        next_point = seg.position(next_longitudinal, next_lateral, next_theta)
        if next_longitudinal > seg.length:
            next_seg, possible_next_seg = self.get_lane_segment(lane, next_point)
            max_in = -float("inf")
            for n_seg in possible_next_seg:
                next_point = n_seg.position(0, lateral, theta)
                delta = (next_point - curr_point) / np.linalg.norm(next_point - curr_point)
                d = np.inner(delta, velocity)
                if d > max_in:
                    max_in = d
                    next_seg = n_seg
            next_point = next_seg.position(0, lateral, theta)
            self.curr_seg[agent_id] = next_seg

        next_waypoint = list(curr_point) + next_point.tolist()

        self.curr_wp[agent_id] = next_waypoint
        if agent_id not in self.wps:
            self.wps[agent_id] = [next_waypoint]
        else:
            self.wps[agent_id].append(next_waypoint)
        # print('next', next_waypoint)
        return next_waypoint

    # def get_time_limit(self, agent_id):
    #     return self.t_v_pair[agent_id][0]
