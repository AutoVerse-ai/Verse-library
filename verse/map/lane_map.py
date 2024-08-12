from typing import Dict, List
import re
from enum import Enum

import numpy as np

from verse.map.lane_segment import AbstractLane
from verse.map.lane import Lane

VIRT_LANE = re.compile(r"M\d\d")
PHYS_LANE = re.compile(r"T\d")
INTERSECT_PHYS_LANE = re.compile(r"[A-Z]+_\d+")
INTERSECT_VIRT_LANE = re.compile(r"[A-Z]+_\d+_\d+")


class LaneMap:
    def __init__(self, lane_seg_list: List[Lane] = []):
        self.lane_dict: Dict[str, Lane] = {}
        self.left_lane_dict: Dict[str, List[str]] = {}
        self.right_lane_dict: Dict[str, List[str]] = {}
        self.h_dict = {}
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def add_lanes(self, lane_seg_list: List[AbstractLane]):
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    @staticmethod
    def get_phys_lane(lane):
        # res = LaneMap._get_phys_lane(lane)
        # print(f"phys({lane}) -> {res}")
        # return res

        # @staticmethod
        # def _get_phys_lane(lane):
        if isinstance(lane, Enum):
            lane = lane.name
        if VIRT_LANE.match(lane):
            return f"T{lane[1]}"
        if INTERSECT_VIRT_LANE.match(lane):
            return lane.rsplit("_", 1)[0]
        # if PHYS_LANE.match(lane) or INTERSECT_PHYS_LANE.match(lane):
        #     return lane
        return lane

    def lane_geometry(self, lane_idx):
        return self.lane_dict[LaneMap.get_phys_lane(lane_idx)].get_geometry()

    def get_longitudinal_position(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        # print(self.lane_dict)
        lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        return lane.get_longitudinal_position(position)

    def get_lateral_distance(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        return lane.get_lateral_distance(position)

    def get_altitude(self, lane_idx, position: np.ndarray) -> float:
        raise NotImplementedError

    def get_lane_heading(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        return lane.get_heading(position)

    def get_lane_segment(self, lane_idx: str, position: np.ndarray) -> AbstractLane:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        seg_idx, segment = lane.get_lane_segment(position)
        return segment

    def get_speed_limit(self, lane_idx: str) -> float:
        lane: Lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        # print(lane.get_speed_limit())
        return lane.get_speed_limit()

    def get_all_speed_limit(self) -> Dict[str, float]:
        ret_dict = {}
        for lane_idx, lane in self.lane_dict.items():
            ret_dict[lane_idx] = lane.get_speed_limit()
        # print(ret_dict)
        return ret_dict

    def get_lane_width(self, lane_idx: str) -> float:
        lane: Lane = self.lane_dict[LaneMap.get_phys_lane(lane_idx)]
        return lane.get_lane_width()

    def h(self, lane_idx, agent_mode_src, agent_mode_dest):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if isinstance(agent_mode_src, Enum):
            agent_mode_src = agent_mode_src.name
        if isinstance(agent_mode_dest, Enum):
            agent_mode_dest = agent_mode_dest.name
        if self.h_dict == {}:
            return None
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx, agent_mode_src, agent_mode_dest):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if isinstance(agent_mode_src, Enum):
            agent_mode_src = agent_mode_src.name
        if isinstance(agent_mode_dest, Enum):
            agent_mode_dest = agent_mode_dest.name
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False
