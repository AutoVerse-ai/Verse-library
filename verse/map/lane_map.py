from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from verse.map.lane_segment import AbstractLane
from verse.map.lane import Lane

class LaneMap:
    def __init__(self, lane_seg_list:List[Lane] = []):
        self.lane_dict:Dict[str, Lane] = {}
        self.left_lane_dict:Dict[str, List[str]] = {}
        self.right_lane_dict:Dict[str, List[str]] = {}
        for lane_seg in lane_seg_list:
            self.lane_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def add_lanes(self, lane_seg_list:List[AbstractLane]):
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
        return len(left_lane_list)>0

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
        return len(right_lane_list)>0

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

    def get_longitudinal_position(self, lane_idx:str, position:np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        return lane.get_longitudinal_position(position)

    def get_lateral_distance(self, lane_idx:str, position:np.ndarray) -> float:
        if position[0]>138 and position[0]<140 and position[1]>-9 and position[1]<-8:
            print("stop") 
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        return lane.get_lateral_distance(position)

    def get_altitude(self, lane_idx, position:np.ndarray) -> float:
        raise NotImplementedError

    def get_lane_heading(self, lane_idx:str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        return lane.get_heading(position)

    def get_lane_segment(self, lane_idx:str, position: np.ndarray) -> AbstractLane:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        seg_idx, segment = lane.get_lane_segment(position)
        return segment

    def get_speed_limit_old(self, lane_idx: str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_dict[lane_idx]
        limit = lane.get_speed_limit_old(position)
        # print(limit)
        # print(position)
        return limit

    def get_speed_limit(self, lane_idx: str) -> float:
        lane = self.lane_dict[lane_idx]
        # print(lane.get_speed_limit())
        return lane.get_speed_limit()

    def get_all_speed_limit(self) -> Dict[str, float]:
        ret_dict = {}
        for lane_idx, lane in self.lane_dict.items():
            ret_dict[lane_idx] = lane.get_speed_limit()
        # print(ret_dict)
        return ret_dict

    def get_lane_width(self, lane_idx: str) -> float:
        lane: Lane = self.lane_dict[lane_idx]
        return lane.get_lane_width() 
