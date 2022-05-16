from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from src.scene_verifier.map.lane_segment import AbstractLane
from src.scene_verifier.map.lane import Lane

class LaneMap:
    def __init__(self, lane_seg_list:List[Lane] = []):
        self.lane_segment_dict:Dict[str, Lane] = {}
        self.left_lane_dict:Dict[str, List[str]] = {}
        self.right_lane_dict:Dict[str, List[str]] = {}
        for lane_seg in lane_seg_list:
            self.lane_segment_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def add_lanes(self, lane_seg_list:List[AbstractLane]):
        for lane_seg in lane_seg_list:
            self.lane_segment_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def has_left(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_segment_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        left_lane_list = self.left_lane_dict[lane_idx]
        return len(left_lane_list)>0

    def left_lane(self, lane_idx):
        assert all((elem in self.left_lane_dict) for elem in self.lane_segment_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.left_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        left_lane_list = self.left_lane_dict[lane_idx]
        return copy.deepcopy(left_lane_list)
        
    def has_right(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.lane_segment_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        right_lane_list = self.right_lane_dict[lane_idx]
        return len(right_lane_list)>0

    def right_lane(self, lane_idx):
        assert all((elem in self.right_lane_dict) for elem in self.lane_segment_dict)
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        if lane_idx not in self.right_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        right_lane_list = self.right_lane_dict[lane_idx]
        return copy.deepcopy(right_lane_list)
        
    def lane_geometry(self, lane_idx):
        if isinstance(lane_idx, Enum):
            lane_idx = lane_idx.name
        return self.lane_segment_dict[lane_idx].get_geometry()

    def get_longitudinal(self, lane_idx:str, position:np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_segment_dict[lane_idx]
        return lane.get_longitudinal(position)

    def get_lateral(self, lane_idx:str, position:np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_segment_dict[lane_idx]
        return lane.get_lateral(position)

    def get_altitude(self, lane_idx, position:np.ndarray) -> float:
        raise NotImplementedError

    def get_lane_heading(self, lane_idx:str, position: np.ndarray) -> float:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_segment_dict[lane_idx]
        return lane.get_heading(position)

    def get_lane_segment(self, lane_idx:str, position: np.ndarray) -> AbstractLane:
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        lane = self.lane_segment_dict[lane_idx]
        return lane.get_lane_segment(position)