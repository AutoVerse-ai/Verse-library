from typing import Dict, List
import copy

from ourtool.map.lane_segment import LaneSegment

class LaneMap:
    def __init__(self, lane_seg_list:List[LaneSegment] = []):
        self.lane_segment_dict:Dict[str, LaneSegment] = {}
        self.left_lane_dict:Dict[str, List[str]] = {}
        self.right_lane_dict:Dict[str, List[str]] = {}
        for lane_seg in lane_seg_list:
            self.lane_segment_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def add_lanes(self, lane_seg_list:List[LaneSegment]):
        for lane_seg in lane_seg_list:
            self.lane_segment_dict[lane_seg.id] = lane_seg
            self.left_lane_dict[lane_seg.id] = []
            self.right_lane_dict[lane_seg.id] = []

    def has_left(self, lane_idx):
        if lane_idx not in self.lane_segment_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        left_lane_list = self.left_lane_dict[lane_idx]
        return len(left_lane_list)>0

    def left_lane(self, lane_idx):
        assert all((elem in self.left_lane_dict) for elem in self.lane_segment_dict)
        if lane_idx not in self.left_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        left_lane_list = self.left_lane_dict[lane_idx]
        return copy.deepcopy(left_lane_list)
        
    def has_right(self, lane_idx):
        if lane_idx not in self.lane_segment_dict:
            Warning(f'lane {lane_idx} not available')
            return False
        right_lane_list = self.right_lane_dict[lane_idx]
        return len(right_lane_list)>0

    def right_lane(self, lane_idx):
        assert all((elem in self.right_lane_dict) for elem in self.lane_segment_dict)
        if lane_idx not in self.right_lane_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        right_lane_list = self.right_lane_dict[lane_idx]
        return copy.deepcopy(right_lane_list)
        
    def lane_geometry(self, lane_idx):
        return self.lane_segment_dict[lane_idx].get_geometry()