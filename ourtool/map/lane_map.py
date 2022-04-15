from typing import Dict 

from ourtool.map.lane_segment import LaneSegment

class LaneMap:
    def __init__(self):
        self.lane_segment_dict:Dict[int:LaneSegment] = {}

    def get_left_lane_idx(self, lane_idx):
        if lane_idx not in self.lane_segment_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        lane_segment:LaneSegment = self.lane_segment_dict[lane_idx]
        return lane_segment.left_lane
        
    def get_left_lane_segment(self,lane_idx):
        left_lane_idx = self.get_left_lane_idx(lane_idx)
        return self.lane_segment_dict[left_lane_idx]

    def get_right_lane_idx(self, lane_idx):
        if lane_idx not in self.lane_segment_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        lane_segment:LaneSegment = self.lane_segment_dict[lane_idx]
        return lane_segment.right_lane
        
    def get_right_lane_segment(self,lane_idx):
        right_lane_idx = self.get_right_lane_idx(lane_idx)
        return self.lane_segment_dict[right_lane_idx]

    def get_next_lane_idx(self, lane_idx):
        if lane_idx not in self.lane_segment_dict:
            raise ValueError(f"lane_idx {lane_idx} not in lane_segment_dict")
        lane_segment:LaneSegment = self.lane_segment_dict[lane_idx]
        return lane_segment.next_segment
        
    def get_next_lane_segment(self,lane_idx):
        next_lane_idx = self.get_next_lane_idx(lane_idx)
        return self.lane_segment_dict[next_lane_idx]
