from typing import List

class LaneSegment:
    def __init__(self, id, left_lane, right_lane, next_segment, lane_parameter = None):
        self.id:int = id
        self.left_lane:int = left_lane
        self.right_lane:int = right_lane 
        self.next_segment:int = next_segment

        self.lane_parameter = None 
        if lane_parameter is not None:
            self.lane_parameter = lane_parameter

        