from typing import List

class LaneSegment:
    def __init__(self, id, lane_parameter = None):
        self.id = id
        # self.left_lane:List[str] = left_lane
        # self.right_lane:List[str] = right_lane 
        # self.next_segment:int = next_segment

        self.lane_parameter = None 
        if lane_parameter is not None:
            self.lane_parameter = lane_parameter

    def get_geometry():
        pass