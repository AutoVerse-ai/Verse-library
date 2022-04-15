from ourtool.map.lane_map import LaneMap
from ourtool.map.lane_segment import LaneSegment

class SingleStraightLaneMap(LaneMap):
    def __init__(self):
        super().__init__()
        segment = LaneSegment(0, None, None, None, None)
        self.lane_segment_dict[segment.id] = segment