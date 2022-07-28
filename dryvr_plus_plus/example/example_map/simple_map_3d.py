from dryvr_plus_plus.scene_verifier.map.lane_map_3d import LaneMap_3d
from dryvr_plus_plus.scene_verifier.map.lane_segment_3d import StraightLane_3d
from dryvr_plus_plus.scene_verifier.map.lane import Lane

import numpy as np


class SimpleMap1(LaneMap_3d):
    def __init__(self, waypoints=[], guard_boxes=[], time_limits=[]):
        super().__init__(waypoints=waypoints, guard_boxes=guard_boxes, time_limits=time_limits)
        segment0 = StraightLane_3d(
            'seg0',
            [0, 0, 0],
            [100, 0, 0],
            3
        )
        lane0 = Lane('Lane1', [segment0])
        self.add_lanes([lane0])
