from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d
from verse.map.lane import Lane

import numpy as np


class SimpleMap1(LaneMap_3d):
    def __init__(self, waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(waypoints=waypoints, guard_boxes=guard_boxes,
                         time_limits=time_limits, box_side=box_side, t_v_pair=t_v_pair)
        segment0 = StraightLane_3d(
            'seg0',
            [0, 0, 0],
            [100, 0, 0],
            3
        )
        segment1 = StraightLane_3d(
            'seg0',
            [0, 3, 0],
            [100, 10, 0],
            3
        )
        lane0 = Lane('Lane1', [segment0])
        lane1 = Lane('Lane2', [segment1])
        self.add_lanes([lane0, lane1])
