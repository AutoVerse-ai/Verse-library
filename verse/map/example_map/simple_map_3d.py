from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d
from verse.map.lane_3d import Lane_3d
from math import pi

import numpy as np


class SimpleMap1(LaneMap_3d):
    def __init__(self, waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(waypoints=waypoints, guard_boxes=guard_boxes,
                         time_limits=time_limits, box_side=box_side, t_v_pair=t_v_pair)
        segment0 = StraightLane_3d(
            'seg0',
            [0, 0, 0],
            [100, 0, 0],
            2
        )
        segment1 = StraightLane_3d(
            'seg0',
            [0, 3, 0],
            [100, 10, 0],
            2
        )
        lane0 = Lane_3d('Lane1', [segment0])
        lane1 = Lane_3d('Lane2', [segment1])
        self.add_lanes([lane0, lane1])


class SimpleMap2(LaneMap_3d):
    def __init__(self, waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(waypoints=waypoints, guard_boxes=guard_boxes,
                         time_limits=time_limits, box_side=box_side, t_v_pair=t_v_pair)
        segment0 = CircularLane_3d(
            'seg0',
            [0, 0, 0],
            5,
            [1, 1, 1],
            0, 2*pi,
            True, 2
        )
        # segment1 = StraightLane_3d(
        #     'seg0',
        #     [0, 3, 0],
        #     [100, 10, 0],
        #     3
        # )
        lane0 = Lane_3d('Lane1', [segment0])
        # lane1 = Lane_3d('Lane2', [segment1])
        # self.add_lanes([lane0, lane1])
        self.add_lanes([lane0])


class SimpleMap3(LaneMap_3d):
    def __init__(self, waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(waypoints=waypoints, guard_boxes=guard_boxes,
                         time_limits=time_limits, box_side=box_side, t_v_pair=t_v_pair)
        segment0 = CircularLane_3d(
            'seg0',
            [0, 0, 0],
            5,
            [1, 1, 1],
            0, 0.8*pi,
            True, 2
        )
        # segment1 = StraightLane_3d(
        #     'seg0',
        #     [0, 3, 0],
        #     [100, 10, 0],
        #     3
        # )
        lane0 = Lane_3d('Lane1', [segment0])
        # lane1 = Lane_3d('Lane2', [segment1])
        # self.add_lanes([lane0, lane1])
        self.add_lanes([lane0])
