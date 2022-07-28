from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from dryvr_plus_plus.scene_verifier.map.lane_segment import AbstractLane
from dryvr_plus_plus.scene_verifier.map.lane import Lane
from dryvr_plus_plus.scene_verifier.map.lane_map import LaneMap


class LaneMap_3d(LaneMap):

    def __init__(self, lane_seg_list: List[Lane] = [], waypoints: List = [], guard_boxes: List = [], time_limits: List = []):
        super().__init__(lane_seg_list)
        # these are for the Follow_Waypoint mode of qurdrotor
        self.waypoints = waypoints
        self.guard_boxes = guard_boxes
        self.time_limits = time_limits

    def get_waypoint_by_id(self, waypoint_id):
        return self.waypoints[waypoint_id]

    def check_guard_box(self, state, waypoint_id):
        if waypoint_id >= len(self.guard_boxes):
            return False
        box = self.guard_boxes[int(waypoint_id)]
        for i in range(len(box[0])):
            if state[i] < box[0][i] or state[i] > box[1][i]:
                return False
        return True

    def get_timelimit_by_id(self, waypoint_id):
        return self.time_limits[waypoint_id]
