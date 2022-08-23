from typing import Dict, List
import copy
from enum import Enum

import numpy as np

from verse.map.lane_segment import AbstractLane
from verse.map.lane import Lane
from verse.map.lane_map import LaneMap


class LaneMap_3d(LaneMap):

    def __init__(self, lane_seg_list: List[Lane] = [], waypoints: dict = {}, guard_boxes: dict = {}, time_limits: dict = {}, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(lane_seg_list)
        # these are for the Follow_Waypoint mode of qurdrotor
        self.waypoints = waypoints
        self.guard_boxes = guard_boxes
        self.time_limits = time_limits
        # these are for the Follow_Lane mode of qurdrotor
        self.box_side = box_side
        self.t_v_pair = t_v_pair

    def get_waypoint_by_id(self, agent_id, waypoint_id):
        return self.waypoints[agent_id][waypoint_id]

    def check_guard_box(self, agent_id, state, waypoint_id):
        if waypoint_id >= len(self.guard_boxes[agent_id]):
            return False
        box = self.guard_boxes[agent_id][int(waypoint_id)]
        for i in range(len(box[0])):
            if state[i] < box[0][i] or state[i] > box[1][i]:
                return False
        return True

    def get_timelimit_by_id(self, agent_id, waypoint_id):
        return self.time_limits[agent_id][waypoint_id]

    def get_next_point(self, lane, agent_id, waypoint_id):
        curr_waypoint = self.waypoints[agent_id][waypoint_id]
        curr_point = np.array(curr_waypoint[0:3])
        longitudinal = self.get_longitudinal_position(lane, curr_point)
        lateral = self.get_lateral_distance(lane, curr_point)
        seg = self.get_lane_segment(lane, curr_point)
        next_point = seg.position(
            longitudinal+self.t_v_pair[agent_id][0]*self.t_v_pair[agent_id][1], lateral)
        next_seg = self.get_lane_segment(lane, next_point)
        if seg == next_seg:
            pass
        else:
            next_point = next_seg.position(0, lateral)

        if len(curr_waypoint) == 3:
            self.waypoints[agent_id][waypoint_id] = curr_point.tolist() + \
                next_point.tolist()
        else:
            self.waypoints[agent_id].append(
                curr_point.tolist()+next_point.tolist())
        return
