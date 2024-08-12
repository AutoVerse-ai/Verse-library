from typing import List

import numpy as np

from verse.map.lane_segment import AbstractLane


class Lane:
    COMPENSATE = 3

    def __init__(self, id, seg_list: List[AbstractLane], speed_limit=None):
        self.id = id
        self.segment_list: List[AbstractLane] = seg_list
        self.speed_limit = speed_limit
        self._set_longitudinal_start()
        self.lane_width = seg_list[0].width

    def _set_longitudinal_start(self):
        longitudinal_start = 0
        for lane_seg in self.segment_list:
            lane_seg.longitudinal_start = longitudinal_start
            longitudinal_start += lane_seg.length

    def get_lane_segment(self, position: np.ndarray) -> AbstractLane:
        min_lateral = float("inf")
        idx = -1
        seg = None
        for seg_idx, segment in enumerate(self.segment_list):
            logitudinal, lateral = segment.local_coordinates(position)
            is_on = 0 - Lane.COMPENSATE <= logitudinal < segment.length
            if is_on:
                if lateral < min_lateral:
                    idx = seg_idx
                    seg = segment
                    min_lateral = lateral
        return idx, seg

    def get_heading(self, position: np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        heading = segment.heading_at(longitudinal)
        return heading

    def get_longitudinal_position(self, position: np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        for i in range(seg_idx):
            longitudinal += self.segment_list[i].length
        return longitudinal

    def get_lateral_distance(self, position: np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        return lateral

    def get_lane_width(self) -> float:
        return self.lane_width

    def get_speed_limit(self):
        return self.speed_limit
