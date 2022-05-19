from typing import List

import numpy as np

from dryvrpy.scene_verifier.map.lane_segment import AbstractLane

class Lane():
    COMPENSATE = 3
    def __init__(self, id, seg_list: List[AbstractLane]):
        self.id = id
        self.segment_list: List[AbstractLane] = seg_list
        self._set_longitudinal_start()

    def _set_longitudinal_start(self):
        longitudinal_start = 0
        for lane_seg in self.segment_list:
            lane_seg.longitudinal_start = longitudinal_start
            longitudinal_start += lane_seg.length

    def get_lane_segment(self, position:np.ndarray) -> AbstractLane:
        for seg_idx, segment in enumerate(self.segment_list):
            logitudinal, lateral = segment.local_coordinates(position)
            is_on = 0-Lane.COMPENSATE <= logitudinal < segment.length
            if is_on:
                return seg_idx, segment
        return -1,None

    def get_heading(self, position:np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        heading = segment.heading_at(longitudinal)
        return heading

    def get_longitudinal_position(self, position:np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        for i in range(seg_idx):    
            longitudinal += self.segment_list[i].length
        return longitudinal

    def get_lateral_distance(self, position:np.ndarray) -> float:
        seg_idx, segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        return lateral
