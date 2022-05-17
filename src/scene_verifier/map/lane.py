from typing import List

import numpy as np

from src.scene_verifier.map.lane_segment import AbstractLane

class Lane():
    COMPENSATE = 3
    def __init__(self, id, seg_list: List[AbstractLane]):
        self.id = id
        self.segment_list: List[AbstractLane] = seg_list

    def get_lane_segment(self, position:np.ndarray) -> AbstractLane:
        for segment in self.segment_list:
            logitudinal, lateral = segment.local_coordinates(position)
            is_on = 0-Lane.COMPENSATE <= logitudinal < segment.length
            if is_on:
                return segment
        return None

    def get_heading(self, position:np.ndarray) -> float:
        segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        heading = segment.heading_at(longitudinal)
        return heading

    def get_longitudinal(self, position:np.ndarray) -> float:
        segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        return longitudinal

    def get_lateral_distance(self, position:np.ndarray) -> float:
        segment = self.get_lane_segment(position)
        longitudinal, lateral = segment.local_coordinates(position)
        return lateral
