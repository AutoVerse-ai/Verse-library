from typing import List

import numpy as np

from src.scene_verifier.map.lane_segment import AbstractLane

class Lane():
    def __init__(self, id, seg_list: List[AbstractLane]):
        self.id = id
        self.segment_list: List[AbstractLane] = seg_list

    def get_lane_segment(self, position:np.ndarray) -> AbstractLane:
        for segment in self.segment_list:
            logitudinal, lateral = segment.local_coordinates(position)
            is_on = 0 <= logitudinal < segment.length
            if is_on:
                return segment
        return None

    def get_longitudinal_error(self, position:np.ndarray) -> float:
        segment = self.get_lane_segment(position)
        longituinal, lateral = segment.local_coordinates(position)
        return longituinal

    def get_lateral_error(self, position:np.ndarray) -> float:
        segment = self.get_lane_segment(position)
        longituinal, lateral = segment.local_coordinates(position)
        return lateral
