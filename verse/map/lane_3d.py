from typing import Tuple, List, Optional

import numpy as np

from verse.map.lane_segment_3d import AbstractLane_3d


class Lane_3d:
    COMPENSATE = 1

    def __init__(self, id, seg_list: List[AbstractLane_3d], plotted=True):
        self.id = id
        self.segment_list: List[AbstractLane_3d] = seg_list
        self._set_longitudinal_start()
        self.lane_width = seg_list[0].width
        self.plotted = plotted

    def _set_longitudinal_start(self):
        longitudinal_start = 0
        for lane_seg in self.segment_list:
            lane_seg.longitudinal_start = longitudinal_start
            longitudinal_start += lane_seg.length

    def get_lane_segment(self, position: np.ndarray) -> Tuple[int, AbstractLane_3d]:
        min_lateral = float("inf")
        idx = -1
        seg = None
        possible = []
        for seg_idx, segment in enumerate(self.segment_list):
            longitudinal, lateral, theta = segment.local_coordinates(position)
            is_on = 0 - Lane_3d.COMPENSATE <= longitudinal < segment.length
            if is_on:
                possible.append(segment)
                if lateral < min_lateral:
                    idx = seg_idx
                    seg = segment
                    min_lateral = lateral
        return idx, seg, possible

    # def get_heading(self, position: np.ndarray) -> float:
    #     seg_idx, segment = self.get_lane_segment(position)
    #     longitudinal, lateral, theta = segment.local_coordinates(position)
    #     heading = segment.heading_at(longitudinal)
    #     return heading

    def get_altitude(self):
        num = len(self.segment_list)
        sum_a = 0
        for seg in self.segment_list:
            sum_a += seg.altitude()
        return sum_a / num

    def get_longitudinal_position(self, position: np.ndarray) -> float:
        seg_idx, segment, poss = self.get_lane_segment(position)
        longitudinal, lateral, theta = segment.local_coordinates(position)
        for i in range(seg_idx):
            longitudinal += self.segment_list[i].length
        return longitudinal

    def get_lateral_distance(self, position: np.ndarray) -> float:
        seg_idx, segment, poss = self.get_lane_segment(position)
        longitudinal, lateral, theta = segment.local_coordinates(position)
        return lateral

    def get_theta_angle(self, position: np.ndarray) -> float:
        seg_idx, segment, poss = self.get_lane_segment(position)
        longitudinal, lateral, theta = segment.local_coordinates(position)
        return theta

    def get_l_r_theta(self, position: np.ndarray) -> Tuple[float, float, float]:
        seg_idx, segment, poss = self.get_lane_segment(position)
        longitudinal, lateral, theta = segment.local_coordinates(position)
        # for i in range(seg_idx):
        #     longitudinal += self.segment_list[i].length
        return longitudinal, lateral, theta

    def get_lane_width(self) -> float:
        return self.lane_width
