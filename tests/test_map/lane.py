import unittest
from verse.map import Lane, AbstractLane, StraightLane, CircularLane
import numpy as np


class testLane(unittest.TestCase):
    # NOTE: If segment_list is initialized as empty list ([]), 
    # the initialization will have errors due to lane_width
    # and _set_longitudinal_start() will also have errors. 
    # In reality, are there any inputs to Verse that can set the segment_list to empty ?

    # NOTE: This test case does not work
    # since the 'width' of AbstractLane is not defined
    # def testInit_simple(self):
    #     abs_lane_1 = AbstractLane("lane_1")
    #     abs_lane_2 = AbstractLane("lane_2")
    #     abs_lane_3 = AbstractLane("lane_3")
    #     lane = Lane(id="simple", seg_list=[abs_lane_1, abs_lane_2, abs_lane_3])
    #     self.assertTrue(lane.lane_width == 4)

    def testInit_simple(self):
        abs_lane_1 = StraightLane("lane_1", start = [0, 0], end= [1, 1])
        abs_lane_2 = StraightLane("lane_2", start = [1, 1], end= [2, 2])
        abs_lane_3 = StraightLane("lane_3", start = [2, 2], end= [3, 3])
        lane = Lane(id="simple", seg_list=[abs_lane_1, abs_lane_2, abs_lane_3])
        self.assertTrue(lane.lane_width == 4)
        self.assertTrue(abs_lane_1.longitudinal_start == 0)
        self.assertTrue(abs_lane_2.longitudinal_start == np.sqrt(2))
        self.assertTrue(abs_lane_3.longitudinal_start == 2 * np.sqrt(2))

    def testInit_mix(self):
        abs_lane_1 = StraightLane("straight_lane_1", start = [0, 0], end = [1, 1])
        abs_lane_2 = CircularLane("circular_lane_1", center = [1, 1], radius=1.0, start_phase = 0, end_phase= np.pi)
        lane = Lane(id="Mixed", seg_list=[abs_lane_1, abs_lane_2])
        





if __name__ == "__main__":
    unittest.main()