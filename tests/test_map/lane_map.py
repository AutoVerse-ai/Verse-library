import unittest
from verse.map import LaneMap, StraightLane
import numpy as np

class testLaneMap(unittest.TestCase):
    def test_init(self):
        lane_map = LaneMap()
        self.assertTrue(lane_map.lane_dict == {})
        self.assertTrue(lane_map.left_lane_dict == {})
        self.assertTrue(lane_map.right_lane_dict == {})

    def test_addLane(self):
        lane_map = LaneMap()
        lane_map.add_lanes([StraightLane("straight_lane_1", start = [0, 0], end = [1, 1])])
        self.assertTrue(len(lane_map.lane_dict) == 1)
        self.assertTrue(len(lane_map.left_lane_dict) == 1)
        self.assertTrue(len(lane_map.right_lane_dict) == 1)

    def test_get_phys_lane(self):
        lane_map = LaneMap()
        lane_map.add_lanes([StraightLane("straight_lane_1", start = [0, 0], end = [1, 1])])
        lane_id = lane_map.get_phys_lane("straight_lane_1")
        self.assertTrue(lane_id == "straight_lane_1")
        self.assertTrue(lane_map.lane_dict[lane_id].width == 4)
        self.assertTrue(len(lane_map.left_lane_dict) == 1)
        self.assertTrue(len(lane_map.right_lane_dict) == 1)
        self.assertTrue(lane_map.left_lane_dict[lane_id] == [])
        self.assertTrue(lane_map.right_lane_dict[lane_id] == [])
    
if __name__ == "__main__":
    unittest.main()