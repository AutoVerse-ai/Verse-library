import unittest
from verse.map import StraightLane
import numpy as np
class testStraight(unittest.TestCase):
    def testPositionSimple(self):
        map = StraightLane("simple", start= [1,2,3,4], end= [3,4,5,6])
        self.assertTrue(map.id == "simple")
        self.assertTrue(map.type == "Straight")
        self.assertTrue(map.width == 4)
        self.assertTrue(map.length != 0)
        self.assertTrue(list(map.start.shape) == list(map.end.shape))
        self.assertTrue(list(map.direction.shape) == list(map.start.shape))

if __name__ == "__main__":
    unittest.main()