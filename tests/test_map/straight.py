import unittest
from verse.map import StraightLane, CircularLane
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

class testCircular(unittest.TestCase):
    def testPositionSimple(self):
        map = CircularLane("simple", center = [1, 2, 3, 4], radius=1.0, start_phase = np.pi, end_phase = 0)


if __name__ == "__main__":
    unittest.main()