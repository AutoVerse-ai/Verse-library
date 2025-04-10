import unittest
from verse.map import CircularLane
import numpy as np

class testCircular(unittest.TestCase):
    def testAssignmentSimple(self):
        map = CircularLane("simple", center = [1, 2, 3, 4], radius=1.0, start_phase = np.pi, end_phase = 0)
        self.assertTrue(map.id == "simple")
        self.assertTrue(map.type == "Circular")
        self.assertTrue(map.radius == 1)
        self.assertTrue(map.start_phase == np.pi)
        self.assertTrue(map.start_phase == np.pi)
        self.assertTrue(map.end_phase == 0)
    
    def testAssignmentMedium_1(self):
         map = CircularLane("simple", center = [1, 2, 3, 4], radius=1.0, start_phase = np.pi, end_phase = 0, clockwise = False)
         self.assertTrue(map.length >= 0) # assert length inside CircularLane to be >= 0
    
    def testAssignmentMedium_2(self):
         map = CircularLane("simple", center = [1, 2, 3, 4], radius=1.0, start_phase = np.pi, end_phase = 0, width = -0.1)
         self.assertTrue(map.width >= 0) # This test should never exist if width is asserted to be >= 0 inside the class


if __name__ == "__main__":
    unittest.main()