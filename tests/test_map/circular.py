import unittest
from verse.map import CircularLane
import numpy as np

class testCircular(unittest.TestCase):
    def testAssignmentSimple(self):
        map = CircularLane("simple", center = [1, 2, 3], radius=1.0, start_phase = np.pi, end_phase = 0)
        self.assertTrue(map.id == "simple")
        self.assertTrue(map.type == "Circular")
        self.assertTrue(map.radius == 1)
        self.assertTrue(map.start_phase == np.pi)
        self.assertTrue(map.end_phase == 0)
    
    def testAssignmentMedium_1(self):
         map = CircularLane("medium_1", center = [1, 2, 3], radius=1.0, start_phase = np.pi, end_phase = 0, clockwise = False)
         self.assertTrue(map.length >= 0) # assert length inside CircularLane to be >= 0
    
    def testAssignmentMedium_2(self):
         map = CircularLane("medium_2", center = [1, 2, 3], radius=1.0, start_phase = np.pi, end_phase = 0, width = -0.1)
         self.assertTrue(map.width >= 0) # This test should never exist if width is asserted to be >= 0 inside the class

    # This test case causes errors
    # This is due to the center has the shape of (3,)
    # and in the returning value of position() function, 
    # it adds two values of shape (3,) and (2,), respectively.
    # def testPosition(self):
    #     map = CircularLane("simple", center = [1, 2, 3], radius=1.0, start_phase = np.pi, end_phase = 0)
    #     position_1 = map.position(longitudinal=1.0, lateral=1.0)
    #     position_2 = map.position(longitudinal=-1.0, lateral=1.0)
    #     position_3 = map.position(longitudinal=1.0, lateral=-1.0)
    #     position_4 = map.position(longitudinal=-1.0, lateral=-1.0)
    
    def testPosition(self):
        # Not sure what to test here
        # TODO: Need some discussion
        map = CircularLane("simple", center = [1, 3], radius=1.0, start_phase = np.pi, end_phase = 0)
        position_1 = map.position(longitudinal=1.0, lateral=1.0)
        position_2 = map.position(longitudinal=-1.0, lateral=1.0)
        position_3 = map.position(longitudinal=1.0, lateral=-1.0)
        position_4 = map.position(longitudinal=-1.0, lateral=-1.0)
    
    def testHeading(self):
        map = CircularLane("simple", center = [1, 2, 3], radius=1.0, start_phase = np.pi, end_phase = 0)
        heading_1 = map.heading_at(longitudinal=1.0)
        heading_2 = map.heading_at(longitudinal=-1.0)
        self.assertTrue(heading_1 >= -np.pi and heading_1 <= np.pi)
        self.assertTrue(heading_2 >= -np.pi and heading_2 <= np.pi)

if __name__ == "__main__":
    unittest.main()