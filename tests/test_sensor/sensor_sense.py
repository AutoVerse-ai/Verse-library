import unittest
from verse.sensor import BaseSensor
from verse.agents.example_agent import NPCAgent, CarAgent
import os
from typing import Dict

# TODO: Need some discussion about how to effectively test the sense function in the base_sensor.py code in sensor director

class TestSense(unittest.TestCase): 
    def test_basic_1(self):
        sensor = BaseSensor()
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        lane_map = None
        val = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )
        state_dict = {"car" : val}
        print(sensor.sense(car, state_dict, lane_map, simulate=True))
        self.assertTrue(True)
if __name__ == "__main__":
    unittest.main()