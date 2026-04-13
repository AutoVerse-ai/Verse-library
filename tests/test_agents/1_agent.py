import unittest

from verse import BaseAgent
from verse.agents.example_agent import BallAgent, NPCAgent, CarAgent
import matplotlib.pyplot as plt
import numpy as np
import os 

class TestBaseAgents(unittest.TestCase):

    def test_initializing_ball(self):
        aball = BallAgent(
        "red_ball", file_name="./demo/ball/ball_bounces.py"
        )
        self.assertTrue(aball.id == "red_ball")
        print("Initilizing test Passed")

    def test_trace_ball(self):
        aball = BallAgent(
        "red_ball", file_name="./demo/ball/ball_bounces.py"
        )
        trace = aball.TC_simulate({"none"}, [5, 10, 2, 2], 10, 0.05)
        self.assertTrue(len(trace) == 201)
        for u, x, y, dx, dy in trace:
            self.assertTrue(x > 0 and y > 0)
            self.assertTrue(dx > 0 and dy > 0)
        print("Test ball Passed")

    def test_car(self):
        script_dir = os.path.dirname(__file__)
        input_code_name = os.path.abspath(os.path.join(script_dir, '..', 'test_controller', 'example_controller5.py'))
        car = CarAgent("car", file_name=input_code_name)

if __name__ == "__main__":
    unittest.main()

