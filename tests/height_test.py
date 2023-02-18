import unittest
from enum import Enum, auto
import sys
import time
import plotly.graph_objects as go
from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M2
from verse import Scenario
from verse.plotter.plotter2D import *

from enum import Enum, auto
import sys
import time
import plotly.graph_objects as go
class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M01 = auto()
    M12 = auto()
    M23 = auto()
    M40 = auto()
    M04 = auto()
    M32 = auto()
    M21 = auto()
    M10 = auto()

class heightTest(unittest.TestCase):
    def setUp(self):
        input_code_name = './example_controller7.py'
        self.scenario = Scenario()

        car = CarAgent('car1', file_name=input_code_name)
        self.scenario.add_agent(car)
        car = NPCAgent('car2')
        self.scenario.add_agent(car)
        car = CarAgent('car3', file_name=input_code_name)
        self.scenario.add_agent(car)
        car = NPCAgent('car4')
        self.scenario.add_agent(car)
        car = NPCAgent('car5')
        self.scenario.add_agent(car)
        car = NPCAgent('car6')
        self.scenario.add_agent(car)
        car = NPCAgent('car7')
        self.scenario.add_agent(car)
        tmp_map = M2()
        self.scenario.set_map(tmp_map)
        self.scenario.set_init(
            [
                [[0, -0.1, 0, 1.0], [0.0, 0.1, 0, 1.0]],
                [[10, -0.1, 0, 0.5], [10, 0.1, 0, 0.5]],
                [[14.5, 2.9, 0, 0.6], [14.5, 3.1, 0, 0.6]],
                [[20, 2.9, 0, 0.5], [20, 3.1, 0, 0.5]],
                [[30, -0.1, 0, 0.5], [30, 0.1, 0, 0.5]],
                [[23, -3.1, 0, 0.5], [23, -2.9, 0, 0.5]],
                [[40, -6.1, 0, 0.5], [40, -5.9, 0, 0.5]],
            ],
            [
                (AgentMode.Normal, TrackMode.T1),
                (AgentMode.Normal, TrackMode.T1),
                (AgentMode.Normal, TrackMode.T0),
                (AgentMode.Normal, TrackMode.T0),
                (AgentMode.Normal, TrackMode.T1),
                (AgentMode.Normal, TrackMode.T2),
                (AgentMode.Normal, TrackMode.T3),
            ]
        )
    def test_exp3_scenario_3(self):
        traces = self.scenario.simulate(80, 0.05, 3)
        self.assertEqual(False, exceedsMax(traces.root,3))
    def test_exp3_verify_3(self):
        traces = self.scenario.verify(80, 0.05,3)
        self.assertEqual(False, exceedsMax(traces.root,3))
    def test_exp3_scenario_none(self):
        traces = self.scenario.simulate(80, 0.05)
        heights = []
        heights = sorted(getHeights(traces.root,heights))
        self.assertEqual([6,8,8,8,8], heights)
    def test_exp3_verify_none(self):
        traces = self.scenario.verify(80, 0.05)
        heights = []
        heights = sorted(getHeights(traces.root,heights))
        self.assertEqual([6,8,8,8,8,8,8], heights)



if __name__ == '__main__':
    unittest.main()


def exceedsMax(root, max_height):
    if root:
        if(root.child == []):
            #print("HEIGHT", root.height)
            if(root.height > max_height):
                return True
                print("Exceeds max height")
        for c in root.child:
            if(exceedsMax(c, max_height)):
                return True
        return False


def getHeights(root,heights):
    if root:
        if(root.child == []):
            #print("HEIGHT", root.height)
            heights.append(root.height)
        for c in root.child:
            getHeights(c,heights)
        return heights
