import unittest
import os
from verse.analysis import AnalysisTree, AnalysisTreeNode, AnalysisTreeNodeType
from verse.agents.example_agent import NPCAgent, CarAgent, BaseAgent

class TestTreeNode(unittest.TestCase):
    def test_simple(self):
        tree = AnalysisTreeNode(
            trace = {},
            init = {},
            mode = {},
            agent = {},
            height = 0,
            child = [],
            start_time = 0.0,
            type = AnalysisTreeNodeType.SIM_TRACE,
            ndigits=0,
            uncertain_param={},
            assert_hits={},
            static={},
            id = 0
        )
        self.assertTrue(tree.start_time >= 0)
        self.assertTrue(tree.height >= 0)

    def test_id(self):
        tree = AnalysisTreeNode(
            trace = {},
            init = {},
            mode = {},
            agent = {},
            height = 0,
            child = [],
            start_time = 0.0,
            type = AnalysisTreeNodeType.SIM_TRACE,
            ndigits=0,
            uncertain_param={},
            assert_hits={},
            static={},
            id = 0
        )
        tree_child = tree.new_child(
            init = {},
            mode = {},
            trace = {},
            start_time = 0.0,
            id = 0
        )
        # self.assertTrue(tree.id != tree.child.id) 
        self.assertTrue(tree_child.height > 0)
    
    def test_get_track(self):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        tree = AnalysisTreeNode(
            trace = {car.id : [] },
            init = {car.id : []},
            mode = {car.id : [
                "Normal",
                "T1"
            ]},
            agent = {car.id : car},
            height = 0,
            child = [],
            start_time = 0.0,
            type = AnalysisTreeNodeType.SIM_TRACE,
            ndigits=0,
            uncertain_param={},
            assert_hits={},
            static={},
            id = 0
        )
        self.assertTrue(tree.get_track(agent_id=car.id, mode=tree.mode[car.id]) == "T1")
    def test_get_mode(self):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        tree = AnalysisTreeNode(
            trace = {car.id : [] },
            init = {car.id : []},
            mode = {car.id : [
                "Accel",
                "T2"
            ]},
            agent = {car.id : car},
            height = 0,
            child = [],
            start_time = 0.0,
            type = AnalysisTreeNodeType.SIM_TRACE,
            ndigits=0,
            uncertain_param={},
            assert_hits={},
            static={},
            id = 0
        )
        self.assertTrue(tree.get_mode(agent_id=car.id, mode=tree.mode[car.id]) == "Accel")
if __name__ == "__main__":
    unittest.main()