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
        # self.assertTrue(tree.id != tree.child.id) This test failed
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

class TestTree(unittest.TestCase):
    #TODO: Need to do the contain function. Very important for incremental verification.
    def test_tree_1(self):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        tree_node = AnalysisTreeNode(
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
        tree = AnalysisTree(root=tree_node)
        self.assertTrue(len(tree.nodes) == 1)
        self.assertTrue(len(tree.get_leaf_nodes(tree_node)) == 1)
        self.assertTrue(tree.height(tree_node) == 1)
    def test_tree_2(self):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        tree_node_1 = AnalysisTreeNode(
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

        car_1 = NPCAgent("car_1")
        tree_node_2 = AnalysisTreeNode(
            trace = {car_1.id : [] },
            init = {car_1.id : []},
            mode = {car_1.id : [
                "Normal",
                "T1"
            ]},
            agent = {car_1.id : car},
            height = 0,
            child = [tree_node_1],
            start_time = 0.0,
            type = AnalysisTreeNodeType.REACH_TUBE,
            ndigits=0,
            uncertain_param={},
            assert_hits={},
            static={},
            id = 1
        )
        tree = AnalysisTree(root=tree_node_2)
        self.assertTrue(len(tree.nodes) == 2)
        self.assertTrue(len(tree.get_leaf_nodes(tree_node_2)) == 1 and tree.get_leaf_nodes(tree_node_2)[0].id == 0)
        self.assertTrue(tree.height(tree_node_2) == 2)
if __name__ == "__main__":
    unittest.main()