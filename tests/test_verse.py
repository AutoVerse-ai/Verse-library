# Introducing unittests for Verse development process
# Read more from https://docs.python.org/3/library/unittest.html

# A scenario is created for testing
import unittest
from ball_bounce_test import ball_bounce_test
from highway_test import highway_test
from verse import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNodeType

from enum import Enum, auto


class TestSimulatorMethods(unittest.TestCase):
    def setUp(self):
        pass

    # def test_m2_2c5n(self):
    #     trace = m2_2c5n_test()
    #     root = trace.root
    #     '''
    #     Test the max height
    #     '''
    #     max_height = 33
    #     assert trace.height(root) <= max_height
    #     print("Max height test passed!")
    #
    #
    #     '''
    #     Test the number of nodes
    #     '''
    #     assert len(trace.nodes) == 33
    #     print("Nodes number test passed!")

    # # def testBallBounce(self):
    #     '''
    #     Test basic ball bounce scenario
    #     Test plotter
    #     '''
    #     trace, _ = ball_bounce_test()
    #
    #     '''
    #     Test properties of root node
    #     '''
    #     root = trace.root
    #     # baseAgent = BaseAgent.__init__()
    #     # assert root.agent == baseAgent
    #     # assert root.mode == ""
    #     # assert root.start_time == 0
    #     # print("Root test passed!")
    #
    #     '''
    #     Test the max height
    #     '''
    #     max_height = 15
    #     assert trace.height(root) <= max_height
    #     print("Max height test passed!")
    #
    #     '''
    #     Test properties of leaf node
    #     '''
    #     # leafs = self.get_leaf_nodes(root)
    #     # for leave in leafs:
    #     #     assert leave.agent == baseAgent
    #     #     assert leave.mode == ""
    #     #     assert leave.start_time == 0
    #     # print("Leave node test passed!")
    #
    #     '''
    #     Test the number of nodes
    #     '''
    #     #assert len(trace.nodes) == 10
    #     #print("Nodes number test passed!")

    def testHighWay(self):
        '''
        Test highway scenario
        Test both simulation and verification function
        '''
        trace_sim, trace_veri = highway_test()

        # assert trace_sim.type == AnalysisTreeNodeType.SIM_TRACE
        # assert trace_veri.type == AnalysisTreeNodeType.REACH_TUBE


if __name__ == "__main__":
    unittest.main()
