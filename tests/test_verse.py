# Introducing unittests for Verse development process
# Read more from https://docs.python.org/3/library/unittest.html

# A scenario is created for testing
import unittest
from ball_bounce_test import ball_bounce_test
from highway_test import highway_test
from verse.analysis.analysis_tree import AnalysisTreeNodeType

from enum import Enum, auto


class TestSimulatorMethods(unittest.TestCase):
    def setUp(self):
        pass

    def testBallBounce(self):
        '''
        Test basic ball bounce scenario
        Test plotter
        '''
        trace,_ = ball_bounce_test()

        assert len(trace.nodes) == 15

    def testHighWay(self):
        '''
        Test highway scenario
        Test both simulation and verification function
        ''' 
        trace_sim, trace_veri = highway_test()

        assert trace_sim.type == AnalysisTreeNodeType.SIM_TRACE
        assert trace_veri.type == AnalysisTreeNodeType.REACH_TUBE

if __name__ == "__main__":
    unittest.main()
