from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent, Scenario
from verse.utils.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
import copy 


### full disclosure, structure of file from mp4_p2

def tree_safe(tree: AnalysisTree):
    for node in tree.nodes:
        if node.assert_hits is not None:
            return False 
    return True

    #   <dai equation="u_dot = -0.9*u*u-u*u*u-0.9*u-v+1"/>
    #   <dai equation="v_dot = u-2*v"/>
    #   <dai equation="u_out = u"/>
    #   <dai equation="v_out = v"/>
    #   <invariant equation="u&lt;0.5"/>
    # </mode>
    # <mode id="1" initial="False" name="stimOff">
    #   <dai equation="u_dot = -0.9*u*u-u*u*u-0.9*u-v"/>
    #   <dai equation="v_dot = u-2*v"/>
    #   <dai equation="u_out = u"/>
    #   <dai equation="v_out = v"/>
    
class CellAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)
         
    @staticmethod
    def dynamics_on(t, state):
        u, v = state
        u_dot = -0.9*u*u-u*u*u-0.9*u-v+1
        v_dot = u-2*v
        return [u_dot, v_dot]
    
    @staticmethod
    def dynamics_off(t, state):
        u, v = state
        u_dot = -0.9*u*u-u*u*u-0.9*u-v
        v_dot = u-2*v
        return [u_dot, v_dot]


    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            if mode[0]=="On":
                r = ode(self.dynamics_on)
            elif mode[0]=="Off":
                r = ode(self.dynamics_off)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace