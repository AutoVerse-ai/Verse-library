# <hyxml type="Model">
#   <automaton name="default_automaton">
#     <variable name="p" scope="LOCAL_DATA" type="Real" />
#     <variable name="lam" scope="LOCAL_DATA" type="Real" />
#     <variable name="pe" scope="LOCAL_DATA" type="Real" />
#     <variable name="ivalue" scope="LOCAL_DATA" type="Real" />
#     <variable name="t" scope="LOCAL_DATA" type="Real" />
#     <mode id="0" initial="True" name="Mode0">
#       <dai equation="p_dot= -476.851246128715*p**2 + 563.63999719734*p - 65.460328416" />
#       <dai equation="lam_dot= -4*lam - 33.3965838336589*p**2 + 99.359272121109*p + 33.8518550719433*pe**2 - 100.713764517065*pe - 4*(-5.05643107459819*p**2 + 15.0435539636327*p - 5.244048)*(-0.240071819537891*pe**2 + 0.714245545738554*pe - 0.248979591836735) + 4*(-4.97822527866553*pe**2 + 14.8108813346529*pe - 5.16294040816327)*(-0.240071819537891*pe**2 + 0.714245545738554*pe - 0.248979591836735) + 56.0441639020408" />
#       <dai equation="pe_dot= -478.163885472*p**2 + 567.545273568*p + 1.45848815920571*pe**2 - 4.33919596739959*pe - 65.309067936" />
#       <dai equation="ivalue_dot= 0" />
#       <dai equation="t_dot= 1" />
#       <dai equation="p_out= p" />
#       <dai equation="lam_out= lam" />
#       <dai equation="pe_out= pe" />
#       <dai equation="ivalue_out= ivalue" />
#       <dai equation="t_out= t" />
#       <invariant equation="t&lt;9.5" />
#     </mode>
#     <mode id="1" initial="False" name="Mode1">
#       <dai equation="p_dot= -476.851246128715*p**2 + 563.63999719734*p - 66.685538304" />
#       <dai equation="lam_dot= -4*lam - 33.3965838336589*p**2 + 99.359272121109*p + 82.9456*(0.0680272108843537*ivalue + 0.00272108843537415*lam + 0.0280272108843537)**2*(-3.529055747207*pe**2 + 10.4994095223567*pe - 0.366)**2 - 4*(0.0680272108843537*ivalue + 0.00272108843537415*lam + 0.0280272108843537)*(-5.05643107459819*p**2 + 15.0435539636327*p - 0.5244048)*(-3.529055747207*pe**2 + 10.4994095223567*pe - 0.366) - 141.0072*(0.0680272108843537*ivalue + 0.00272108843537415*lam + 0.0280272108843537)*(-3.529055747207*pe**2 + 10.4994095223567*pe - 0.366) + 52.10842488" />
#       <dai equation="pe_dot= -478.163885472*p**2 + 567.545273568*p + 1.45848815920571*pe**2 - 4.33919596739959*pe - 66.670412256" />
#       <dai equation="ivalue_dot= 0.14*lam - 2.058" />
#       <dai equation="t_dot= 1" />
#       <dai equation="p_out= p" />
#       <dai equation="lam_out= lam" />
#       <dai equation="pe_out= pe" />
#       <dai equation="ivalue_out= ivalue" />
#       <dai equation="t_out= t" />
#     </mode>
#     <transition destination="1" id="1" source="0">
#       <guard equation="t&gt;=9.5" />
#       <action equation="t = 0" />
#     </transition>
#   </automaton>
#   <composition automata="default_automaton" />
#   <property initialSet="Mode0:p==0.6353&amp;&amp;lam==14.7&amp;&amp;pe==0.5573&amp;&amp;ivalue==0.017&amp;&amp;t==0" name="Prop1" type="Safety" unsafeSet="lam&gt;=100">
#     <parameters kvalue="4000.0" timehorizon="15.0" timestep="0.001" />
#   </property>
# </hyxml>

from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent, Scenario
from verse.analysis.utils import wrap_to_pi 
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
    def dynamic_on(t, state):
        u, v = state
        u_dot = -0.9*u*u-u*u*u-0.9*u-v+1
        v_dot = u-2*v
        return [u_dot, v_dot]
    
    @staticmethod
    def dynamic_off(t, state):
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
                r = ode(self.dynamic_on)
            elif mode[0]=="Off":
                r = ode(self.dynamic_off)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace