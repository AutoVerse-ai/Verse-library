
from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.utils.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
import copy 

from enum import Enum, auto

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go

from verse.utils.fixed_points import *

### adapted from c2e2 

class NavAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamics_z1(t, state):
        x, y, vx, vy = state
        vx_dot = -1.2*vx+0.1*vy-0.1
        vy_dot = 0.1*vx-1.2*vy+1.2
        return [vx, vy, vx_dot, vy_dot]

    @staticmethod
    def dynamics_z2(t, state):
        x, y, vx, vy = state
        vx_dot = -1.2*vx+0.1*vy-4.8
        vy_dot = 0.1*vx-1.2*vy+0.4
        return [vx, vy, vx_dot, vy_dot]
    
    @staticmethod
    def dynamics_z3(t, state):
        x, y, vx, vy = state
        vx_dot = -1.2*vx+0.1*vy+2.4
        vy_dot = 0.1*vx-1.2*vy-0.2
        return [vx, vy, vx_dot, vy_dot]
    
    @staticmethod
    def dynamics_z4(t, state):
        x, y, vx, vy = state
        vx_dot = -1.2*vx+0.1*vy+3.9
        vy_dot = 0.1*vx-1.2*vy-3.9
        return [vx, vy, vx_dot, vy_dot]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            if mode[0]=="Zone1":
                r = ode(self.dynamics_z1)
            elif mode[0]=="Zone2":
                r = ode(self.dynamics_z2)
            elif mode[0]=="Zone3":
                r = ode(self.dynamics_z3)
            elif mode[0]=="Zone4":
                r = ode(self.dynamics_z4)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class NavMode(Enum):
    Zone1=auto()
    Zone2=auto()
    Zone3=auto()
    Zone4=auto()

class State:
    x: float
    y: float
    vx: float
    vy: float
    agent_mode: NavMode 

    def __init__(self, x, y, vx, vy, agent_mode: NavMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)

    if ego.agent_mode == NavMode.Zone1 and ego.x>=1 and ego.vx>0:
        output.agent_mode = NavMode.Zone2
    if ego.agent_mode == NavMode.Zone2 and ego.x<=1 and ego.vx<0:
        output.agent_mode = NavMode.Zone1
    if ego.agent_mode == NavMode.Zone2 and ego.y>=1 and ego.vy>0:
        output.agent_mode = NavMode.Zone4
    if ego.agent_mode == NavMode.Zone4 and ego.y<=1 and ego.vy<0:
        output.agent_mode = NavMode.Zone2
    if ego.agent_mode == NavMode.Zone1 and ego.y>=1 and ego.vy>0:
        output.agent_mode = NavMode.Zone3
    if ego.agent_mode == NavMode.Zone3 and ego.y<=1 and ego.vy<0:
        output.agent_mode = NavMode.Zone1
    if ego.agent_mode == NavMode.Zone3 and ego.x>=1 and ego.vx>0:
        output.agent_mode = NavMode.Zone4
    if ego.agent_mode == NavMode.Zone4 and ego.x<=1 and ego.vx<0:
        output.agent_mode = NavMode.Zone3

    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "nav_sys.py")
    Nav = NavAgent('nav', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Nav) ### need to add breakpoint around here to check decision_logic of agents

    init_nav = [[0.5, 0.5, 0, 0],[0.55, 0.55, 0, 0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'nav', init_nav, (NavMode.Zone1,)
    )

    # for t>2 seconds (I think around 3.7 seconds) verify starts taking a bit -- not that relevant considering x>0.5 is unsafe and that occurs pretty much constantly
    trace = scenario.verify(2, 0.01)

    # pp_fix(reach_at_fix(trace, 0, 10))

    ### fixed points eventually reached at t=120, not quite at t=60 though
    print(f'Fixed points exists? {fixed_points_fix(trace, 2, 0.01)}')

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace") 
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()