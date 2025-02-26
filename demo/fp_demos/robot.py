
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

class RobotAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamics(t, state):
        x0, x1, x2, x3 = state
        x0_dot = x2
        x1_dot = x3
        x2_dot = (-2*1.0*x1*x2*x3-2.0*x0-2.0*x2)/(1.0*x1*x1 + 1.0)+(4)/(1.0*x1*x1+1.0)
        x3_dot = x1*x2*x2 - 1.0*x1 - 1.0*x3 + 1.0
        return [x0_dot, x1_dot, x2_dot, x3_dot]

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class RobotMode(Enum):
    Normal=auto()

class State:
    x0: float
    x1: float
    x2: float
    x3: float
    agent_mode: RobotMode 

    def __init__(self, x0, x1, x2, x3, agent_mode: RobotMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)
    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "robot.py")
    Robot = RobotAgent('robot', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Robot) ### need to add breakpoint around here to check decision_logic of agents

    init_robot = [[1.5, 1.5, 0, 0],[1.51, 1.51, 0.01, 0.01]]
    # # -----------------------------------------

    scenario.set_init_single(
        'robot', init_robot, (RobotMode.Normal,)
    )

    trace = scenario.verify(10, 0.01)

    # pp_fix(reach_at_fix(trace, 0, 10))

    # no fixed points reached by t=60, x0, x1 seem to converge but x3 at least doesn't 
    # fixed point reached by t=120, trying to plot it out will take a bit
    print(f'Fixed points exists? {fixed_points_fix(trace, 10, 0.01)}')

    fig = go.Figure()
    for i in range(1, 5):
        ### due to how the reachtube_tree is constructed, the final fig will just be zoomed in on x2 and x3, need to manually refit window
        fig = reachtube_tree(trace, None, fig, 0, i, [0, i], "fill", "trace", plot_color=colors[i:]) 
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()