
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

class TMAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamics_sd(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.01*sx -0.01*10.3 + 0.3*2.8 - 0.3*vx - 0.5*ax
        omega_dot = -2*omega
        vy_dot = -2*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]

    @staticmethod
    def dynamics_st1(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.5*vx+1.4-0.5*ax
        omega_dot = 3 - 3*0.05*omega + 0.2-0.01*sy
        vy_dot = 2.5*3 - 0.15*3*omega + 0.5 - 0.025*sy - 0.05*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]
    
    @staticmethod
    def dynamics_et1(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.5*vx+1.4-0.5*ax
        omega_dot = -0.1*omega + 0.2 - 0.01*sy
        vy_dot = -0.1*2.5*omega + 0.5 - 0.025*sy - 0.05*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]
    
    @staticmethod
    def dynamics_et2(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.5*vx+1.4-0.5*ax
        omega_dot = -0.1*omega + 0.2 - 0.01*sy
        vy_dot = -0.1*2.5*omega + 0.5 - 0.025*sy - 0.05*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]

    @staticmethod
    def dynamics_st2(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.5*vx+1.4-0.5*ax
        omega_dot = -3 - 3*0.05*omega + 0.2-0.01*sy
        vy_dot = -2.5*3 - 0.15*3*omega + 0.5 - 0.025*sy - 0.05*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]
    
    @staticmethod
    def dynamics_su(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.01*sx + 0.01*10.3 + 0.3*2.8 - 0.3*vx - 0.5*ax
        omega_dot = -2*omega
        vy_dot = -2*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]
    
    @staticmethod
    def dynamics_cont(t, state):
        sx, vx, ax, vy, omega, sy = state
        vx_dot = 0.1*ax
        sx_dot = vx-2.5
        ax_dot = -0.5*vx+1.4-0.5*ax
        omega_dot = -2*omega
        vy_dot = -2*vy
        sy_dot = 0.1*vy
        return [sx_dot, vx_dot, ax_dot, vy_dot, omega_dot, sy_dot]

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            if mode[0]=="SlowDown":
                r = ode(self.dynamics_sd)
            elif mode[0]=="StartTurn1":
                r = ode(self.dynamics_st1)
            elif mode[0]=="EndTurn1":
                r = ode(self.dynamics_et1)
            elif mode[0]=="EndTurn2":
                r = ode(self.dynamics_et2)
            elif mode[0]=="StartTurn2":
                r = ode(self.dynamics_st2)
            elif mode[0]=="SpeedUp":
                r = ode(self.dynamics_su)
            elif mode[0]=="Continue":
                r = ode(self.dynamics_cont)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class TMMode(Enum):
    SlowDown = auto()
    StartTurn1 = auto()
    EndTurn1 = auto()           
    EndTurn2 = auto()            
    StartTurn2 = auto()            
    SpeedUp = auto()            
    Continue = auto()


class State:
    sx: float
    vx: float
    ax: float
    vy: float
    omega: float
    sy: float
    agent_mode: TMMode 

    def __init__(self, x, y, vx, vy, agent_mode: TMMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)

    if ego.agent_mode == TMMode.SlowDown and ego.sx+10>=0:
        output.agent_mode = TMMode.StartTurn1
    if ego.agent_mode == TMMode.StartTurn1 and ego.sy>=12:
        output.agent_mode = TMMode.EndTurn1
    if ego.agent_mode == TMMode.StartTurn2 and ego.sy<=3.5:
        output.agent_mode = TMMode.EndTurn2
    if ego.agent_mode == TMMode.SpeedUp and ego.sx>=10:
        output.agent_mode = TMMode.StartTurn2    
    if ego.agent_mode == TMMode.EndTurn1 and ego.vy<=0.05:
        output.agent_mode = TMMode.SpeedUp
    if ego.agent_mode == TMMode.EndTurn2 and ego.vy+0.05>=0:
        output.agent_mode = TMMode.Continue

    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "total_motion.py")
    TM = TMAgent('tm', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(TM) ### need to add breakpoint around here to check decision_logic of agents

    init_tm = [[-15, 3.25, 0, 0, 0, 0],[-14.95, 3.3, 0, 0, 0, 0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'tm', init_tm, (TMMode.SlowDown,)
    )

    # for t>2 seconds (I think around 3.7 seconds) verify starts taking a bit -- not that relevant considering x>0.5 is unsafe and that occurs pretty much constantly
    trace = scenario.verify(40, 0.1)

    # pp_fix(reach_at_fix(trace, 0, 10))

    ### fixed point will never be reached due to structure of scenario 
    print(f'Fixed points exists? {fixed_points_fix(trace, 40, 0.1)}')

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 5, [0, 5], "fill", "trace") 
    fig = reachtube_tree(trace, None, fig, 0, 6, [0, 6], "fill", "trace", plot_color=colors[1:]) 
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()