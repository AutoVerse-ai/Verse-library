from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent 
from verse.analysis.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType 
from verse.parser import ControllerIR

class PedestrainAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None 
    ):
        super().__init__(
            id, code, file_name
        )

    @staticmethod
    def dynamic(t, state):
        x, y, theta, v, _ = state
        x_dot = 0
        y_dot = v
        theta_dot = 0
        v_dot = 0
        return [x_dot, y_dot, theta_dot, v_dot, 0]    

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            r = ode(self.dynamic)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            if init[3] < 0:
                init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace
