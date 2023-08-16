from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent 
from verse.analysis.utils import wrap_to_pi 
from verse.analysis.analysis_tree import TraceType 
from verse.parser import ControllerIR

class VehicleAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None, 
        accel = 5,
        speed = 10
    ):
        super().__init__(
            id, code, file_name
        )
        self.accel = accel
        self.speed = speed
         
    @staticmethod
    def dynamic(t, state, u):
        x, y, theta, v, _ = state
        delta, a = u
        x_dot = v * np.cos(theta + delta)
        y_dot = v * np.sin(theta + delta)
        theta_dot = v / 1.75 * np.sin(delta)
        v_dot = a
        return [x_dot, y_dot, theta_dot, v_dot, 0]
    
    def action_handler(self, mode: List[str], state) -> Tuple[float, float]:
        x, y, theta, v, _ = state
        vehicle_mode,  = mode
        vehicle_pos = np.array([x, y])
        a = 0
        lane_width = 3
        d = -y
        if vehicle_mode == "Normal" or vehicle_mode == "Stop":
            pass
        elif vehicle_mode == "SwitchLeft":
            d += lane_width
        elif vehicle_mode == "SwitchRight":
            d -= lane_width
        elif vehicle_mode == "Brake":
            a = max(-self.accel, -v)
        elif vehicle_mode == "Accel":
            a = min(self.accel, self.speed - v)
        else:
            raise ValueError(f"Invalid mode: {vehicle_mode}")

        heading = 0
        psi = wrap_to_pi(heading - theta)
        steering = psi + np.arctan2(0.45 * d, v)
        steering = np.clip(steering, -0.61, 0.61)
        return steering, a

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            steering, a = self.action_handler(mode, init)
            r = ode(self.dynamic)
            r.set_initial_value(init).set_f_params([steering, a])
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            if init[3] < 0:
                init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace
