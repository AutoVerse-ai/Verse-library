# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.utils.utils import wrap_to_pi
from verse.analysis.analysis_tree import TraceType
from verse.parser import ControllerIR



class CarAgent(BaseAgent):
    def __init__(self, id, initial_state=None, initial_mode=None):
        self.id = id
        self.decision_logic = ControllerIR.empty()
        self.set_initial_state(initial_state)
        self.set_initial_mode(initial_mode)
        self.set_static_parameter(None)
        self.set_uncertain_parameter(None)
    
    @staticmethod
    def dynamic(t, state, u, w):
        # x : position of the vehicle over x-axis
        # y : position of the vehicle over y-axis
        # steering_angle 
        # heading_angle
        # velocity
        # state = [x, y, steering_angle, heading_angle, velocity, time_dot]
        # l_wb = 2.578
        print("state is", state)
        # x = state[0]
        # y = state[1]
        steering_angle = state[2]
        heading_angle = state[3]
        velocity = state[4]
        
        l_wb = 2.578

        u1, u2 = u
        w1, w2 = w

        steering_angle_dot = u1 + w1
        velocity_dot = u2 + w2

        heading_angle_dot = (velocity / l_wb) * np.tan(steering_angle)

        x_dot = velocity * np.cos(heading_angle)
        y_dot = velocity * np.sin(heading_angle)
        time_dot = 1
        dots = [x_dot, y_dot, steering_angle_dot, heading_angle_dot, velocity_dot]

        if len(state) == 5:
            return dots
        return dots + [1]
    
    def action_handler(self, mode: List[str], state, lane_map: LaneMap):
        steering = 0 
        a = 0
        return steering, a

    def TC_simulate(
        self, mode: Tuple[str], init, time_bound, time_step, lane_map: LaneMap = None
    ) -> TraceType:
        print("init", init)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        for i in range(num_points):
            steering, a = self.action_handler(mode, init, lane_map)
            r = ode(self.dynamic)
            u = [0.7, 11]
            w = [-0.02, 0.3]
            r.set_initial_value(init).set_f_params(u, w)
            res: np.ndarray = r.integrate(r.t + time_step)
            print("res:", res)
            init = res.flatten()
            # if init[3] < 0:
            #     init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
            trace[i + 1, 5] = time_step * (i + 1)
        return trace



    