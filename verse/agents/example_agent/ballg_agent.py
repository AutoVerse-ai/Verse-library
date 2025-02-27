# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap


class Ball_g_agent(BaseAgent):
    """Dynamics of point object falling under gravity"""
    g_acc: float

    def __init__(self, id, code=None, file_name=None, g_acc_val=9.8):
        """Contructor
            id: A string which will be treated as the name of the agent
            EXACTLY one of the following should be given
            file_name: name of the file containing the decision logic (DL)
            code: pyhton string defining the decision logic (DL)
            Then
            g_acc_val: value of acceleration due to gravity
        """
        super().__init__(id, code, file_name)
        self.g_acc = g_acc_val

    @staticmethod
    def dynamics(self, t, state):
        """Defines the RHS of the ODE used to simulate trajectories"""
        x, y, vx, vy = state
        x_dot = vx
        y_dot = vy
        vx_dot = -self.g_acc  # Does not work.
        vy_dot = 0
        return [x_dot, y_dot, vx_dot, vy_dot]


if __name__ == "__main__":
    aball = Ball_g_agent(
        "red_ball", file_name="/Users/mitras/Dpp/Verse-library/demo/ball/ball_bounces.py")
    trace = aball.TC_simulate({"none"}, [5, 10, 2, 2], 10, 0.05)
    print(trace)
