# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode 
from scipy.spatial.transform import Rotation as R, Slerp

from verse.agents import BaseAgent
import random 

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
omega_e = np.array([[0, 0, n]]).T # set angular momentum of chief to be constant pointing in the inertial z-axis
J = np.diag([0.001, 0.001, 0.002]) # modeling a 1U cubesat
K = np.linalg.inv(J)

class OrbitalAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
    
    @staticmethod
    def dynamics(t, state):
        x, y, z, vx, vy, vz = state
        vx_dot = 3*(n**2)*x + 2*n*vy
        vy_dot = -2*n*vx
        vz_dot = -(n**2)*z
        # hvx_dot = 3*(n**2)*hx + 2*n*hvy
        # hvy_dot = -2*n*hvx
        # hvz_dot = -(n**2)*hz

        return [vx, vy, vz, vx_dot, vy_dot, vz_dot, # 0-5
                # hvx, hvy, hvz, hvx_dot, hvy_dot, hvz_dot, #6-11
                ]

    def TC_simulate(self, mode, initial_condition, time_horizon, time_step, map=None):
        time_horizon = float(time_horizon)
        number_points = int(np.ceil(time_horizon / time_step))
        t = [round(i * time_step, 10) for i in range(0, number_points)]
        init = initial_condition
        trace = [[0]+list(init)]
        for i in range(len(t)):
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step) # pretty sure r.t is always 0 but confirm later
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)