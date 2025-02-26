# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse.agents import BaseAgent
from verse.map import LaneMap
from math import pi, tan, cos, sin

# Configuration Parameters
zeta = 0.9  # coefficient of restitution
ms = 3.2  # mass of sleeve(kg)
mg2 = 18.1  # mass of second gear(kg)
Jg2 = 0.7  # inertia of second gear(kg*m^2)
ig2 = 3.704  # gear ratio of second gear
Rs = 0.08  # radius of sleeve(m)
theta = pi * (36 / 180)  # included angle of gear(rad)
b = 0.01  # width of gear spline(m)
deltap = -0.003  # axial position where sleeve engages with second gear(m)

# Model inputs
Fs = 70
# shifting force(N)

# Disturbances
Tf = 1
# resisting moment(N*m), its domain is [1,5]


class sleeve_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamics_free(t, state):
        px, py, vx, vy, i = state
        vx_dot = Fs / ms
        vy_dot = -Rs * Tf / Jg2
        px_dot = vx
        py_dot = vy
        i_dot = 0
        return [px_dot, py_dot, vx_dot, vy_dot, i_dot]

    @staticmethod
    def dynamics_meshed(t, state):
        px, py, vx, vy, i = state
        vx_dot = 0
        vy_dot = 0
        px_dot = vx
        py_dot = vy
        i_dot = 0
        return [px_dot, py_dot, vx_dot, vy_dot, i_dot]

    def TC_simulate(
        self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None
    ) -> np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound / time_step))
        t = [round(i * time_step, 10) for i in range(0, number_points)]

        init = initialCondition
        trace = [[0] + init]
        for i in range(len(t)):
            if mode[0] == "Free":
                r = ode(self.dynamics_free)
            elif mode[0] == "Meshed":
                r = ode(self.dynamics_meshed)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)
