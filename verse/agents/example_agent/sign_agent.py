from verse import BaseAgent
import numpy as np
from verse.parser import ControllerIR

class SignAgent(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.controller = ControllerIR.empty()

    def TC_simulate(self, mode, init, time_horizon, time_step, map=None):
        number_points = int(np.ceil(float(time_horizon)/time_step))
        t = [i*time_step for i in range(0,number_points)]
        trace = [[0] + init] + [[i + time_step] + init for i in t]
        return np.array(trace)
