# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from dryvr_plus_plus.scene_verifier.agents.base_agent import BaseAgent
from dryvr_plus_plus.scene_verifier.map.lane_map import LaneMap


class vanderpol_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state):
        x, y = state
        x = float(x)
        y = float(y)
        x_dot = y
        y_dot = (1-x**2)*y - x
        return [x_dot, y_dot]

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        # note: digit of time
        init = initialCondition
        trace = [[0]+init]
        for i in range(len(t)):
            r = ode(self.dynamic)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)


class thermo_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state, rate):
        temp, total_time, cycle_time = state
        temp = float(temp)
        total_time = float(total_time)
        cycle_time = float(cycle_time)
        temp_dot = temp*rate
        total_time_dot = 1
        cycle_time_dot = 1
        return [temp_dot, total_time_dot, cycle_time_dot]

    def action_handler(self, mode):
        if mode == 'ON':
            rate = 0.1
        elif mode == 'OFF':
            rate = -0.1
        else:
            print(mode)
            raise ValueError(f'Invalid mode: {mode}')
        return rate

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]

        init = initialCondition
        trace = [[0]+init]
        for i in range(len(t)):
            rate = self.action_handler(mode[0])
            r = ode(self.dynamic)
            r.set_initial_value(init).set_f_params(rate)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)


if __name__ == '__main__':
    aball = vanderpol_agent('agent1')
    trace = aball.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05)
    print(trace)
