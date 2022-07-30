# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse.agents import BaseAgent
from verse.map import LaneMap


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


class craft_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def ProxA_dynamics(t, state):
        xp, yp, xd, yd, total_time, cycle_time = state
        xp_dot = xd
        yp_dot = yd
        xd_dot = -2.89995083970656*xd - 0.0576765518445905*xp + 0.00877200894463775*yd + 0.000200959896519766 * \
            yp - (1.43496e+18*xp + 6.050365344e+25)*pow(pow(yp, 2) +
                                                        pow(xp + 42164000, 2), -1.5) + 807.153595726846
        yd_dot = -0.00875351105536225*xd - 0.000174031357370456*xp - 2.90300269286856*yd - \
            1.43496e+18*yp*pow(pow(yp, 2) + pow(xp + 42164000,
                               2), -1.5) - 0.0664932019993982*yp
        total_time_dot = 1
        cycle_time_dot = 1
        return [xp_dot, yp_dot, xd_dot, yd_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def ProxB_dynamics(t, state):
        xp, yp, xd, yd, total_time, cycle_time = state
        xp_dot = xd
        yp_dot = yd
        xd_dot = -19.2299795908647*xd - 0.576076729033652*xp + 0.00876275931760007*yd + 0.000262486079431672 * \
            yp - (1.43496e+18*xp + 6.050365344e+25)*pow(pow(yp, 2) +
                                                        pow(xp + 42164000, 2), -1.5) + 807.153595726846
        yd_dot = -0.00876276068239993*xd - 0.000262486080737868*xp - 19.2299765959399*yd - \
            1.43496e+18*yp*pow(pow(yp, 2) + pow(xp + 42164000,
                               2), -1.5) - 0.575980743701182*yp
        total_time_dot = 1
        cycle_time_dot = 1
        return [xp_dot, yp_dot, xd_dot, yd_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Passive_dynamics(t, state):
        xp, yp, xd, yd, total_time, cycle_time = state
        xp_dot = xd
        yp_dot = yd
        xd_dot = 0.0000575894721132000*xp+0.00876276*yd
        yd_dot = -0.00876276*xd
        total_time_dot = 1
        cycle_time_dot = 1
        return [xp_dot, yp_dot, xd_dot, yd_dot, total_time_dot, cycle_time_dot]

    def action_handler(self, mode):
        if mode == 'ProxA':
            return ode(self.ProxA_dynamics)
        elif mode == 'ProxB':
            return ode(self.ProxB_dynamics)
        elif mode == 'Passive':
            return ode(self.Passive_dynamics)
        else:
            raise ValueError

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]

        init = initialCondition
        trace = [[0]+init]
        for i in range(len(t)):
            r = self.action_handler(mode[0])
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)


if __name__ == '__main__':
    aball = vanderpol_agent('agent1')
    trace = aball.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05)
    print(trace)
