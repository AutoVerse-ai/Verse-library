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

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
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
class laub_loomis_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state):
        x1, x2, x3, x4,x5,x6, x7 = state
        x1 = float(x1)
        x2 = float(x2)
        x3 = float(x3)
        x4 = float(x4)
        x5 = float(x5)
        x6 = float(x6)
        x7 = float(x7)
        x1_dot = 1.4*x3 - 0.9*x1

        x2_dot = 2.5*x5 - 1.5*x2
        x3_dot =  0.6*x7 - 0.8*x2*x3
        x4_dot =  2 - 1.3*x3*x4
        x5_dot = 0.7*x1 - x4*x5
        x6_dot =  0.3*x1 - 3.1*x6
        x7_dot = 1.8*x6 - 1.5*x2*x7
        #x7_dot = 1.8*x6 − 1.5*x2*x7


        return [x1_dot, x2_dot,x3_dot, x4_dot, x5_dot, x6_dot, x7_dot]

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
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

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
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

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
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

class spacecraft_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class

        super().__init__(id, code, file_name)

    @staticmethod
    def Approaching_dynamics(t, state):
        x, y, vx, vy, total_time, cycle_time = state
        mu = 3.986e14 * (60 ** 2)
        r = 43164e3
        mc = 500
        n = (mu / (r ** 3)) ** .5
        rc = ((r + x)**2 + y**2)**.5


        x_dot = vx
        y_dot = vy
        K1 = np.array([[-28.8287, .1005, -1449.9754, .0046],[-.087, -33.2562, .00462, -1451.5013]])
        temp = K1@np.array([x,y,vx,vy]).T
        ux = temp[0]
        uy = temp[1]
        vx_dot = x*n**2 + 2*n*vy +mu/(r**2) - (mu/(rc**3))*(r+x) +ux/mc

        vy_dot = y*n**2 - 2*n*vx - (mu/rc**3)*y +uy/mc
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Rendezvous_dynamics(t, state):
        x, y, vx, vy, total_time, cycle_time = state
        mu = 3.986e14 * 60 ** 2
        r = 43164e3
        mc = 500
        n = (mu / (r ** 3)) ** .5
        rc = ((r + x )**2 + y ** 2) ** .5

        x_dot = vx
        y_dot = vy
        K1 = np.array([[-288.0288, 0.1312, -9614.9898 ], [-0.1312, -288, 0, -9614.9883]])
        temp = K1 @ np.array([x, y, vx, vy]).T
        ux = temp[0]
        uy = temp[1]
        vx_dot = x*n**2 + 2*n*vy +mu/(r**2) - (mu/(rc**3))*(r+x) +ux/mc

        vy_dot = y*n**2 - 2*n*vx - (mu/rc**3)*y +uy/mc
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Aborting_dynamics(t, state):
        x, y, vx, vy, total_time, cycle_time = state
        mu = 3.986e14 * 60 ** 2
        r = 43164e3
        mc = 500
        n = (mu / (r ** 3)) ** .5
        rc = ((r + x)**2 + y ** 2) ** .5

        x_dot = vx
        y_dot = vy
        ux = 0
        uy = 0
        vx_dot = x*n**2 + 2*n*vy +mu/(r**2) - (mu/(rc**3))*(r+x) +ux/mc

        vy_dot = y*n**2 - 2*n*vx - (mu/rc**3)*y +uy/mc
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    def action_handler(self, mode):
        if mode == 'Approaching':
            return ode(self.Approaching_dynamics)
        elif mode == 'Rendezvous':
            return ode(self.Rendezvous_dynamics)
        elif mode == 'Aborting':
            return ode(self.Aborting_dynamics)
        else:
            raise ValueError

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
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
