# Example agent.
import math
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
from scipy.integrate import solve_ivp

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

class coupled_vanderpol_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state):
        x1, y1, x2, y2,b = state
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        b = float(b)
        mu =1
        x1_dot = y1
        y1_dot = mu*(1 - x1**2)*y1 + b*(x2 - x1) - x1
        x2_dot = y2
        y2_dot = mu*(1 - x2**2)*y2 - b*(x2 - x1) - x2
        bdot =0
        return [x1_dot, y1_dot, x2_dot, y2_dot,bdot]

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
class robertson_agent(BaseAgent):
    def __init__(self, id, beta, gamma, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)
        self.beta = beta 
        self.gamma = gamma 

    def dynamic(self, t, state):

        x, y,z = state
        x = float(x)
        y = float(y)
        z = float(z)

        x_dot = -.4*x +self.beta*y*z
        y_dot = .4*x - self.beta*y*z - self.gamma*y**2
        z_dot = self.gamma*y**2

        return [x_dot, y_dot,z_dot]

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        # # note: digit of time
        # init = initialCondition
        # trace = [[0]+init]
        # for i in range(len(t)):
        #     r = ode(self.dynamic)
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        # return np.array(trace)
        t_span = [0, time_bound]
        res = solve_ivp(self.dynamic, t_span = t_span, y0 = initialCondition, method='Radau', t_eval=t, rtol=4e-14, atol=1e-12)
        trace = np.vstack((res.t, res.y)).T
        return trace



class laub_loomis_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state):
        x1, x2, x3, x4,x5,x6, x7,W = state
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


        return [x1_dot, x2_dot,x3_dot, x4_dot, x5_dot, x6_dot, x7_dot,0]

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

class volterra_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state):
        x, y, t_loc = state
        x = float(x)
        y = float(y)

        x_dot = 3*x - 3*x*y
        y_dot = x*y - y


        return [x_dot, y_dot, 1]

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None) -> np.ndarray:
        # time_bound = float(time_bound)
        # number_points = int(np.ceil(time_bound/time_step))
        # t = [round(i*time_step, 10) for i in range(0, number_points)]
        # # note: digit of time
        # init = initialCondition
        # trace = [[0]+init]
        # for i in range(len(t)):
        #     r = ode(self.dynamic)
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        # return np.array(trace)
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        # # note: digit of time
        # init = initialCondition
        # trace = [[0]+init]
        # for i in range(len(t)):
        #     r = ode(self.dynamic)
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        # return np.array(trace)
        t_span = [0, time_bound]
        res = solve_ivp(self.dynamic, t_span = t_span, y0 = initialCondition, method='Radau', t_eval=t, atol=1e-4)
        if not res.success:
            print("stop here")
        trace = np.vstack((res.t, res.y)).T
        return trace


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
        trace = np.array([0, *init])
        for i in range(len(t)):
            rate = self.action_handler(mode[0])
            r = ode(self.dynamic)
            r.set_initial_value(init).set_f_params(rate)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace = np.vstack((trace, np.insert(init, 0, time_step * (i + 1))))
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
        K1 = np.array([[-28.8287, 0.1005, -1449.9754, 0.0046],[-.087, -33.2562, 0.00462, -1451.5013]])
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
        K2 = np.array([[-288.0288, 0.1312, -9614.9898,0], [-0.1312, -288, 0, -9614.9883]])
        temp = K2 @ np.array([x, y, vx, vy]).T
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

zeta = 0.9 # coefficient of restitution
ms = 3.2 # mass of sleeve(kg)
mg2 = 18.1 # mass of second gear(kg)
Jg2 = 0.7 # inertia of second gear(kg*m^2)
ig2 = 3.704 # gear ratio of second gear
Rs = 0.08 # radius of sleeve(m)
theta = math.pi*(36/180) # included angle of gear(rad)
b = 0.01 # width of gear spline(m)
deltap = -0.003 # axial position where sleeve engages with second gear(m)

# Model inputs
Fs = 70; # shifting force(N)

# Disturbances
Tf = 1; # resisting moment(N*m), its domain is [1,5]
class gearbox_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)
    @staticmethod
    def dynamic_free(t, state):
        px, py, vx, vy, i, trans, one = state
        vx_dot = Fs / ms
        vy_dot = -Rs * Tf / Jg2
        px_dot = vx
        py_dot = vy
        i_dot = 0
        return [px_dot, py_dot, vx_dot, vy_dot, i_dot, 0, 0]

    @staticmethod
    def dynamic_meshed(t, state):
        px, py, vx, vy, trans, one, i = state
        vx_dot = 0
        vy_dot = 0
        px_dot = vx
        py_dot = vy
        i_dot = 0
        return [px_dot, py_dot, vx_dot, vy_dot, i_dot, 0, 0]

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step,
                    track_map: LaneMap = None) -> np.ndarray:
        # time_bound = float(time_bound)
        # number_points = int(np.ceil(time_bound / time_step))
        # t = [round(i * time_step, 10) for i in range(0, number_points)]

        # init = initialCondition
        # trace = [[0] + init]
        # for i in range(len(t)):
        #     if mode[0] == 'Free':
        #         r = ode(self.dynamic_free)
        #     elif mode[0] == 'Meshed':
        #         r = ode(self.dynamic_meshed)
        #     else:
        #         raise ValueError
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        t_span = [0, time_bound]
        if  mode[0] == 'Free':
            res = solve_ivp(self.dynamic_free, t_span = t_span, y0 = initialCondition, method='RK45', t_eval=t)
        elif mode[0] == 'Meshed':
            res = solve_ivp(self.dynamic_meshed, t_span = t_span, y0 = initialCondition, method='RK45', t_eval=t)
        else:
            raise ValueError
        trace = np.vstack((res.t, res.y)).T
        return trace

class powertrain_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class

        super().__init__(id, code, file_name)

    @staticmethod
    def negAngle_dynamics(t, state):
        x1, x2, x3, x4, x5,x6, x7,x8,x9 ,total_time = state
        x1_dot= 1.0 / 12.0 * x7 - x9
        x2_dot= (0.5 * (12.0 * x4 - x7) + 0.5 * (12.0 * x3 - 12.0 * (x1 + x8)) + 0.5 * (12.0 * 5.0 - 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)) - x2) / 0.1
        x3_dot= x4
        x4_dot= 5.0
        x5_dot= x6
        x6_dot= 1.0 / 140.0 * (100000.0 * (x8 - x5) - 5.6 * x6)
        x7_dot= 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)
        x8_dot= x9
        x9_dot= 0.01 * (10000.0 * (x1 - 0.03) - 100000.0 * (x8 - x5) - 1.0 * x9)
        total_time_dot = 1
        return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot,x7_dot,x8_dot,x9_dot, total_time_dot]

    @staticmethod
    def deadzone_dynamics(t, state):
        x1, x2, x3, x4, x5,x6, x7,x8,x9 ,total_time = state
        x1_dot= 1.0 / 12.0 * x7 - x9
        x2_dot= (0.5 * (12.0 * x4 - x7) + 0.5 * (12.0 * x3 - 12.0 * (x1 + x8)) + 0.5 * (12.0 * 5.0 - 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 0.0 * (x1 - 0.03) - 0.0 * x7)) - x2) / 0.1
        x3_dot= x4
        x4_dot= 5.0
        x5_dot= x6
        x6_dot= 1.0 / 140.0 * (100000.0 * (x8 - x5) - 5.6 * x6)
        x7_dot= 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 0.0 * (x1 - 0.03) - 0.0 * x7)
        x8_dot= x9
        x9_dot= 0.01 * (0.0 * (x1 - 0.03) - 100000.0 * (x8 - x5) - 1.0 * x9)

        total_time_dot = 1
        return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot,x7_dot,x8_dot,x9_dot, total_time_dot]

    @staticmethod
    def posAngle_dynamics(t, state):
        x1, x2, x3, x4, x5,x6, x7,x8,x9, total_time = state
        x1_dot = 1.0 / 12.0 * x7 - x9
        x2_dot= (0.5 * (12.0 * x4 - x7) + 0.5 * (12.0 * x3 - 12.0 * (x1 + x8)) + 0.5 * (12.0 * 5.0 - 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)) - x2) / 0.1
        x3_dot= x4
        x4_dot= 5.0
        x5_dot= x6
        x6_dot= 1.0 / 140.0 * (100000.0 * (x8 - x5) - 5.6 * x6)
        x7_dot= 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)
        x8_dot= x9
        x9_dot= 0.01 * (10000.0 * (x1 - 0.03) - 100000.0 * (x8 - x5) - 1.0 * x9)
        total_time_dot = 1
        return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot,x7_dot,x8_dot,x9_dot, total_time_dot]

    @staticmethod
    def negAngleInit_dynamics(t, state):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, total_time = state
        x1_dot = 1.0 / 12.0 * x7 - x9
        x2_dot = (0.5 * (12.0 * x4 - x7) + 0.5 * (12.0 * x3 - 12.0 * (x1 + x8)) + 0.5 * (12.0 * -5.0 - 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)) - x2) / 0.1
        x3_dot = x4
        x4_dot = -5.0
        x5_dot = x6
        x6_dot = 1.0 / 140.0 * (100000.0 * (x8 - x5) - 5.6 * x6)
        x7_dot = 1.0 / 0.3 * (x2 - 1.0 / 12.0 * 10000.0 * (x1 - 0.03) - 0.0 * x7)
        x8_dot= x9
        x9_dot= 0.01 * (10000.0 * (x1 - 0.03) - 100000.0 * (x8 - x5) - 1.0 * x9)


        total_time_dot = 1
        return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot,x7_dot,x8_dot,x9_dot, total_time_dot]

    def action_handler(self, mode):
        if mode == 'negAngle':
            return ode(self.negAngle_dynamics)
        elif mode == 'deadzone':
            return ode(self.deadzone_dynamics)
        elif mode == 'posAngle':
            return ode(self.posAngle_dynamics)
        elif mode == 'negAngleInit':
            return ode(self.negAngleInit_dynamics)
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
