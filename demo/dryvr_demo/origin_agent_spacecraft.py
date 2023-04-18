# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
from scipy.integrate import solve_ivp

from verse.agents import BaseAgent
from verse.map import LaneMap


class spacecraft_linear_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class

        super().__init__(id, code, file_name)

    @staticmethod
    def Approaching_dynamics(t, state):
        x, y, vx, vy, total_time, cycle_time = state
        #mu = 3.986e14 * (60 ** 2)
        #r = 43164e3
        #mc = 500
        #n = (mu / (r ** 3)) ** .5
        #rc = ((r + x)**2 + y**2)**.5


        x_dot = vx
        y_dot = vy
        #K1 = np.array([[-28.8287, 0.1005, -1449.9754, 0.0046],[-.087, -33.2562, 0.00462, -1451.5013]])
        #temp = K1@np.array([x,y,vx,vy]).T
        #ux = #temp[0]
        #uy = #temp[1]
        # vx_dot = 5.75894721132e-5*x+0.00876276*vy-0.002*ux
        # vy_dot = -0.00876276*vx-0.002*uy

        vx_dot= -2.89995083970656*vx - 0.0575997658817729*x + 0.00877200894463775*vy + 0.000200959896519766*y
        vy_dot= -0.00875351105536225*vx - 0.000174031357370456*x - 2.90300269286856*vy - 0.0665123984901026*y
        #ux_dot = 28.8286776769430*vx-0.100479948259883*vy+1449.97541985328*(5.75894721132e-5*x+0.00876276*vy-0.002*ux)-0.00462447231887482*(-0.00876276*vx-0.002*uy)
        #uy_dot = 0.0870156786852279*vx+33.2561992450513*vy-0.00462447231887482*(5.75894721132e-5*x+0.00876276*vy-0.002*ux)+1451.50134643428*(-0.00876276*vx-0.002*uy)
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Rendezvous_dynamics(t, state):
        x, y, vx, vy,total_time, cycle_time = state
        #mu = 3.986e14 * 60 ** 2
        #r = 43164e3
        #mc = 500
        #n = (mu / (r ** 3)) ** .5
        #rc = ((r + x )**2 + y ** 2) ** .5

        x_dot = vx
        y_dot = vy
        #K2 = np.array([[-288.0288, 0.1312, -9614.9898,0], [-0.1312, -288, 0, -9614.9883]])
        #temp = K2 @ np.array([x, y, vx, vy]).T
        # ux = temp[0]
        # uy = temp[1]
        vx_dot= -19.2299795908647*vx - 0.575999943070835*x + 0.00876275931760007*vy + 0.000262486079431672*y
        vy_dot= -0.00876276068239993*vx - 0.000262486080737868*x - 19.2299765959399*vy - 0.575999940191886*y
        # vx_dot = 5.75894721132e-5*x+0.00876276*vy-0.002*ux
        # vy_dot = -0.00876276*vx-0.002*uy
        # ux_dot = 288.028766271474*vx-0.131243039715836*vy+9614.98979543236*(5.75894721132e-5*x+0.00876276*vy-0.002*ux)+3.41199965400404e-7*(-0.00876276*vx-0.002*uy)
        # uy_dot = 0.131243040368934*vx+287.999970095943*vy+3.41199965400404e-7*(5.75894721132e-5*x+0.00876276*vy-0.002*ux)+9614.98829796995*(-0.00876276*vx-0.002*uy)
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Aborting_dynamics(t, state):
        x, y, vx, vy ,total_time, cycle_time = state
        mu = 3.986e14 * 60 ** 2
        r = 43164e3
        mc = 500
        n = (mu / (r ** 3)) ** .5
        rc = ((r + x)**2 + y ** 2) ** .5

        x_dot = vx
        y_dot = vy
        vx_dot = 0.0000575894721132000*x+0.00876276*vy
        vy = -0.00876276*vx

        vy_dot = y*n**2 - 2*n*vx - (mu/rc**3)*y
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