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
        x, y, vx, vy, t, cycle_time = state


        x_dot = vx
        y_dot = vy

        vx_dot= -2.89995083970656*vx - 0.0575997658817729*x + 0.00877200894463775*vy + 0.000200959896519766*y
        vy_dot= -0.00875351105536225*vx - 0.000174031357370456*x - 2.90300269286856*vy - 0.0665123984901026*y

        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Rendezvous_dynamics(t, state):
        x, y, vx, vy,t, cycle_time = state


        x_dot = vx
        y_dot = vy

        vx_dot= -19.2299795908647*vx - 0.575999943070835*x + 0.00876275931760007*vy + 0.000262486079431672*y
        vy_dot= -0.00876276068239993*vx - 0.000262486080737868*x - 19.2299765959399*vy - 0.575999940191886*y

        t_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, t_dot, cycle_time_dot]

    @staticmethod
    def Aborting_dynamics(t, state):
        x, y, vx, vy ,t, cycle_time = state
        x_dot = vx
        y_dot = vy
        vx_dot = 0.0000575894721132000*x+0.00876276*vy
        vy_dot = -0.00876276*vx

        t_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, t_dot, cycle_time_dot]

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
        '''
        # time_bound = float(time_bound)
        # number_points = int(np.ceil(time_bound/time_step))
        # t = [round(i*time_step, 10) for i in range(0, number_points)]

        # init = initialCondition
        # trace = [[0]+init]
        # for i in range(len(t)):
        #     r = self.action_handler(mode[0])
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        # return np.array(trace)
        '''
        init = initialCondition

        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        for i in range(num_points):
            r = self.action_handler(mode[0])
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            # if init[3] < 0:
            #     init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class spacecraft_linear_agent_nd(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class

        super().__init__(id, code, file_name)

    @staticmethod
    def Approaching_dynamics(t, state):
        x, y, vx, vy, total_time, cycle_time = state



        x_dot = vx
        y_dot = vy


        vx_dot= -2.89995083970656*vx - 0.0575997658817729*x + 0.00877200894463775*vy + 0.000200959896519766*y
        vy_dot= -0.00875351105536225*vx - 0.000174031357370456*x - 2.90300269286856*vy - 0.0665123984901026*y

        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Rendezvous_dynamics(t, state):
        x, y, vx, vy,total_time, cycle_time = state


        x_dot = vx
        y_dot = vy

        vx_dot= -19.2299795908647*vx - 0.575999943070835*x + 0.00876275931760007*vy + 0.000262486079431672*y
        vy_dot= -0.00876276068239993*vx - 0.000262486080737868*x - 19.2299765959399*vy - 0.575999940191886*y
        total_time_dot = 1
        cycle_time_dot = 1
        return [x_dot, y_dot, vx_dot, vy_dot, total_time_dot, cycle_time_dot]

    @staticmethod
    def Aborting_dynamics(t, state):
        x, y, vx, vy ,total_time, cycle_time = state


        x_dot = vx
        y_dot = vy
        vx_dot = 0.0000575894721132000*x+0.00876276*vy
        vy_dot = -0.00876276*vx

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
        '''
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
        '''
    
        init = initialCondition

        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        
        for i in range(num_points):
            r = self.action_handler(mode[0])
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            # if init[3] < 0:
            #     init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace