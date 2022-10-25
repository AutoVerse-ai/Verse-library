# Example agent.
from typing import Tuple, List

import numpy as np 
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.parser import ControllerIR

class Agent1(BaseAgent):
    def __init__(self, id, code = None, file_name = None, initial_state = None, initial_mode = None):
        super().__init__(id, code, file_name, initial_state=initial_state, initial_mode=initial_mode)
        self.switch_duration = 0

    @staticmethod
    def dynamic(t, state, u):
        x, y, theta, v = state
        delta, a = u  
        x_dot = v*np.cos(theta+delta)
        y_dot = v*np.sin(theta+delta)
        theta_dot = v/1.75*np.sin(delta)
        v_dot = a 
        return [x_dot, y_dot, theta_dot, v_dot]

    def action_handler(self, mode: List[str], state, lane_map:LaneMap)->Tuple[float, float]:
        x,y,theta,v = state
        vehicle_mode = mode[0]
        vehicle_lane = mode[1]
        vehicle_pos = np.array([x,y])
        a = 0
        if vehicle_mode == "Normal":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
            self.switch_duration = 0
        elif vehicle_mode == "Brake":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
            if v>0:
                a = -1    
                if v<0.01:
                    a=0
            else:
                a = 1
                if v>-0.01:
                    a=0
            self.switch_duration = 0
        else:
            raise ValueError(f'Invalid mode: {vehicle_mode}')

        psi = lane_map.get_lane_heading(vehicle_lane, vehicle_pos)-theta
        steering = psi + np.arctan2(0.45*d, v)
        steering = np.clip(steering, -0.61, 0.61)
        return steering, a  

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map:LaneMap=None)->np.ndarray:
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step,10) for i in range(0,number_points)]

        init = initialCondition
        trace = [[0]+init]
        for i in range(len(t)):
            steering, a = self.action_handler(mode, init, lane_map)
            r = ode(self.dynamic)    
            r.set_initial_value(init).set_f_params([steering, a])      
            res:np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            if init[3] < 0:
                init[3] = 0
            trace.append([t[i] + time_step] + init) 

        return np.array(trace)
