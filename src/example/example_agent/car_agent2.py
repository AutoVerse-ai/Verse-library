from typing import Tuple

import numpy as np 
from scipy.integrate import ode

from src.scene_verifier.agents.base_agent import BaseAgent
from src.scene_verifier.map.lane_map import LaneMap

class CarAgent2(BaseAgent):
    def __init__(self, id, code = None, file_name = None):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic(t, state, u):
        x, y, theta, v = state
        delta, a = u  
        x_dot = v*np.cos(theta+delta)
        y_dot = v*np.sin(theta+delta)
        theta_dot = v/1.75*np.sin(delta)
        v_dot = a 
        return [x_dot, y_dot, theta_dot, v_dot]

    def action_handler(self, mode, state, lane_map:LaneMap)->Tuple[float, float]:
        x,y,theta,v = state
        vehicle_mode = mode[0]
        vehicle_lane = mode[1]
        vehicle_pos = np.array([x,y])
        if vehicle_mode == "Normal":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
        elif vehicle_mode == "SwitchLeft":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos) + 3
        elif vehicle_mode == "SwitchRight":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos) - 3
        psi = lane_map.get_lane_heading(vehicle_lane, vehicle_pos)-theta
        steering = psi + np.arctan2(0.45*d, v)
        steering = np.clip(steering, -0.61, 0.61)
        a = 0
        return steering, a  

    def TC_simulate(self, mode, initialCondition, time_bound, lane_map:LaneMap=None)->np.ndarray:
        mode = mode.split(',')
        time_step = 0.01
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [i*time_step for i in range(0,number_points)]

        init = initialCondition
        trace = [[0]+init]
        for i in range(len(t)):
            steering, a = self.action_handler(mode, init, lane_map)
            r = ode(self.dynamic)    
            r.set_initial_value(init).set_f_params([steering, a])      
            res:np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init) 

        return np.array(trace)