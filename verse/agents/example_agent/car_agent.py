# Example agent.
from typing import Tuple, List

import numpy as np 
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.parser import ControllerIR

class NPCAgent(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.controller = ControllerIR.empty()

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
        ''' Computes steering and acceleration based on current lane, target lane and
            current state using a Stanley controller-like rule'''
        x,y,theta,v = state
        vehicle_mode = mode[0]
        vehicle_lane = mode[1]
        vehicle_pos = np.array([x,y])
        d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
        psi = lane_map.get_lane_heading(vehicle_lane, vehicle_pos)-theta
        steering = psi + np.arctan2(0.45*d, v)
        steering = np.clip(steering, -0.61, 0.61)
        a = 0
        return steering, a  

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map:LaneMap=None)->np.ndarray:
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

class CarAgent(BaseAgent):
    def __init__(self, id, code = None, file_name = None):
        super().__init__(id, code, file_name)
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
        elif vehicle_mode == "SwitchLeft":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos) + lane_map.get_lane_width(vehicle_lane) 
            self.switch_duration += 0.1
        elif vehicle_mode == "SwitchRight":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos) - lane_map.get_lane_width(vehicle_lane)
            self.switch_duration += 0.1
        elif vehicle_mode == "Brake":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
            a = -1    
            self.switch_duration = 0
        elif vehicle_mode == "Accel":
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
            a = 1
            self.switch_duration = 0
        elif vehicle_mode == 'Stop':
            d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
            a = 0
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

class WeirdCarAgent(CarAgent):
    def __init__(self, id, code = None, file_name = None):
        super().__init__(id, code, file_name)
        self.gain = 0.2
        self.thres = 0.8

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        vehicle_mode = mode[0]
        steering, a = self._action_handler(mode, state, lane_map)
        # print("agnt", vehicle_mode, state, steering, a)
        return steering, a
        
    def _action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        vehicle_mode = mode[0]
        theta = state[2]
        if abs(theta) > self.thres:
            return 0, 0
        if vehicle_mode == "SwitchLeft" and theta >= 0:
            return self.gain, 0
        elif vehicle_mode == "SwitchRight" and theta <= 0:
            return -self.gain, 0
        else:
            return 0, 0
