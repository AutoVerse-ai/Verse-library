from src.scene_verifier.agents.base_agent import BaseAgent
import numpy as np 
from scipy.integrate import ode
from src.scene_verifier.map.lane_map import LaneMap

class CarAgent(BaseAgent):
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

    def TC_simulate(self, mode, initialCondition, time_bound, lane_map:LaneMap=None):
        mode = mode.split(',')
        vehicle_mode = mode[0]
        vehicle_lane = mode[1]
        time_step = 0.01
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [i*time_step for i in range(0,number_points)]

        init = initialCondition
        trace = [[0]+init]
        lane_parameter = lane_map.lane_geometry(vehicle_lane)
        if vehicle_mode == "Normal":
            for i in range(len(t)):
                x,y,theta,v = init
                d = -y+lane_parameter
                psi = -theta
                steering = psi + np.arctan2(0.45*d, v)
                steering = np.clip(steering, -0.61, 0.61)
                a = 0
                r = ode(self.dynamic)    
                r.set_initial_value(init).set_f_params([steering, a])      
                res:np.ndarray = r.integrate(r.t + time_step)
                init = res.flatten().tolist()
                trace.append([t[i] + time_step] + init) 
        elif vehicle_mode == "SwitchLeft":
            for i in range(len(t)):
                x,y,theta,v = init
                d = -y+3+lane_parameter
                psi = -theta
                steering = psi + np.arctan2(0.45*d, v)
                steering = np.clip(steering, -0.61, 0.61)
                a = 0
                r = ode(self.dynamic)    
                r.set_initial_value(init).set_f_params([steering, a])      
                res:np.ndarray = r.integrate(r.t + time_step)
                init = res.flatten().tolist()
                trace.append([t[i] + time_step] + init) 
        elif vehicle_mode == "SwitchRight":
            for i in range(len(t)):
                x,y,theta,v = init
                d = -y-3+lane_parameter
                psi = -theta
                steering = psi + np.arctan2(0.45*d, v)
                steering = np.clip(steering, -0.61, 0.61)
                a = 0
                r = ode(self.dynamic)    
                r.set_initial_value(init).set_f_params([steering, a])      
                res:np.ndarray = r.integrate(r.t + time_step)
                init = res.flatten().tolist()
                trace.append([t[i] + time_step] + init) 
        return np.array(trace)