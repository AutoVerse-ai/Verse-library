import numpy as np
from typing import List, Tuple

def car_dynamics(t, state, u):
    x, y, theta, v = state
    delta, a = u  
    x_dot = v*np.cos(theta+delta)
    y_dot = v*np.sin(theta+delta)
    theta_dot = v/1.75*np.sin(delta)
    v_dot = a 
    return [x_dot, y_dot, theta_dot, v_dot]

def car_action_handler(self, mode: List[str], state, lane_map)->Tuple[float, float]:
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