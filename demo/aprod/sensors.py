from typing import List, Union, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

class GenericSensor:
    def __init__(self):
        pass
    
    @staticmethod
    def sense(state: List, env: List[List] = None ,t: float = None) -> Optional[List]:
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        raise NotImplementedError()
    
class GroundSensor(GenericSensor):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def sense(state, env = None, t = None):
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        true_lat = [x, y, z, vx, vy, vz] 
        hlat = [h*np.random.uniform(0.95, 1.05) for h in true_lat]
        # hlat = true_lat
        return [x,y,z,vx,vy,vz] + hlat + [q0, q1, q2, q3, np.nan, np.nan, np.nan, np.nan, om_x, om_y, om_z]

class AOSensor(GenericSensor):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def sense(state, env = None, t = None):
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        hq = add_quat_noise([q1, q2, q3, q0])
        hq = [hq[-1]] + hq[:-1]
        return [x, y, z, vx, vy, vz] + [np.nan for _ in range(6)] + [q0, q1, q2, q3] + hq + [om_x, om_y, om_z]

class ProximitySensor(GenericSensor):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def sense(state, env = None, t = None):
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        true_pos = [x, y, z]
        hpos = [h*np.random.uniform(0.99, 1.01) for h in true_pos]
        hq = add_quat_noise([q1, q2, q3, q0])
        hq = [hq[-1]] + hq[:-1]
        return [x,y,z,vx,vy,vz] + hpos + [np.nan, np.nan, np.nan] + [q0, q1, q2, q3] + hq + [om_x, om_y, om_z]

def add_quat_noise(q: List, std: float = 0.05) -> List:
    dtheta = np.random.normal(0, std, 3) # R3 rotation vector
    theta = np.linalg.norm(dtheta)

    if theta<1e-8:
        return q # no rotation if the magnitude of the rotation is too small

    axis = dtheta/theta
    dq = np.concatenate((axis * np.sin(theta/2), [np.cos(theta/2)]))
    q_new = R.from_quat(q)*R.from_quat(dq)
    return q_new.as_quat().tolist()

def combine_sensors(state: List) -> List:
    sensed_states = []
    state = list(state)
    sensors: List[GroundSensor] = [AOSensor, ProximitySensor, GroundSensor]
    for sensor in sensors:
        sensed_states.append(sensor.sense(state))
    sensed_states = np.array(sensed_states)
    fused_orb = np.nanmean(sensed_states[:,6:12], axis=0)
    fused_orb = np.where(np.isnan(fused_orb), state[:6], fused_orb).tolist()

    sensed_att = sensed_states[:-1, 16:20]
    fused_att = state[16:20]
    if not np.isnan(sensed_att[0][0]):
        if not np.isnan(sensed_att[1][0]):
            hq = list(sensed_att[0][:-1]) + [sensed_att[0][-1]]
            hq_prox = list(sensed_att[1][:-1]) + [sensed_att[1][-1]]
            slerp = Slerp([0,1], R.from_quat([hq, hq_prox]))
            fused_att = slerp(0.5).as_quat() # interpolated value
            fused_att = [fused_att[-1]] + fused_att[:-1].tolist()
        else:
            fused_att = sensed_att[0]
    elif not np.isnan(sensed_att[1][0]):
        fused_att = sensed_att[1]
    fused_att = list(fused_att)

    return state[:6] + fused_orb + state[12:16] + fused_att + state[20:]

def apply_sensor(state: np.ndarray, sensor: GenericSensor) -> List:
    sensed_state = sensor.sense(state)
    return np.where(np.isnan(sensed_state), state, sensed_state)

def generic_combine_sensors(state: List, sensors: List[GenericSensor]) -> List:
    '''
    Generic version of a combined sensor function
    '''
    pass