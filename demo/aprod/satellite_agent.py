# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode 
from scipy.spatial.transform import Rotation as R, Slerp

from verse.agents import BaseAgent
import random 

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
omega_e = np.array([[0, 0, n]]).T # set angular momentum of chief to be constant pointing in the inertial z-axis
J = np.diag([0.001, 0.001, 0.002]) # modeling a 1U cubesat
K = np.linalg.inv(J)

class SatelliteAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
    
    @staticmethod
    def dynamics(t, state):
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        vx_dot = 3*(n**2)*x + 2*n*vy
        vy_dot = -2*n*vx
        vz_dot = -(n**2)*z
        hvx_dot = 3*(n**2)*hx + 2*n*hvy
        hvy_dot = -2*n*hvx
        hvz_dot = -(n**2)*hz


        rho = np.array([[q1, q2, q3]]).T 
        s_rho = np.array([[0, -q3, q2], 
                          [q3, 0, -q1], 
                          [-q2, q1, 0]])
        
        hrho = np.array([[q1, q2, q3]]).T
        s_hrho = np.array([[0, -hq3, hq2], 
                          [hq3, 0, -hq1], 
                          [-hq2, hq1, 0]])
        
        om = np.array([[om_x, om_y, om_z]]).T
        rot_do: np.ndarray = R.from_quat([q1, q2, q3, q0]).as_matrix() # q is actually the rotation from body to hill frame 
        rot = rot_do.T # so take transpose to get from hill from to body rotation
        omega_ed = rot @ omega_e # angular momentum of chief in inertial frame rotated to body frame



        q0_dot = (rho.T @ om).item()/2
        rho_dot = -((q0*np.eye(3)-s_rho) @ om)/2

        hq0_dot = (hrho.T @ om).item()/2
        hrho_dot = -((hq0*np.eye(3)-s_hrho) @ om)/2

        om_dot = np.cross(om.flatten(), omega_ed.flatten())-K@np.cross((om+omega_ed).flatten(), (J@(om+omega_ed)).flatten())

        return [vx, vy, vz, vx_dot, vy_dot, vz_dot, # 0-5
                hvx, hvy, hvz, hvx_dot, hvy_dot, hvz_dot, #6-11
                q0_dot, rho_dot[0][0], rho_dot[1][0], rho_dot[2][0], #12-15
                hq0_dot, hrho_dot[0][0], hrho_dot[1][0], hrho_dot[2][0], #16-19
                om_dot[0], om_dot[1], om_dot[2] #20-22
                ]
    
    # assumes q is in scalar-last format as scipy expects
    @staticmethod
    def add_quat_noise(q: List, std: float = 0.05) -> List:
        dtheta = np.random.normal(0, std, 3) # R3 rotation vector
        theta = np.linalg.norm(dtheta)

        if theta<1e-8:
            return q # no rotation if the magnitude of the rotation is too small

        axis = dtheta/theta
        dq = np.concatenate((axis * np.sin(theta/2), [np.cos(theta/2)]))
        q_new = R.from_quat(q)*R.from_quat(dq)
        return q_new.as_quat().tolist()

    @staticmethod
    def apply_sensors(state: List, t: float = 0) -> List: # this should really take mode as an input
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz, q0, q1, q2, q3, hq0, hq1, hq2, hq3, om_x, om_y, om_z = state
        # ground based sensor
        hlat = [hx, hy, hz, hvx, hvy, hvz]
        hlat = [h*random.uniform(0.95, 1.05) for h in hlat]

        # angles-only sensor -- assume no obstacles 
        hq = SatelliteAgent.add_quat_noise([q1, q2, q3, q0]) # scalar last
        # hq = [hq[-1]] + hq[:-1] # re-ordering to get scalar first

        # proximity sensor -- realistically should only be able some distance away from the chief/other objects, but assume always active for now
        hpos = [hx, hy, hz]
        hpos = [h*random.uniform(0.99, 1.01) for h in hpos]
        
        hq_prox = SatelliteAgent.add_quat_noise([q1, q2, q3, q0]) # scalar last
        # hq_prox = [hq_prox[-1]] + hq_prox[:-1]

        # naively interpolating between proximity and AO sensor measurements
        slerp = Slerp([0,1], R.from_quat([hq, hq_prox]))
        hq_int = slerp(0.5).as_quat() # interpolated value
        hq_int = [hq_int[-1]] + hq_prox[:-1]

        # naively averaging positional estimates
        hpos_int = ((np.array(hlat[:3]) + np.array(hpos))/2).tolist()

        return [x, y, z, vx, vy, vz] + hpos_int + hlat[3:] + [q0, q1, q2, q3] + hq_int + [om_x, om_y, om_z]

    def TC_simulate(self, mode, initial_condition, time_horizon, time_step, map=None):
        time_horizon = float(time_horizon)
        number_points = int(np.ceil(time_horizon / time_step))
        t = [round(i * time_step, 10) for i in range(0, number_points)]
        init = initial_condition
        T_obs = 1 # observation interval
        last_obs = 0 # time of last observation
        init =  SatelliteAgent.apply_sensors(init, 0) # TODO: implement this
        trace = [[0]+list(init)]
        for i in range(len(t)):
            if i*time_step - last_obs > T_obs:
                init = SatelliteAgent.apply_sensors(init, i*time_step) 
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step) # pretty sure r.t is always 0 but confirm later
            init = res.flatten().tolist()
            
            q = np.array(init[12:16]) # normalize both of these quaternions
            hq = np.array(init[16:20])
            q_norm = (q/np.linalg.norm(q)).tolist()
            hq_norm = (hq/np.linalg.norm(hq)).tolist()
            init[12:16] = q_norm
            init[16:20] = hq_norm

            trace.append([t[i] + time_step] + init)
        return np.array(trace)