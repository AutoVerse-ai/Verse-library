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

class OrbitalAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
    
    @staticmethod
    def dynamics(t, state):
        x, y, z, vx, vy, vz = state
        vx_dot = 3*(n**2)*x + 2*n*vy
        vy_dot = -2*n*vx
        vz_dot = -(n**2)*z
        # hvx_dot = 3*(n**2)*hx + 2*n*hvy
        # hvy_dot = -2*n*hvx
        # hvz_dot = -(n**2)*hz

        return [vx, vy, vz, vx_dot, vy_dot, vz_dot, # 0-5
                # hvx, hvy, hvz, hvx_dot, hvy_dot, hvz_dot, #6-11
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
        true_lat = [x, y, z, vx, vy, vz]
        hlat = [h*random.uniform(0.95, 1.05) for h in true_lat]

        # angles-only sensor -- assume no obstacles 
        hq = SatelliteAgent.add_quat_noise([q1, q2, q3, q0]) # scalar last
        # hq = [hq[-1]] + hq[:-1] # re-ordering to get scalar first

        # proximity sensor -- realistically should only be able some distance away from the chief/other objects, but assume always active for now
        true_pos = [x, y, z]
        hpos = [h*random.uniform(0.99, 1.01) for h in true_pos]
        
        hq_prox = SatelliteAgent.add_quat_noise([q1, q2, q3, q0]) # scalar last
        # hq_prox = [hq_prox[-1]] + hq_prox[:-1]

        # naively interpolating between proximity and AO sensor measurements
        slerp = Slerp([0,1], R.from_quat([hq, hq_prox]))
        hq_int = slerp(0.5).as_quat() # interpolated value
        hq_int = [hq_int[-1]] + hq_prox[:-1]
        """
        Visualizing all or even most q values is not straightforward due to nonlinearity in quaternion computations -- consider just doing a bunch of simulations and taking the extrema in each dimension
        """

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
        # init =  SatelliteAgent.apply_sensors(init, 0) # TODO: implement this
        trace = [[0]+list(init)]
        for i in range(len(t)):
            # if t[i] + time_step - last_obs > T_obs:
            #     init = SatelliteAgent.apply_sensors(init, i*time_step) 
            #     last_obs = t[i] + time_step
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step) # pretty sure r.t is always 0 but confirm later
            init = res.flatten().tolist()

            trace.append([t[i] + time_step] + init)
        return np.array(trace)