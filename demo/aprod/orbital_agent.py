# Example agent.
from typing import Tuple, List

import numpy as np
import cvxpy as cp
from scipy.integrate import ode, solve_ivp
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.linalg import expm

from verse.agents import BaseAgent
import random 

n = 0.0012 # constant equal to (\mu/r_e^3)^{-3}
omega_e = np.array([[0, 0, n]]).T # set angular momentum of chief to be constant pointing in the inertial z-axis
J = np.diag([0.001, 0.001, 0.002]) # modeling a 1U cubesat
K = np.linalg.inv(J)
m = 10000 # mass of satellite 
q = 200              # bounding boxes: inside
r = 50               # outside
M = 1000  # for big-M disjunction 
u_max = 100 # to cap how large the input can be (note that due to the effect of mass, the actual input is limited to 0.01 m/s^2)
Q = np.diag([100, 100, 100, 1, 1, 1])  # cost function for state -- high on positional error 
R = 0.01 * np.eye(3) # try smaller penalty on control for tracking gain 

A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
        ])
B = np.vstack([np.zeros((3, 3)), np.eye(3)/m])

class OrbitalAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
    
    @staticmethod
    def dynamics(t, state):
        """
        Just for reference, should not be used to generate trajectory
        """
        x, y, z, vx, vy, vz, hx, hy, hz, hvx, hvy, hvz = state
        vx_dot = 3*(n**2)*x + 2*n*vy
        vy_dot = -2*n*vx
        vz_dot = -(n**2)*z
        hvx_dot = 3*(n**2)*hx + 2*n*hvy
        hvy_dot = -2*n*hvx
        hvz_dot = -(n**2)*hz
        # hvx_dot = 3*(n**2)*hx + 2*n*hvy
        # hvy_dot = -2*n*hvx
        # hvz_dot = -(n**2)*hz

        return [vx, vy, vz, vx_dot, vy_dot, vz_dot, # 0-5
                hvx, hvy, hvz, hvx_dot, hvy_dot, hvz_dot, #6-11
                ]

    @staticmethod
    def discretize_dynamics(dt: float) -> Tuple[np.ndarray, np.ndarray]:
        M_aug = np.zeros((9, 9))
        M_aug[:6, :6] = A
        M_aug[:6, 6:] = B
        exp_M = expm(M_aug * dt)
        return exp_M[:6, :6], exp_M[:6, 6:] # A, B
    
    @staticmethod
    def compute_ref(dt: float, x0: np.ndarray, N: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # need dt * N = T
        x = [cp.Variable(6) for _ in range(N + 1)]
        u = [cp.Variable(3) for _ in range(N)]

        # Binary variables for square exclusion
        z = cp.Variable(4, boolean=True)

        constraints = []

        # Initial condition
        constraints.append(x[0] == x0)

        # Dynamics
        A_d, B_d = OrbitalAgent.discretize_dynamics(dt)
        for k in range(N):
            constraints.append(x[k+1] == A_d @ x[k] + B_d @ u[k])
            constraints += [cp.abs(u[k]) <= u_max]  # Elementwise control constraint

        # Terminal NMT constraint
        constraints.append(x[N][4] + 2 * n * x[N][0] == 0)     # v_y + 2\eta r_x = 0
        constraints.append(x[N][3] - (n / 2) * x[N][1] == 0)   # v_x - \eta /2 r_y = 0

        # Terminal position in square Q
        constraints += [
            x[N][0] <= q,
            x[N][0] >= -q,
            x[N][1] <= q,
            x[N][1] >= -q,
        ]

        # Exclude inner square R using big-M disjunction
        # z[0] => r_x <= -r
        # z[1] => r_x >=  r
        # z[2] => r_y <= -r
        # z[3] => r_y >=  r
        constraints += [
            x[N][0] <= -r + M * (1 - z[0]),
            x[N][0] >=  r - M * (1 - z[1]),
            x[N][1] <= -r + M * (1 - z[2]),
            x[N][1] >=  r - M * (1 - z[3]),
            cp.sum(z) >= 1
        ]

        objective = cp.Minimize(cp.sum([cp.norm1(u_k) for u_k in u]))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.HIGHS)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            # print("Found a feasible trajectory.")
            x_sol = np.array([xk.value for xk in x])
            u_sol = np.array([uk.value for uk in u])
            return x_sol, u_sol
        else:
            # print("No feasible solution found.")
            raise Exception('No reference trajectory found')

    @staticmethod
    def compute_x_ref_t(t: float, t_k: float, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray: # k is some index between [0, N-1]
        """
        Computes the exact x_ref(t) using the fact that for a given index k, we start at x_k and u_k is constant
        """
        delta = t - t_k
        n = A.shape[0]
        m = B.shape[1]

        # Augmented matrix exponential
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        M_exp = expm(M * delta) # van loan's method

        Phi = M_exp[:n, :n]
        Gamma = M_exp[:n, n:]

        return Phi @ x_k + Gamma @ u_k

    @staticmethod
    def x_ref_fn(t: float, dt: float, x_sol: np.ndarray, u_sol: np.ndarray):
        # Assumes t is within [0, T]
        k = int(t // dt) # since N*dt = T, each t \in [0, T] corresponds to a specific starting index x_k and input u_k where all times [0, dt) should correspond to x_0, u_0, ..., etc. up to [T-dt, T)
        return OrbitalAgent.compute_x_ref_t(t, k * dt, x_sol[k], u_sol[k])

    @staticmethod
    def u_ref_fn(t: float, dt: float, u_sol: np.ndarray):
        k = int(t // dt) # assume t within [0, T]
        return u_sol[k]

    def simulate_tracking(K: np.ndarray, x0: np.ndarray, x_ref_fn: function, u_ref_fn: function, T: float, dt: float, x_ref: np.ndarray, u_ref: np.ndarray):
        def ode(t, x):
            x_ref = x_ref_fn(t, dt, x_sol, u_sol)
            u_ref = u_ref_fn(t, dt, u_sol)
            u = u_ref + K @ (x_ref - x)
            dxdt = A @ x + B @ u
            return dxdt

        t_eval = np.arange(0, T + dt, dt)
        sol = solve_ivp(ode, [0, T], x0, t_eval=t_eval, method='RK45')
        return sol.t, sol.y.T  # Return times and x(t)

    def TC_simulate(self, mode, initial_condition, time_horizon, time_step, map=None):
        time_horizon = float(time_horizon)
        number_points = int(np.ceil(time_horizon / time_step))
        t = [round(i * time_step, 10) for i in range(0, number_points)]
        init = initial_condition
        trace = [[0]+list(init)]
        for i in range(len(t)):
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step) # pretty sure r.t is always 0 but confirm later
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)
    
if __name__ == "__main__":
    x_sol, u_sol = OrbitalAgent.compute_ref(20, np.zeros(6), 10)
    print(x_sol, u_sol)