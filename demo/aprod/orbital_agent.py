# Example agent.
from typing import Tuple, List

import numpy as np
import cvxpy as cp
from scipy.integrate import ode, solve_ivp
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.linalg import expm, solve_continuous_are
import pickle

from verse.agents import BaseAgent
import random 
import plotly.graph_objects as go

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

S = solve_continuous_are(A, B, Q, R)
g = np.linalg.inv(R) @ B.T @ S # 3x3 x 3x6 x 6x6 -> 3x6 (g@(x_ref-x)+u_ref -- 3x6 x 6x6 + 3x3 good)

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

        objective = cp.Minimize(cp.sum([cp.norm1(u_k) for u_k in u])) # this doesn't need to exist
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
        if k>=len(u_sol):
            u_sol[-1]
        return u_sol[k]

    def simulate_tracking(x0: np.ndarray, x_ref_fn, u_ref_fn, T: float, dt: float, time_step: float, x_sol: np.ndarray, u_sol: np.ndarray):
        def ode(t, x):
            true_x, hat_x = x[:6], x[6:]
            x_ref = x_ref_fn(t, dt, x_sol, u_sol)
            u_ref = u_ref_fn(t, dt, u_sol)
            u = u_ref + g @ (x_ref - hat_x)
            dot_x = A @ true_x + B @ u
            dot_hat_x = A @ hat_x + B @ u
            return np.concatenate([dot_x, dot_hat_x])

        t_eval = np.arange(0, T+time_step, time_step)
        sol = solve_ivp(ode, [0, T], x0, t_eval=t_eval, method='RK45')
        return sol.t, sol.y.T  # Return times and x(t)

    # def TC_simulate(self, mode, initial_condition, time_horizon, time_step, map=None):
    #     time_horizon = float(time_horizon)
    #     number_points = int(np.ceil(time_horizon / time_step))
    #     t = [round(i * time_step, 10) for i in range(0, number_points)]
    #     init = initial_condition
    #     trace = [[0]+list(init)]
    #     for i in range(len(t)):
    #         r = ode(self.dynamics)
    #         r.set_initial_value(init)
    #         res: np.ndarray = r.integrate(r.t + time_step) # pretty sure r.t is always 0 but confirm later
    #         init = res.flatten().tolist()
    #         trace.append([t[i] + time_step] + init)
    #     return np.array(trace)
    
    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
        x0 = initialSet
        T = time_horizon
        N = 10
        dt = T/N
        x_sol, u_sol = OrbitalAgent.compute_ref(dt, x0[6:], N)

        with open('demo/aprod/ref_traj.pkl','wb') as f:
            pickle.dump(x_sol, f)

        u_sol = np.vstack([u_sol, u_sol[-1]]) # holding last input 
        x_ref_fn, u_ref_fn = OrbitalAgent.x_ref_fn, OrbitalAgent.u_ref_fn
        ts, trace = OrbitalAgent.simulate_tracking(x0, x_ref_fn, u_ref_fn, T, dt, time_step, x_sol, u_sol)
        timed_trace = np.concatenate((ts.reshape(-1, 1), trace), axis=1)
        return timed_trace

if __name__ == "__main__":
    T = 1500
    time_step = 1
    N = 10 # this is a hyperparameter to determine the number of intervals to separate the continuous trajectory into
    dt = T/N
    # x0 = np.zeros(12)
    base = [10,20,0,1,2,0]
    # x0 = np.array([0 for _ in range(6)] + [np.random.rand()-0.5 for _ in range(6)])
    # x0 = np.array(base+base)
    x0 = np.array(base + [np.random.rand()-0.5+base[i] for i in range(6)])
    x_sol, u_sol = OrbitalAgent.compute_ref(dt, x0[6:], N) # compute based on estimated state
    u_sol = np.vstack([u_sol, u_sol[-1]]) # holding last input 
    x_ref_fn, u_ref_fn = OrbitalAgent.x_ref_fn, OrbitalAgent.u_ref_fn
    ts, trace = OrbitalAgent.simulate_tracking(x0, x_ref_fn, u_ref_fn, T, dt, time_step, x_sol, u_sol)
    # print(trace)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trace[:, 0],
            y=trace[:,1],
            mode="lines",
            line_color="#0000CC",
            showlegend=False,
    ))
    fig.add_trace(
        go.Scatter(
            x=trace[:, 6],
            y=trace[:,7],
            mode="lines",
            line_color="#CC0000",
            showlegend=False,
    ))
    fig.add_trace(
        go.Scatter(
            x=x_sol[:, 0],
            y=x_sol[:,1],
            mode="lines",
            line_color="#000000",
            showlegend=False,
    ))
    fig.show()