import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.linalg import expm, solve_continuous_are
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 

n = 0.0012 
# u_max = 25 # this is the true maximum action allowed
m = 10000
A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
        ])
B = np.vstack([np.zeros((3, 3)), np.eye(3)/m])
Q = np.diag([100, 100, 100, 1, 1, 1])  # cost function for state -- high on positional error 
R = 0.01 * np.eye(3) # try smaller penalty on control for tracking gain 
S = solve_continuous_are(A, B, Q, R)
g: np.ndarray = np.linalg.inv(R) @ B.T @ S # 3x3 x 3x6 x 6x6 -> 3x6 (g@(x_ref-x)+u_ref -- 3x6 x 6x6 + 3x3 good)

def discretize_dynamics(dt: float) -> tuple[np.ndarray, np.ndarray]:
    M_aug = np.zeros((9, 9))
    M_aug[:6, :6] = A
    M_aug[:6, 6:] = B
    exp_M = expm(M_aug * dt)
    return exp_M[:6, :6], exp_M[:6, 6:] # A, B

def compute_ref_nmt(dt: float, T, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Use a given NMT as a reference trajectory instead of solving for one
    """
    N = int(np.ceil(T/dt))
    T = dt*N
    t_eval = np.linspace(0, N*dt, N+1)
    sol = solve_ivp((lambda t, x: A @ x), [0, T], x0, t_eval=t_eval, method='RK45')
    x_ref = sol.y.T 
    u_ref = np.zeros((N, 3))
    return x_ref, u_ref

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
    
def x_ref_fn(t: float, dt: float, x_sol: np.ndarray, u_sol: np.ndarray):
    # Assumes t is within [0, T]
    k = int(t // dt) # since N*dt = T, each t \in [0, T] corresponds to a specific starting index x_k and input u_k where all times [0, dt) should correspond to x_0, u_0, ..., etc. up to [T-dt, T)
    return compute_x_ref_t(t, k * dt, x_sol[k], u_sol[k])

def u_ref_fn(t: float, dt: float, u_sol: np.ndarray):
    k = int(t // dt) # assume t within [0, T]
    if k>=len(u_sol):
        u_sol[-1]
    return u_sol[k]

def get_ref_action(x_ref: np.ndarray, u_ref: np.ndarray, x: torch.Tensor, g: np.ndarray = g):
    x_ref = torch.tensor(x_ref, dtype=x.dtype, device = x.device)
    u_ref = torch.tensor(u_ref, dtype=x.dtype, device = x.device)
    g = torch.tensor(g, dtype=x.dtype, device = x.device)
    err = x_ref.unsqueeze(0).expand(x.size(0), -1) - x
    return u_ref.unsqueeze(0) + err @ g.T # (1, M) + (B,N)@(N,M) 

def get_ref_action_traj(states_traj, x_ref=np.zeros(6), u_ref=np.zeros(3)):
    """
    states_traj: (B, T, state_dim)
    returns: (B, T, action_dim)
    Basic idea is to convert from (B,T) to (B*T) to make sure dimensions match -- again, at some
    """
    B, T, state_dim = states_traj.shape

    # Flatten batch and time: (B*T, state_dim)
    states_flat = states_traj.reshape(B * T, state_dim)

    # Call the existing expert function that handles (batch, state_dim)
    actions_flat = get_ref_action(x_ref, u_ref, states_flat)

    # Reshape back to (B, T, action_dim)
    actions_traj = actions_flat.reshape(B, T, -1)
    return actions_traj

def simulate_tracking(x0: np.ndarray, T: float, dt: float, time_step: float, x_sol: np.ndarray, u_sol: np.ndarray, u_max: float, x_ref_fn = x_ref_fn, u_ref_fn = u_ref_fn):
    def ode(t, x):
        x_ref = x_ref_fn(t, dt, x_sol, u_sol)
        u_ref = u_ref_fn(t, dt, u_sol)
        u = u_ref + g @ (x_ref - x)

        u_mag = np.linalg.norm(u) # clamping input
        try:
            if u_mag > u_max:
                u = u * u_max/u_mag
        except:
            u_mag = np.inf

        dot_x = A @ x + B @ u
        return dot_x

    t_eval = np.arange(0, T+time_step, time_step)
    sol = solve_ivp(ode, [0, T], x0, t_eval=t_eval, method='Radau')
    return sol.t, sol.y.T  # Return times and x(t)
    
def dynamics(state, action, dt=1.0, u_max = 25):
    """
    state: (batch, n)
    action: (batch, m)
    dt: time step length
    u_max: the true maximum input allowed by dynamics, should be equal to umax, the maximum input allowed by NN controller
    """
    A_d, B_d, = discretize_dynamics(dt)
    A_d = torch.tensor(A_d, dtype=state.dtype, device=state.device)
    B_d = torch.tensor(B_d, dtype=state.dtype, device=state.device)

    # (batch, n) @ (n, n)^T → (batch, n)
    clamped_action = torch.clamp(action, -u_max, u_max)  # enforce control bounds
    next_state = state @ A_d.T + clamped_action @ B_d.T

    return next_state

class SatelliteCTRL(nn.Module):
    def __init__(self, state_dim, action_dim, umax=25):
        """
        """
        super().__init__()
        self.umax = umax
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # squash between -1 and 1
        )

    def forward(self, state):
        # return self.umax * self.net(state)
        return self.net(state)

def rollout(actor, x0, x_goal, T=3000, dt=10, actor_u_max=25, true_u_max=25, p_expert: float = 0): 
    state = x0
    states, actions = [], []
    N = int(np.ceil(T/dt))
    for _ in range(N):
        mask = torch.bernoulli(torch.full((state.shape[0], 1), p_expert, dtype=state.dtype))
        action = actor(state) * actor_u_max
        action_expert = get_ref_action(np.zeros(6), np.zeros(3), state) # in the future u_ref and x_ref should be given as a parameter
        final_actions = mask * action_expert + (1 - mask) * action
        next_state = dynamics(state, final_actions, dt, true_u_max)
        states.append(state)
        actions.append(action)
        state = next_state
    states = torch.stack(states, dim=1)   # (batch, N, state_dim)
    actions = torch.stack(actions, dim=1) # (batch, N, action_dim)
    return states, actions

def loss_fn(states, actions, x_goal, w_traj=1.0, w_goal=1.0, w_ctrl=1e-2, schedule="linear"):
    """
    Compute loss for LVLH docking with time-varying trajectory weights and state weighting matrix Q.

    Args:
        states: [batch, N, 6]  trajectory of states
        actions: [batch, N, m]  trajectory of actions
        x_goal: [6]  goal state (usually zeros)
        w_traj: float, weight for trajectory loss
        w_goal: float, weight for terminal loss
        w_ctrl: float, weight for control loss
        schedule: str, weighting schedule for trajectory loss ("linear", "quadratic", "exponential")

    Returns:
        scalar loss
    """

    device = states.device
    B, N, d = states.shape
    assert d == 6, "Expected 6D LVLH state"

    # State weighting (Q matrix diag)
    # Typical scales: pos ~100, vel ~1, out-of-plane ~10x smaller
    q = torch.tensor([1/100**2, 1/100**2, 1/10**2,
                      1/1**2,   1/1**2,   1/0.1**2],
                      device=device, dtype=states.dtype)
    
    # Time-varying trajectory weights
    if schedule == "linear":
        weights = torch.linspace(0.1, 1.0, N, device=device, dtype=states.dtype)
    elif schedule == "quadratic":
        weights = torch.linspace(0.1, 1.0, N, device=device, dtype=states.dtype) ** 2
    elif schedule == "exponential":
        weights = torch.logspace(-1, 0, N, base=10.0, device=device, dtype=states.dtype)
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    
    # Normalize so sum(weights)=1 -- not necessary but makes scaling w.r.t. goal easier
    weights = weights / weights.sum()

    traj_loss = torch.mean(torch.sum(weights[None, :, None] * q[None, None, :] * (states - x_goal[None, None, :])**2, dim=(1,2)))
    goal_loss = torch.mean(torch.sum(q[None, :] * (states[:, -1, :] - x_goal[None, :])**2, dim=1))
    ctrl_loss = torch.mean(torch.sum(actions**2, dim=(1,2)))

    loss = w_goal * goal_loss + w_traj * traj_loss + w_ctrl * ctrl_loss
    return loss

def dagger_loss(actor, states, actions_expert):
    """
    states: (batch, N, state_dim)
    actions_expert: (batch, N, action_dim)
    """
    # Compute actor outputs at the visited states
    batch, N, state_dim = states.shape
    _, action_dim = actions_expert.shape[0], actions_expert.shape[2]

    # Flatten batch and time to evaluate actor in one go
    states_flat = states.reshape(batch * N, state_dim)        # (B*N, state_dim)
    actions_pred_flat = actor(states_flat)                    # (B*N, action_dim)

    # Reshape back to (B, N, action_dim)
    actions_pred = actions_pred_flat.reshape(batch, N, action_dim)

    # L2 loss against expert actions
    loss = torch.mean((actions_pred - actions_expert)**2)
    return loss

def sample_initial(lb: np.ndarray, ub: np.ndarray, batch_size: int = 32):
    d = lb.shape[0]
    # Uniform [0,1] samples
    rand = np.random.rand(batch_size, d)
    # Scale + shift to [lb, ub]
    samples = lb + (ub - lb) * rand
    return torch.tensor(samples, dtype=torch.float32) # may need to check if this data type is actually correct

def epoch_schedule_prob(epoch, total_epochs, p_start=1.0, p_end=0.0, schedule="exp"): # for behavioral cloning
    frac = epoch / max(1, total_epochs - 1)
    if schedule == "linear":
        return p_start + (p_end - p_start) * frac
    elif schedule == "exp":
        # slower early decay
        return p_start * ((p_end/p_start) ** (frac**2))
    else:
        return p_start + (p_end - p_start) * frac
    
if __name__ == "__main__":
    T = 1500
    # overwrite = False
    overwrite = True
    ts = 1
    dt = 1 # some time step >= ts
    N = int(np.ceil(T/dt))
    # this sample and lb, ub are from docking point in docking scenario
    x0 = np.array([  0.0624 , -75.01119,   0.     ,  -0.04506,  -0.00003,   0.     ])
    x_goal = torch.tensor([0.,0.,0.,0.,0.,0.]).unsqueeze(0)  # goal state for docking
    num_epochs = 100
    # lb, ub = [base[i]-2.5 for i in range(6)], [base[i]+2.5 for i in range(6)] 
    lb, ub = [ -1.10722, -75.82266,  -0.27715,  -0.04657,  -0.00163,  -0.0005 ], [  1.23202, -74.19972,   0.27715,  -0.04356,   0.00157,   0.0005 ]
    state_dim = 6  # e.g. 3D pos + 3D vel
    action_dim = 3
    actor = SatelliteCTRL(state_dim, action_dim, umax=25)
    opt = optim.Adam(actor.parameters(), lr=3e-4)
    actor_umax_end   = 25 # actual u_max to be enforced
    # actor_umax_end   = 100 # actual u_max to be enforced
    actor_umax_start = actor_umax_end*2.5    # initial generous limit
    loss_dagger_weight = 1
    # x_sol, u_sol = np.zeros((3001,6)), np.zeros((3000,3))
    # u_sol = np.vstack([u_sol, u_sol[-1]]) # holding last input 
    # ts, trace = simulate_tracking(x0, T, dt, ts, x_sol, u_sol, actor_umax_end) 
    # plt.plot(trace[:,0], trace[:,1])
    # plt.show()
    # exit()

    if overwrite or not os.path.exists("./demo/aprod/model_weights.pth"): # in the future filename should be associated with T and initial set -- figure out some hashing scheme
        for epoch in tqdm(range(num_epochs)):
            frac = epoch / num_epochs
            actor_umax = actor_umax_start * (1 - frac) + actor_umax_end * frac

            x0 = sample_initial(np.array(lb), np.array(ub), batch_size=32)
            p_expert = epoch_schedule_prob(epoch, num_epochs, schedule='exp')
            states, actions = rollout(actor, x0, x_goal, T, dt, actor_u_max=actor_umax, true_u_max = actor_umax_end, p_expert=p_expert)
            loss_traj = loss_fn(states, actions, x_goal)
            loss_dagger = dagger_loss(actor, states, get_ref_action_traj(states, x_ref=np.zeros(6,), u_ref=np.zeros(3,)))
            loss = total_loss = loss_traj + p_expert * loss_dagger * loss_dagger_weight 

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss {loss.item():.4f}")

        torch.save(actor.state_dict(), "./demo/aprod/model_weights.pth")
    else:
        trained_model = torch.load("./demo/aprod/model_weights.pth")
        actor.load_state_dict(trained_model)

    actor.eval()
    x0 = sample_initial(np.array(lb), np.array(ub), batch_size=1)
    state, actions = rollout(actor, x0, x_goal, T, dt, actor_umax_end)
    states = state[0].detach().numpy()
    plt.plot(states[:,0], states[:,1])
    plt.show()