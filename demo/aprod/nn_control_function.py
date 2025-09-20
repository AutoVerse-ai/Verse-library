import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.linalg import expm
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
ry = 75
r_inner = ry - 20 
x0_nmt = np.array([0, ry, 0, n/2*ry, 0, 0])
x0_inner = np.array([0, r_inner, 0, n/2*r_inner, 0, 0])
x0_nmt_ahead = np.array( [ 8.91431, 72.85074,  0.     ,  0.04371, -0.02139,  0.     ]) # these are both the same amount ahead [ 8.91431, 72.85074,  0.     ,  0.04371, -0.02139,  0.     ]
x0_inner_ahead = np.array([ 6.53691, 53.42363,  0.     ,  0.03205, -0.01569,  0.     ]) # [ 6.53691, 53.42363,  0.     ,  0.03205, -0.01569,  0.     ]


def discretize_dynamics(dt: float) -> tuple[np.ndarray, np.ndarray]:
    M_aug = np.zeros((9, 9))
    M_aug[:6, :6] = A
    M_aug[:6, 6:] = B
    exp_M = expm(M_aug * dt)
    return exp_M[:6, :6], exp_M[:6, 6:] # A, B

def compute_ref_nmt(dt: float, T, x0: np.ndarray = x0_nmt) -> tuple[np.ndarray, np.ndarray]:
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

def rollout(actor, x0, x_goal, T=3000, dt=1.0, actor_u_max=25, true_u_max=25): 
    state = x0
    states, actions = [], []
    N = int(np.ceil(T/dt))
    for _ in range(N):
        action = actor(state) * actor_u_max
        next_state = dynamics(state, action, dt, true_u_max)
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

def sample_initial(lb: np.ndarray, ub: np.ndarray, batch_size: int = 32):
    d = lb.shape[0]
    # Uniform [0,1] samples
    rand = np.random.rand(batch_size, d)
    # Scale + shift to [lb, ub]
    samples = lb + (ub - lb) * rand
    return torch.tensor(samples, dtype=torch.float32) # may need to check if this data type is actually correct

if __name__ == "__main__":
    T = 3000
    # overwrite = False
    overwrite = True
    dt = 1
    N = int(np.ceil(T/dt))
    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x_goal = torch.tensor([0.,0.,0.,0.,0.,0.]).unsqueeze(0)  # goal state for docking
    num_epochs = 100
    ### TODO: change these bounds to the actual bounds I'd get from the docking scenario
    lb, ub = [base[i]-2.5 for i in range(6)], [base[i]+2.5 for i in range(6)] 
    state_dim = 6  # e.g. 3D pos + 3D vel
    action_dim = 3
    actor = SatelliteCTRL(state_dim, action_dim, umax=25)
    opt = optim.Adam(actor.parameters(), lr=3e-4)
    actor_umax_end   = 25 # actual u_max to be enforced
    actor_umax_start = actor_umax_end*2.5    # initial generous limit

    if overwrite or not os.path.exists("./demo/aprod/model_weights.pth"): # in the future filename should be associated with T and initial set -- figure out some hashing scheme
        for epoch in tqdm(range(num_epochs)):
            frac = epoch / num_epochs
            actor_umax = actor_umax_start * (1 - frac) + actor_umax_end * frac

            x0 = sample_initial(np.array(lb), np.array(ub), batch_size=32)
            states, actions = rollout(actor, x0, x_goal, T, dt, actor_u_max=actor_umax, true_u_max = actor_umax_end)
            loss = loss_fn(states, actions, x_goal)

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