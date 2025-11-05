import numpy as np
from scipy.optimize import minimize
from typing import Tuple
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def dist_extrema(agent: np.ndarray, obstacle: np.ndarray) -> Tuple[float, float]:
    """
    Compute extrema with an agent and an obstacle that are both represented by bounding boxes
    Expecting bounds as 3x2 array and obstacle as 3x2 array
    There's a closed form way to do this under the assumption both boxes are disjoint, but this is simple
    """
    dist = lambda r: np.linalg.norm(r[:3] - r[3:])
    neg_dist = lambda r: -dist(r)
    combined_bounds = np.vstack((agent, obstacle))
    guess = np.mean(combined_bounds, axis=1).T # take average of min and max along each dimension
    tuple_bounds = tuple(map(tuple, combined_bounds)) # converting each row into a tuple and converting the map into a tuple to get a tuple of tuples
    res_min = minimize(dist, guess, bounds=tuple_bounds)
    res_max = minimize(neg_dist, guess, bounds=tuple_bounds)
    return res_min.fun, -res_max.fun

def psi_extrema(pos_min, pos_max, obs_pos_min, obs_pos_max, theta_min, theta_max):
    """
    Compute bounds on psi angle for given position and heading bounds
    
    Args:
        pos_min, pos_max: ego position bounds [x_min, y_min], [x_max, y_max]
        obs_pos_min, obs_pos_max: obstacle position bounds  
        theta_min, theta_max: ego heading bounds
    
    Returns:
        (psi_min, psi_max): bounds on psi angle
    """
    
    def psi_function(state):
        ego_x, ego_y, other_x, other_y, theta = state
        rel_x = other_x - ego_x
        rel_y = other_y - ego_y
        
        # Compute psi with normalization
        psi = ((np.arctan2(rel_y, rel_x) - theta + np.pi) % (2*np.pi)) - np.pi
        return psi
    
    # Create bounds for all variables
    bounds = [
        (pos_min[0], pos_max[0]),  # ego_x
        (pos_min[1], pos_max[1]),  # ego_y
        (obs_pos_min[0], obs_pos_max[0]),  # other_x
        (obs_pos_min[1], obs_pos_max[1]),  # other_y
        (theta_min, theta_max)  # theta
    ]
    
    # Use central point as initial guess
    guess = np.array([
        (pos_min[0] + pos_max[0]) / 2,
        (pos_min[1] + pos_max[1]) / 2,
        (obs_pos_min[0] + obs_pos_max[0]) / 2,
        (obs_pos_min[1] + obs_pos_max[1]) / 2,
        (theta_min + theta_max) / 2
    ])
    
    # Find minimum and maximum psi
    from scipy.optimize import minimize
    
    res_min = minimize(psi_function, guess, bounds=bounds)
    res_max = minimize(lambda x: -psi_function(x), guess, bounds=bounds)
    
    return res_min.fun, -res_max.fun

class SquaredNormDiff(torch.nn.Module):
    def forward(self, x, y):
        diff = x - y               # shape (batch, dim)
        return torch.sum(diff * diff, dim=1, keepdim=True)  # shape (batch, 1)

def dist_extrema_crown(agent: np.ndarray, obstacle: np.ndarray) -> Tuple[float, float]:
    ego_l, ego_u = agent.T
    other_l, other_u = obstacle.T
    ego_l, ego_u = torch.tensor(ego_l).float(), torch.tensor(ego_u).float()
    other_l, other_u = torch.tensor(other_l).float(), torch.tensor(other_u).float()
    shape_x = shape_y = torch.zeros(1,other_l.shape[0]).float()
    sensor_model = BoundedModule(SquaredNormDiff(), (shape_x, shape_y), device="cpu")
    ego_center, other_center = ((ego_l + ego_u) / 2).unsqueeze(0), ((other_l + other_u) / 2).unsqueeze(0)
    ego_delta = PerturbationLpNorm(x_L=ego_l.unsqueeze(0), x_U=ego_u.unsqueeze(0))
    other_delta = PerturbationLpNorm(x_L=other_l.unsqueeze(0), x_U=other_u.unsqueeze(0))
    ego_bounded, other_bounded = BoundedTensor(ego_center, ego_delta), BoundedTensor(other_center, other_delta)
    
    dist_sq_l, dist_sq_u =  sensor_model.compute_bounds(x=(ego_bounded, other_bounded), method="backward")
    return dist_sq_l.clamp_min(0).sqrt().item(), dist_sq_u.clamp_min(0).sqrt().item()

if __name__ == '__main__':
    # bounds = np.array([[-1,1], [1, 2], [1,2]])
    bounds = np.array([[0,1], [1, 2], [1,2]])
    obstacle = np.zeros((3,2))
    print(dist_extrema(bounds, obstacle))