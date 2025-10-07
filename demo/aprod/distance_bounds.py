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