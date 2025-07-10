import numpy as np
from scipy.optimize import minimize
from typing import Tuple

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

if __name__ == '__main__':
    # bounds = np.array([[-1,1], [1, 2], [1,2]])
    bounds = np.array([[0,1], [1, 2], [1,2]])
    obstacle = np.zeros((3,2))
    print(dist_extrema(bounds, obstacle))