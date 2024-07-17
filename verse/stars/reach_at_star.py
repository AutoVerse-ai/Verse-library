import numpy as np
from starset import StarSet, HalfSpace
from verse.analysis.utils import sample_rect
from typing_extensions import List, Callable, Dict
from scipy.integrate import ode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy as cp
from z3 import *

### assuming mode is not a parameter
def reach_star(trace: Dict[str, List], t_l: float = 0, t_u: float = None) -> Dict[str, List[StarSet]]: 
    reach = {}
    if t_u is None:
        for agent in trace:
            t_u = trace[agent][-1][0] # T
            break
    
    for agent in trace:
        for i in range(len(trace[agent])):
            cur = trace[agent][i]
            if cur[0]<t_l:
                continue
            if cur[0]>t_u:
                break
            if agent not in reach:
                reach[agent] = []
            reach[agent].append(cur[1]) # just store the star set
    
    return reach

if __name__ == "__main__":
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    example = {'agent': [[0, StarSet(np.array([0, 0]), np.array([[1, 0], [0, 1]]), C, np.array([1, 1, 1, 1]))]]}
    print(reach_star(example))