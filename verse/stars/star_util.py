import numpy as np
from starset import StarSet, HalfSpace
from verse.analysis.utils import sample_rect
from typing_extensions import List
from scipy.integrate import ode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy as cp
from z3 import *

def containment_poly(star: StarSet, point: np.ndarray) -> bool:
    if star.dimension() != point.shape[0]:
        raise ValueError(f'Dimension of point does not match the dimenions of the starset')
    
    center, basis, C, g = star.center, star.basis, star.C, star.g
    # print(basis, basis.shape, C, C.shape)
    alpha = RealVector('a', C.shape[1]) # holds list of alphas, should we be taking length from C or basis?
    s = Solver()

    ### add equality constraints
    for i in range(star.dimension()):
        exp = center[i]
        for j in range(len(alpha)):
            exp += alpha[j]*basis[j][i] # from the jth alpha/v, grab the ith dimension
        # print(exp)
        s.add(exp == point[i])

    ### add alpha constraints
    for i in range(C.shape[0]): # iterate over each row
        exp = 0 # there's probably a better way to do this, but this works
        for j in range(len(alpha)): # iterate over alphas
            exp += C[i][j]*alpha[j]
        # print(exp)
        s.add(exp <= g[i])

    return s.check()==sat

### N is the number of points, tol is how many misses consecutively we can see before raising an error  
def sample_star(star: StarSet, N: int, tol: float = 0.1) -> List[List[float]]:
    rect = star.overapprox_rectangle()
    points = []
    misses = 0
    while len(points)<N:
        point = np.array(sample_rect(rect))
        if containment_poly(star, point):
            points.append(point)
            misses = 0
        else:
            misses+=1
            if misses>int(N*tol):
                raise Exception("Too many consecutive misses, halting function. Call smple_rect instead.")
    return points