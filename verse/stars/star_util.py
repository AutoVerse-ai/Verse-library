import numpy as np
from starset import StarSet
from starset import HalfSpace
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
        s.add(exp == point[i])

    ### add alpha constraints
    for i in range(C.shape[0]): # iterate over each row
        exp = 0 # there's probably a better way to do this, but this works
        for j in range(len(alpha)): # iterate over alphas
            exp += C[i][j]*alpha[j]
        s.add(exp <= g[i])


    return s.check()==sat