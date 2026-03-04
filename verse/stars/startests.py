import numpy as np
from starset import StarSet
from starset import *
from verse.utils.utils import sample_rect
from typing_extensions import List
from scipy.integrate import ode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy as cp
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

def plot_stars(stars: List[StarSet], dim1: int = None, dim2: int = None):
    for star in stars:
        x, y = np.array(star.get_verts(dim1, dim2))
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    plt.show()

def sim_ugly(vec, t):
    x = vec[0]
    y = vec[1]
    
    out_vec = []
    out_vec.append(x*y)
    if y == 0:
        out_vec.append(0)
    else:
        out_vec.append(x/y)
    return out_vec
def sim_simple(vec, t):
    x = vec[0]
    y = vec[1]
    
    out_vec = []
    out_vec.append(x+0.7)
    out_vec.append(y+0.7)
    return out_vec

### testing dynamic fnctions like this

def dynamics_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    ### vanderpol
    x_dot = y
    y_dot = (1 - x**2) * y - x

    ### cardiac cell
    # x_dot = -0.9*x*x-x*x*x-0.9*x-y+1
    # y_dot = x-2*y

    ### jet engine
    # x_dot = -y-1.5*x*x-0.5*x*x*x-0.5
    # y_dot = 3*x-y

    ### brusselator 
    # x_dot = 1+x**2*y-2.5*x
    # y_dot = 1.5*x-x**2*y

    ### bucking col -- change center to around -0.5 and keep basis size low
    # x_dot = y
    # y_dot = 2*x-x*x*x-0.2*y+0.1
    return [x_dot, y_dot]

### TO-DO: add another dynamic function(s) to test out dynamics with more than 2 dimension

def dyn_3d(vec, t):
    x, y, z = t

    ### purely for testing, doesn't correspond to any model
    x_dot = y
    y_dot = (1 - x**2) * y - x
    z_dot = x

    return [x_dot, y_dot, z_dot]

def sim_test(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dynamics_test)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

def sim_test_3d(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dyn_3d)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

###

basis = np.array([[1, 0], [0, 1]])
center = np.array([3,3])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])


def sim(vec, t):
    A = np.array([[0.1,0],[0,0.1]])
    i = 0
    while i <= t:
        i += 1
        vec = np.matmul(A, vec)
    return vec

star = StarSet(center, basis, C, g)
print(star.sample_h())
point = np.array([2, 4.00001])
print(containment_poly(star, point))