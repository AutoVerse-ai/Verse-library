import numpy as np
from starset import StarSet, HalfSpace
from verse.analysis.utils import sample_rect
from typing_extensions import List, Callable
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
def sample_star(star: StarSet, N: int, tol: float = 0.2) -> List[List[float]]:
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

# def post_cont_pca(old_star: StarSet, new_center: np.ndarray, derived_basis: np.ndarray,  points: np.ndarray) -> StarSet:
def post_cont_pca(old_star: StarSet, derived_basis: np.ndarray,  points: np.ndarray) -> StarSet:
    if points.size==0:
        raise ValueError(f'No points given as input')
    if old_star.dimension() != points.shape[1]:
        raise ValueError(f'Dimension of points does not match the dimenions of the starset')
    if  old_star.basis.shape != derived_basis.shape:
        raise ValueError(f'Dimension of given basis does not match basis of original starset')

    center, basis, C, g = old_star.center, old_star.basis, old_star.C, old_star.g
    alpha = [RealVector(f'a_{i}', C.shape[1]) for i in range(points.shape[0])]
    u = Real('u')
    c = RealVector('i', old_star.dimension())
    # u = RealVector('u', C.shape[0])

    o = Optimize()
    ### add equality constraints
    ### this makes no sense, need to be able to add constraints such that checking a set of points instead of just a single points makes sense instead of this mess
    for p in range(len(points)):
        point = points[p]
        for i in range(old_star.dimension()):
            # exp = new_center[i]
            exp = c[i]
            for j in range(len(alpha[p])):
                exp += alpha[p][j]*derived_basis[j][i] # from the jth alpha/v, grab the ith dimension
            # print(exp)
            o.add(exp == point[i])

        ### add alpha constraints
        for i in range(C.shape[0]): # iterate over each row
            exp = 0 # there's probably a better way to do this, but this works
            for j in range(len(alpha[p])): # iterate over alphas
                exp += C[i][j]*alpha[p][j]
            # print(exp)
            o.add(exp <= u*g[i])
            # o.add(exp <= u[i]*g[i])
    
    o.minimize(u)

    model = None
    new_basis = derived_basis
    if o.check() == sat:
        model = o.model()
        new_basis = derived_basis * float(model[u].as_fraction())
    else:
        raise RuntimeError(f'Optimizer was unable to find a valid mu') # this is also hit if the function is interrupted

    print(model[u].as_decimal(10))
    # return StarSet(new_center, derived_basis, C, g * float(model[u].as_fraction()))
    new_center = np.array([float(model[c[i]].as_fraction()) for i in range(len(c))])
    return StarSet(new_center, derived_basis, C, g * float(model[u].as_fraction()))
    # return old_star.superposition(new_center, new_basis)

### from a set of points at a given time, generate a starset -- could possible reformat or remake this function to be more general
### expects an input with shape N (num points) x n (state dimenion) NOT N x n+1 (state dimension + time)
### this may get an exception if mu can't be generated, either figure out what to do about that or modify post_cont_pca s.t. it doesn't throw an error
def gen_starset(points: np.ndarray, old_star: StarSet) -> StarSet:
    new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
    pca: PCA = PCA(n_components=points.shape[1])
    pca.fit(points)
    scale = np.sqrt(pca.explained_variance_)
    derived_basis = (pca.components_.T @ np.diag(scale)).T # scaling each component by sqrt of dimension
    
    return post_cont_pca(old_star, derived_basis, points)

### doing post_computations using simulation then constructing star sets around each set of points afterwards -- not iterative
def gen_starsets_post_sim(old_star: StarSet, sim: Callable, T: float = 7, ts: float = 0.05, N: int = 100, no_init: bool = False) -> List[StarSet]:
    points = np.array(sample_star(old_star, N))
    post_points = []
    if no_init: 
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist()[1:])
    else:
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist())
    post_points = np.array(post_points)
    stars: List[StarSet] = []
    for t in range(post_points.shape[1]): # pp has shape N x (T/dt) x (n + 1), so index using first 
        stars.append(gen_starset(post_points[:, t, 1:], old_star)) ### something is up with this 
    return stars

### doing sim and post_cont iteratively to construct new starsets and get new points from them every ts
### this takes a decent amount of time -- guessing coming from post_cont_pca and/or sample_star as sim should be pretty fast
### sample star may be taking a while
### also size of star set blows up quickly, check what's going on -- probably need a better/different plotter function now
def sim_star(init_star: StarSet, sim: Callable, T: int = 7, ts: float = 0.05, N: int = 100) -> List[StarSet]:
    t = 0
    stars: List[StarSet] = []
    old_star = init_star
    while t<T:
        new_star = gen_starsets_post_sim(old_star, sim, ts, ts, N, True)[0] # gen_starset should return a list including only one starset
        stars.append(new_star)
        t += ts
        old_star = copy.deepcopy(new_star)
    return stars



'''
Visualization functions
'''

def plot_stars_points(stars: List[StarSet], points: np.ndarray):
    for star in stars:
        x, y = np.array(star.get_verts())
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    plt.scatter(points[:, 0], points[:, 1])
    # plt.show()

def gen_starsets_post_sim_vis(old_star: StarSet, sim: Callable, T: float = 7, ts: float = 0.05, N: int = 100, no_init: bool = False) -> List[StarSet]:
    points = np.array(sample_star(old_star, N, tol=10)) ### sho
    post_points = []
    if no_init: 
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist()[1:])
    else:
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist())
    post_points = np.array(post_points)
    stars: List[StarSet] = []
    for t in range(post_points.shape[1]): # pp has shape N x (T/dt) x (n + 1), so index using first 
        stars.append(gen_starset(post_points[:, t, 1:], old_star))
    # print(post_points)
    plot_stars_points(stars, post_points[:, 0, 1:]) # this only makes sense if points is 2D, i.e., only simulated one ts
    return stars

def plot_stars_points_nonit(stars: List[StarSet], points: np.ndarray):
    for star in stars:
        x, y = np.array(star.get_verts())
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    for t in range(points.shape[1]): # pp has shape N x (T/dt) x (n + 1), so index using first 
        plt.scatter(points[:, t, 1], points[:, t, 2])
    # plt.show()

def gen_starsets_post_sim_vis_nonit(old_star: StarSet, sim: Callable, T: float = 7, ts: float = 0.05, N: int = 100, no_init: bool = False) -> None:
    points = np.array(sample_star(old_star, N, tol=10)) ### sho
    post_points = []
    if no_init: 
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist()[1:])
    else:
        for point in points:
            post_points.append(sim(mode=None, initialCondition=point, time_bound=T, time_step=ts).tolist())
    post_points = np.array(post_points)
    stars: List[StarSet] = []
    for t in range(post_points.shape[1]): # pp has shape N x (T/dt) x (n + 1), so index using first 
        stars.append(gen_starset(post_points[:, t, 1:], old_star))
    # print(post_points)
    plot_stars_points_nonit(stars, post_points) # this only makes sense if points is 2D, i.e., only simulated one ts
    plt.show()

def sim_star_vis(init_star: StarSet, sim: Callable, T: int = 7, ts: float = 0.05, N: int = 100) -> None:
    t = 0
    stars: List[StarSet] = []
    old_star = init_star
    while t<T:
        new_star = gen_starsets_post_sim_vis(old_star, sim, ts, ts, N, True)[0] # gen_starset should return a list including only one starset
        stars.append(new_star)
        t += ts
        old_star = copy.deepcopy(new_star)
    plt.show()
    # return stars