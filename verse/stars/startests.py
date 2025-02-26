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
    # x_dot = y
    # y_dot = (1 - x**2) * y - x

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
    x_dot = y
    y_dot = 2*x-x*x*x-0.2*y+0.1
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

#def pred(alpha_vec):
    #print("in predicate")
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
    # intermediate = C @ alpha_vec
    #print(alpha_vec)
    #print(np.multiply(C, alpha_vec))
    #print(intermediate)
    #if (np.less_equal(intermediate,g)).all() == True:
    #    return True
    #return False 

def sim(vec, t):
    A = np.array([[0.1,0],[0,0.1]])
    i = 0
    while i <= t:
        i += 1
        vec = np.matmul(A, vec)
    return vec

#basis_rot = np.array([[0.707,0.707],[-0.707,0.707]])


test = StarSet(center,basis, C, g)
basis = np.array([[1.0, 0.0], [0.0, 1.0]])
center = np.array([3.0,3.0])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
test1 = StarSet(center,basis, C, g)
test_transformed = test1.post_cont(sim_simple, 1)
test_transformed2 = test1.post_cont(sim_ugly, 1)

### my tests


# p1 = np.array([2, 1.9])
# p2 = np.array([4, 4.1])
# ### from testing, know it works for rectangles, need to check more complex shapes
# print(containment_poly(test, p1))
# print(containment_poly(test, p2))
# points = np.array(sample_star(test, 100))

# plt.scatter(points[:, 0], points[:, 1])
# plot_stars([test])

# basis = np.array([[3, 1/3], [3, -1/4]]) * np.diag([0.01, 0.01])
# # basis = np.array([[3, 1/3], [3, -1/4]]) * np.diag([0.1, 0.1])
# # basis = np.array([[3, 1/3], [3, -1/4]]) 
# # center = np.array([1.35,2.25]) ### vanderpol, everything else unless listed otherwise
# center = np.array([-0.5, -0.5])

# C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
# # C = np.transpose(np.array([[1,-1,0,0, 1],[0,0,1,-1, 1]]))
# # g = np.array([1,1,1,1, 1.5])
# g = np.array([1,1,1,1])
# test_nrect = StarSet(center, basis, C, g)

# # # stars = gen_starsets_post_sim(test_nrect, sim_test)
# # # stars = sim_star(test_nrect, sim_test, T=0.25)
# # sim_star_vis(test_nrect, sim_test, T=1)
# gen_starsets_post_sim_vis_nonit(test_nrect, sim_test, 7) ### may need to modify get verts 

basis = np.array([[3, 1/3, -1], [3, -1/4, 0], [3, 0, 1]]) * np.diag([0.1, 0.1, 0.1])
center = np.array([1, 1, 1])
C = np.transpose(np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0], [0,0,0,0,1,-1]]))
g = np.ones(6)
test_3d = StarSet(center, basis, C, g)
# plot_stars([test_3d])
new_stars = gen_starsets_post_sim_vis_nonit_nd(test_3d, sim_test_3d)

# points = np.array(sample_star(test_nrect, 100))

# post_points = []
# for point in points:
#     # print(point)
#     post_points.append(sim_test(mode=None, initialCondition=point, time_bound=1, time_step=0.05).tolist()[-1][1:])
#     # post_points.append(sim_test(mode=None, initialCondition=point, time_bound=7, time_step=0.05).tolist())
# # new_center = np.array(sim_test(mode=None, initialCondition=center, time_bound=7, time_step=0.1).tolist()[-1][1:])
# post_points = np.array(post_points)
# # print(post_points.shape)

# new_center = np.mean(post_points, axis=0)
# scaler = StandardScaler()

# pca = PCA(n_components=2) ### in the future, this process should be done in relation to dimension
# pca.fit(post_points)
# scale_factor = np.sqrt(pca.explained_variance_)
# derived_basis: np.ndarray = (pca.components_.T @ np.diag(scale_factor)).T
# db_unscaled = np.array([pca.components_[0], pca.components_[1]]) # this especially should be a loop
# print(db_unscaled, '\n ---- \n', derived_basis, '\n ---- \n', pca.components_ @ np.diag(scale_factor), '\n ---- \n', np.diag(scale_factor))

# # post_test = post_cont_pca(test_nrect, derived_basis, post_points)
# post_test = gen_starset(post_points, test_nrect)
# unbloated_test = StarSet(new_center, derived_basis, C, g)

# # print(test_nrect.basis, test_nrect.center, test_nrect.C, test_nrect.g)
# # print(post_test.basis, post_test.center, post_test.C, post_test.g)

# # plt.quiver(*new_center, db_unscaled[0][0], db_unscaled[0][1], color='r', scale=3, label='PC1')
# # plt.quiver(*new_center, db_unscaled[1][0], db_unscaled[1][1], color='g', scale=3, label='PC2')
# plt.quiver(*new_center, derived_basis[0][0], derived_basis[0][1], color='r', scale=30, label='PC1')
# plt.quiver(*new_center, derived_basis[1][0], derived_basis[1][1], color='g', scale=30, label='PC2')
# plt.scatter(post_points[:, 0], post_points[:, 1])
# plt.scatter(points[:, 0], points[:, 1])
# plot_stars([post_test, unbloated_test, test_nrect])


exit()


# rect_nrect = StarSet.rect_to_star(*test_nrect.overapprox_rectangle()) ### note to self, never use overapprox_rectangles(), at least not for this purpose
# print(rect_nrect.basis, rect_nrect.center, rect_nrect.C, rect_nrect.g)

# plot_stars([test_nrect, StarSet.rect_to_star(*test_nrect.overapprox_rectangle())])

print(test1.overapprox_rectangles())

print(test_transformed.overapprox_rectangles())

print(test_transformed2.overapprox_rectangles())

plot_stars([test1, test_transformed])
plot_stars([test1, test_transformed2])
test1.show()
print('__________')
test_transformed.show()
print('__________')
test_transformed2.show()
print('__________')

basis = np.array([[1, 0], [0, 1]])
center = np.array([0,0])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([4,-2,4,-2])
test2 = StarSet(center,basis, C, g)
#plot_star(test)
test_transformed = test2.post_cont(sim_simple, 1)
test_transformed2 = test2.post_cont(sim_ugly, 1)

print(test2.overapprox_rectangles())

print(test_transformed.overapprox_rectangles())

print(test_transformed2.overapprox_rectangles())


plot_stars([test2, test_transformed])
plot_stars([test_transformed2])
test1.show()
print('__________')
test_transformed.show()
print('__________')
test_transformed2.show()
#plot_star(test)

exit()
basis = np.array([[3, 1/3], [3, -1/4]])
center = np.array([9,1])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
test1post = StarSet(center,basis, C, g)
plot_stars([test1, test1post])


basis = np.array([[1, 0], [0, 1]])
center = np.array([3.7,3.7])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
test1post = StarSet(center,basis, C, g)
plot_stars([test1, test1post])


#test = StarSet(center,basis, C, g)
#basis = np.array([[1, 0], [0, 1]])
#center = np.array([3,3])
#C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
#g = np.array([1,1,1,1])
#test1lin = StarSet(center,basis, C, g)
#plot_stars([test1, test1lin])

basis = np.array([[0, 0], [0, 0]])
center = np.array([0,0])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([4,-2,4,-2])
test2post = StarSet(center,basis, C, g)
plot_stars([test2, test2post])

basis = np.array([[1, 0], [0, 1]])
center = np.array([0.7,0.7])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([4,-2,4,-2])
test2post = StarSet(center,basis, C, g)
plot_stars([test2, test2post])


#print(test.get_max_min(1))

exit()

test.contains_point(np.array([3,2]))
print("done with contains test")
exit()
test.plot()


test.is_empty()
print("orig")
test.show()

print(test.satisfies(np.array([1,0]), -2))
print(test.satisfies(np.array([1,0]), 10))

print("test star set after")
test.show()

test.intersection_halfspace(np.array([5,5]), 3)
print("add single const")
test.show()
print("add multiple constr")
test.intersection_poly(np.array([[8,8],[9,9]]), np.array([4,5]))
test.show()

#print("results!")

#print(test.contains_point(np.array([1,0])))
#print(test.contains_point(np.array([3,3])))
#print(test.contains_point(np.array([2,2])))
#print(test.contains_point(np.array([4,2])))
#print(test.contains_point(np.array([4,1])))


new_test = test.post_cont(sim, 1)
new_test.show()


print("from poly test")
new_star = StarSet.from_poly(np.array([[8,8],[9,9]]), np.array([4,5]))
new_star.show()


#print("to poly test")
polystar = StarSet(center, basis,C, g)
#mat, rhs = polystar.to_poly()
#print(mat)
#print(rhs)
#print(np.matmul(mat, [3,3]))

#print(polystar.satisfies(np.array([[1,1]]),np.array([7])))

#print("verts test")
#print(StarSet.get_verts(polystar))

#new = test.superposition([2], [[4]])
#new.show()

#new = new.superposition([2], [[5,6]])

#test_half = HalfSpace(np.array([1,1]), 2)

#foo = np.array([3,3])
#bar = 
#test_star = StarSet(np.array([3,3]), np.array([[0,5],[0,5]]), C, g)

#result = test_star.intersection_halfspace(test_half)
#result.show()