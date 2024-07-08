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
import matplotlib.pyplot as plt 

def plot_stars(stars):
    for star in stars:
        x, y = np.array(star.get_verts())
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

def dynamic_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    x = y
    y = (1 - x**2) * y - x
    return [x, y]

def sim_test(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = initialCondition
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dynamic_test)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

###

test = StarSet(center,basis, C, g)
basis = np.array([[1.0, 0.0], [0.0, 1.0]])
center = np.array([3.0,3.0])
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
test1 = StarSet(center,basis, C, g)
test_transformed = test1.post_cont(sim_simple, 1)
test_transformed2 = test1.post_cont(sim_ugly, 1)

rect = test.overapprox_rectangles()
N = 100 # parameter to control num samples
rect_t = []
for i in range(N):
    point = sample_rect(rect)
    rect_t.append(sim_test(mode=None, initialCondition=point, time_bound=7, time_step=0.1).tolist()[-1][1:])
rect_t = np.array(rect_t)

pca = PCA(n_components=2)
pca.fit(rect_t)
# print(rect_t)
# print(pca.components_, pca.explained_variance_ratio_)
pc1 = pca.components_[0]
pc2 = pca.components_[1]
scale_factor = np.sqrt(pca.explained_variance_)*3  # Scale factor based on explained variance
angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])  # Angle of the first principal component

plt.scatter(rect_t[:, 0], rect_t[:, 1])
origin = np.mean(rect_t, axis=0)
plt.quiver(*origin, pc1[0], pc1[1], color='r', scale=3, label='PC1')
plt.quiver(*origin, pc2[0], pc2[1], color='g', scale=3, label='PC2')
ellipse = Ellipse(pca.mean_, width=2*scale_factor[0], height=2*scale_factor[1], angle=np.degrees(angle), fill=False, color='b', linestyle='--', linewidth=2, label='Ellipsoid')
ax = plt.gca()
ax.add_patch(ellipse)
# print(angle)

bloat = cp.Variable()
constraints = []
for i in range(rect_t.shape[0]):
    width = bloat*scale_factor[0]
    height = bloat*scale_factor[1] # in real implementation, would need to do this using a list somehow either through comprehension or looping
    constraints.append(((rect_t[i, 0] - origin[0]) / width)**2 + ((rect_t[i, 1] - origin[1]) / height)**2 <= 1) # these constraints don't work, just multiplying won't help as w**2*h**2 is also non-convex
    ### solution: overapprox here as rect, turn into polytope -- checking should be easier -- just check half-spaces

# objective = cp.Minimize(bloat) # again, for larger dimenions will have to use determinant
# prob = cp.Problem(objective, constraints)
# prob.solve()
# scaled_factors = scale_factor*bloat.value
# opt_ellipse = Ellipse(pca.mean_, width=2*scaled_factors[0], height=2*scaled_factors[1], angle=np.degrees(angle), fill=False, color='b', linestyle='--', linewidth=2, label='Ellipsoid Opt')
# ax.add_patch(opt_ellipse)

plt.show()

### have ellipsoid from PCA -- axes, lengths, now how to optimize such that volume is minimized while the only parameter is coefficient of all axes? 
### i.e., twiddle bloating factor in all directions at once instead of doing so for each axis

exit()

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