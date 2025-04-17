import numpy as np
import scipy.optimize as sop
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from verse.stars import StarSet

def star_manhattan_distance(center, basis, C, g):
    #order of x: [x1, x2, ..., y1, y2, ..., z1, z2, ..., b1, b2, ..., a11, a12, ..., a21, a22, ...]

    # x1 = c1 + a11*v11 + a12*v21 + ...
    # x2 = c2 + a12*v12 + a12*v22 + ...

    # y1 = c1 + a21*v11 + a22*v12 + ...
    # y2 = c2 + a21*v12 + a22*v22 + ...

    #define N; large number for now
    N = 1048575

    #definitions
    dimension = len(center) #number of dimensions for the set
    num_vec = len(basis) #numver of vectors in the basis
    num_cons = len(g)
    x_len = dimension*4 + num_vec*2 #length of x, the vector of decision variables


    #define c to minimize c @ x
    c = np.zeros(x_len)
    #set the z's to -1
    z_start = dimension*2
    for i in range(dimension):
        c[i + z_start] = -1 
        #dimension*2 because z's start after x and y

    #define integrality
    integrality = np.zeros(x_len)
    b_start = dimension*3
    for i in range(dimension):
        integrality[i + b_start] = 1 #integrality of b is 1; rest is 0

    #create the A matrix
    xy_cons = dimension*2
    abs_cons = dimension*4
    a_cons = len(C)*2
    A = np.zeros((xy_cons + abs_cons + a_cons, x_len),dtype=float)

    #define the constraints for the points
    #constraints for xy
    #in the form: x - a*v - ... = c
    alpha_start = dimension*4
    for i in range(xy_cons):
        for j in range(x_len):
            if i == j:
                A[i,j] = 1
            #for each dimension, the corresponding basis element is set
            alpha_start_index = int(j - alpha_start - (np.floor(i/dimension))*num_vec)
            if(alpha_start_index >= 0 and alpha_start_index < num_vec):
                A[i,j] = -1*basis[alpha_start_index][i%dimension]
            
    
    #each dimension has 4 constraints
    for i in range(dimension):
        #first constraint
        A[z_start + i*4, i] = 1 #x
        A[z_start + i*4, i + dimension] = -1 #y
        A[z_start + i*4, i + dimension*2] = -1 #z
        A[z_start + i*4, i + dimension*3] = N #b

        #second constraint
        A[z_start + i*4 + 1, i] = -1 #x
        A[z_start + i*4 + 1, i + dimension] = 1 #y
        A[z_start + i*4 + 1, i + dimension*2] = -1 #z
        A[z_start + i*4 + 1, i + dimension*3] = -1*N #b

        #third constraint
        A[z_start + i*4 + 2, i] = 1 #x
        A[z_start + i*4 + 2, i + dimension] = -1 #y
        A[z_start + i*4 + 2, i + dimension*2] = -1 #z

        #fourth constraint
        A[z_start + i*4 + 3, i] = -1 #x
        A[z_start + i*4 + 3, i + dimension] = 1 #y
        A[z_start + i*4 + 3, i + dimension*2] = -1 #z

    
    #alpha constraints
    min_a_row = dimension*2 + dimension*4
    min_a_col = dimension*4

    for i in range(2):
        for j in range(num_cons):
            for k in range(num_vec):
                A[min_a_row + i*num_cons + j, min_a_col + i*num_vec + k] = C[j][k]
    

    #upper bound and lower bound
    bu = np.zeros(xy_cons + abs_cons + a_cons)
    bl = np.zeros(xy_cons + abs_cons + a_cons)

    #x and y constraints; must equal center, so both upper and lower bound are set
    for i in range(dimension):
        bl[i] = center[i]
        bl[i + dimension] = center[i]

        bu[i] = center[i]
        bu[i + dimension] = center[i]

    #z constraints
    #ask about -z
    for i in range(dimension):
        #upper constraint
        bu[z_start + i*4 + 0] = np.inf #x
        bu[z_start + i*4 + 1] = np.inf #y
        bu[z_start + i*4 + 2] = 0 #z
        bu[z_start + i*4 + 3] = 0 #b

        bl[z_start + i*4 + 0] = 0 #x
        bl[z_start + i*4 + 1] = -1*N #y
        bl[z_start + i*4 + 2] = -1*np.inf #z
        bl[z_start + i*4 + 3] = -1*np.inf #b


    min_row = dimension*2 + dimension*4
    for i in range(min_row, xy_cons + abs_cons + a_cons):
        index = i - min_row
        bu[i] = g[index % num_cons]
        bl[i] = -1*np.inf


    lower = -1*np.inf*np.ones(x_len)
    upper = np.inf*np.ones(x_len)

    for i in range(dimension):
        lower[b_start + i] = 0
        upper[b_start + i] = 1

    constraints = LinearConstraint(A, bl, bu)
    bounds = Bounds(lb = lower, ub = upper)
    result = sop.milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    if result.status != 0:
        print("diam_failed")
        return 0, 0

    return -1*result.fun, 1



'''
center = [0,0]
basis = [[1,0],[0,1]]
C = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
g = [4,4,0,0]

star = StarSet(center, basis, C, g)
star.plot()

print(star_manhattan_distance(center, basis, C, g))

center = [0,0]
basis = [[1,0],[0,1]]
C = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1]])
g = [4,4,0,0,4]

star = StarSet(center, basis, C, g)
star.plot()

print(star_manhattan_distance(center, basis, C, g))


#3d 



center = [0,  0,  0,  0,  0.1, 0.1]
basis = [[ 0.99974, -0,      -0.005,   0,       0,       0],
 [ 0,       0.9997,   0.00002,  0,       0,       0     ],
 [ 0.0868,  -0.00004,  0.74803,  0,       0,       0     ],
 [ 0.00004, 0.08678,  0.00066,  0.74776,  0,       0     ],
 [ 0,       0,       0,       0,       1,       0     ],
 [ 0,       0,       0,       0,       0,       1     ]]
C = np.array([[ 1,  0,  0,  0,  0,  0],
 [ 0,  1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0,  0],
 [ 0,  0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1,  0],
 [ 0,  0,  0,  0,  0,  1],
 [-1, -0, -0, -0, -0, -0],
 [-0, -1, -0, -0, -0, -0],
 [-0, -0, -1, -0, -0, -0],
 [-0, -0, -0, -1, -0, -0],
 [-0, -0, -0, -0, -1, -0],
 [-0, -0, -0, -0, -0, -1]])
g = [-875, -375,    0,    0,    0,    0,  925,  425,    0,    0,    0,    0]

star = StarSet(center, basis, C, g)
star.plot()

print(star_manhattan_distance(center, basis, C, g))
'''

# center = [0,0,0]
# basis = [[1,0,0],[0,1,0], [0,0,1]]
# C = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
# g = [0.96117,0,0,-0.96117,0,0]

# star = StarSet(center, basis, C, g)
# star.plot()

# print(star_manhattan_distance(center, basis, C, g))


    

    

    


            
        






            
            
            

    


