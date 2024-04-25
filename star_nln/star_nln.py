import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
import itertools 
from scipy.optimize import linprog
import copy 

def vanderpol(s,t):
    mu = 1
    x1, y1, x2, y2, b = s 
    x1_dot = y1 
    y1_dot = mu*(1-x1**2)*y1+b*(x2-x1)-x1 
    x2_dot = y2 
    y2_dot = mu*(1-x2**2)*y2-b*(x2-x1)-x2 
    b_dot = 0 
    return [x1_dot, y1_dot, x2_dot, y2_dot, b_dot]

def laubloomis(x,t):
    pass 

class Star:
    def __init__(
        self,
        center: np.ndarray,
        basis: np.ndarray,
        C: np.ndarray,
        g: np.ndarray
    ):
        self.center = center
        self.basis = basis 
        self.C = C 
        self.g = g 

    def in_star(self, point):
        m = self.n_basis()
        c = np.zeros(m)
        A_eq = self.basis.T 
        b_eq = point.squeeze() - self.center.squeeze()
        A_ub = self.C 
        b_ub = self.g 
        bounds = [(-np.inf, np.inf)]*(m)
        res = linprog(
            c = c,
            A_eq = A_eq,
            b_eq = b_eq,
            A_ub = A_ub,
            b_ub = b_ub,
            bounds = bounds
        )
        return res.success

    def n_basis(self):
        return self.basis.shape[0] 
    
    def n_dim(self):
        return len(self.center)

    def n_pred(self):
        return len(self.g)

    def visualize(self, fig: plt.axes, dim1, dim2) -> plt.axes:
        xs, ys = self.get_verts(dim1, dim2)
        #print(verts)
        fig.plot(xs, ys)
        # plt.show()
        return fig

    # def normalize(self) -> Star:
    #     tmp_center = self.center 
    #     tmp_basis = self.basis 
    #     tmp_C = self.C 
    #     tmp_g = self.g 
    #     norm_factor = np.linalg.norm(tmp_basis, axis=1)
    #     tmp_basis = tmp_basis/norm_factor

    @classmethod
    def from_rect(cls, rect: np.ndarray):
        n = rect.shape[1]
        center = (rect[0,:]+rect[1,:])/2.0
        basis = np.diag((rect[1,:]-rect[0,:])/2.0)
        C = np.zeros((n*2,n))
        g = np.ones((n*2,1))
        for i in range(n):
            C[i*2,i] = -1
            C[i*2+1,i] = 1
        return cls(center, basis, C, g) 

    #stanley bak code
    def get_verts(self, dim1, dim2):
        """get the vertices of the stateset"""
        #TODO: generalize for n dimensional
        verts = []
        x_pts = []
        y_pts = []
        extra_dims_ct = len(self.center) - 2
        zeros = []
        for i in range(0, extra_dims_ct):
            zeros.append([0])
        # sample the angles from 0 to 2pi, 100 samples
        for angle in np.linspace(0, 2*np.pi, 100):
            x_component = np.cos(angle)
            y_component = np.sin(angle)
            #TODO: needs to work for 3d and any dim of non-graphed state
            direction = [[x_component], [y_component]]
            direction.extend(zeros)
            direction = np.array(direction)
            #for i in range(0, extra_dims_ct):
            #    direction.append([0])
            #direction.extend(zeros)

            pt = self.maximize(direction)

            verts.append(pt)
            #print(pt)
            x_pts.append(pt[0][0])
            #print(pt[0][0])
            y_pts.append(pt[0][1])
            #print(pt[1][0])
        x_pts.append(x_pts[0])
        y_pts.append(y_pts[0])
        return (x_pts, y_pts)

    def min_max(self, dim):
        n = self.n_dim()
        m = self.n_basis()
        p = self.n_pred()
        # A_ub = np.zeros((n+p, m+m+n))
        # b_ub = np.zeros((n+p, 1))
        # A_ub[:n,:n] = np.eye(n)
        # A_ub[:n,n:n+m] = self.basis.T 
        # A_ub[n:,m+n:] = self.C
        # b_ub[:n,:] = center.reshape((-1,1))
        # b_ub[n:,:] = self.g
        # # Constraints enforcing equality of alphas
        # A_eq = np.zeros((m, n+m+m))
        # b_eq = np.zeros((m, 1))
        # A_eq[:,n:n+m] = np.eye(m)
        # A_eq[:,n+m:n+m+m] = -np.eye(m)
        A_eq = np.zeros((n, m+n))
        b_eq = np.zeros((n, 1))
        A_eq[:,:n] = np.eye(n)
        A_eq[:,n:] = self.basis.T
        b_eq = self.center.reshape((-1,1))
        A_ub = np.zeros((p, n+m))
        b_ub = np.zeros((p,1))
        A_ub[:,n:] = self.C
        b_ub = self.g.reshape((-1,1))
        c = np.zeros(n+m)
        c[dim] = 1
        bounds = [(-np.inf, np.inf)]*(n+m)
        res_min = linprog(
            c = c,
            A_ub = A_ub,
            b_ub = b_ub,
            A_eq = A_eq,
            b_eq = b_eq,
            bounds = bounds
        )
        res_max = linprog(
            c = -c,
            A_ub = A_ub,
            b_ub = b_ub,
            A_eq = A_eq,
            b_eq = b_eq,
            bounds = bounds
        )

        return (res_min.x[dim], res_max.x[dim])

    def maximize(self, opt_direction):
        """return the point that maximizes the direction"""

        opt_direction *= -1

        # convert opt_direction to domain
        #print(self.basis)
        domain_direction = opt_direction.T @ self.basis


        # use LP to optimize the constraints
        res = linprog(c=domain_direction, 
                A_ub=self.C, 
                b_ub=self.g, 
                bounds=(None, None))
        
        # convert point back to range
        #print(res.x)
        domain_pt = res.x.reshape((res.x.shape[0], 1))
        #if domain_direction[0][0] == -1 and domain_direction[0][1] == 0:
        #    print(domain_pt)
        #range_pt = self.center + self.basis @ domain_pt
        range_pt = self.center + domain_pt.T @ self.basis 

        #if domain_direction[0][0] == -1 and domain_direction[0][1] == 0:
        #    print(self.basis)
        #    print(range_pt)
        # return the point
        return range_pt

def find_alpha(star:Star, point:np.ndarray):
    point = point.reshape((-1,1))
    p = star.n_pred() 
    n = star.n_dim()
    m = star.n_basis()
    tmp = np.linalg.det(star.basis)
    # if tmp<1e-8:
    #     tmp = np.random.randint(0,2,(5,5))*2-1
    #     star.basis = star.basis*tmp
    A_eq = np.zeros((n, 2*m))
    A_eq[:,:m] = star.basis.T
    b_eq = point - star.center.reshape((-1,1))
    A_ub = np.zeros((2*m,2*m))
    A_ub[:m,:m] = np.eye(m)
    A_ub[:m,m:2*m] = -np.eye(m)
    A_ub[m:2*m,:m] = -np.eye(m)
    A_ub[m:2*m,m:2*m] = -np.eye(m)
    b_ub = np.zeros((2*m, 1))
    c = np.zeros(2*m)
    c[m:] = 1
    bounds = [(-1000,1000)]*2*m
    res = linprog(
        c = c,
        A_ub = A_ub,
        b_ub = b_ub,
        A_eq = A_eq,
        b_eq = b_eq,
        bounds = bounds,
        method= 'highs-ipm'
    )
    return res.x[:m]

def sample_rect(rect, n=1):
    res = np.random.uniform(rect[0], rect[1], (n, len(rect[0])))
    return res

if __name__ == "__main__":
    init_rect = [
        [1.25, 2.35, 1.25, 2.35, 1],
        [1.55, 2.45, 1.55, 2.45, 3]
    ] 

    init = sample_rect(init_rect, 100)
    t = np.arange(0, 7, 0.01)
    fig0 = plt.figure(0)
    plt.plot([0,0],[1.25,1.55])
    fig1 = plt.figure(1)
    plt.plot([0,0],[2.35,2.45])
    fig2 = plt.figure(2)
    plt.plot([0,0],[1.25,1.55])
    fig3 = plt.figure(3)
    plt.plot([0,0],[2.35,2.45])
    traces = []
    for i in range(init.shape[0]):
        trace = odeint(vanderpol, init[i,:], t)
        plt.figure(0)
        plt.plot(t, trace[:,0],'r')
        plt.figure(1)
        plt.plot(t, trace[:,1],'r')
        plt.figure(2)
        plt.plot(t, trace[:,2],'r')
        plt.figure(3)
        plt.plot(t, trace[:,3],'r')
        traces.append(trace)
    
    inits = itertools.product([1.25,1.55],[2.35,2.45],[1.25,1.55],[2.35,2.45],[1.0,3.0])
    
    # corner_traces = []
    # for init in inits:
    #     trace = odeint(vanderpol, init, t)
    #     plt.figure(0)
    #     plt.plot(t, trace[:,0],'b')
    #     plt.figure(1)
    #     plt.plot(t, trace[:,1],'b')
    #     plt.figure(2)
    #     plt.plot(t, trace[:,2],'b')
    #     plt.figure(3)
    #     plt.plot(t, trace[:,3],'b')
    #     corner_traces.append(trace)
    # plt.show()

    # Compute star set for each axis assuming linearality 
    init_star = Star.from_rect(np.array(init_rect))
    num_basis = init_star.n_basis()
    center = init_star.center
    center_trace = odeint(vanderpol, center, t)
    basis_trace = []
    for i in range(num_basis):
        x0 = center + init_star.basis[i,:]
        tmp = odeint(vanderpol, x0, t)
        basis_trace.append(tmp)
    # star_vertices = 
    star_list = []
    for i in range(center_trace.shape[0]):
        tmp_center = center_trace[i]
        tmp_basis = np.zeros(init_star.basis.shape)
        for j in range(len(basis_trace)):
            tmp_basis[j,:] = basis_trace[j][i,:] - tmp_center
        tmp_C = init_star.C 
        tmp_g = init_star.g 
        star_list.append(Star(tmp_center, tmp_basis, tmp_C, tmp_g))

    trace0 = []
    trace1 = []
    trace2 = []
    trace3 = []
    for i in range(len(star_list)):
        star = star_list[i]
        trace0.append(star.min_max(0))
        trace1.append(star.min_max(1))
        trace2.append(star.min_max(2))
        trace3.append(star.min_max(3))
    trace0 = np.array(trace0)
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    trace3 = np.array(trace3)
    plt.figure(0)
    plt.plot(t, trace0[:,0], 'g')
    plt.plot(t, trace0[:,1], 'g')
    plt.figure(1)
    plt.plot(t, trace1[:,0], 'g')
    plt.plot(t, trace1[:,1], 'g')
    plt.figure(2)
    plt.plot(t, trace2[:,0], 'g')
    plt.plot(t, trace2[:,1], 'g')
    plt.figure(3)
    plt.plot(t, trace3[:,0], 'g')
    plt.plot(t, trace3[:,1], 'g')
    # plt.show()

    # Bloat each stars
    # 1) Collect simulation trajectories
    # 2) Normalize basis for all the stars (skip)
    # 3) At each time step, find alpha for each trajectories
    new_star_list = []
    for i in range(len(t)):
        print(i)
        if i==305:
            print("stop here")
        curr_star = star_list[i]
        # curr_star.normalize()
        alpha_list = []
        for traj in traces:
            alpha = find_alpha(curr_star, traj[i,:])
            alpha_list.append(alpha)
        # 4) Find g such that all the alphas at that time step satisfy Calpha\leq g
        g = copy.deepcopy(curr_star.g)
        for alpha in alpha_list:
            tmp_g = curr_star.C@alpha.reshape((-1,1)) 
            g = np.maximum(g, tmp_g)
        new_star = Star(curr_star.center, curr_star.basis, curr_star.C, g)
        new_star_list.append(new_star)

    trace0 = []
    trace1 = []
    trace2 = []
    trace3 = []
    for i in range(len(new_star_list)):
        star = new_star_list[i]
        trace0.append(star.min_max(0))
        trace1.append(star.min_max(1))
        trace2.append(star.min_max(2))
        trace3.append(star.min_max(3))
    trace0 = np.array(trace0)
    trace1 = np.array(trace1)
    trace2 = np.array(trace2)
    trace3 = np.array(trace3)
    plt.figure(0)
    plt.plot(t, trace0[:,0], 'y')
    plt.plot(t, trace0[:,1], 'y')
    plt.figure(1)
    plt.plot(t, trace1[:,0], 'y')
    plt.plot(t, trace1[:,1], 'y')
    plt.figure(2)
    plt.plot(t, trace2[:,0], 'y')
    plt.plot(t, trace2[:,1], 'y')
    plt.figure(3)
    plt.plot(t, trace3[:,0], 'y')
    plt.plot(t, trace3[:,1], 'y')

    fig4 = plt.figure(4)
    ax4 = fig4.gca()
    # for i in range(new_star_list):
    star = new_star_list[305]
    ax4 = star.visualize(ax4, None, None)
    plt.plot(star.center[0], star.center[1], 'g*')
    for trace in traces:
        plt.plot(trace[305,0], trace[305,1], 'r*')
    for i in range(star.basis.shape[0]):
        base = star.basis[i,:]*50
        x = [star.center[0], star.center[0]+base[0]]
        y = [star.center[1], star.center[1]+base[1]]
        plt.plot(copy.deepcopy(x),copy.deepcopy(y),'b')
    plt.show()
    print("here")