from typing import List

from scipy.optimize import minimize
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify
from sympy.core import *
import numpy as np

import matplotlib.pyplot as plt 

def find_min(expr, var_range):
    bounds = []
    vars = list(expr.free_symbols)
    x0 = []
    jac = []
    for var in vars:
        bounds.append(var_range[var])
        x0.append(var_range[var][0])
        jac.append(diff(expr, var))
    expr_func = lambdify([vars], expr)
    jac_func = lambdify([vars], jac)
    res = minimize(
        expr_func, 
        x0,
        bounds = bounds,
        jac = jac_func,
        method = 'L-BFGS-B'
    )
    return res.fun

def find_max(expr, var_range):
    tmp = -expr 
    res = find_min(tmp, var_range)
    return -res

def computeD(exprs, symbol_x, symbol_w, x, w, x_hat, w_hat):
    d = []
    for expr in exprs:
        if all(a<=b for a,b in zip(x, x_hat)) and all(a<=b for a,b in zip(w,w_hat)):
            var_range = {}
            for i, var in enumerate(symbol_x):
                var_range[var] = (x[i], x_hat[i])
            for i, var in enumerate(symbol_w):
                var_range[var] = (w[i], w_hat[i])
            res = find_min(expr, var_range)
            d.append(res)
        elif all(a>=b for a,b in zip(x,x_hat)) and all(a>=b for a,b in zip(w,w_hat)):
            var_range = {}
            for i,var in enumerate(symbol_x):
                var_range[var] = (x_hat[i], x[i])
            for i, var in enumerate(symbol_w):
                var_range[var] = (w_hat[i], w[i])
            res = find_max(expr, var_range)
            d.append(res)
        else:
            raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}")
    return d

def compute_reachtube(
        initial_set: List[List[float]],
        uncertain_var_bound: List[List[float]],
        time_horizon: float,
        time_step: float,
        exprs: List[Expr],
        vars: List[Symbol],
        uncertain_vars: List[Symbol]
    ):
    '''
    Example systems that we can handle x_{k+1} = F(x_k, w_k)
    exprs: List[Expr]. The expressions defining function F
    '''
    assert len(uncertain_var_bound) == len(initial_set) == 2
    assert len(uncertain_var_bound[0]) == len(uncertain_vars)
    assert len(initial_set[0]) == len(vars)

    number_points = int(np.ceil(time_horizon/time_step))
    t = [round(i*time_step,10) for i in range(0,number_points)]

    trace = [[0] + initial_set[0] + initial_set[1]]
    num_var = len(vars)
    num_uncertain_vars = len(uncertain_vars) 
    for i in range(len(t)):
        xk = trace[-1]
        x = xk[1:1+num_var]
        x_hat = xk[1+num_var:]
        w = uncertain_var_bound[0]
        w_hat = uncertain_var_bound[1]
        d = computeD(exprs, vars, uncertain_vars, x, w, x_hat, w_hat)
        d_hat = computeD(exprs, vars, uncertain_vars, x_hat, w_hat, x, w)
        trace.append([round(t[i]+time_step,10)]+d+d_hat)
    return trace

if __name__ == "__main__":
    x1 = Symbol('x1',real=True)
    x2 = Symbol('x2',real=True)
    # x3 = Symbol('x3',real=True)
    # x4 = Symbol('x4',real=True)
    w1 = Symbol('w1',real=True)
    w2 = Symbol('w2',real=True)

    time_horizon = 10
    dt = 0.01
    # expr = [
    #     x1 + dt*(-2*x1+x2*(1+x1)+x3+w1),
    #     x2 + dt*(-x2+x1*(1-x2)+0.1),
    #     x3 + dt*(-x4),
    #     x4 + dt*(x3)
    # ]
    expr = [
        x1+dt*(x1*(1.1+w1-x1-0.1*x2)),
        x2+dt*(x2*(4+w2-3*x1-x2)),
    ]
    expr_func = lambdify([(x1,x2,w1,w2)], expr)
    for j in range(20):
        # init = [np.random.uniform(1,1.5),np.random.uniform(1,1.5),1,0]
        init = [1,1]
        number_points = int(np.ceil(time_horizon/dt))
        t = [round(i*dt,10) for i in range(0,number_points)]
        trace = [[0]+init]
        for i in range(number_points):
            x_k = trace[-1][1:]
            w1_val = np.random.uniform(-0.1, 0.1)
            w2_val = np.random.uniform(-0.1, 0.1)
            x_k1 = expr_func(x_k+[w1_val,w2_val])
            trace.append([t[i]+dt]+x_k1)
        trace = np.array(trace)
        plt.figure(0)
        plt.plot(trace[:,0], trace[:,1], 'b')

        plt.figure(1)
        plt.plot(trace[:,0], trace[:,2], 'b')

        plt.figure(2)
        plt.plot(trace[:,1], trace[:,2], 'b')
    
    tube = compute_reachtube(
        [[1,1], [1,1]],
        [[-0.1, -0.1],[0.1, 0.1]],
        time_horizon,
        dt,
        expr,
        [x1,x2],
        [w1,w2]
    )
    tube = np.array(tube)

    plt.figure(0)
    plt.plot(tube[:,0], tube[:,1], 'r')
    plt.plot(tube[:,0], tube[:,3], 'g')

    plt.figure(1)
    plt.plot(tube[:,0], tube[:,2], 'r')
    plt.plot(tube[:,0], tube[:,4], 'g')

    # plt.figure(2)
    # for i in range(tube.shape[0]):
    #     plt.plot([tube[i,1], tube[i,5]], [tube[i,2],tube[i,2]], 'g')
    #     plt.plot([tube[i,5], tube[i,5]], [tube[i,2],tube[i,6]], 'g')
    #     plt.plot([tube[i,5], tube[i,1]], [tube[i,6],tube[i,6]], 'g')
    #     plt.plot([tube[i,1], tube[i,1]], [tube[i,6],tube[i,2]], 'g')
    plt.show()
