from scipy.integrate import ode
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify
from sympy.core import *

dt = 0.01
x1 = Symbol('x1',real=True)
x2 = Symbol('x2',real=True)
# x3 = Symbol('x3',real=True)
# x4 = Symbol('x4',real=True)
w1 = Symbol('w1',real=True)
w2 = Symbol('w2',real=True)

exprs = [
    x1*(1.1+w1-x1-0.1*x2),
    x2*(4+w2-3*x1-x2),
]

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

def computeD(symbol_x, symbol_w, x, w, x_hat, w_hat):
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

def dynamics(t, state, args):
    symbol_x, symbol_w, w, what = args
    x = state[:len(symbol_x)]
    xhat = state[len(symbol_x):]
    assert len(state) == len(symbol_x*2), f"{len(state)}, {len(symbol_x*2)}"

    d = computeD(symbol_x, symbol_w, x, w, xhat, what)
    dhat = computeD(symbol_x, symbol_w, xhat, what, x, w)
    return d+dhat

def simulate(initial_condition, time_bound, time_step):
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound/time_step))
    t = [round(i*time_step, 10) for i in range(0, number_points)]
    
    init = initial_condition 
    trace = [[0]+init]
    r = ode(dynamics).set_integrator('dopri5', nsteps=2000).set_initial_value(init).set_f_params(([x1, x2], [w1, w2], [-0.1, -0.1], [0.1, 0.1]))
    for i in range(len(t)):
        print(i)
        res = r.integrate(r.t+time_step)
        init = res.flatten().tolist()
        trace.append([t[i]+time_step]+init)
    return np.array(trace)

if __name__ == "__main__":
    res = simulate([1,1,1,1], 1.15, 0.01)

    plt.figure(0)
    plt.plot(res[:,0], res[:,1],'r')
    plt.plot(res[:,0], res[:,3],'g')

    plt.figure(1)
    plt.plot(res[:,0], res[:,2],'r')
    plt.plot(res[:,0], res[:,4],'g')
    plt.show()