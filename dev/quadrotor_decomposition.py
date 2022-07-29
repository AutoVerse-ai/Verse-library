from typing import List

from scipy.optimize import minimize
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify
from sympy.core import *
import numpy as np

import matplotlib.pyplot as plt 

from dynamics_autograd import dynamicsL1, dynamicsL1_jac, dynamicsL1_1d, dynamicsL1_jac_1d
from dynamics_autograd import dynamicsgeo, dynamicsgeo_jac, dynamicsgeo_1d, dynamicsgeo_jac_1d, dynamicsgeounroll_1d, dynamicsgeounroll_jac_1d
import jax.scipy as jscipy
import jax.numpy as jnp
from jax import jit

def find_min(expr_func, jac_func, var_range, args, idx):
    bounds = []
    x0 = []
    for i in range(var_range.shape[1]):
        bounds.append((var_range[0,i], var_range[1,i]))
        x0.append(var_range[0,i])
    # print(expr_func(x0, args, idx))
    # print(jac_func(x0, args, idx).shape)
    res = minimize(
        expr_func, 
        x0,
        args = (args, idx),
        bounds = bounds,
        jac = jac_func,
        method = 'L-BFGS-B',
        tol = 1e-15
    )
    # print(res)
    return res.fun

def negate_dynamicsL1(combined_state, args, idx):
    res = -dynamicsL1_1d(combined_state, args, idx)
    return res

def negate_dynamicsL1_jac(combined_state, args, idx):
    res = -dynamicsL1_jac_1d(combined_state, args, idx)
    return res

def find_maxL1(expr_func, jac_func, var_range, args, idx):
    res = find_min(negate_dynamicsL1, negate_dynamicsL1_jac, var_range, args, idx)
    return -res

def negate_dynamicsgeo(combined_state, args, idx):
    res = -dynamicsgeo_1d(combined_state, args, idx)
    return res

def negate_dynamicsgeo_jac(combined_state, args, idx):
    res = -dynamicsgeo_jac_1d(combined_state, args, idx)
    return res

def find_maxgeo(expr_func, jac_func, var_range, args, idx):
    res = find_min(negate_dynamicsgeo, negate_dynamicsgeo_jac, var_range, args, idx)
    return -res

def negate_dynamicsgeounroll(combined_state, args, idx):
    res = -dynamicsgeounroll_1d(combined_state, args, idx)
    return res

def negate_dynamicsgeounroll_jac(combined_state, args, idx):
    res = -dynamicsgeounroll_jac_1d(combined_state, args, idx)
    return res

def find_maxgeounroll(expr_func, jac_func, var_range, args, idx):
    res = find_min(negate_dynamicsgeounroll, negate_dynamicsgeounroll_jac, var_range, args, idx)
    return -res

# def computeD(exprs, symbol_x, symbol_w, x, w, x_hat, w_hat):
#     d = []
#     for expr in exprs:
#         if all(a<=b for a,b in zip(x, x_hat)) and all(a<=b for a,b in zip(w,w_hat)):
#             var_range = {}
#             for i, var in enumerate(symbol_x):
#                 var_range[var] = (x[i], x_hat[i])
#             for i, var in enumerate(symbol_w):
#                 var_range[var] = (w[i], w_hat[i])
#             res = find_min(expr, var_range)
#             d.append(res)
#         elif all(a>=b for a,b in zip(x,x_hat)) and all(a>=b for a,b in zip(w,w_hat)):
#             var_range = {}
#             for i,var in enumerate(symbol_x):
#                 var_range[var] = (x_hat[i], x[i])
#             for i, var in enumerate(symbol_w):
#                 var_range[var] = (w_hat[i], w[i])
#             res = find_max(expr, var_range)
#             d.append(res)
#         else:
#             raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}")
#     return d

# def compute_reachtube(
#         initial_set: List[List[float]],
#         uncertain_var_bound: List[List[float]],
#         time_horizon: float,
#         time_step: float,
#         exprs: List[Expr],
#         vars: List[Symbol],
#         uncertain_vars: List[Symbol]
#     ):
#     '''
#     Example systems that we can handle x_{k+1} = F(x_k, w_k)
#     exprs: List[Expr]. The expressions defining function F
#     '''
#     assert len(uncertain_var_bound) == len(initial_set) == 2
#     assert len(uncertain_var_bound[0]) == len(uncertain_vars)
#     assert len(initial_set[0]) == len(vars)

#     number_points = int(np.ceil(time_horizon/time_step))
#     t = [round(i*time_step,10) for i in range(0,number_points)]

#     trace = [[0] + initial_set[0] + initial_set[1]]
#     num_var = len(vars)
#     num_uncertain_vars = len(uncertain_vars) 
#     for i in range(len(t)):
#         xk = trace[-1]
#         x = xk[1:1+num_var]
#         x_hat = xk[1+num_var:]
#         w = uncertain_var_bound[0]
#         w_hat = uncertain_var_bound[1]
#         d = computeD(exprs, vars, uncertain_vars, x, w, x_hat, w_hat)
#         d_hat = computeD(exprs, vars, uncertain_vars, x_hat, w_hat, x, w)
#         trace.append([round(t[i]+time_step,10)]+d+d_hat)
#     return trace

# def dynamics_jit(combined_state, args):

def test():
    t = 0
    dt = 0.001

    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    
    As_v = -5.0
    As_omega = -5.0
    dt_L1 = dt

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    state_init_lower = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    din_init_lower = state_init_lower[3:6]+state_init_lower[15:18]+state_init_lower[6:15]+\
        [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]

    state_init_upper = [1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]    
    din_init_upper = state_init_upper[3:6]+state_init_upper[15:18]+state_init_upper[6:15]+\
        [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
    
    var_range = np.array([state_init_lower+din_init_lower, state_init_upper + din_init_upper])

    args = t, dt, L1_params
    for i in range(var_range.shape[1]):
        res_min = find_min(dynamicsL1_1d, dynamicsL1_jac_1d, var_range=var_range, args=args, idx=i)
        res_max = find_maxL1(dynamicsL1_1d, dynamicsL1_jac_1d, var_range=var_range, args=args, idx=i)
        print(res_min, res_max)

def computeDL1(dynamics, dynamics_jac, args, x, w, x_hat, w_hat):
    assert len(x) == len(x_hat)
    assert len(w) == len(w_hat)
    
    d = []
    if all(a<=b for a,b in zip(x,x_hat)) and all(a<=b for a,b in zip(w,w_hat)):
        var_range = np.array([x+w, x_hat+w_hat])
        for i in range(len(x)):
            val = find_min(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    elif all(a>=b for a,b in zip(x,x_hat)) and all(a>=b for a,b in zip(w,w_hat)):
        var_range = np.array([x_hat+w_hat, x+w])
        for i in range(len(x)):
            val = find_maxL1(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    else:
            raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}")
    
    return d

def computeDgeo(dynamics, dynamics_jac, args, x, w, x_hat, w_hat):
    assert len(x) == len(x_hat)
    assert len(w) == len(w_hat)
    
    d = []
    if all(a<=b for a,b in zip(x,x_hat)) and all(a<=b for a,b in zip(w,w_hat)):
        var_range = np.array([x+w, x_hat+w_hat])
        for i in range(len(x)):
            val = find_min(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    elif all(a>=b for a,b in zip(x,x_hat)) and all(a>=b for a,b in zip(w,w_hat)):
        var_range = np.array([x_hat+w_hat, x+w])
        for i in range(len(x)):
            val = find_maxgeo(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    else:
            raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}")
    return d

def computeDgeoUnroll(dynamics, dynamics_jac, args, x, w, x_hat, w_hat):
    assert len(x) == len(x_hat)
    assert len(w) == len(w_hat)
    
    d = []
    if all(a<=b for a,b in zip(x,x_hat)) and all(a<=b for a,b in zip(w,w_hat)):
        var_range = np.array([x+w, x_hat+w_hat])
        for i in range(len(x)):
            val = find_min(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    elif all(a>=b for a,b in zip(x,x_hat)) and all(a>=b for a,b in zip(w,w_hat)):
        var_range = np.array([x_hat+w_hat, x+w])
        for i in range(len(x)):
            val = find_maxgeounroll(dynamics, dynamics_jac, var_range, args, i)
            if isinstance(val, jnp.DeviceArray):
                val = float(val)
            d.append(val)
    else:
            raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}")
    return d

def compute_reachtubeL1(
    initial_set: List[List[float]],
    uncertain_var_bound: List[List[float]],
    time_horizon: float, 
    time_step: float,
    dynamics,
    dynamics_jac,
    num_var: int
):

    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    
    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    number_points = int(np.ceil(time_horizon/time_step))
    t = [round(i*time_step, 10) for i in range(0, number_points)]

    trace = [[0]+initial_set[0]+initial_set[1]]
    for i in range(len(t)):
        print(i, trace[-1])
        xk = trace[-1]
        x = xk[1:1+num_var]
        x_hat = xk[1+num_var:]
        w = uncertain_var_bound[0]
        w_hat = uncertain_var_bound[1]
        args = (t[i], time_step, L1_params)
        d = computeDL1(dynamics, dynamics_jac, args, x, w, x_hat, w_hat)
        d_hat = computeDL1(dynamics, dynamics_jac, args, x_hat, w_hat, x, w)
        trace.append([round(t[i]+time_step,10)]+d+d_hat)
    return trace

def compute_reachtubegeo(
    initial_set: List[List[float]],
    uncertain_var_bound: List[List[float]],
    time_horizon: float, 
    time_step: float,
    dynamics,
    dynamics_jac,
    num_var: int
):

    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    
    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    number_points = int(np.ceil(time_horizon/time_step))
    t = [round(i*time_step, 10) for i in range(0, number_points)]

    trace = [[0]+initial_set[0]+initial_set[1]]
    for i in range(len(t)):
        print(i, trace[-1])
        xk = trace[-1]
        x = xk[1:1+num_var]
        x_hat = xk[1+num_var:]
        w = uncertain_var_bound[0]
        w_hat = uncertain_var_bound[1]
        args = (t[i], time_step, L1_params)
        d = computeDgeo(dynamics, dynamics_jac, args, x, w, x_hat, w_hat)
        d_hat = computeDgeo(dynamics, dynamics_jac, args, x_hat, w_hat, x, w)
        trace.append([round(t[i]+time_step,10)]+d+d_hat)
    return trace

def computeReachtubeGeoUnroll(
    initial_set: List[List[float]],
    uncertain_var_bound: List[List[float]],
    time_horizon: float, 
    time_step: float,
    dynamics,
    dynamics_jac,
    num_var: int
):

    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    
    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    number_points = int(np.ceil(time_horizon/time_step))
    t = [round(i*time_step, 10) for i in range(0, number_points)]

    trace = [[0]+initial_set[0]+initial_set[1]]
    for i in range(len(t)):
        print(i, trace[-1])
        x0 = trace[0]
        x = x0[1:1+num_var]
        x_hat = x0[1+num_var:]
        w = uncertain_var_bound[0]
        w_hat = uncertain_var_bound[1]
        args = (t[i], time_step, L1_params, i+1)
        d = computeDgeoUnroll(dynamics, dynamics_jac, args, x, w, x_hat, w_hat)
        d_hat = computeDgeoUnroll(dynamics, dynamics_jac, args, x_hat, w_hat, x, w)
        trace.append([round(t[i]+time_step,10)]+d+d_hat)
    return trace


def simulateL1(init, time_bound, time_step):
    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    N_step = int((time_bound)/time_step)
    state_init = init[0:18]
    din_init = init[18:]

    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    traj: jnp.ndarray = jnp.array([[0] + state_init + din_init])
    dynamics_jit = jit(dynamicsL1)
    dynamics_jac_jit = jit(dynamicsL1_jac)
    for i in range(N_step):
        # if i%1000 == 0:
        print(i)
        t = traj[-1,0]
        state = traj[-1,1:]
        state_plus = dynamics_jit(state, (t, time_step, L1_params,))
        # J = dynamics_jac_jit(state, (t, time_step, L1_params,))
        t = t+time_step
        state_plus = jnp.concatenate((jnp.array([t]), state_plus), axis=0)
        traj = jnp.concatenate((traj, jnp.reshape(state_plus, (1,62))), axis=0)
    return traj

def simulategeo(init, time_bound, time_step):
    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    N_step = int((time_bound)/time_step)
    state_init = init[0:18]
    din_init = init[18:]

    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    traj: jnp.ndarray = jnp.array([[0] + state_init + din_init])
    dynamics_jit = jit(dynamicsgeo)
    dynamics_jac_jit = jit(dynamicsgeo_jac)
    for i in range(N_step):
        # if i%1000 == 0:
        print(i)
        t = traj[-1,0]
        state = traj[-1,1:]
        state_plus = dynamics_jit(state, (t, time_step, L1_params,))
        # J = dynamics_jac_jit(state, (t, time_step, L1_params,))
        t = t+time_step
        state_plus = jnp.concatenate((jnp.array([t]), state_plus), axis=0)
        traj = jnp.concatenate((traj, jnp.reshape(state_plus, (1,19))), axis=0)
    return traj

def testReachtubeL1():
    import matplotlib.pyplot as plt 

    for i in range(5):
        print(f">>>>>> {i}")
        state_init = [np.random.uniform(0,0.01),0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
        din_init = state_init[3:6]+state_init[15:18]+state_init[6:15]+\
            [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
            [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
        trace = simulateL1(
            state_init + din_init,
            0.01,
            0.001,
        )
        for i in range(1,trace.shape[1]):
            plt.figure(i)
            plt.plot(trace[:,0], trace[:,i],'b')
    
    # plt.show()
    state_init_lower = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    din_init_lower = state_init_lower[3:6]+state_init_lower[15:18]+state_init_lower[6:15]+\
        [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]

    state_init_upper = [0.01,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]    
    din_init_upper = state_init_upper[3:6]+state_init_upper[15:18]+state_init_upper[6:15]+\
        [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
    
    trace = compute_reachtubeL1(
        [state_init_lower+din_init_lower, state_init_upper+din_init_upper],
        [[],[]],
        0.01,
        0.001,
        dynamicsL1_1d,
        dynamicsL1_jac_1d,
        61
    )
    # print(trace)

    trace = np.array(trace)
    for i in range(1, 62):
        plt.figure(i)
        plt.plot(trace[:,0], trace[:,i],'r')
        plt.plot(trace[:,0], trace[:,i+61],'g')

    plt.show()

def testReachtubeGeo():
    import matplotlib.pyplot as plt 

    for i in range(2):
        print(f">>>>>> {i}")
        state_init = [np.random.uniform(0,0.1),0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
        # din_init = state_init[3:6]+state_init[15:18]+state_init[6:15]+\
        #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
        trace = simulategeo(
            state_init,
            0.1,
            0.001,
        )

        for i in range(1,19):
            plt.figure(i)
            plt.plot(trace[:,0], trace[:,i],'b')
    
        # plt.figure(19)
        # plt.plot(trace[:,1], trace[:,2], 'b')
    # plt.show()
    state_init_lower = [0,0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
    # din_init_lower = state_init_lower[3:6]+state_init_lower[15:18]+state_init_lower[6:15]+\
    #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
    #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]

    state_init_upper = [0.1,0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
    # din_init_upper = state_init_upper[3:6]+state_init_upper[15:18]+state_init_upper[6:15]+\
    #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
    #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
    
    trace = compute_reachtubegeo(
        [state_init_lower, state_init_upper],
        [[],[]],
        0.1,
        0.001,
        dynamicsgeo_1d,
        dynamicsgeo_jac_1d,
        18
    )
    # print(trace)

    trace = np.array(trace)
    for i in range(1, 19):
        plt.figure(i)
        plt.plot(trace[:,0], trace[:,i],'r')
        plt.plot(trace[:,0], trace[:,i+18],'g')

    plt.show()

def testReachtubeGeoUnroll():
    import matplotlib.pyplot as plt 

    for i in range(2):
        print(f">>>>>> {i}")
        state_init = [np.random.uniform(0,0.1),0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
        # din_init = state_init[3:6]+state_init[15:18]+state_init[6:15]+\
        #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
        trace = simulategeo(
            state_init,
            1.0,
            0.001,
        )

        for i in range(1,19):
            plt.figure(i)
            plt.plot(trace[:,0], trace[:,i],'b')
    
        # plt.figure(19)
        # plt.plot(trace[:,1], trace[:,2], 'b')
    # plt.show()
    state_init_lower = [0,0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
    # din_init_lower = state_init_lower[3:6]+state_init_lower[15:18]+state_init_lower[6:15]+\
    #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
    #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]

    state_init_upper = [0.1,0,1,0,2,1,0.98,0.1,-0.2,-0.01,1,0,0.2,0,0.98,-0.2,0,0.03]
    # din_init_upper = state_init_upper[3:6]+state_init_upper[15:18]+state_init_upper[6:15]+\
    #     [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
    #     [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
    
    trace = computeReachtubeGeoUnroll(
        [state_init_lower, state_init_upper],
        [[],[]],
        1.0,
        0.001,
        dynamicsgeounroll_1d,
        dynamicsgeounroll_jac_1d,
        18
    )
    # print(trace)

    trace = np.array(trace)
    for i in range(1, 19):
        plt.figure(i)
        plt.plot(trace[:,0], trace[:,i],'r')
        plt.plot(trace[:,0], trace[:,i+18],'g')

    plt.show()

if __name__ == "__main__":
    # testReachtubeL1()
    # testReachtubeGeo()
    testReachtubeGeoUnroll()