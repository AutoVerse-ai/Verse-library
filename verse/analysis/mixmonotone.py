import numpy as np
from scipy.optimize import minimize

def find_min(expr_func, jac_func, var_range, num_var, args, idx):
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
        args = (num_var, args, idx),
        bounds = bounds,
        jac = jac_func,
        method = 'L-BFGS-B',
        tol = 1e-20
    )
    # print(res)
    return res.fun

def find_max(expr_func, jac_func, var_range, num_var, args, idx):
    neg_expr_func = lambda x, num_var, args, idx: -expr_func(x, num_var, args, idx)
    neg_jac_func = lambda x, num_var, args, idx: -jac_func(x, num_var, args, idx)
    res = find_min(neg_expr_func, neg_jac_func, var_range, num_var, args, idx)
    return -res

def compute_reachtube_mixmono_disc(
        initial_set,
        uncertain_var_bound,
        time_horizon,
        time_step,
        decomposition
):
    initial_set = initial_set[0]
    number_points = int(np.ceil(time_horizon / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    trace = [[0] + initial_set[0] + initial_set[1]]
    num_var = len(initial_set[0])
    for i in range(len(t)):
        xk = trace[-1]
        x = xk[1:1 + num_var]
        xhat = xk[1 + num_var:]
        w = uncertain_var_bound[0]
        what = uncertain_var_bound[1]
        d = decomposition(x, w, xhat, what, time_step)
        dhat = decomposition(xhat, what, x, w, time_step)
        trace.append([round(t[i] + time_step, 10)] + d + dhat)

    res = []
    for i in range(len(trace) - 1):
        res0 = [trace[i][0]] + np.minimum.reduce(
            [trace[i][1:1 + num_var], trace[i + 1][1:1 + num_var], trace[i][1 + num_var:],
            trace[i + 1][1 + num_var:]]
        ).tolist()
        res.append(res0)
        res1 = [trace[i + 1][0]] + np.maximum.reduce(
            [trace[i][1:1 + num_var], trace[i + 1][1:1 + num_var], trace[i][1 + num_var:],
            trace[i + 1][1 + num_var:]]
        ).tolist()
        res.append(res1)
    return res

def calculate_bloated_tube_mixmono_disc(
            mode,
            init,
            uncertain_param,
            time_horizon,
            time_step,
            agent,
            lane_map
    ):
        if hasattr(agent, 'dynamics') and hasattr(agent, 'decomposition'):
            decomposition = agent.decomposition
            res = compute_reachtube_mixmono_disc(init, uncertain_param, time_horizon, time_step, decomposition)
        elif hasattr(agent, 'dynamics') and hasattr(agent, 'dynamics_jac'):
            def decomposition_func(x, w, xhat, what, dt):
                expr_func = lambda x, num_var, args, idx: agent.dynamics(x[:num_var], x[num_var:]+args)[idx]
                jac_func = lambda x, num_var, args, idx: agent.dynamics_jac(x[:num_var], x[num_var:]+args)[idx, :]

                assert len(x) == len(xhat)
                assert len(w) == len(what)

                d = []
                num_var = len(x)
                args = [dt]
                if all(a <= b for a, b in zip(x, xhat)) and all(a <= b for a, b in zip(w, what)):
                    var_range = np.array([x + w, xhat + what])
                    for i in range(len(x)):
                        val = find_min(expr_func, jac_func, var_range, num_var, args, i)
                        d.append(val)
                elif all(b <= a for a, b in zip(x, xhat)) and all(b <= a for a, b in zip(w, what)):
                    var_range = np.array([xhat + what, x + w])
                    for i in range(len(x)):
                        val = find_max(expr_func, jac_func, var_range, [dt], i)
                        d.append(val)
                else:
                    raise ValueError(f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, xhat, what]}")
                return d

            res = compute_reachtube_mixmono_disc(init, uncertain_param, time_horizon, time_step,
                                                      decomposition_func)
        elif hasattr(agent, 'dynamics'):
            pass
        else:
            raise ValueError('Not enough information to apply discrete time mixed monotone algorithm.')
        return res