import numpy as np
from scipy.optimize import minimize
import inspect, ast, textwrap, inspect, astunparse
import warnings
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify
from sympy.core import *

from scipy.integrate import ode


def find_min(expr_func, jac_func, var_range, num_var, args, idx):
    bounds = []
    x0 = []
    for i in range(var_range.shape[1]):
        bounds.append((var_range[0, i], var_range[1, i]))
        x0.append(var_range[0, i])
    # print(expr_func(x0, args, idx))
    # print(jac_func(x0, args, idx).shape)
    res = minimize(
        expr_func, x0, args=(num_var, args, idx), bounds=bounds, jac=jac_func, method="L-BFGS-B"
    )
    # print(res)
    return res.fun


def find_max(expr_func, jac_func, var_range, num_var, args, idx):
    neg_expr_func = lambda x, num_var, args, idx: -expr_func(x, num_var, args, idx)
    neg_jac_func = lambda x, num_var, args, idx: -jac_func(x, num_var, args, idx)
    res = find_min(neg_expr_func, neg_jac_func, var_range, num_var, args, idx)
    return -res


def find_min_symbolic(expr, var_range):
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
    res = minimize(expr_func, x0, bounds=bounds, jac=jac_func, method="L-BFGS-B")
    return res.fun


def find_max_symbolic(expr, var_range):
    tmp = -expr
    res = find_min_symbolic(tmp, var_range)
    return -res


def compute_reachtube_mixmono_disc(
    initial_set, uncertain_var_bound, time_horizon, time_step, decomposition
):
    initial_set = initial_set[0]
    number_points = int(np.ceil(time_horizon / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    trace = [[0] + initial_set[0] + initial_set[1]]
    num_var = len(initial_set[0])
    for i in range(len(t)):
        xk = trace[-1]
        x = xk[1 : 1 + num_var]
        xhat = xk[1 + num_var :]
        w = uncertain_var_bound[0]
        what = uncertain_var_bound[1]
        d = decomposition(x, w, xhat, what, time_step)
        dhat = decomposition(xhat, what, x, w, time_step)
        trace.append([round(t[i] + time_step, 10)] + d + dhat)

    res = []
    for i in range(len(trace) - 1):
        res0 = [trace[i][0]] + np.minimum.reduce(
            [
                trace[i][1 : 1 + num_var],
                trace[i + 1][1 : 1 + num_var],
                trace[i][1 + num_var :],
                trace[i + 1][1 + num_var :],
            ]
        ).tolist()
        res.append(res0)
        res1 = [trace[i + 1][0]] + np.maximum.reduce(
            [
                trace[i][1 : 1 + num_var],
                trace[i + 1][1 : 1 + num_var],
                trace[i][1 + num_var :],
                trace[i + 1][1 + num_var :],
            ]
        ).tolist()
        res.append(res1)
    return res


def compute_reachtube_mixmono_cont(
    initial_set, uncertain_var_bound, time_horizon, time_step, decomposition
):
    initial_set = initial_set[0]
    num_var = len(initial_set[0])
    num_uncertain_var = len(uncertain_var_bound[0])

    def decomposition_dynamics(t, state, u):
        x, xhat = state[:num_var], state[num_var:]
        w, what = u[:num_uncertain_var], u[num_uncertain_var:]

        d = decomposition(x, w, xhat, what)
        dhat = decomposition(xhat, what, x, w)

        return np.concatenate((d, dhat))

    time_bound = float(time_horizon)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    init = initial_set[0] + initial_set[1]
    uncertain = uncertain_var_bound[0] + uncertain_var_bound[1]
    trace = [[0] + init]
    r = (
        ode(decomposition_dynamics)
        .set_integrator("dopri5", nsteps=2000)
        .set_initial_value(init)
        .set_f_params(uncertain)
    )
    for i in range(len(t)):
        res: np.ndarray = r.integrate(r.t + time_step)
        res = res.flatten().tolist()
        trace.append([t[i] + time_step] + res)

    res = []
    for i in range(len(trace) - 1):
        res0 = [trace[i][0]] + np.minimum.reduce(
            [
                trace[i][1 : 1 + num_var],
                trace[i + 1][1 : 1 + num_var],
                trace[i][1 + num_var :],
                trace[i + 1][1 + num_var :],
            ]
        ).tolist()
        res.append(res0)
        res1 = [trace[i + 1][0]] + np.maximum.reduce(
            [
                trace[i][1 : 1 + num_var],
                trace[i + 1][1 : 1 + num_var],
                trace[i][1 + num_var :],
                trace[i + 1][1 + num_var :],
            ]
        ).tolist()
        res.append(res1)
    return res


def calculate_bloated_tube_mixmono_cont(
    mode, init, uncertain_param, time_horizon, time_step, agent, lane_map
):
    if hasattr(agent, "dynamics") and hasattr(agent, "decomposition"):
        decomposition = agent.decomposition
        res = compute_reachtube_mixmono_cont(
            init, uncertain_param, time_horizon, time_step, decomposition
        )
    else:
        raise ValueError("Not enough information to apply discrete time mixed monotone algorithm.")
    return res


def calculate_bloated_tube_mixmono_disc(
    mode, init, uncertain_param, time_horizon, time_step, agent, lane_map
):
    if hasattr(agent, "dynamics") and hasattr(agent, "decomposition"):
        decomposition = agent.decomposition
        res = compute_reachtube_mixmono_disc(
            init, uncertain_param, time_horizon, time_step, decomposition
        )
    elif hasattr(agent, "dynamics") and hasattr(agent, "dynamics_jac"):

        def decomposition_func(x, w, xhat, what, dt):
            expr_func = lambda x, num_var, args, idx: agent.dynamics(
                list(x[:num_var]), list(x[num_var:]) + args
            )[idx]
            jac_func = lambda x, num_var, args, idx: agent.dynamics_jac(
                list(x[:num_var]), list(x[num_var:]) + args
            )[idx, :]

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
                    val = find_max(expr_func, jac_func, var_range, num_var, args, i)
                    d.append(val)
            else:
                raise ValueError(
                    f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, xhat, what]}"
                )
            return d

        res = compute_reachtube_mixmono_disc(
            init, uncertain_param, time_horizon, time_step, decomposition_func
        )
    elif hasattr(agent, "dynamics"):
        dynamics_func = agent.dynamics
        lines = inspect.getsource(dynamics_func)
        function_body = ast.parse(textwrap.dedent(lines)).body[0].body
        if not isinstance(function_body, list):
            raise ValueError(f"Failed to extract dynamics for {agent}")

        text_exprs = []
        extract = False
        x_var = []
        w_var = []
        for i, elem in enumerate(function_body):
            if isinstance(elem, ast.Expr):
                if isinstance(elem.value, ast.Constant) and elem.value.value == "Begin Dynamic":
                    extract = True
            elif isinstance(elem, ast.Expr):
                if isinstance(elem.value, ast.Constant) and elem.value.value == "End Dynamic":
                    extract = False
            elif extract:
                if isinstance(elem, ast.Assign):
                    text_exprs.append(astunparse.unparse(elem.value))
                    # x_var_name = elem.targets[0].id.replace('_plus','')
                    # x_var.append(x_var_name)
            else:
                if isinstance(elem, ast.Assign):
                    if elem.value.id == "args":
                        var_list = elem.targets[0].elts
                        for var in var_list[:-1]:
                            w_var.append(var.id)
                    elif elem.value.id == "x":
                        var_list = elem.targets[0].elts
                        for var in var_list:
                            x_var.append(var.id)

        if len(text_exprs) != len(x_var):
            raise ValueError(f"Failed to extract dynamics for {agent}")

        symbol_x = [Symbol(elem, real=True) for elem in x_var]
        symbol_w = [Symbol(elem, real=True) for elem in w_var]
        dt = Symbol("dt", real=True)

        tmp = [sympify(elem).subs("dt", time_step) for elem in text_exprs]
        expr_symbol = []
        for expr in tmp:
            for symbol in symbol_x:
                expr = expr.subs(symbol.name, symbol)
            for symbol in symbol_w:
                expr = expr.subs(symbol.name, symbol)
            expr_symbol.append(expr)
        exprs = expr_symbol

        def computeD(x, w, x_hat, w_hat, dt):
            d = []
            for expr in exprs:
                if all(a <= b for a, b in zip(x, x_hat)) and all(a <= b for a, b in zip(w, w_hat)):
                    var_range = {}
                    for i, var in enumerate(symbol_x):
                        var_range[var] = (x[i], x_hat[i])
                    for i, var in enumerate(symbol_w):
                        var_range[var] = (w[i], w_hat[i])
                    res = find_min_symbolic(expr, var_range)
                    d.append(res)
                elif all(a >= b for a, b in zip(x, x_hat)) and all(
                    a >= b for a, b in zip(w, w_hat)
                ):
                    var_range = {}
                    for i, var in enumerate(symbol_x):
                        var_range[var] = (x_hat[i], x[i])
                    for i, var in enumerate(symbol_w):
                        var_range[var] = (w_hat[i], w[i])
                    res = find_max_symbolic(expr, var_range)
                    d.append(res)
                else:
                    raise ValueError(
                        f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}"
                    )
            return d

        res = compute_reachtube_mixmono_disc(
            init, uncertain_param, time_horizon, time_step, computeD
        )
    else:
        raise ValueError("Not enough information to apply discrete time mixed monotone algorithm.")
    return res
