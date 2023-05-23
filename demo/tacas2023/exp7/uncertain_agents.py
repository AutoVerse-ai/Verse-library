from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse.agents import BaseAgent
from verse.parser import ControllerIR

from scipy.optimize import minimize
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify
from sympy.core import *


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
    res = minimize(expr_func, x0, bounds=bounds, jac=jac_func, method="L-BFGS-B")
    return res.fun


def find_max(expr, var_range):
    tmp = -expr
    res = find_min(tmp, var_range)
    return -res


def computeD(exprs, symbol_x, symbol_w, x, w, x_hat, w_hat):
    d = []
    for expr in exprs:
        if all(a <= b for a, b in zip(x, x_hat)) and all(a <= b for a, b in zip(w, w_hat)):
            var_range = {}
            for i, var in enumerate(symbol_x):
                var_range[var] = (x[i], x_hat[i])
            for i, var in enumerate(symbol_w):
                var_range[var] = (w[i], w_hat[i])
            res = find_min(expr, var_range)
            d.append(res)
        elif all(b <= a for a, b in zip(x, x_hat)) and all(b <= a for a, b in zip(w, w_hat)):
            var_range = {}
            for i, var in enumerate(symbol_x):
                var_range[var] = (x_hat[i], x[i])
            for i, var in enumerate(symbol_w):
                var_range[var] = (w_hat[i], w[i])
            res = find_max(expr, var_range)
            d.append(res)
        else:
            raise ValueError(
                f"Condition for x, w, x_hat, w_hat not satisfied: {[x, w, x_hat, w_hat]}"
            )
    return d


class Agent1(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.decision_logic = ControllerIR.empty()

    def dynamics(self, x, args):
        w1, w2, dt = args
        x1, x2 = x

        x1_plus = x1 + dt * (x1 * (1.1 + w1 - x2 - 0.1 * x2))
        x2_plus = x2 + dt * (x2 * (4 + w2 - 3 * x1 - x2))
        return [x1_plus, x2_plus]

    def decomposition(self, x, w, xhat, what, params):
        dt = params
        x1 = Symbol("x1", real=True)
        x2 = Symbol("x2", real=True)

        w1 = Symbol("w1", real=True)
        w2 = Symbol("w2", real=True)

        exprs = [
            x1 + dt * x1 * (1.1 + w1 - x1 - 0.1 * x2),
            x2 + dt * x2 * (4 + w2 - 3 * x1 - x2),
        ]
        res = computeD(exprs, [x1, x2], [w1, w2], x, w, xhat, what)
        return res


class Agent2(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.decision_logic = ControllerIR.empty()

    def dynamics(self, x, args):
        w1, w2, dt = args
        x1, x2 = x

        x1_plus = x1 + dt * (x1 * (1.1 + w1 - x1 - 0.1 * x2))
        x2_plus = x2 + dt * (x2 * (4 + w2 - 3 * x1 - x2))
        return [x1_plus, x2_plus]

    def dynamics_jac(self, x, args):
        w1, w2, dt = args
        x1, x2 = x

        j1x1 = dt * (w1 - x1 - x2 / 10 + 11 / 10) - dt * x1 + 1
        j1x2 = -(dt * x1) / 10
        j1w1 = dt * x1
        j1w2 = 0
        j2x1 = -3 * dt * x2
        j2x2 = dt * (w2 - 3 * x1 - x2 + 4) - dt * x2 + 1
        j2w1 = 0
        j2w2 = dt * x2

        return np.array([[j1x1, j1x2, j1w1, j1w2], [j2x1, j2x2, j2w1, j2w2]])


class Agent3(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.decision_logic = ControllerIR.empty()

    def dynamics(self, x, args):
        w1, w2, dt = args
        x1, x2 = x
        """Begin Dynamic"""
        x1_plus = x1 + dt * (x1 * (1.1 + w1 - x1 - 0.1 * x2))
        x2_plus = x2 + dt * (x2 * (4 + w2 - 3 * x1 - x2))
        """End Dynamic"""
        return [x1_plus, x2_plus]


class Agent4(BaseAgent):
    def __init__(self, id):
        self.id = id
        self.decision_logic = ControllerIR.empty()

    def dynamics(self, x, args):
        w1, dt = args
        x1, x2, x3, x4 = x

        x1_plus = x1 + dt * (-2 * x1 + x2 * (1 + x1) + x3 + w1)
        x2_plus = x2 + dt * (-x2 + x1 * (1 - x2) + 0.1)
        x3_plus = x3 + dt * (-x4)
        x4_plus = x4 + dt * (x3)
        return [x1_plus, x2_plus, x3_plus, x4_plus]

    def dynamics_jac(self, x, args):
        w1, dt = args
        x1, x2, x3, x4 = x

        j1x1, j1x2, j1x3, j1x4, j1w1 = [dt * (x2 - 2) + 1, dt * (x1 + 1), dt, 0, dt]
        j2x1, j2x2, j2x3, j2x4, j2w1 = [-dt * (x2 - 1), 1 - dt * (x1 + 1), 0, 0, 0]
        j3x1, j3x2, j3x3, j3x4, j3w1 = [0, 0, 1, -dt, 0]
        j4x1, j4x2, j4x3, j4x4, j4w1 = [0, 0, dt, 1, 0]

        return np.array(
            [
                [j1x1, j1x2, j1x3, j1x4, j1w1],
                [j2x1, j2x2, j2x3, j2x4, j2w1],
                [j3x1, j3x2, j3x3, j3x4, j3w1],
                [j4x1, j4x2, j4x3, j4x4, j4w1],
            ]
        )


class Agent5(BaseAgent):
    def __init__(self, id):
        # super().__init__(id, code, file_name)
        self.id = id
        self.decision_logic = ControllerIR.empty()
        self.init_cont = None
        self.init_disc = None
        self.static_parameters = None
        self.uncertain_parameters = None

    def dynamics(self, x, args):
        w1 = args
        x1, x2, x3, x4 = x

        dx1 = -2 * x1 + x2 * (1 + x1) + x3 + w1
        dx2 = -x2 + x1 * (1 - x2) + 0.1
        dx3 = -x4
        dx4 = x3
        return [dx1, dx2, dx3, dx4]

    def decomposition(self, x, w, xhat, what):
        x1, x2, x3, x4 = x
        w1 = w
        x1hat, x2hat, x3hat, x4hat = xhat
        w1hat = what

        d1 = -2 * x1 + self._a(x, xhat) + x3 + w1
        d2 = -x2 + self._b(x, xhat) + 0.1
        d3 = -x4hat
        d4 = x3

        return [d1, d2, d3, d4]

    def _a(self, x, xhat):
        x1, x2, x3, x4 = x
        x1hat, x2hat, x3hat, x4hat = xhat
        if all(a <= b for a, b in zip(x, xhat)):
            return min(x2 * (1 + x1), x2hat * (1 + x1))
        elif all(a <= b for a, b in zip(xhat, x)):
            return max(x2 * (1 + x1), x2hat * (1 + x1))

    def _b(self, x, xhat):
        x1, x2, x3, x4 = x
        x1hat, x2hat, x3hat, x4hat = xhat
        if all(a <= b for a, b in zip(x, xhat)):
            return min(x1 * (1 - x2), x1hat * (1 - x2))
        elif all(a <= b for a, b in zip(xhat, x)):
            return max(x1 * (1 - x2), x1hat * (1 - x2))


class Agent6(BaseAgent):
    def __init__(self, id):
        # super().__init__(id, code, file_name)
        self.id = id
        self.decision_logic = ControllerIR.empty()
        self.init_cont = None
        self.init_disc = None
        self.static_parameters = None
        self.uncertain_parameters = None

    def dynamics(self, x, args):
        w1, w2 = args
        x1, x2 = x

        dx1 = x1 * (1.1 + w1 - x1 - 0.1 * x2)
        dx2 = x2 * (4 + w2 - 3 * x1 - x2)
        return [dx1, dx2]

    def decomposition(self, x, w, xhat, what):
        x1, x2 = x
        w1, w2 = w
        x1hat, x2hat = xhat
        w1hat, w2hat = what

        d1 = x1 * (1.1 + w1 - x1 - 0.1 * x2hat)
        d2 = x2 * (4 + w2 - 3 * x1hat - x2)

        return [d1, d2]
