"""
This file contains guard class for DryVR
"""

import random

import sympy
from z3 import *

from dryvr_plus_plus.common.utils import handleReplace


class Guard:
    """
    This is class to calculate the set in the 
    reach tube that intersect with the guard
    """

    def __init__(self, variables):
        """
        Guard checker class initialization function.

        Args:
            variables (list): list of variable name
        """
        self.varDic = {'t': Real('t')}
        self.variables = variables
        for var in variables:
            self.varDic[var] = Real(var)

    def _build_guard(self, guard_str):
        """
        Build solver for current guard based on guard string

        Args:
            guard_str (str): the guard string.
            For example:"And(v>=40-0.1*u, v-40+0.1*u<=0)"

        Returns:
            A Z3 Solver obj that check for guard.
            A symbol index dic obj that indicates the index
            of variables that involved in the guard.
        """
        cur_solver = Solver()
        # This magic line here is because SymPy will evaluate == to be False
        # Therefore we are not be able to get free symbols from it
        # Thus we need to replace "==" to something else
        sympy_guard_str = guard_str.replace("==", ">=")

        symbols = list(sympy.sympify(
            sympy_guard_str, evaluate=False).free_symbols)
        symbols = [str(s) for s in symbols]
        symbols_idx = {s: self.variables.index(
            s) + 1 for s in symbols if s in self.variables}
        if 't' in symbols:
            symbols_idx['t'] = 0

        guard_str = handleReplace(guard_str, list(self.varDic.keys()))
        # TODO use an object instead of `eval` a string
        cur_solver.add(eval(guard_str))
        return cur_solver, symbols_idx

    def guard_sim_trace(self, trace, guard_str):
        """
        Check the guard for simulation trace.
        Note we treat the simulation trace as the set as well.
        Consider we have a simulation trace as following
        [0.0, 1.0, 1.1]
        [0.1, 1.02, 1.13]
        [0.2, 1.05, 1.14]
        ...
        We can build set like
        lower_bound: [0.0, 1.0, 1.1]
        upper_bound: [0.1, 1.02, 1.13]

        lower_bound: [0.1, 1.02, 1.13]
        upper_bound: [0.2, 1.05, 1.14]
        And check guard for these set. This is because if the guard
        is too small, it is likely for simulation point ignored the guard.
        For example:
            .     .     .     . |guard| .    .   .
            In this case, the guard gets ignored

        Args:
            trace (list): the simulation trace
            guard_str (str): the guard string.
            For example:"And(v>=40-0.1*u, v-40+0.1*u<=0)"

        Returns:
            A initial point for next mode,
            The truncated simulation trace
        """
        if not guard_str:
            return None, trace
        cur_solver, symbols = self._build_guard(guard_str)
        guard_set = {}

        for idx in range(len(trace) - 1):
            lower = trace[idx]
            upper = trace[idx + 1]
            cur_solver.push()
            for symbol in symbols:
                cur_solver.add(self.varDic[symbol] >= min(
                    lower[symbols[symbol]], upper[symbols[symbol]]))
                cur_solver.add(self.varDic[symbol] <= max(
                    lower[symbols[symbol]], upper[symbols[symbol]]))
            if cur_solver.check() == sat:
                cur_solver.pop()
                guard_set[idx] = upper
            else:
                cur_solver.pop()
                if guard_set:
                    # Guard set is not empty, randomly pick one and return
                    # idx, point = random.choice(list(guard_set.items()))
                    idx, point = list(guard_set.items())[0]
                    # Return the initial point for next mode, and truncated trace
                    return point[1:], trace[:idx + 1]

        if guard_set:
            # Guard set is not empty, randomly pick one and return
            # idx, point = random.choice(list(guard_set.items()))
            idx, point = list(guard_set.items())[0]
            # Return the initial point for next mode, and truncated trace
            return point[1:], trace[:idx + 1]

        # No guard hits for current tube
        return None, trace

    def guard_sim_trace_time(self, trace, guard_str):
        """
        Return the length of the truncated traces

        Args:
            trace (list): the simulation trace
            guard_str (str): the guard string.
            For example:"And(v>=40-0.1*u, v-40+0.1*u<=0)"

        Returns:
            the length of the truncated traces.
        """
        next_init, trace = self.guard_sim_trace(trace, guard_str)
        return len(trace)

    def guard_reachtube(self, tube, guard_str):
        """
        Check the guard intersection of the reach tube


        Args:
            tube (list): the reach tube
            guard_str (str): the guard string.
            For example:"And(v>=40-0.1*u, v-40+0.1*u<=0)"

        Returns:
            Next mode initial set represent as [upper_bound, lower_bound],
            Truncated tube before the guard,
            The time when elapsed in current mode.

        """
        if not guard_str:
            return None, tube

        cur_solver, symbols = self._build_guard(guard_str)
        guard_set_lower = []
        guard_set_upper = []
        for i in range(0, len(tube), 2):
            cur_solver.push()
            lower_bound = tube[i]
            upper_bound = tube[i + 1]
            for symbol in symbols:
                cur_solver.add(self.varDic[symbol] >=
                               lower_bound[symbols[symbol]])
                cur_solver.add(self.varDic[symbol] <=
                               upper_bound[symbols[symbol]])
            if cur_solver.check() == sat:
                # The reachtube hits the guard
                cur_solver.pop()
                guard_set_lower.append(lower_bound)
                guard_set_upper.append(upper_bound)

                tmp_solver = Solver()
                tmp_solver.add(Not(cur_solver.assertions()[0]))
                for symbol in symbols:
                    tmp_solver.add(
                        self.varDic[symbol] >= lower_bound[symbols[symbol]])
                    tmp_solver.add(
                        self.varDic[symbol] <= upper_bound[symbols[symbol]])
                if tmp_solver.check() == unsat:
                    print("Full intersect, break")
                    break
            else:
                cur_solver.pop()
                if guard_set_lower:
                    # Guard set is not empty, build the next initial set and return
                    # At some point we might further reduce the initial set for next mode
                    init_lower = guard_set_lower[0][1:]
                    init_upper = guard_set_upper[0][1:]
                    for j in range(1, len(guard_set_lower)):
                        for k in range(1, len(guard_set_lower[0])):
                            init_lower[k - 1] = min(init_lower[k - 1],
                                                    guard_set_lower[j][k])
                            init_upper[k - 1] = max(init_upper[k - 1],
                                                    guard_set_upper[j][k])
                    # Return next initial Set, the result tube, and the true transit time
                    return [init_lower, init_upper], tube[:i], guard_set_lower[0][0]

        # Construct the guard if all later trace sat the guard condition
        if guard_set_lower:
            # Guard set is not empty, build the next initial set and return
            # At some point we might further reduce the initial set for next mode
            init_lower = guard_set_lower[0][1:]
            init_upper = guard_set_upper[0][1:]
            for j in range(1, len(guard_set_lower)):
                for k in range(1, len(guard_set_lower[0])):
                    init_lower[k - 1] = min(init_lower[k - 1], guard_set_lower[j][k])
                    init_upper[k - 1] = max(init_upper[k - 1], guard_set_upper[j][k])
            # init_upper[0] = init_lower[0]

            # Return next initial Set, the result tube, and the true transit time
            return [init_lower, init_upper], tube[:i], guard_set_lower[0][0]

        return None, tube, tube[-1][0]
