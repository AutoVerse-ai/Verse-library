"""
This file contains uniform checker class for DryVR
"""
import sympy
from z3 import *

from dryvr_plus_plus.common.constant import *
from dryvr_plus_plus.common.utils import handleReplace, neg


class UniformChecker:
    """
    This is class for check unsafe checking
    """

    def __init__(self, unsafe, variables):
        """
        Reset class initialization function.

        Args:
            unsafe (str): unsafe constraint
            variables (list): list of variable name
        """
        self.varDic = {'t': Real('t')}
        self._variables = variables
        self._solver_dict = {}
        for var in variables:
            self.varDic[var] = Real(var)

        if not unsafe:
            return

        original = unsafe

        unsafe = handleReplace(unsafe, list(self.varDic.keys()))
        unsafe_list = unsafe[1:].split('@')
        for unsafe in unsafe_list:
            mode, cond = unsafe.split(':')
            self._solver_dict[mode] = [Solver(), Solver()]
            self._solver_dict[mode][0].add(eval(cond))
            self._solver_dict[mode][1].add(eval(neg(cond)))

        unsafe_list = original[1:].split('@')
        for unsafe in unsafe_list:
            mode, cond = unsafe.split(':')
            # This magic line here is because SymPy will evaluate == to be False
            # Therefore we are not be able to get free symbols from it
            # Thus we need to replace "==" to something else, which is >=
            cond = cond.replace("==", ">=")
            symbols = list(sympy.sympify(cond).free_symbols)
            symbols = [str(s) for s in symbols]
            symbols_idx = {s: self._variables.index(
                s) + 1 for s in symbols if s in self._variables}
            if 't' in symbols:
                symbols_idx['t'] = 0
            self._solver_dict[mode].append(symbols_idx)  # TODO Fix typing

    def check_sim_trace(self, traces, mode):
        """
        Check the simulation trace

        Args:
            traces (list): simulation traces
            mode (str): mode need to be checked

        Returns:
            An int for checking result SAFE = 1, UNSAFE = -1
        """
        if mode in self._solver_dict:
            cur_solver = self._solver_dict[mode][0]
            symbols = self._solver_dict[mode][2]
        elif 'Allmode' in self._solver_dict:
            cur_solver = self._solver_dict['Allmode'][0]
            symbols = self._solver_dict['Allmode'][2]
        else:
            # Return True if we do not check this mode
            return SAFE

        for t in traces:
            cur_solver.push()
            for symbol in symbols:
                cur_solver.add(self.varDic[symbol] == t[symbols[symbol]])

            if cur_solver.check() == sat:
                cur_solver.pop()
                return UNSAFE
            else:
                cur_solver.pop()
        return SAFE

    def check_reachtube(self, tube, mode):
        """
        Check the bloated reach tube

        Args:
            tube (list): reach tube
            mode (str): mode need to be checked

        Returns:
            An int for checking result SAFE = 1, UNSAFE = -1, UNKNOWN = 0
        """
        if mode not in self._solver_dict and 'Allmode' not in self._solver_dict:
            # Return True if we do not check this mode
            return SAFE

        safe = SAFE
        for i in range(0, len(tube), 2):
            lower = tube[i]
            upper = tube[i + 1]
            if self._check_intersection(lower, upper, mode):
                if self._check_containment(lower, upper, mode):
                    # The unsafe region is fully contained
                    return UNSAFE
                else:
                    # We do not know if it is truly unsafe or not
                    safe = UNKNOWN
        return safe

    def cut_tube_till_unsafe(self, tube):
        """
        Truncated the tube before it intersect with unsafe set

        Args:
            tube (list): reach tube

        Returns:
            truncated tube
        """
        if not self._solver_dict:
            return tube
        # Cut the reach tube till it intersect with unsafe
        for i in range(0, len(tube), 2):
            lower = tube[i]
            upper = tube[i + 1]
            if self._check_intersection(lower, upper, 'Allmode'):
                # we need to cut here
                return tube[:i]

        return tube

    def _check_intersection(self, lower, upper, mode):
        """
        Check if current set intersect with the unsafe set

        Args:
            lower (list): lower bound of the current set
            upper (list): upper bound of the current set
            mode (str): the mode need to be checked

        Returns:
            Return a bool to indicate if the set intersect with the unsafe set
        """
        if mode in self._solver_dict:
            cur_solver = self._solver_dict[mode][0]
            symbols = self._solver_dict[mode][2]
        elif 'Allmode' in self._solver_dict:
            cur_solver = self._solver_dict['Allmode'][0]
            symbols = self._solver_dict['Allmode'][2]
        else:
            raise ValueError("Unknown mode '" + mode + "'")

        cur_solver.push()
        for symbol in symbols:
            cur_solver.add(self.varDic[symbol] >= lower[symbols[symbol]])
            cur_solver.add(self.varDic[symbol] <= upper[symbols[symbol]])

        check_result = cur_solver.check()

        if check_result == sat:
            cur_solver.pop()
            return True
        if check_result == unknown:
            print("Z3 return unknown result")
            exit()  # TODO Proper return instead of exit
        else:
            cur_solver.pop()
            return False

    def _check_containment(self, lower, upper, mode):
        """
        Check if the current set is fully contained in unsafe region

        Args:
            lower (list): lower bound of the current set
            upper (list): upper bound of the current set
            mode (str): the mode need to be checked

        Returns:
            Return a bool to indicate if the set is fully contained in unsafe region
        """
        if mode in self._solver_dict:
            cur_solver = self._solver_dict[mode][1]
            symbols = self._solver_dict[mode][2]
        elif 'Allmode' in self._solver_dict:
            cur_solver = self._solver_dict['Allmode'][1]
            symbols = self._solver_dict['Allmode'][2]
        else:
            raise ValueError("Unknown mode '" + mode + "'")

        cur_solver.push()
        for symbol in symbols:
            cur_solver.add(self.varDic[symbol] >= lower[symbols[symbol]])
            cur_solver.add(self.varDic[symbol] <= upper[symbols[symbol]])
        check_result = cur_solver.check()

        if check_result == sat:
            cur_solver.pop()
            return False
        if check_result == unknown:
            print("Z3 return unknown result")
            exit()  # TODO Proper return instead of exit
        else:
            cur_solver.pop()
            return True
