"""
This file contains reset class for DryVR
"""

import sympy

from dryvr_plus_plus.common.utils import randomPoint


class Reset:
    """
    This is class for resetting the initial set
    """

    def __init__(self, variables):
        """
        Reset class initialization function.

        Args:
            variables (list): list of varibale name
        """
        self.variables = variables

    def reset_set(self, raw_eqs_str, lower_bound, upper_bound):
        """
        Reset the initial set based on reset expressions

        Args:
            raw_eqs_str (str): reset expressions separated by ';'
            lower_bound (list): lower bound of the initial set
            upper_bound (list): upper bound of the initial set

        Returns:
            lower bound and upper bound of the initial set after reset
        """
        if not raw_eqs_str:
            return lower_bound, upper_bound

        raw_eqs = raw_eqs_str.split(';')
        lb_list = []
        ub_list = []
        for rawEqu in raw_eqs:
            lb, ub = self._handle_reset(rawEqu, lower_bound, upper_bound)
            lb_list.append(lb)
            ub_list.append(ub)

        return self._merge_result(lb_list, ub_list, lower_bound, upper_bound)

    def reset_point(self, raw_eqs_str, point):
        """
        Reset the initial point based on reset expressions

        Args:
            raw_eqs_str (str): list of reset expression
            point (list): the initial point need to be reset

        Returns:
            a point after reset
        """
        if point == [] or not point:
            return point
        lower, upper = self.reset_set(raw_eqs_str, point, point)
        return randomPoint(lower, upper)

    @staticmethod
    def _merge_result(lb_list, ub_list, lower_bound, upper_bound):
        """
        Merge the a list of reset result
        Since we allow multiple reset per transition,
        we get list of reset result, each result corresponding to one reset expression
        We need to merge all reset result together

        Args:
            lb_list (list): list of reset lower bound results
            ub_list (list): list of reset upper bound results
            lower_bound(list): original lower bound
            upper_bound(list): original upper bound

        Returns:
            Upper bound and lower bound after merge the reset result
        """
        ret_lb = list(lower_bound)
        ret_ub = list(upper_bound)

        for i in range(len(lb_list)):
            cur_lb = lb_list[i]
            cur_ub = ub_list[i]
            for j in range(len(cur_lb)):
                if cur_lb[j] != lower_bound[j]:
                    ret_lb[j] = cur_lb[j]
                if cur_ub[j] != upper_bound[j]:
                    ret_ub[j] = cur_ub[j]
        return ret_lb, ret_ub

    def _build_all_combo(self, symbols, lower_bound, upper_bound):
        """
        This function allows us to build all combination given symbols
        For example, if we have a 2-dimension set for dim A and B.
        symbols = [A,B]
        lowerBound = [1.0, 2.0]
        upperBound = [3.0, 4.0]
        Then the result should be all possible combination of the value of A and B
        result:
            [[1.0, 2.0], [3.0, 4.0], [3.0, 2.0], [1.0, 4.0]]

        Args:
            symbols (list): symbols we use to create combo
            lower_bound (list): lower bound of the set
            upper_bound (list): upper bound of the set

        Returns:
            List of combination value
        """
        if not symbols:
            return []

        cur_symbol = str(symbols[0])
        idx = self.variables.index(cur_symbol)
        lo = lower_bound[idx]
        up = upper_bound[idx]
        ret = []
        next_level = self._build_all_combo(symbols[1:], lower_bound, upper_bound)
        if next_level:
            for n in next_level:
                ret.append(n + [(cur_symbol, lo)])
                ret.append(n + [(cur_symbol, up)])
        else:
            ret.append([cur_symbol, lo])
            ret.append([cur_symbol, up])
        return ret

    def _handle_wrapped_reset(self, raw_eq, lower_bound, upper_bound):
        """
        This is a function to handle reset such as V = [0, V+1]

        Args:
            raw_eq (str): reset equation
            lower_bound (list): lower bound of the set
            upper_bound (list): upper bound of the set

        Returns:
            Upper bound and lower bound after the reset
        """
        final_equ = sympy.sympify(raw_eq)
        rhs_symbols = list(final_equ.free_symbols)
        combos = self._build_all_combo(rhs_symbols, lower_bound, upper_bound)
        min_reset = float('inf')
        max_reset = float('-inf')
        if combos:
            for combo in combos:
                if len(combo) == 2:
                    result = float(final_equ.subs(combo[0], combo[1]))
                else:
                    result = float(final_equ.subs(combo))
                min_reset = min(min_reset, float(result))
                max_reset = max(max_reset, float(result))
        else:
            min_reset = float(final_equ)
            max_reset = float(final_equ)
        return (min_reset, max_reset)

    def _handle_reset(self, raw_equ, lower_bound, upper_bound):
        """
        Handle the reset with single reset expression

        Args:
            raw_equ (str): reset equation
            lower_bound (list): lower bound of the set
            upper_bound (list): upper bound of the set

        Returns:
            Upper bound and lower bound after the reset
        """
        equ_split = raw_equ.split('=')
        lhs, rhs = equ_split[0], equ_split[1]
        target = sympy.sympify(lhs)
        # Construct the equation
        final_equ = sympy.sympify(rhs)
        if not isinstance(final_equ, list):
            rhs_symbols = list(sympy.sympify(rhs).free_symbols)
        else:
            rhs_symbols = None
        # print target, rhs_symbols
        combos = self._build_all_combo(rhs_symbols, lower_bound, upper_bound)
        # final_equ = solve(equ, target)[0]

        min_reset = float('inf')
        max_reset = float('-inf')
        if combos:
            for combo in combos:
                if len(combo) == 2:
                    result = float(final_equ.subs(combo[0], combo[1]))
                else:
                    result = float(final_equ.subs(combo))
                min_reset = min(min_reset, float(result))
                max_reset = max(max_reset, float(result))
        elif isinstance(final_equ, list):
            min_reset = min(self._handle_wrapped_reset(final_equ[0], lower_bound, upper_bound))
            max_reset = max(self._handle_wrapped_reset(final_equ[1], lower_bound, upper_bound))
        else:
            min_reset = float(final_equ)
            max_reset = float(final_equ)

        ret_lb = list(lower_bound)
        ret_ub = list(upper_bound)
        target_idx = self.variables.index(str(target))
        ret_lb[target_idx] = min_reset
        ret_ub[target_idx] = max_reset
        return ret_lb, ret_ub
