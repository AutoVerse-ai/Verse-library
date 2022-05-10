"""
This file contains uniform checker class for DryVR
"""

from src.common.utils import handleReplace, neg
from z3 import *


class GoalChecker:
    """
    This is class to check if the goal set is reached
    by reach tube
    """

    def __init__(self, goal, variables):
        """
        Goal checker class initialization function.

        Args:
            goal (str): a str describing the goal set.
            For example: "And(x_1>=19.5, x_1<=20.5, x_2>=-1.0, x_2<=1.0)"
            variables (list): list of variable name
        """
        self.varDic = {'t': Real('t')}
        self.variables = variables
        for var in variables:
            self.varDic[var] = Real(var)

        goal = handleReplace(goal, list(self.varDic.keys()))

        self.intersectChecker = Solver()
        self.containChecker = Solver()

        self.intersectChecker.add(eval(goal))
        self.containChecker.add(eval(neg(goal)))

    def goal_reachtube(self, tube):
        """
        Check if the reach tube satisfied the goal

        Args:
            tube (list): the reach tube.

        Returns:
            A bool indicates if the goal is reached
            The truncated tube if the goal is reached, otherwise the whole tube
        """
        for i in range(0, len(tube), 2):
            lower = tube[i]
            upper = tube[i + 1]
            if self._check_intersection(lower, upper):
                if self._check_containment(lower, upper):
                    return True, tube[:i + 2]
        return False, tube

    def _check_intersection(self, lower, upper):
        """
        Check if the goal set intersect with the current set
        #FIXME Maybe this is not necessary since we only want to check
        the fully contained case
        Bolun 02/13/2018

        Args:
            lower (list): the list represent the set's lowerbound.
            upper (list): the list represent the set's upperbound.

        Returns:
            A bool indicates if the set intersect with the goal set
        """
        cur_solver = self.intersectChecker
        cur_solver.push()
        cur_solver.add(self.varDic["t"] >= lower[0])
        cur_solver.add(self.varDic["t"] <= upper[0])
        for i in range(1, len(lower)):
            cur_solver.add(self.varDic[self.variables[i - 1]] >= lower[i])
            cur_solver.add(self.varDic[self.variables[i - 1]] <= upper[i])
        if cur_solver.check() == sat:
            cur_solver.pop()
            return True
        else:
            cur_solver.pop()
            return False

    def _check_containment(self, lower, upper):
        """
        Check if the current set contained in goal set.

        Args:
            lower (list): the list represent the set's lowerbound.
            upper (list): the list represent the set's upperbound.

        Returns:
            A bool indicates if the set if contained in the goal set
        """
        cur_solver = self.containChecker
        cur_solver.push()
        cur_solver.add(self.varDic["t"] >= lower[0])
        cur_solver.add(self.varDic["t"] <= upper[0])
        for i in range(1, len(lower)):
            cur_solver.add(self.varDic[self.variables[i - 1]] >= lower[i])
            cur_solver.add(self.varDic[self.variables[i - 1]] <= upper[i])
        if cur_solver.check() == sat:
            cur_solver.pop()
            return False
        else:
            cur_solver.pop()
            return True
