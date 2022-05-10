"""
This file contains a distance checker class for controller synthesis
"""
from math import sqrt


class DistChecker:
    """
    This is the class to calculate the distance between
    current set and the goal set for DryVR controller synthesis.
    The distance it calculate is the euclidean distance.
    """

    def __init__(self, goal_set, variables):
        """
        Distance checker class initialization function.

        Args:
            goal_set (list): a list describing the goal set.
                [["x_1","x_2"],[19.5,-1],[20.5,1]]
                which defines a goal set
                19.5<=x_1<=20.5 && -1<=x_2<=1
            variables (list): list of variable name
        """
        var, self.lower, self.upper = goal_set
        self.idx = []
        for v in var:
            self.idx.append(variables.index(v) + 1)

    def calc_distance(self, lower_bound, upper_bound):
        """
        Calculate the euclidean distance between the
        current set and goal set.

        Args:
            lower_bound (list): the lower bound of the current set.
            upper_bound (list): the upper bound of the current set.

        Returns:
            the euclidean distance between current set and goal set

        """
        dist = 0.0
        for i in range(len(self.idx)):
            max_val = max(
                (self.lower[i] - lower_bound[self.idx[i]]) ** 2,
                (self.upper[i] - lower_bound[self.idx[i]]) ** 2,
                (self.lower[i] - upper_bound[self.idx[i]]) ** 2,
                (self.upper[i] - upper_bound[self.idx[i]]) ** 2
            )
            dist += max_val
        return sqrt(dist)
