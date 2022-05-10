"""
This file contains initial set class for DryVR
"""


class InitialSet:
    """
    This is class to represent the initial set
    """

    def __init__(self, lower, upper):
        """
        Initial set class initialization function.

        Args:
            lower (list): lower bound of the initial set
            upper (list): upper bound of the initial set
        """

        self.lower_bound = lower
        self.upper_bound = upper
        self.delta = [(upper[i] - lower[i]) / 2.0 for i in range(len(upper))]
        # Child point points to children InitialSetStack obj
        # This it how it works
        # Since a initial set can generate a reach tube that intersect
        # with different guards
        # So there can be multiple children pointers
        # Therefore this is a dictionary obj
        # self.child["MODEA"] = InitialSetStack for MODEA
        self.child = {}
        self.bloated_tube = []

    def refine(self):
        """
        This function refine the current initial set into two smaller set

        Returns:
            two refined initial set

        """
        # Refine the initial set into two smaller set
        # based on index with largest delta
        idx = self.delta.index(max(self.delta))
        # Construct first smaller initial set
        init_set_one_ub = list(self.upper_bound)
        init_set_one_lb = list(self.lower_bound)
        init_set_one_lb[idx] += self.delta[idx]
        # Construct second smaller initial set
        init_set_two_ub = list(self.upper_bound)
        init_set_two_lb = list(self.lower_bound)
        init_set_two_ub[idx] -= self.delta[idx]

        return (
            InitialSet(init_set_one_lb, init_set_one_ub),
            InitialSet(init_set_two_lb, init_set_two_ub),
        )

    def __str__(self):
        """
        Build string representation for the initial set

        Returns:
            A string describes the initial set
        """
        ret = ""
        ret += "Lower Bound: " + str(self.lower_bound) + "\n"
        ret += "Upper Bound: " + str(self.upper_bound) + "\n"
        ret += "Delta: " + str(self.delta) + "\n"
        return ret
