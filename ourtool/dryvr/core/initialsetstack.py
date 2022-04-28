"""
This file contains initial set class stack for DryVR
"""
import random


class InitialSetStack:
    """
    This is class is for list of initial sets for a single node
    """

    def __init__(self, mode, threshold, remain_time, start_time):
        """
        Initial set class initialization function.

        Args:
            mode (str): the mode name for current node
            threshold (int): threshold for refinement
            remain_time (float): remaining time of the current node
        """

        # Mode name for the current mode
        self.mode = mode
        # Threshold (int) for the refinement
        self.threshold = threshold
        # Pointer point to the parent node
        # If you wonder where child pointers are
        # It's actually in the InitialSet class.........
        self.parent = None
        # A stack for InitialSet object
        self.stack = []
        self.remain_time = remain_time
        # Stores bloated tube result for all tubes start from initial sets in stack
        self.bloated_tube = []
        self.start_time = start_time

    def is_valid(self):
        """
        Check if current node is still valid or not.
        (if number of initial sets in stack are more than threshold)

        Returns:
            A bool to check if the stack is valid or not valid
        """
        return len(self.stack) <= self.threshold

    def __str__(self):
        """
        Build string representation for the initial set stack

        Args:
            None
        Returns:
            A string describes the stack
        """
        ret = "===============================================\n"
        ret += "Mode: " + str(self.mode) + "\n"
        ret += "stack size: " + str(len(self.stack)) + "\n"
        ret += "remainTime: " + str(self.remain_time) + "\n"
        return ret


class GraphSearchNode:
    """
    This is class for graph search node
    Contains some helpful stuff in graph search tree
    """

    def __init__(self, mode, remain_time, min_time_thres, level):
        """
        GraphSearchNode class initialization function.

        Args:
            mode (str): the mode name for current node
            remain_time (float): remaining time of the current node
            min_time_thres (float): minimal time should stay in this mode
            level (int): tree depth
        """
        self.mode = mode
        # Pointer point to parent node
        self.parent = None
        # Initial set
        self.initial = None
        self.bloated_tube = []
        self.remain_time = remain_time
        self._min_time_thres = min_time_thres
        # Pointer point to children
        self.children = {}
        # keep track which child is visited
        self.visited = set()
        # select number of candidates initial set for children
        self.candidates = []
        self.level = level

    def random_picker(self, k):
        """
        Randomly pick k candidates initial set

        Args:
            k (int): number of candidates
        Returns:
            A list of candidate initial sets
        """
        i = 0
        ret = []
        while i < len(self.bloated_tube):
            if self.bloated_tube[i][0] >= self._min_time_thres:
                ret.append((self.bloated_tube[i], self.bloated_tube[i + 1]))
            i += 2

        random.shuffle(ret)
        return ret[:k]

    def __str__(self):
        """
        Build string representation for the Graph Search Node

        Returns:
            A string describes the stack
        """
        ret = ""
        ret += "Level: " + str(self.level) + "\n"
        ret += "Mode: " + str(self.mode) + "\n"
        ret += "Initial: " + str(self.initial) + "\n"
        ret += "visited: " + str(self.visited) + "\n"
        ret += "num candidates: " + str(len(self.candidates)) + "\n"
        ret += "remaining time: " + str(self.remain_time) + "\n"
        if self.bloated_tube:
            ret += "bloatedTube: True" + "\n"
        else:
            ret += "bloatedTube: False" + "\n"
        return ret
