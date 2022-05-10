"""
This is a data structure to hold reachtube data per node
"""


class LinkedNode:
    def __init__(self, node_id, file_name):
        self.file_name = file_name.strip()
        self.nodeId = node_id
        self.lower_bound = {}
        self.upper_bound = {}
        self.child = {}

    def print_tube(self):
        for key in sorted(self.lower_bound):
            if key in self.upper_bound:
                print(self.upper_bound[key])
            print(self.lower_bound[key])
