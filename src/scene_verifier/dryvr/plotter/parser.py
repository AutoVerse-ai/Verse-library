"""
This file consist parser code for DryVR reachtube output
"""

from src.plotter.linkednode import LinkedNode


def parse(data):
    init_node = None
    prev_node = None
    cur_node = None
    lower_bound = {}
    upper_bound = {}
    y_min = [float("inf") for _ in range(len(data[2].strip().split()))]
    y_max = [float("-inf") for _ in range(len(data[2].strip().split()))]

    for line in data:
        # This is a mode indicator
        if ',' in line or '->' in line or line.strip().isalpha() or len(line.strip()) == 1:
            insert_data(cur_node, lower_bound, upper_bound)
            # There is new a transition
            if '->' in line:
                mode_list = line.strip().split('->')
                prev_node = init_node
                for i in range(1, len(mode_list) - 1):
                    prev_node = prev_node.child[mode_list[i]]
                cur_node = prev_node.child.setdefault(mode_list[-1], LinkedNode(mode_list[-1], line))
            else:
                cur_node = LinkedNode(line.strip(), line)
                if not init_node:
                    init_node = cur_node
                else:
                    cur_node = init_node
            # Using dictionary because we want to concat data
            lower_bound = {}
            upper_bound = {}
            LOWER = True

        else:
            line = list(map(float, line.strip().split()))
            if len(line) <= 1:
                continue
            if LOWER:
                LOWER = False
                # This data appeared in lower_bound before, concat the data
                if line[0] in lower_bound:
                    for i in range(1, len(line)):
                        lower_bound[line[0]][i] = min(lower_bound[line[0]][i], line[i])
                else:
                    lower_bound[line[0]] = line

                for i in range(len(line)):
                    y_min[i] = min(y_min[i], line[i])
            else:
                LOWER = True
                if line[0] in upper_bound:
                    for i in range(1, len(line)):
                        upper_bound[line[0]][i] = max(upper_bound[line[0]][i], line[i])
                else:
                    upper_bound[line[0]] = line

                for i in range(len(line)):
                    y_max[i] = max(y_max[i], line[i])
    insert_data(cur_node, lower_bound, upper_bound)
    return init_node, y_min, y_max


def insert_data(node, lower_bound, upper_bound):
    if not node or len(lower_bound) == 0:
        return

    for key in lower_bound:
        if key in node.lower_bound:
            for i in range(1, len(lower_bound[key])):
                node.lower_bound[key][i] = min(node.lower_bound[key][i], lower_bound[key][i])
        else:
            node.lower_bound[key] = lower_bound[key]

    for key in sorted(upper_bound):
        if key in node.upper_bound:
            for i in range(1, len(upper_bound[key])):
                node.upper_bound[key][i] = max(node.upper_bound[key][i], upper_bound[key][i])
        else:
            node.upper_bound[key] = upper_bound[key]
