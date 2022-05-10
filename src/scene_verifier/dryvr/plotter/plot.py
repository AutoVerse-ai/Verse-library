"""
This file consist main plotter code for DryVR reachtube output
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue', 'yellow', 'black']


def plot(node, dim, y_min, y_max, x_dim):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot('111')
    lower_bound = []
    upper_bound = []
    for key in sorted(node.lower_bound):
        lower_bound.append(node.lower_bound[key])
    for key in sorted(node.upper_bound):
        upper_bound.append(node.upper_bound[key])

    for i in range(min(len(lower_bound), len(upper_bound))):
        lb = list(map(float, lower_bound[i]))
        ub = list(map(float, upper_bound[i]))

        for ci, d in enumerate(dim):
            rect = patches.Rectangle((lb[x_dim], lb[d]), ub[x_dim] - lb[x_dim], ub[d] - lb[d],
                                     color=colors[ci % len(colors)], alpha=0.7)
            ax1.add_patch(rect)

    y_axis_min = min([y_min[i] for i in dim])
    y_axis_max = max([y_max[i] for i in dim])
    ax1.set_title(node.nodeId, fontsize=12)
    ax1.set_ylim(bottom=y_axis_min, top=y_axis_max)
    ax1.plot()
    # TODO Eliminate hardcoded folders
    fig1.savefig('output/' + node.file_name + '.png', format='png', dpi=200)


def rrt_plot(lower_bound, upper_bound, x_dim, y_dim, goal, unsafe_list, region, initial):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot('111')
    x_min = region[0][0]
    y_min = region[0][1]
    x_max = region[1][0]
    y_max = region[1][1]

    # Draw the path
    for i in range(min(len(lower_bound), len(upper_bound))):
        lb = list(map(float, lower_bound[i]))
        ub = list(map(float, upper_bound[i]))

        rect = patches.Rectangle((lb[x_dim], lb[y_dim]), ub[x_dim] - lb[x_dim], ub[y_dim] - lb[y_dim], color='blue',
                                 alpha=0.7)
        ax1.add_patch(rect)

    # Draw the goal
    if goal:
        lb, ub = goal
        rect = patches.Rectangle((lb[0], lb[1]), ub[0] - lb[0], ub[1] - lb[1], color='green', alpha=0.7)
        ax1.add_patch(rect)

    if initial:
        lb, ub = initial
        rect = patches.Rectangle((lb[0], lb[1]), ub[0] - lb[0], ub[1] - lb[1], color='yellow', alpha=0.7)
        ax1.add_patch(rect)

    # Draw the unsafe
    if unsafe_list:
        for unsafe in unsafe_list:
            lb, ub = unsafe
            rect = patches.Rectangle((lb[0], lb[1]), ub[0] - lb[0], ub[1] - lb[1], color='red', alpha=0.7)
            ax1.add_patch(rect)

    ax1.set_title("RRT", fontsize=12)
    ax1.set_ylim(bottom=y_min, top=y_max)
    ax1.set_xlim(left=x_min, right=x_max)
    ax1.plot()
    fig1.savefig('output/rrt.png', format='png', dpi=200)
