# Plot polytope in 3d

import numpy as np
import polytope as pc
import pyvista as pv

from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode

import vtk

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)


def plot3dReachtubeSingle(tube, x_dim, y_dim, z_dim, ax, color, edge=True):
    for lb, ub in tube:
        box = [[lb[x_dim], lb[y_dim], lb[z_dim]], [ub[x_dim], ub[y_dim], ub[z_dim]]]
        poly = pc.box2poly(np.array(box).T)
        ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge)
    return ax


def plot3dMap(lane_map, color="k", ax=None, width=0.1, num=20):
    if ax is None:
        ax = pv.Plotter()
    for lane_idx in lane_map.lane_dict:
        lane = lane_map.lane_dict[lane_idx]
        if lane.plotted:
            for lane_seg in lane.segment_list:
                # if lane_seg.type == 'Straight':
                oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                points = np.vstack((oc_x, oc_y, oc_z)).T
                spline = pv.Spline(points, 400)
                tube = spline.tube(radius=width)
                ax.add_mesh(tube, color=color)
                # elif lane_seg.type == 'Circular':
                #     oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)

    return ax


def plot3dReachtube(root, agent_id, x_dim, y_dim, z_dim, color="b", ax=None, edge=True):
    if isinstance(root, AnalysisTree):
        root = root.root

    if ax is None:
        ax = pv.Plotter()

    queue = [root]
    idx = 0
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0, len(trace), 2):
            data.append([trace[i], trace[i + 1]])
        # for key in sorted(node.lower_bound):
        #     lower_bound.append(node.lower_bound[key])
        # for key in sorted(node.upper_bound):
        #     upper_bound.append(node.upper_bound[key])
        ax = plot3dReachtubeSingle(data, x_dim, y_dim, z_dim, ax, color, edge=edge)
        queue += node.child
        idx += 1

    return ax
    # for i in range(min(len(lower_bound), len(upper_bound))):
    #     lb = list(map(float, lower_bound[i]))
    #     ub = list(map(float, upper_bound[i]))

    #     box = [[lb[x_dim], lb[y_dim], lb[z_dim]],[ub[x_dim], ub[y_dim], ub[z_dim]]]
    #     poly = pc.box2poly(np.array(box).T)
    #     plot_polytope_3d(poly.A, poly.b, ax = ax, color = '#b3de69')


def plot_polytope_3d(A, b, ax=None, color="red", trans=0.2, edge=True):
    if ax is None:
        ax = pv.Plotter()

    poly = pc.Polytope(A=A, b=b)
    vertices = pc.extreme(poly)
    cloud = pv.PolyData(vertices)
    volume = cloud.delaunay_3d()
    shell = volume.extract_geometry()
    if len(shell.points) <= 0:
        return ax
    ax.add_mesh(shell, opacity=trans, color=color)
    if edge:
        edges = shell.extract_feature_edges(20)
        ax.add_mesh(edges, color="k", line_width=0.1, opacity=0.5)
    return ax


def plot_line_3d(start, end, ax=None, color="blue", line_width=1):
    if ax is None:
        ax = pv.Plotter()

    a = start
    b = end

    # Preview how this line intersects this mesh
    line = pv.Line(a, b)

    ax.add_mesh(line, color=color, line_width=line_width)
    return ax


def plot_point_3d(points, ax=None, color="blue", point_size=100):
    if ax is None:
        ax = pv.Plotter()
    ax.add_points(points, render_points_as_spheres=True, point_size=point_size, color=color)
    return ax


if __name__ == "__main__":
    A = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
    b = np.array([[1], [1], [1], [1], [1], [1]])
    b2 = np.array([[-1], [2], [-1], [2], [-1], [2]])
    fig = pv.Plotter()
    fig = plot_polytope_3d(A, b, ax=fig, color="red")
    fig = plot_polytope_3d(A, b2, ax=fig, color="green")
    fig.show()
