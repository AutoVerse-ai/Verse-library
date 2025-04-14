import polytope as pc
import pyvista as pv
import numpy as np
import itertools 
import time 

def plot_polytope_3d(lb, ub, ax=None, color="red", trans=0.2, edge=True):
    if ax is None:
        ax = pv.Plotter()

    vertices = np.array(list(itertools.product(*zip(lb, ub))))
    # vertices = pc.extreme(poly)
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

def generate_rectangles(
    init_lb = np.array([-1,-1,-1]),
    init_ub = np.array([1,1,1]),
    step = 0.05,
    nums = 10000
):
    res = []
    for i in range(nums):
        res.append([init_lb+step*i, init_ub+step*i])
    return [[[-19.66666666666667, 1983.7948713721926, -1.261882267474506], [116.255273947809, 2119.6666666666665, 1.0]], 
[[-19.6666666626099, 1985.933333337213, -1.0], [115.6666666626099, 2122.0666666627853, 1.0]], 
[[-18.50172926904947, 1979.9722381634679, -2.4032613649541528], [116.87728455265028, 2116.106067469855, -0.27240932460296774]], 
[[-19.6666666626099, 1989.9333333372124, -1.0], [115.6666666626099, 2126.0666666627835, 1.0]], 
[[-18.532815735748905, 1975.972820217436, -2.5548572049939593], [116.80715202306835, 2112.109834923588, -0.5479419065759259]], 
[[-19.6666666626099, 1993.9333333372117, -1.0], [115.6666666626099, 2130.0666666627812, 1.0]], 
[[-18.50432285427152, 1971.9729609459007, -2.5620267635098557], [116.83998322399884, 2108.109939336325, -0.5611430647150912]], 
[[-19.6666666626099, 1997.933333337211, -1.0], [115.6666666626099, 2134.0666666627794, 1.0]], 
[[-18.470739436141102, 1967.9731223672086, -2.5623495993098606], [116.87577111090948, 2104.110078461039, -0.5617480461209943]], 
[[-19.6666666626099, 2001.9333333372103, -1.0], [115.6666666626099, 2138.066666662777, 1.0]]] 

if __name__ == "__main__":
    rects = generate_rectangles()

    fig = pv.Plotter()
    start_time = time.time()
    vertices = []
    for i in range(len(rects)):
        lb = rects[i][0]
        ub = rects[i][1]
        # tmp = list(itertools.product(*zip(lb, ub)))
        x0, x1 = lb[0], ub[0]
        y0, y1 = lb[1], ub[1]
        z0, z1 = lb[2], ub[2]

        corners = [
            (x0, y0, z0),  # index 0
            (x1, y0, z0),  # index 1
            (x1, y1, z0),  # index 2
            (x0, y1, z0),  # index 3,
            (x0, y0, z1),  # index 4
            (x1, y0, z1),  # index 5
            (x1, y1, z1),  # index 6
            (x0, y1, z1),  # index 7
        ]
        vertices = vertices+corners
    print(time.time()-start_time)
    start_time = time.time()
    vertices = np.array(vertices)

    print(vertices.shape)

    N = len(rects)
    cells = np.zeros((N, 9), dtype=np.int64)
    cells[:, 0] = 8

    cell_indices = np.arange(N * 8).reshape(N, 8)
    cells[:, 1:] = cell_indices

    cells = cells.ravel()

    celltypes = np.full(N, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, vertices)

    fig.add_mesh(grid, show_edges=True)

    # cloud = pv.PolyData(vertices)
    # volume = cloud.delaunay_3d()
    # shell = volume.extract_geometry()
    # fig.add_mesh(shell, opacity=0.2, color="red")
    print(time.time()-start_time)
    start_time = time.time()
    fig.render()
    fig.show()

    print(time.time()-start_time)
