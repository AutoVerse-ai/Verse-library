import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import pyvista as pv 
import json 
script_dir = os.path.dirname(os.path.realpath(__file__))
import polytope as pc

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

def plot3dRect(lb, ub, ax, color, edge=True):
    box = [lb, ub]
    poly = pc.box2poly(np.array(box).T)
    ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge)
    return ax

with open(os.path.join(script_dir, 'static_obstacles.json'), 'r') as f:
    obstacle_data = json.load(f)
plotter = pv.Plotter()

for i in range(4):
    df = pd.read_csv(os.path.join(script_dir, f'extracted_{i}','./_minihawk_pose.csv'))
    tx = np.array(df['tx'])
    ty = np.array(df['ty'])
    tz = np.array(df['tz'])
    pos = np.zeros((tx.shape[0], 3))
    pos[:,0] = tx
    pos[:,1] = ty
    pos[:,2] = tz


    # print(df['tx'].shape, df['ty'].shape)
    # plt.plot(tx, ty)
    # plt.show()

    # for i in range(tx.shape[0]-1):
    #     start = np.array([tx[i], ty[i], tz[i]])
    #     end = np.array([tx[i+1], ty[i+1], tz[i+1]])
    #     line = pv.Line(start, end)
    #     plotter.add_mesh(line, color='blue', line_width=2)
    trajectory = pv.lines_from_points(pos)
    plotter.add_mesh(trajectory, color='blue', line_width=3)

min_ub = np.inf
for i in range(0, len(obstacle_data)):
    rect = obstacle_data[i]
    lb = np.array([rect['x_min'], -rect['y_max'], rect['z_min']])
    ub = np.array([rect['x_max'], -rect['y_min'], rect['z_max']])
    dist_lb = np.linalg.norm(lb - pos, axis=1)
    dist_ub = np.linalg.norm(ub - pos, axis=1)
    # if np.all(dist_lb> 100)  and np.all(dist_ub>100):
    #     continue
    if ub[2]<-75:
        continue
    if (not np.max((ub-lb))>1):
        continue
    if ub[2]<min_ub:
        min_ub = ub[2]
    print(i)
    lb[2] += 77.01757264137268
    ub[2] += 77.01757264137268
    plot3dRect(lb, ub, plotter, 'r', edge=True) 
print(min_ub)
plotter.show()