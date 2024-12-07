import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import pyvista as pv 
import json 
script_dir = os.path.dirname(os.path.realpath(__file__))
import polytope as pc
import sys 

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
        ax.add_mesh(edges, color="k", line_width=0.1, opacity=trans)
    return ax

def plot3dRect(lb, ub, ax, color, edge=True, trans=0.2):
    box = [lb, ub]
    poly = pc.box2poly(np.array(box).T)
    ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge, trans=trans)
    return ax

# output_folder = sys.argv[1]
output_folder = 'Scenario-02/'
output_folder = os.path.join(script_dir, output_folder)

with open(os.path.join(script_dir, 'static_obstacles_GUAM_below.json'), 'r') as f:
    obstacle_data = json.load(f)

tmp = []
for i in range(0, len(obstacle_data)):
    rect = obstacle_data[i]
    lb = np.array([rect['x_min'], rect['y_min'], rect['z_min']])
    ub = np.array([rect['x_max'], rect['y_max'], rect['z_max']])
    # if ub[2]<0:
    #     continue
    if lb[0] < -1000:
        continue
    # if (not np.max((ub-lb))>0.5):
    #     continue
    tmp.append(obstacle_data[i])
obstacle_data_removed = tmp

obs_lb = np.zeros((len(obstacle_data_removed), 3))
obs_ub = np.zeros((len(obstacle_data_removed), 3))
for i, obs in enumerate(obstacle_data_removed):
    obs_lb[i,:] = np.array([obs['x_min'],obs['y_min'],obs['z_min']])
    obs_ub[i,:] = np.array([obs['x_max'],obs['y_max'],obs['z_max']])

plotter = pv.Plotter()
obs_intersect = np.zeros(len(obstacle_data_removed))
for i, name in enumerate(os.listdir(output_folder)):
    if name.startswith('extracted'):
        df = pd.read_csv(os.path.join(output_folder, name,'./_minihawk_pose.csv'))
        tx = np.array(df['tx'])
        ty = np.array(df['ty'])
        tz = np.array(df['tz'])
        pos = np.zeros((tx.shape[0], 3))
        pos[:,0] = tx
        pos[:,1] = ty
        pos[:,2] = tz
        idx_start= 0
        prev_intersect = False
        for j in range(pos.shape[0]):
            # if j>620:
            #     print("stop")
            if np.any(np.all(obs_lb<=pos[j,:],axis=1) & np.all(pos[j,:]<=obs_ub, axis=1)):
                obs_idx = np.where(np.all(obs_lb<=pos[j,:],axis=1) & np.all(pos[j,:]<=obs_ub, axis=1))[0]
                # print(f"unsafe: {obs_idx}")
                obs_intersect[obs_idx] = 1
                if prev_intersect == False:
                    trajectory = pv.lines_from_points(pos[idx_start:j])
                    plotter.add_mesh(trajectory, color='blue', line_width=3)
                    idx_start = j-1
                prev_intersect = True
            else:
                if prev_intersect == True:
                    trajectory = pv.lines_from_points(pos[idx_start:j])
                    plotter.add_mesh(trajectory, color='red', line_width=3)
                    idx_start = j-1 
                prev_intersect = False 

        if idx_start < pos.shape[0]:
            color = 'red' if prev_intersect else 'blue'
            trajectory = pv.lines_from_points(pos[idx_start:])
            if prev_intersect:
                plotter.add_mesh(trajectory, color='red', line_width=3)
            else:
                plotter.add_mesh(trajectory, color='blue', line_width=3)        

min_ub = np.inf
for i in range(0, len(obstacle_data_removed)):
    rect = obstacle_data_removed[i]
    lb = np.array([rect['x_min'], rect['y_min'], rect['z_min']])
    ub = np.array([rect['x_max'], rect['y_max'], rect['z_max']])
    dist_lb = np.linalg.norm(lb - pos, axis=1)
    dist_ub = np.linalg.norm(ub - pos, axis=1)
    # if np.all(dist_lb> 100)  and np.all(dist_ub>100):
    #     continue
    # if ub[2]<0:
    #     continue
    if (not np.max((ub-lb))>0.5):
        continue
    if ub[2]<min_ub:
        min_ub = ub[2]
    print(i)
    # lb[2] += 77.01757264137268
    # ub[2] += 77.01757264137268
    if obs_intersect[i] == 1:
        plot3dRect(lb, ub, plotter, 'orange', edge=True, trans=0.3)
    else:
        plot3dRect(lb, ub, plotter, 'gray', edge=True, trans=1.0)
print(obs_lb[obs_intersect>0.5,:],obs_ub[obs_intersect>0.5,:])
plotter.show()