from minihawk_agent import MiniHawkAgent
from verse import Scenario, ScenarioConfig
from enum import Enum, auto 
import pyvista as pv 
import polytope as pc
import json 
import os 
script_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np 
import sys 
import matplotlib.pyplot as plt 
import yaml
import pandas as pd 

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

def plot3dRect(lb, ub, ax, color, trans=0.2, edge=True):
    box = [lb, ub]
    poly = pc.box2poly(np.array(box).T)
    ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge, trans=trans)
    return ax

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(parallel=False))

    data_dir = os.path.join(script_dir, sys.argv[1])
    with open(os.path.join(data_dir, 'vnv_scitech.yml'), 'r') as f:
        config = yaml.full_load(f)

    init_x = config['ego_vehicle']['location']['x']
    init_y = config['ego_vehicle']['location']['y']
    init_z = config['ego_vehicle']['location']['z']

    quad = MiniHawkAgent('quad', folder_name=data_dir)
    scenario.add_agent(quad) 

    scenario.set_init(
        [
            [[init_x-5.0, init_y-5.0, init_z-5.0, 0.0, 0.0, 0.0],
             [init_x+5.0, init_y+5.0, init_z+5.0, 0.0, 0.0, 0.0]]
        ],
        [
            (AgentMode.Default, )
        ]
    )

    traces = scenario.verify(100, 0.25)
    
    # for i, name in enumerate(os.listdir(data_dir)):
    #     if name.startswith('extracted'):
    #         df = pd.read_csv(os.path.join(data_dir, name,'./_minihawk_pose.csv'))
    #         tx = np.array(df['tx'])
    #         ty = np.array(df['ty'])
    #         tz = np.array(df['tz'])
    #         pos = np.zeros((tx.shape[0], 3))
    #         pos[:,0] = tx
    #         pos[:,1] = ty
    #         pos[:,2] = tz
            
    for i in range(quad.all_traces.shape[0]):
        plt.figure(0)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,1],'b')
        plt.figure(1)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,2],'b')
        plt.figure(2)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,3],'b')

    trace = np.array(traces.nodes[0].trace['quad'])
    plt.figure(0)
    plt.plot(trace[::2,0], trace[::2,1], 'b')
    plt.plot(trace[1::2,0], trace[1::2,1], 'b')
    plt.title('x')
    plt.figure(1)
    plt.plot(trace[::2,0], trace[::2,2], 'b')
    plt.plot(trace[1::2,0], trace[1::2,2], 'b')
    plt.title('y')
    plt.figure(2)
    plt.plot(trace[::2,0], trace[::2,3], 'b')
    plt.plot(trace[1::2,0], trace[1::2,3], 'b')
    plt.title('z')
    plt.show()
    plotter = pv.Plotter()
    
    trace = traces.nodes[0].trace['quad']
    for i in range(0, len(trace), 2):
        lb = trace[i][1:4]
        ub = trace[i+1][1:4]
        plot3dRect(lb, ub, plotter, color='blue')

    data_dir = os.path.join(script_dir, sys.argv[1])
    with open(os.path.join(data_dir, 'static_obstacles.json'), 'r') as f:
        obstacle_data = json.load(f)
    total = 0
    # for i in range(0, len(obstacle_data)):
    #     rect = obstacle_data[i]
    #     lb = np.array([rect['x_min'], rect['y_min'], rect['z_min']])
    #     ub = np.array([rect['x_max'], rect['y_max'], rect['z_max']])
    #     # dist_lb = np.linalg.norm(lb - pos, axis=1)
    #     # dist_ub = np.linalg.norm(ub - pos, axis=1)
    #     # if np.all(dist_lb> 100)  and np.all(dist_ub>100):
    #     #     continue
    #     if ub[2]<-75:
    #         continue
    #     if lb[2]<-75:
    #         continue
    #     if (not np.max((ub-lb))>0.5):
    #         continue
    #     print(i)
    #     plot3dRect(lb, ub, plotter, color='gray', edge=True, trans=1)
    #     total+=1 
    # print(total)
    plotter.show()