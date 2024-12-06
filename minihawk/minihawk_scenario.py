from minihawk_agent import MiniHawkAgent
from verse import Scenario, ScenarioConfig
from enum import Enum, auto 
import pyvista as pv 
import polytope as pc
import json 
import os 
script_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np 

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

    quad = MiniHawkAgent('quad')
    scenario.add_agent(quad) 

    scenario.set_init(
        [
            [[0-5.0, 75-5.0, 75-5.0, 0.0, 0.0, 0.0],
             [0+5.0, 75+5.0, 75+5.0, 0.0, 0.0, 0.0]]
        ],
        [
            (AgentMode.Default, )
        ]
    )

    traces = scenario.verify(100, 0.25)
    
    plotter = pv.Plotter()
    
    trace = traces.nodes[0].trace['quad']
    for i in range(0, len(trace), 2):
        lb = trace[i][1:4]
        ub = trace[i+1][1:4]
        plot3dRect(lb, ub, plotter, color='blue')

    with open(os.path.join(script_dir, 'static_obstacles.json'), 'r') as f:
        obstacle_data = json.load(f)
    total = 0
    for i in range(0, len(obstacle_data)):
        rect = obstacle_data[i]
        lb = np.array([rect['x_min'], -rect['y_max'], rect['z_min']])
        ub = np.array([rect['x_max'], -rect['y_min'], rect['z_max']])
        # dist_lb = np.linalg.norm(lb - pos, axis=1)
        # dist_ub = np.linalg.norm(ub - pos, axis=1)
        # if np.all(dist_lb> 100)  and np.all(dist_ub>100):
        #     continue
        if ub[2]<-75:
            continue
        if (not np.max((ub-lb))>1):
            continue
        print(i)
        lb[2] += 77.01757264137268
        ub[2] += 77.01757264137268
        plot3dRect(lb, ub, plotter, color='black', edge=True, trans=0.5)
        total+=1 
    print(total)
    plotter.show()