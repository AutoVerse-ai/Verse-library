from minihawk_agent import MiniHawkAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
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
    scenario = Scenario(ScenarioConfig(
        parallel=False, 
        reachability_method=ReachabilityMethod.DRYVR_DISC
    ))

    # output_folder = sys.argv[1]
    output_folder = './Scenario-02/'

    data_dir = os.path.join(script_dir, output_folder)
    # with open(os.path.join(data_dir, 'vnv_scitech.yml'), 'r') as f:
    #     config = yaml.full_load(f)

    # init_x = config['ego_vehicle']['location']['x']
    # init_y = config['ego_vehicle']['location']['y']
    # init_z = config['ego_vehicle']['location']['z']

    quad = MiniHawkAgent('quad', folder_name=data_dir)
    scenario.add_agent(quad) 

    all_z = quad.all_traces[:,:,3]
    z_mean = np.mean(all_z, axis=0)
    z_diff = np.linalg.norm(all_z-z_mean, axis=1)
    idx = np.argsort(z_diff)
    # z_max = np.where(z_diff)[0]
    quad.all_traces = np.delete(quad.all_traces, idx[-3:], axis=0)
    quad.generate_nominal_trace()

    traces = quad.all_traces
    inits = traces[:,0,1:]
    init_mean = np.mean(inits,axis=0)
    # init_idx = np.argmin(np.linalg.norm(inits-init_mean, axis=1))
    # init_c = inits[init_idx]
    init_r = np.max(np.abs(init_mean-inits), axis=0)
    scenario.set_init(
        [
            [[init_mean[0]-init_r[0], init_mean[1]-init_r[1], init_mean[2]-init_r[2]],
             [init_mean[0]+init_r[0], init_mean[1]+init_r[1], init_mean[2]+init_r[2]]]
        ],
        [
            (AgentMode.Default, )
        ]
    )

    traces = scenario.verify(
        100, 
        0.25,
        # params={'bloating_method':'GLOBAL'}
    )
    
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
        # if np.linalg.norm(quad.all_traces[i,0,1:]-init_c)<=0.01:
        #     plt.figure(0)
        #     plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,1],'g')
        #     plt.figure(1)
        #     plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,2],'g')
        #     plt.figure(2)
        #     plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,3],'g')
        # else:
        plt.figure(0)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,1],'b')
        plt.figure(1)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,2],'b')
        plt.figure(2)
        plt.plot(quad.all_traces[i,:,0],quad.all_traces[i,:,3],'b')

    plt.figure(0)
    plt.plot(quad.nominal_trace[:,0],quad.nominal_trace[:,1],'g')
    plt.figure(1)
    plt.plot(quad.nominal_trace[:,0],quad.nominal_trace[:,2],'g')
    plt.figure(2)
    plt.plot(quad.nominal_trace[:,0],quad.nominal_trace[:,3],'g')


    trace = np.array(traces.nodes[0].trace['quad'])
    plt.figure(0)
    plt.plot(trace[::2,0], trace[::2,1], 'r')
    plt.plot(trace[::2,0], trace[1::2,1], 'r')
    plt.title('x')
    plt.figure(1)
    plt.plot(trace[::2,0], trace[::2,2], 'r')
    plt.plot(trace[::2,0], trace[1::2,2], 'r')
    plt.title('y')
    plt.figure(2)
    plt.plot(trace[::2,0], trace[::2,3], 'r')
    plt.plot(trace[::2,0], trace[1::2,3], 'r')
    plt.title('z')
    plt.show()
    plotter = pv.Plotter()

    trace_array = np.array(traces.nodes[0].trace['quad'])
    trace_lb = trace_array[::2,1:4]
    trace_ub = trace_array[1::2,1:4]

    with open(os.path.join(data_dir, 'static_obstacles.json'),'r') as f:
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

    # plotter = pv.Plotter()
    obs_intersect = np.zeros(len(obstacle_data_removed))
    # for i, name in enumerate(os.listdir(output_folder)):
    #     if name.startswith('extracted'):
    #         df = pd.read_csv(os.path.join(output_folder, name,'./_minihawk_pose.csv'))
    #         tx = np.array(df['tx'])
    #         ty = np.array(df['ty'])
    #         tz = np.array(df['tz'])
    #         pos = np.zeros((tx.shape[0], 3))
    #         pos[:,0] = tx
    #         pos[:,1] = ty
    #         pos[:,2] = tz
    #         idx_start= 0
    #         prev_intersect = False
    for j in range(trace_lb.shape[0]):
        if np.any(
            (np.all(obs_lb<=trace_lb, axis=1) & np.all(trace_ub<=obs_ub, axis=1)) |
            (np.all(obs_lb<=trace_ub, axis=1) & np.all(trace_ub<=obs_ub, axis=1)) |
            (np.all(trace_lb<=obs_lb, axis=1) & np.all(obs_lb<=trace_ub, axis=1)) | 
            (np.all(trace_lb<=obs_ub, axis=1) & np.all(obs_ub<=trace_ub, axis=1)) 
        ):
            obs_idx = np.where(
                (np.all(obs_lb<=trace_lb, axis=1) & np.all(trace_ub<=obs_ub, axis=1)) |
                (np.all(obs_lb<=trace_ub, axis=1) & np.all(trace_ub<=obs_ub, axis=1)) |
                (np.all(trace_lb<=obs_lb, axis=1) & np.all(obs_lb<=trace_ub, axis=1)) | 
                (np.all(trace_lb<=obs_ub, axis=1) & np.all(obs_ub<=trace_ub, axis=1)) 
            )[0]
            obs_intersect[obs_idx] = 1
            plot3dRect(trace_lb, trace_ub, plotter, color='red')
        else:
            plot3dRect(trace_lb, trace_ub, plotter, color='blue')
        # if j>620:
        #     print("stop")
        # Condition for intersection
        # if np.any(np.all(obs_lb<=pos[j,:],axis=1) & np.all(pos[j,:]<=obs_ub, axis=1)):
        #     obs_idx = np.where(np.all(obs_lb<=pos[j,:],axis=1) & np.all(pos[j,:]<=obs_ub, axis=1))[0]
        #     # print(f"unsafe: {obs_idx}")
        #     obs_intersect[obs_idx] = 1
        #     if prev_intersect == False:
        #         trajectory = pv.lines_from_points(pos[idx_start:j])
        #         plotter.add_mesh(trajectory, color='blue', line_width=3)
        #         idx_start = j-1
        #     prev_intersect = True
        # else:
        #     if prev_intersect == True:
        #         trajectory = pv.lines_from_points(pos[idx_start:j])
        #         plotter.add_mesh(trajectory, color='red', line_width=3)
        #         idx_start = j-1 
        #     prev_intersect = False 

    # if idx_start < pos.shape[0]:
    #     color = 'red' if prev_intersect else 'blue'
    #     trajectory = pv.lines_from_points(pos[idx_start:])
    #     if prev_intersect:
    #         plotter.add_mesh(trajectory, color='red', line_width=3)
    #     else:
    #         plotter.add_mesh(trajectory, color='blue', line_width=3)        

    min_ub = np.inf
    for i in range(0, len(obstacle_data_removed)):
        rect = obstacle_data_removed[i]
        lb = np.array([rect['x_min'], rect['y_min'], rect['z_min']])
        ub = np.array([rect['x_max'], rect['y_max'], rect['z_max']])
        # dist_lb = np.linalg.norm(lb - pos, axis=1)
        # dist_ub = np.linalg.norm(ub - pos, axis=1)
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

    # trace = traces.nodes[0].trace['quad']
    # for i in range(0, len(trace), 2):
    #     lb = trace[i][1:4]
    #     ub = trace[i+1][1:4]
    #     plot3dRect(lb, ub, plotter, color='blue')

    # data_dir = os.path.join(script_dir, output_folder)
    # with open(os.path.join(data_dir, 'static_obstacles.json'), 'r') as f:
    #     obstacle_data = json.load(f)
    # total = 0
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