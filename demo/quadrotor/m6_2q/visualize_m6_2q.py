from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
import os 
import pyvista as pv 
from verse.plotter.plotter3D import plot3dReachtube, plot3dMap, plot_polytope_3d
from verse.map.example_map.map_tacas import M6
import warnings
import sys
from typing import Dict
import polytope as pc
import numpy as np 

import vtk 
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

warnings.filterwarnings("ignore")

# def get_total_length(node:AnalysisTreeNode):
#     trace = list(node.trace.values())[0]
#     if node.child==[]:
#         return len(trace)
#     else:
#         subtree_length_list = []
#         for child_node in node.child:
#             subtree_length = get_total_length(child_node)
#             subtree_length_list.append(subtree_length)
#         max_subtree_length = max(subtree_length_list)
#         return len(trace) + max_subtree_length

def get_time_dict(tree:AnalysisTree) -> Dict:
    time_dict = {}
    node_queue = [tree.root]
    while node_queue!=[]:
        node = node_queue.pop(0)
        trace_length = len(list(node.trace.values())[0])
        for i in range(0,trace_length,2):
            drone1_trace = node.trace['test1']
            drone2_trace = node.trace['test2']
            key = round(drone1_trace[i][0],10)
            if key not in time_dict:
                time_dict[key] = [{'test1':drone1_trace[i:i+2], 'test2':drone2_trace[i:i+2]}]
            else:
                time_dict[key].append({'test1':drone1_trace[i:i+2], 'test2':drone2_trace[i:i+2]})
        node_queue += node.child
    return time_dict

script_dir = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(script_dir, 'output1.json')

traces = AnalysisTree.load(res)

tmp_map = M6()

param = 0

# total_length = get_total_length(traces.root)    
# print(total_length)
time_dict = get_time_dict(traces)

key_list = list(time_dict.keys())

for i in range(len(key_list)):
    print(i)
    fig = pv.Plotter(off_screen=True)
    fig = plot3dMap(tmp_map, ax = fig, width=0.05)
    for j in range(0,i):
        for content in time_dict[key_list[j]]:
            lb1, ub1 = content['test1']
            lb2, ub2 = content['test2']
            b1 = [[lb1[1], lb1[2], lb1[3]], [ub1[1], ub1[2], ub1[3]]]
            b2 = [[lb2[1], lb2[2], lb2[3]], [ub2[1], ub2[2], ub2[3]]]
            poly1 = pc.box2poly(np.array(b1).T)
            fig = plot_polytope_3d(poly1.A, poly1.b, ax=fig, color='r', edge=True)
            poly2 = pc.box2poly(np.array(b2).T)
            fig = plot_polytope_3d(poly2.A, poly2.b, ax=fig, color='b', edge=True)
    fig.set_background("#e0e0e0")
    fig.camera.view_angle = 20.0
    fig.camera.azimuth = 60
    fig.camera.elevation = -15
    fig.show(screenshot=f'./gif/quadrotor_{i}.png')



# fig = pv.Plotter()
# fig = plot3dMap(tmp_map, ax=fig, width=0.05)
# fig = plot3dReachtube(traces, "test1", 1, 2, 3, "r", fig, edge=True)
# fig = plot3dReachtube(traces, "test2", 1, 2, 3, "b", fig, edge=True)
# print(fig.camera.view_angle)
# print(fig.camera.elevation)
# fig.camera.view_angle = 20.0
# fig.camera.azimuth = 60
# fig.camera.elevation = -15
# fig.show()

