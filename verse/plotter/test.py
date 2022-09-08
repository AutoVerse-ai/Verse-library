import plotly.graph_objects as go
import numpy as np
from math import pi, cos, sin, acos, asin
from plotly.subplots import make_subplots
from enum import Enum, auto
# import time
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map_3d import LaneMap_3d

start = np.array([1, 1, 1])
end = np.array([2, 3, 4])
width = 10
length = np.linalg.norm(end - start)
n = (end-start)/length
l1 = (n[0]**2+n[1]**2)**0.5
l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
a = np.array([n[1]/l1, -n[0]/l1, 0])
b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
r = width/2

num = 3

thetas, ls = np.mgrid[0:2*pi:num*1j, 0:length:num*1j]
xnew = start[0]+ls*n[0]+r*(a[0]*np.cos(thetas)+b[0]*np.sin(thetas))
ynew = start[1]+ls*n[1]+r*(a[1]*np.cos(thetas)+b[1]*np.sin(thetas))
znew = start[2]+ls*n[2]+r*(a[2]*np.cos(thetas)+b[2]*np.sin(thetas))

print(xnew)
print(ynew)
print(znew)

oc = np.zeros((num, num, 3))
oc[:, :, 0] = xnew
oc[:, :, 1] = ynew
oc[:, :, 2] = znew

print(oc)
print(oc-start)
# fig = go.Figure()
# fig.add_trace(go.Surface(x=xnew, y=ynew, z=znew, opacity=0.5,
#               colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,255,255)']]))
# fig.update_traces(showscale=False)

# fig.show()
