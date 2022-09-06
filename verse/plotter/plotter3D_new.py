import plotly.graph_objects as go
import numpy as np
from math import pi, cos, sin, acos, asin
from plotly.subplots import make_subplots
from enum import Enum, auto
import time

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

num = 100

thetas, ls = np.mgrid[0:2*pi:num*1j, 0:length:num*1j]
xnew = start[0]+ls*n[0]+r*(a[0]*np.cos(thetas)+b[0]*np.sin(thetas))
ynew = start[1]+ls*n[1]+r*(a[1]*np.cos(thetas)+b[1]*np.sin(thetas))
znew = start[2]+ls*n[2]+r*(a[2]*np.cos(thetas)+b[2]*np.sin(thetas))

fig = go.Figure()
fig.add_trace(go.Surface(x=xnew, y=ynew, z=znew, opacity=0.5,
              colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,255,255)']]))
fig.update_traces(showscale=False)

fig.show()

# a, b, d = 1.32, 1., 0.8
# c = a**2 - b**2
# u, v = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
# x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 *
#      np.cos(u)) / (a - c * np.cos(u) * np.cos(v))
# y = b * np.sin(u) * (a - d*np.cos(v)) / (a - c * np.cos(u) * np.cos(v))
# z = b * np.sin(v) * (c*np.cos(u) - d) / (a - c * np.cos(u) * np.cos(v))

# # print(x)
# # print(y)
# # print(z)

# fig = make_subplots(rows=1, cols=2,
#                     specs=[[{'is_3d': True}, {'is_3d': True}]],
#                     subplot_titles=['Color corresponds to z',
#                                     'Color corresponds to distance to origin'],
#                     )

# fig.add_trace(go.Surface(x=x, y=y, z=z), 1, 1)
# fig.add_trace(go.Surface(x=x, y=y, z=z), 1, 2)
# fig.update_layout(title_text="Ring cyclide")
# fig.show()
