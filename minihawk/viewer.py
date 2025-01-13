from verse.analysis import AnalysisTree
from verse.plotter.plotter2D import *
import plotly.graph_objects as go 

res = AnalysisTree.load('sc1.json')

fig = go.Figure()
fig = reachtube_tree(res, None, fig, 0, 1)
fig.show()