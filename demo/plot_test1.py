import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
values = [1, 2, 3]
fig.add_trace(go.Scatter(
    x=values,
    y=values,
    marker=dict(
        symbol='square',
        size=16,
        cmax=39,
        cmin=0,
        color=values,
        colorbar=dict(
            title="Colorbar"
        ),
        colorscale="Viridis"
    ),
    # marker_symbol='square', marker_line_color="midnightblue", marker_color="lightskyblue",
    # marker_line_width=2, marker_size=15,
    mode="markers"))
fig.show()
