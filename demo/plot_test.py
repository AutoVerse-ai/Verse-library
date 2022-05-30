import plotly.graph_objects as go
import numpy as np


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_rev = x[::-1]

# Line 1
y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y1_lower = y1_lower[::-1]

# Line 2
y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
y2_lower = y2_lower[::-1]

# Line 3
y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
y3_lower = y3_lower[::-1]


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x+x_rev,
    y=y1_upper+y1_lower,
    # fill='toself',
    # fillcolor='rgba(0,100,80,0.2)',
    # line_color='rgba(255,255,255,0)',
    # showlegend=False,
    name='Fair',
))
# fig.add_trace(go.Scatter(
#     x=x+x_rev,
#     y=y2_upper+y2_lower,
#     fill='toself',
#     fillcolor='rgba(0,176,246,0.2)',
#     line_color='rgba(255,255,255,0)',
#     name='Premium',
#     showlegend=False,
# ))
# fig.add_trace(go.Scatter(
#     x=x+x_rev,
#     y=y3_upper+y3_lower,
#     fill='toself',
#     fillcolor='rgba(231,107,243,0.2)',
#     line_color='rgba(255,255,255,0)',
#     showlegend=False,
#     name='Ideal',
# ))
fig.add_trace(go.Scatter(
    x=x, y=y1,
    line_color='rgb(0,100,80)',
    name='Fair',
))
# fig.add_trace(go.Scatter(
#     x=x, y=y2,
#     line_color='rgb(0,176,246)',
#     name='Premium',
# ))
# fig.add_trace(go.Scatter(
#     x=x, y=y3,
#     line_color='rgb(231,107,243)',
#     name='Ideal',
# ))

fig.update_traces(mode='lines')
fig.show()
print(x+x_rev)
print(y1_upper+y1_lower)
