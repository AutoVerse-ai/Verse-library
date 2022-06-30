# Plotly-based Plotter Development Notes

Now the latest version is placed in plotter2D_new.py. All functions in the plotter2D.py still work. Every function are in developemnt and might change.

## Current work & Todo
- **Animation with trails** supported in simulation_anime_trail() and will be tested further.
- **Modified accelerating mode** modified, will be tested
- **new quadrotor agent** next
- **different color for segments of trace** done.

## Functions
Belows are the functions currently used. Some of the functions in the file are deprecated.

#### reachtube_anime(root, map, fig, x_dim, y_dim, map_type)

The genernal plotter for reachtube animation. It draws the all traces of reachtubes and the map. Animation is implemented as rectangles.

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.verify().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The Default value is 1.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The Default value is 2.
- **map_type** the way to draw the map. It should be 'lines' or 'fill'. For the 'lines' mode, map is only drawn by margins of lanes. For the 'fill' mode, the lanes will be filled semitransparent colors.

#### draw_reachtube_tree_v2(root, agent_id, fig, x_dim, y_dim, color_id, map_type)

The genernal static plotter for reachtube tree. It draws the all traces of reachtubes and the map.
The original version is implemented with rectangle and very inefficient.

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.verify().
- **agent_id:** the id of target agent. It should a string, which is the id/name of agent.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The Default value is 1.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The Default value is 2.
- **color_id:** a int indicating the color. Now 10 kinds of colors are supported. If it is None, the colors of segments will be auto-assigned. The default value is None.
- **map_type** the way to draw the map. It should be 'lines' or 'fill' or 'detailed'. For the 'lines' mode, map is only drawn by margins of lanes. For the 'fill' mode, the lanes will be filled semitransparent colors. For the 'detailed' mode, it will vistualize the speed limit information (if exists) by fill the lanes with different colors. The Default value is 'lines'.

#### draw_map(map, color, fig, fill_type)

The genernal static plotter for map. It is called in many functions drawing traces, so it is often unnecessary to call it separately.

**parameters:**
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **color** the color of the margin of the lanes, should be a string like 'black' or in rgb/rgba format, like 'rgb(0,0,0)' or 'rgba(0,0,0,1)'. The default value is 'rgba(0,0,0,1)' which is non-transparent black.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **fill_type** the way to draw the map. It should be 'lines' or 'fill'. For the 'lines' mode, map is only drawn by margins of lanes. For the 'fill' mode, the lanes will be filled semitransparent colors.

#### plotly_map(map, color, fig):

The old ungenernal static plotter for map which support visualization of speed limit of lanes which is a pending feature.

**parameters:**
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py. It doesn't handle the map with wrong format. Only SimpleMap3_v2() class is supported now.
- **color** the color of the margin of the lanes, should be a string like 'black' or in rgb/rgba format, like 'rgb(0,0,0)' or 'rgba(0,0,0,1)'. The default value is 'rgba(0,0,0,1)' which is non-transparent black.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().

#### simulation_tree(root, map, fig, x_dim, y_dim, map_type, scale_type):

The genernal static plotter for simulation trees. It draws the traces of agents and map.

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The Default value is 1.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The Default value is 2.
- **map_type** the way to draw the map. It should be 'lines' or 'fill'. For the 'lines' mode, map is only drawn by margins of lanes. For the 'fill' mode, the lanes will be filled semitransparent colors.
- **scale_type** the way to draw the map. It should be 'trace' or 'map'. For the 'trace' mode, the plot will be scaled to show all traces. For the 'map' mode, the plot will be scaled to show the whole map. The Default value is 'trace'.

#### simulation_tree_single(root, agent_id, x_dim, y_dim, color_id, fig):

The genernal static plotter for simulation tree. It draws the  traces of one specific agent.

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.simulate().
- **agent_id:** the id of target agent. It should a string, which is the id/name of agent.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The Default value is 1.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The Default value is 2.
- **color_id:** a int indicating the color. Now 10 kinds of colors are supported. If it is None, the colors of segments will be auto-assigned. The default value is None.

#### draw_simulation_anime(root, map, fig)

The old ungenernal plotter for simulation animation. It draws the all traces and the map. Animation is implemented as points and arrows. 

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py. It doesn't handle the map with wrong format. Only SimpleMap3_v2() class is supported now.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().

#### simulation_anime(root, map, fig, x_dim, y_dim, map_type, scale_type):

The genernal plotter for simulation animation. It draws the all traces and the map. Animation is implemented as points. Since arrow is hard to generalize.

**parameters:**
- **root:** the root node of the trace, should be the return value of Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The Default value is 1.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The Default value is 2.
- **map_type** the way to draw the map. It should be 'lines' or 'fill'. For the 'lines' mode, map is only drawn by margins of lanes. For the 'fill' mode, the lanes will be filled semitransparent colors. The Default value is 'lines'.
- **scale_type** the way to draw the map. It should be 'trace' or 'map'. For the 'trace' mode, the plot will be scaled to show all traces. For the 'map' mode, the plot will be scaled to show the whole map. The Default value is 'trace'.

