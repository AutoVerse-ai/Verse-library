# Plotly-based Plotter Development Notes

Now the latest version of the plotter is placed in plotter2D.py. Every function is in developement and might change.

## Notes
- fix slight bugs on the slider bar
- support the display of unsafe condition
- merge animation with/without trail into one function
- resupport the feature of plotting full trace in anime

## Quick start
### Four high-level plotting functions
- **simulation_anime** gives the animation of the simulation with/without trail.
- **simulation_tree** statically shows all the traces of the simulation.
- **reachtube_anime** gives the animation of the verfication.
- **reachtube_tree** statically shows all the traces of the verfication.
### API
#### reachtube_anime(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, time_step=None, speed_rate=1, combine_rect=None)
- **root:** the root node of the trace, normally should be the return value of Scenario.verify() or Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The default value is **1**.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The default value is **2**.
- **print_dim_list** the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. The default value is **None**. And then all dimensions will be shown.
- **map_type** the way to draw the map. It should be **'lines'** or **'fill'** or **'detailed'**. The default value is **'lines'**.
  - For the **'lines'** mode, map is only drawn by margins of lanes.
  - For the **'fill'** mode, the lanes will be filled with semitransparent colors.
  - For the **'detailed'** mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). Otherwise, it is the same as the 'lines' mode.
- **scale_type** the way to scale the coordinate axises. It should be **'trace'** or **'map'**. The default value is **'trace'**.
  -  For the **'trace'** mode, the traces will be in the center of the plot with an appropriate scale.
  - For the **'map'** mode, the map will be in the center of the plot with an appropriate scale.
- **label_mode** the mode to display labels or not. if it is 'None', then labels will not be displayed. Otherwise, labels will be displayed. The default value is **'None'**.
- **sample_rate** it determines the points used in the plot. if sample_rate = n which is a positive integer, it means that the plotter sample a point within n points. it is useful when the points are too much and the response of the plot is slow. The default value is **1**.
- **time_step** it is used to determine the num of digits of time points. If it's None, then the num of digits is set as 3. Otherwise, the num of digits is set as the num of digits of the given time_step. Normally, it should be the time step of simulation/verification. The default value is **None**.
- **speed_rate** it determines the speed od anime. Due to the performance, it maybe be limited when the response of the plot is slow. The default value is **1**.
- **combine_rect** it determines the way of displaying reachtube. Specifically, it can combine specified number of reachtubes as a rectangle. The default value is **None** here, which means no combination.

#### reachtube_tree(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, combine_rect=1):
- **root:** the root node of the trace, normally should be the return value of Scenario.verify() or Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The default value is **1**.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The default value is **2**.
- **print_dim_list** the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. The default value is **None**. And then all dimensions will be shown.
- **map_type** the way to draw the map. It should be **'lines'** or **'fill'** or **'detailed'**. The default value is **'lines'**.
  - For the **'lines'** mode, map is only drawn by margins of lanes.
  - For the **'fill'** mode, the lanes will be filled with semitransparent colors.
  - For the **'detailed'** mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). Otherwise, it is the same as the 'lines' mode.
- **scale_type** the way to scale the coordinate axises. It should be **'trace'** or **'map'**. The default value is **'trace'**.
  -  For the **'trace'** mode, the traces will be in the center of the plot with an appropriate scale.
  - For the **'map'** mode, the map will be in the center of the plot with an appropriate scale.
- **label_mode** the mode to display labels or not. if it is 'None', then labels will not be displayed. Otherwise, labels will be displayed. The default value is **'None'**.
- **sample_rate** it determines the points used in the plot. if sample_rate = n which is a positive integer, it means that the plotter sample a point within n points. it is useful when the points are too much and the response of the plot is slow. The default value is **1**.
- **combine_rect** it determines the way of displaying reachtube. Specifically, it can combine specified number of reachtubes as a rectangle. The default value is **1** here.

#### simulation_tree(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1):
- **root:** the root node of the trace, normally should be the return value of Scenario.verify() or Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The default value is **1**.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The default value is **2**.
- **print_dim_list** the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. The default value is **None**. And then all dimensions will be shown.
- **map_type** the way to draw the map. It should be **'lines'** or **'fill'** or **'detailed'**. The default value is **'lines'**.
  - For the **'lines'** mode, map is only drawn by margins of lanes.
  - For the **'fill'** mode, the lanes will be filled with semitransparent colors.
  - For the **'detailed'** mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). Otherwise, it is the same as the 'lines' mode.
- **scale_type** the way to scale the coordinate axises. It should be **'trace'** or **'map'**. The default value is **'trace'**.
  -  For the **'trace'** mode, the traces will be in the center of the plot with an appropriate scale.
  - For the **'map'** mode, the map will be in the center of the plot with an appropriate scale.
- **label_mode** the mode to display labels or not. if it is 'None', then labels will not be displayed. Otherwise, labels will be displayed. The default value is **'None'**.
- **sample_rate** it determines the points used in the plot. if sample_rate = n which is a positive integer, it means that the plotter sample a point within n points. it is useful when the points are too much and the response of the plot is slow. The default value is **1**.

#### simulation_anime(root: Union[AnalysisTree, AnalysisTreeNode], map=None, fig=go.Figure(), x_dim: int = 1, y_dim: int = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, time_step=None, speed_rate=1, anime_mode='normal', full_trace=False):
- **root:** the root node of the trace, normally should be the return value of Scenario.verify() or Scenario.simulate().
- **map:** the map of the scenario, templates are in dryvr_plus_plus.example.example_map.simple_map2.py.
- **fig:** the object of the figure, its type should be plotly.graph_objects.Figure().
- **x_dim:** the dimension of x coordinate in the trace list of every time step. The default value is **1**.
- **y_dim:** the dimension of y coordinate in the trace list of every time step. The default value is **2**.
- **print_dim_list** the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. The default value is **None**. And then all dimensions will be shown.
- **map_type** the way to draw the map. It should be **'lines'** or **'fill'** or **'detailed'**. The default value is **'lines'**.
  - For the **'lines'** mode, map is only drawn by margins of lanes.
  - For the **'fill'** mode, the lanes will be filled with semitransparent colors.
  - For the **'detailed'** mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). Otherwise, it is the same as the 'lines' mode.
- **scale_type** the way to scale the coordinate axises. It should be **'trace'** or **'map'**. The default value is **'trace'**.
  -  For the **'trace'** mode, the traces will be in the center of the plot with an appropriate scale.
  - For the **'map'** mode, the map will be in the center of the plot with an appropriate scale.
- **label_mode** the mode to display labels or not. if it is 'None', then labels will not be displayed. Otherwise, labels will be displayed. The default value is **'None'**.
- **sample_rate** it determines the points used in the plot. if sample_rate = n which is a positive integer, it means that the plotter sample a point within n points. it is useful when the points are too much and the response of the plot is slow. The default value is **1**.
- **time_step** it is used to determine the num of digits of time points. If it's None, then the num of digits is set as 3. Otherwise, the num of digits is set as the num of digits of the given time_step. Normally, it should be the time step of simulation/verification. The default value is **None**.
- **speed_rate** it determines the speed od anime. Due to the performance, it maybe be limited when the response of the plot is slow. The default value is **1**.
- **anime_mode** it determines if the trails are displayed or not. if it's 'normal', then the trails will not be displayed. Otherwise, displayed. The default value is **'normal'**.
- **full_trace** it determines if the full trace is displayed or not. if it's False, then the full trace will not be displayed. Otherwise, displayed. The default value is **False**.