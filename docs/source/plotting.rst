"""""""""""""""""
Visualization
"""""""""""""""""
Verse has two different sets of visualization functions for generating plots and animations. Functions for 2D plots use `Plotly Open Source Graphing Library for Python <https://plotly.com/python/>`_.


.. contents:: Function list
   :depth: 3
===================
Common paramteres
===================

* ``time_step`` determines the num of digits of time points. Normally, it should be the time step used to compute  simulation/verification. Default value is ``None`` which is set as 3. 

===================
2D Visualization
===================
----------------------
simulation_tree
----------------------
Shows simulation traces, possibly with multiple brances.

Note: Since the plotter functions have similar APIs, 
in this document, we treat this function as a base function. 
Its parameters are general and occurred in all remaining functions. 
We will omit them in the remaining functions and only list some specific parameters. 

Usage::

  simulation_tree(root, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1)

Parameters:

* ``root``: root node of a simulation trace. Typically, return value of ``Scenario.simulate()``.

* ``map``: the map of the scenario plotted as a background. Use ``None`` if there is no map. You may check tutorial for more details. 

* ``fig``: figure object of type ``plotly.graph_objects.Figure()``.

* ``x_dim:`` the dimension of x coordinate in the trace list of every time step. The default value is ``1``.

* ``y_dim:`` the dimension of y coordinate in the trace list of every time step. The default value is ``2``.

* ``print_dim_list`` the list containing the dimensions of data which will be shown directly or indirectly when the mouse hovers on the point. The default value is ``None``. And then all dimensions will be shown.

* ``map_type`` the way to draw the map. It should be ``'lines'`` or ``'fill'`` or ``'detailed'``. The default value is ``'lines'``.
   * For the ``'lines'`` mode, map is only drawn by margins of lanes. 
   * For the ``'fill'`` mode, the lanes will be filled with semitransparent colors. 
   * For the ``'detailed'`` mode, the lanes will be filled some colors according to the speed limits of lanes(if the information is given). Otherwise, it is the same as the 'lines' mode.

* ``scale_type`` the way to scale the coordinate axises. It should be ``'trace'`` or ``'map'``. The default value is ``'trace'``. 
   * For the ``'trace'`` mode, the traces will be in the center of the plot with an appropriate scale. 
   * For the ``'map'`` mode, the map will be in the center of the plot with an appropriate scale. 

* ``label_mode`` the mode to display labels which indicate mode transitions or not. The default value is ``'None'``. 
   * If it is ``'None'``, then labels will not be displayed. 
   * Otherwise, labels will be displayed. 
  
* ``sample_rate`` it determines the points used in the plot. It is useful when the points are too much and the response of the plot is slow. The default value is ``1``.  
   * If ``sample_rate = n`` where ``n`` is a positive integer, then the plotter samples one point for every ``n`` points. 
  

----------------------
simulation_anime
----------------------
It shows the animation of the simulation with/without trail.

Usage::

  simulation_anime(root, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, time_step=None, speed_rate=1, anime_mode='normal', full_trace=False)

Parameters not occurred in ``simulation_tree``:

* ``time_step`` it is used to determine the num of digits of time points. Normally, it should be the time step of simulation/verification and provoided by the user. (May improve to auto determine) The default value is ``None``.
   * If it's ``None``, then the num of digits is set as 3. 
   * Otherwise, the num of digits is set as the num of digits of the given ``time_step``. 

* ``speed_rate`` it determines the speed up rate of anime. Due to the performance, it maybe be limited when the response of the plot is slow. The default value is ``1``.  

* ``anime_mode`` it determines if the trails are displayed or not. The default value is ``'normal'``.  
   * If it's ``'normal'``, then the trails will not be displayed. 
   * Otherwise, the trails will be displayed. 

* ``full_trace`` it determines if the full trace is displayed or not. The default value is ``False``.  
   * if it's ``False``, then the full trace will not be displayed. 
   * Otherwise, the full trace will be displayed. 

----------------------
reachtube_tree
----------------------
It statically shows the reachtubes, possibly with multiple brances. 

Usage::

	reachtube_tree(root, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, combine_rect=1):

Parameters not occurred in ``simulation_tree``:

* ``combine_rect`` it determines the way of displaying reachtube. Specifically, it can combine specified number of reachtubes as a rectangle. The default value is ``1`` here.

----------------------
reachtube_anime
----------------------
It shows the animation of the reachtube.

Usage::

  reachtube_anime(root, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1, time_step=None, speed_rate=1, combine_rect=None)

Parameters not occurred in ``simulation_tree``:



* ``speed_rate`` it determines the speed up rate of anime. Due to the performance, it maybe be limited when the response of the plot is slow. The default value is ``1``.  

* ``combine_rect`` it determines the way of displaying reachtube. Specifically, it can combine specified number of reachtubes as a rectangle. The default value is ``None`` here, which means no combination.  

===================
3D Visualization
===================
