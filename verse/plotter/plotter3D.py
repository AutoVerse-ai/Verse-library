# Plot polytope in 3d

import numpy as np
import polytope as pc
import pyvista as pv

from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
import ast
import vtk
import os
import json
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)


int_to_color = {1:'b', 2:'r', 3:'g', 4: 'purple', 5:'orange'}
color_map = {}

node_rect_cache ={}
node_idx = 0

def plot3dReachtubeSingle(tube, ax, x_dim=1, y_dim=2, z_dim=3, edge=False,  log_file="bounding_boxes.txt", save_to_file = True, step =1000):
    if os.path.exists('plotter_config.json'):
        with open('plotter_config.json', 'r') as f:
            config = json.load(f)
            x_dim = config['x_dim']
            y_dim = config['y_dim']
            z_dim = config['z_dim']
            step = int(config['speed'])
            save_to_file = bool(config['save'])
            log_file = config['log_file']
            node_batch = config['node_batch']
    global node_idx
    rects_dict = {}
    length = len(tube[list(tube.keys())[0]])

    for i in range(0, length,2):
        for agent_id in tube:
            if agent_id not in color_map:
                color_map[agent_id] = int_to_color[len(color_map) +1]
            
            trace = tube[agent_id]
            lb = trace[i]
            ub =  trace[i + 1]
            box = [[lb[x_dim]-1, lb[y_dim]-1, lb[z_dim]-1], [ub[x_dim]+1, ub[y_dim]+1, ub[z_dim]+1]]
            if(save_to_file):
                with open(log_file, "a") as f:
                    f.write(f"{box}, {color_map[agent_id]}\n")

            if(agent_id not in rects_dict):
                 rects_dict[agent_id] = []
            
            rects_dict[agent_id].append(box)

            if not node_batch and (i % step == 0 or i >= length-2):
                plotGrid(ax, color_map[agent_id], rects_dict[agent_id]  )
                rects_dict[agent_id] = []


    for agent_id in tube:
        if(agent_id not in node_rect_cache):

            node_rect_cache[agent_id] = []
        node_rect_cache[agent_id] += rects_dict[agent_id]

    if( node_batch and node_idx % step==0  ):

        for agent_id in tube:

            plotGrid(ax, color_map[agent_id], node_rect_cache[agent_id] )
            node_rect_cache[agent_id] = []

    node_idx+=1
    return ax
    
    
    # if os.path.exists('plotter_config.json'):
    #     with open('plotter_config.json', 'r') as f:
    #         config = json.load(f)
    #         x_dim = config['x_dim']
    #         y_dim = config['y_dim']
    #         z_dim = config['z_dim']
    #         step = int(config['speed'])
    #         save_to_file = bool(config['save'])
    #         log_file = config['log_file']
    # if save_to_file:
    #     # Open file in write mode ONCE to clear existing content
    #     with open(log_file, "w") as f:
    #         f.write("")  # Optional: just clears the file

    
    # rects_dict = {}
    # length = len(tube[list(tube.keys())[0]])
    # for i in range(0, length, 2):

    #     for agent_id in tube:
    #         if agent_id not in color_map:
    #             color_map[agent_id] = int_to_color[len(color_map) +1]
            
    #         trace = tube[agent_id]
    #         lb = trace[i]
    #         ub =  trace[i + 1]
    #         box = [[lb[x_dim]-1, lb[y_dim]-1, lb[z_dim]-1], [ub[x_dim]+1, ub[y_dim]+1, ub[z_dim]+1]]
    #         if(save_to_file):
    #             with open(log_file, "a") as f:
    #                 f.write(f"{box}, {color_map[agent_id]}\n")

    #         if(agent_id not in rects_dict):
    #             rects_dict[agent_id] = []
            
    #         rects_dict[agent_id].append(box)
    #         if i % step == 0 or i >= length-2:
    #             plotGrid(ax, color_map[agent_id] , rects=rects_dict[agent_id])
    #             rects_dict[agent_id] = []

           

              

def load_and_plot(ax, step=None, log_file="boxes1.txt"):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            line = lines[0]
            box_str, color = line.rsplit(",", 1)  # Split into box and color
            box = ast.literal_eval(box_str.strip())  # Convert box string to list
            if( len(np.array(box).shape) !=2):
                load_and_plot_simulations(ax,step, log_file)
            else:
                load_and_plot_boxes(ax,step, log_file )

        if not lines:
            print("Nothing found in the file.")
            return ax
    except FileNotFoundError:
        print(f"File {log_file} not found.")


def load_and_plot_boxes(ax, step=1000, log_file="boxes1.txt"):
    with open(log_file, "r") as f:
        lines = f.readlines()
    rect_dict = {}
    for i, line in enumerate(lines):
        # Extract the box and color from each line
        box_str, color = line.rsplit(",", 1)  # Split into box and color
        box = ast.literal_eval(box_str.strip())  # Convert box string to list
        color = color.strip().strip("'\"")  # Clean up the color string
        if( color not in rect_dict):
            rect_dict[color] = []
        rect_dict[color].append(box)

        #poly = pc.box2poly(np.array(box).T)
    for color, rects in rect_dict.items():
        plotGrid(ax, color, rects)
    
    
    return ax


def load_and_plot_simulations(ax, step = None, log_file = "boxes1.txt"):
    with open(log_file, "r") as f:
        lines = f.readlines()
    rect_dict = {}
    for i, line in enumerate(lines):
        # Extract the box and color from each line
        box_str, color = line.rsplit(",", 1)  # Split into box and color
        box = ast.literal_eval(box_str.strip())  # Convert box string to list
        color = color.strip().strip("'\"")  # Clean up the color string
        if( color not in rect_dict):
            rect_dict[color] = []
        rect_dict[color].append(box)

        #poly = pc.box2poly(np.array(box).T)
    for color, points in rect_dict.items():
        plotPolyLine(points, color, ax)
    
    
    return ax

def plotGrid(ax,color, rects):
    vertices = []
    cell_indices = []

    
    
    # Add vertices for each rectangle and track indices
    for i in range(len(rects)):
        lb = rects[i][0]
        ub = rects[i][1]
        
        x0, x1 = lb[0], ub[0]
        y0, y1 = lb[1], ub[1]
        z0, z1 = lb[2], ub[2]
        
        corners = [
            (x0, y0, z0),  # index 0
            (x1, y0, z0),  # index 1
            (x1, y1, z0),  # index 2
            (x0, y1, z0),  # index 3
            (x0, y0, z1),  # index 4
            (x1, y0, z1),  # index 5
            (x1, y1, z1),  # index 6
            (x0, y1, z1),  # index 7
        ]
        
        start_idx = len(vertices)
        vertices.extend(corners)
        
        cell_indices.append([start_idx + j for j in range(8)])
    
    vertices = np.array(vertices)
    
    N = len(rects)
    cells = np.zeros((N, 9), dtype=np.int64)
    cells[:, 0] = 8  # 8 points per cell
    
    for i in range(N):
        cells[i, 1:] = cell_indices[i]
    
    # Flatten cells array
    cells = cells.ravel()
    
    # Define cell types
    celltypes = np.full(N, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    
    # Create and display the grid
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)
    ax.add_mesh(grid, show_edges=False, color=color, opacity=0.5)


def plot3dMap(lane_map, color="k", ax=None, width=0.1, num=20):
    if ax is None:
        ax = pv.Plotter()
    for lane_idx in lane_map.lane_dict:
        lane = lane_map.lane_dict[lane_idx]
        if lane.plotted:
            for lane_seg in lane.segment_list:
                # if lane_seg.type == 'Straight':
                oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)
                points = np.vstack((oc_x, oc_y, oc_z)).T
                spline = pv.Spline(points, 400)
                tube = spline.tube(radius=width)
                ax.add_mesh(tube, color=color)
                # elif lane_seg.type == 'Circular':
                #     oc, oc_x, oc_y, oc_z = lane_seg.get_lane_center(num)

    return ax


def plot3dReachtube(root, agent_id, x_dim, y_dim, z_dim, color="b", ax=None, edge=False):
    if isinstance(root, AnalysisTree):
        root = root.root

    if ax is None:
        ax = pv.Plotter()

    queue = [root]
    idx = 0

    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = traces[agent_id]
        data = []
        for i in range(0, len(trace), 2):
            data.append([trace[i], trace[i + 1]])
        # for key in sorted(node.lower_bound):
        #     lower_bound.append(node.lower_bound[key])
        # for key in sorted(node.upper_bound):
        #     upper_bound.append(node.upper_bound[key])
        ax = plot3dReachtubeSingle(data, x_dim, y_dim, z_dim, ax, color, edge=edge)

        queue += node.child
        idx += 1

    return ax
    # for i in range(min(len(lower_bound), len(upper_bound))):
    #     lb = list(map(float, lower_bound[i]))
    #     ub = list(map(float, upper_bound[i]))

    #     box = [[lb[x_dim], lb[y_dim], lb[z_dim]],[ub[x_dim], ub[y_dim], ub[z_dim]]]
    #     poly = pc.box2poly(np.array(box).T)
    #     plot_polytope_3d(poly.A, poly.b, ax = ax, color = '#b3de69')


def plot_polytope_3d(A, b, ax=None, color="red", trans=0.2, edge=False, render = False):
    if ax is None:
        ax = pv.Plotter()

    poly = pc.Polytope(A=A, b=b)
    vertices = pc.extreme(poly)
    cloud = pv.PolyData(vertices)
    volume = cloud.delaunay_3d()
    shell = volume.extract_geometry()
    if len(shell.points) <= 0:
        return ax
    ax.add_mesh(shell, opacity=trans, color=color, render= render)
    if edge:
        edges = shell.extract_feature_edges(20)
        ax.add_mesh(edges, color="k", line_width=0.1, opacity=0.5, render= render)
        
    return ax

def plot3dSimulationSingle(tube, ax, line_width=5, step = 1000, x_dim=1, y_dim=2, z_dim=3, save_to_file = True, log_file = "simulations.txt" ):
    if os.path.exists('plotter_config.json'):
        with open('plotter_config.json', 'r') as f:
            config = json.load(f)
            x_dim = config['x_dim']
            y_dim = config['y_dim']
            z_dim = config['z_dim']
            step = int(config['speed'])
            save_to_file = bool(config['save'])
            log_file = config['log_file']
            node_batch = config['node_batch']
    global node_idx

    rects_dict = {}
    length = len(tube[list(tube.keys())[0]])-1

    for i in range(0, length ):
        for agent_id in tube:
            if agent_id not in color_map:
                color_map[agent_id] = int_to_color[len(color_map) +1]
            
            trace = tube[agent_id]
            lb = trace[i]
            point = [lb[x_dim]-1, lb[y_dim]-1, lb[z_dim]-1]
            if(save_to_file):
                with open(log_file, "a") as f:
                    f.write(f"{point}, {color_map[agent_id]}\n")

           
            
            if(agent_id not in rects_dict):
                rects_dict[agent_id] = []
            rects_dict[agent_id].append(point)

            if not node_batch and (i % step == 0 or i >= length-1):
                plotPolyLine(rects_dict[agent_id], color_map[agent_id], ax)
                rects_dict[agent_id] = [rects_dict[agent_id][-1]]


    for agent_id in tube:
        if(agent_id not in node_rect_cache):

            node_rect_cache[agent_id] = []
        node_rect_cache[agent_id] += rects_dict[agent_id]

    if( node_batch and node_idx % step==0  ):

        for agent_id in tube:

            plotPolyLine(node_rect_cache[agent_id], color_map[agent_id], ax)
            node_rect_cache[agent_id] = []

    node_idx+=1
    return ax

def plotRemaining(ax, verify):
    for agent_id, rects in  node_rect_cache.items():
        if (len(rects) > 0 ):
            if(verify):
                plotGrid(ax, color_map[agent_id], rects)
            else:
                plotPolyLine(rects, color_map[agent_id], ax)
            


def plotPolyLine(points, color, ax):
    n_points = len(points)
    cells = np.full((n_points-1, 3), 2, dtype=np.int64)

    # Set the point indices for each line segment
    # Each line connects point i to point i+1
    cells[:, 1] = np.arange(0, n_points-1)
    cells[:, 2] = np.arange(1, n_points)
    
    # Flatten the cells array
    cells = cells.ravel()
    
    # Create the polyline
    poly_line = pv.PolyData(points, lines=cells)
    ax.add_mesh(poly_line, color=color, line_width=5)
def plot_line_3d(start, end, ax=None, color="blue", line_width=1):
    if ax is None:
        ax = pv.Plotter()

    a = start
    b = end

    # Preview how this line intersects this mesh
    line = pv.Line(a, b)

    ax.add_mesh(line, color=color, line_width=line_width, render=False)
    #ax.reset_camera()

    return ax


def plot_point_3d(points, ax=None, color="blue", point_size=100):
    if ax is None:
        ax = pv.Plotter()
    ax.add_points(points, render_points_as_spheres=True, point_size=point_size, color=color)
    return ax


if __name__ == "__main__":
    A = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
    b = np.array([[1], [1], [1], [1], [1], [1]])
    b2 = np.array([[-1], [2], [-1], [2], [-1], [2]])
    fig = pv.Plotter()
    fig = plot_polytope_3d(A, b, ax=fig, color="red")
    fig = plot_polytope_3d(A, b2, ax=fig, color="green")
    fig.show()
    
