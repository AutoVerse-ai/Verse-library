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
from collections import OrderedDict

color_map = {}

node_rect_cache ={}
node_idx = 0


load_time=0
#load_idx = 0


plotted = []
not_plotted = []

offset = 0
prev_time = 0

#refine_cache = []


def plot3dReachtubeSingleLive(tube, ax, x_dim=1, y_dim=2, z_dim=3, edge=False,  log_file="bounding_boxes.txt", save_to_file = True, step =1000):
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
    global offset
    global prev_time
    rects_dict = {}
    length = len(tube[list(tube.keys())[0]])

    for i in range(0, length,2):
        for agent_id in tube:
            if agent_id not in color_map:
                color_map[agent_id] = agent_id.split('_')[1]

            
            trace = tube[agent_id]
            lb = trace[i]
            if lb[0] ==0:
                offset = prev_time
            ub =  trace[i + 1]
            box = [[lb[x_dim]-1, lb[y_dim]-1, lb[z_dim]-1], [ub[x_dim]+1, ub[y_dim]+1, ub[z_dim]+1]]

            if(save_to_file):
                with open(log_file, "a") as f:
    
                    save_time = offset + ub[0]
                    f.write(f"{box}, {color_map[agent_id]}, {save_time}\n")
                    prev_time = save_time   

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
    
           
def preprocess_file(ax, log_file="boxes1.txt"):
   
    with open(log_file, "r") as f:
        lines = f.readlines()
        line = lines[0]

        box_str, color,time = line.rsplit(",", 2)  # Split into box and color and time
        box = ast.literal_eval(box_str.strip())  # Convert box string to list
        if( len(np.array(box).shape) !=2):
            verify = False
        else:
            verify = True
    if os.path.exists('plotter_config.json'):
        with open('plotter_config.json', 'r') as f:
            config = json.load(f)
            step = int(config['speed'])

    rect_dict = {}

    for i, line in enumerate(lines):
        # Extract the box and color from each line
        box_str, color,time = line.rsplit(",", 2)  # Split into box and color
        if(box_str == "END"):
            rect_dict = {}
            continue

        box = ast.literal_eval(box_str.strip())  # Convert box string to list
        color = color.strip().strip("'\"")  # Clean up the color string
        if( color not in rect_dict):
            rect_dict[color] = []
        rect_dict[color].append(box)

        if (len(rect_dict[color]) % step == 0 or i >= len(lines)-1):
            if(verify):
                plotted.append((plotGrid(ax, color, rect_dict[color]), float(time)))
                rect_dict[color] = []

            else:
                plotted.append((   plotPolyLine(rect_dict[color], color, ax)   , float(time)))
                rect_dict[color] = [rect_dict[color][-1]]

    def merge_actors_by_timestamp(ax, plotted):
        # Extract and prepare meshes for merging
        meshes = []
        for actor, _ in plotted:
            # Get the mesh directly from the actor
            mesh = pv.wrap(actor.GetMapper().GetInput())
            # Scale if needed
            meshes.append(mesh)
    
        # Merge all meshes
        if meshes:
            merged_mesh = meshes[0].copy()
            for mesh in meshes[1:]:
                merged_mesh = merged_mesh.merge(mesh)
            
            # Add the merged mesh to the plotter
            ax.add_mesh(merged_mesh, reset_camera=False, show_edges=False, opacity=0.0)
            return merged_mesh
        return None
      

    plotted.sort(key=lambda x: x[1])
    print(len(plotted))
    merge_actors_by_timestamp(ax, plotted)




    # print("Plotted: ", len(plotted), plotted[0][1], plotted[-2][1], plotted[-1][1])
    # print("Not Plotted: ", len(not_plotted))
    


def load_and_plot(ax, target_time=0):
    global load_time

    if target_time > load_time:
        while len(not_plotted):
            actor, timestamp = not_plotted[-1]  # Peek (don't pop yet)
            if timestamp > target_time:
                break  # Stop if next frame is past target_time
            actor, timestamp = not_plotted.pop()
            ax.add_actor(actor, reset_camera=False)
            plotted.append((actor, timestamp))
            load_time = timestamp

    elif target_time < load_time:
        while len(plotted):
            actor, timestamp = plotted[-1]  # Peek
            if timestamp < target_time:
                break  # Stop if next frame is earlier than target_time
            actor, timestamp = plotted.pop()
            ax.remove_actor(actor, reset_camera=False)
            not_plotted.append((actor, timestamp))
            load_time = timestamp


def plotPolyLine(points, color, ax):
    n_points = len(points)
    cells = np.full((n_points-1, 3), 2, dtype=np.int64)
    #print(points)

    # Set the point indices for each line segment
    # Each line connects point i to point i+1
    cells[:, 1] = np.arange(0, n_points-1)
    cells[:, 2] = np.arange(1, n_points)
    
    # Flatten the cells array
    cells = cells.ravel()
    
    # Create the polyline
    poly_line = pv.PolyData(points, lines=cells)
    return  (ax.add_mesh(poly_line, color=color, line_width=3   ))


last_plotted = {}  # color -> (center, actor)

def plotGrid(ax, color, rects):
    global last_plotted
    
    vertices = []
    cell_indices = []

    for i in range(len(rects)):
        lb = rects[i][0]
        ub = rects[i][1]

        x0, x1 = lb[0], ub[0]
        y0, y1 = lb[1], ub[1]
        z0, z1 = lb[2], ub[2]

        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
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
    cells = cells.ravel()

    celltypes = np.full(N, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)
    ax.add_mesh(grid, show_edges=False, color=color, opacity=0.5)

    # Plane placement for the last box only
    new_lb, new_ub = rects[-1]
    new_center = (np.array(new_lb) + np.array(new_ub)) / 2

    # Remove old plane if exists
    # if color in last_plotted and last_plotted[color][1] is not None:
    #     ax.remove_actor(last_plotted[color][1])

    # if color in last_plotted:
    #     prev_center = last_plotted[color][0]
    #     direction = new_center - prev_center
    #     norm = np.linalg.norm(direction)
    #     if norm > 1e-6:
    #         direction /= norm
    #     else:
    #         direction = np.array([1, 0, 0])  # Default direction
    # else:
    #     direction = np.array([1, 0, 0])  # Default direction if first time

    # Create a simple triangular plane icon
    plane_size = np.linalg.norm(np.array(new_ub) - np.array(new_lb))
    #plane_mesh = create_plane_icon(new_center, direction, size=plane_size)

    # actor = ax.add_mesh(plane_mesh, color=color+"80", show_edges=True, line_width=1.5)
    # last_plotted[color] = (new_center, actor)

    # return actor

def create_plane_icon(center, direction, size=1.0):
    # Create a triangle representing the plane
    perp1 = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(perp1) < 1e-3:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1  = np.float64(perp1)/np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    tip = center + direction * size
    left = center - perp1 * (size * 0.5) - direction * (size * 0.5)
    right = center + perp1 * (size * 0.5) - direction * (size * 0.5)

    faces = [3, 0, 1, 2]  # Triangle face
    points = np.array([tip, left, right])
    return pv.PolyData(points, faces)
     

def plot3dSimulationSingleLive(tube, ax, line_width=3, step = 1000, x_dim=1, y_dim=2, z_dim=3, save_to_file = True, log_file = "simulations.txt" ):
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
    global offset
    global prev_time

    rects_dict = {}
    length = len(tube[list(tube.keys())[0]])-1

    for i in range(length ):
        for agent_id in tube:
            if agent_id not in color_map:
                color_map[agent_id] = agent_id.split('_')[1]
            
            trace = tube[agent_id]
            lb = trace[i]
            if(lb[0] ==0):
                offset = prev_time

                
            point = [lb[x_dim]-1, lb[y_dim]-1, lb[z_dim]-1]
            if(save_to_file):
                with open(log_file, "a") as f:
                    save_time = offset + lb[0]
                    f.write(f"{point}, {color_map[agent_id]}, {save_time}\n")
                    prev_time = save_time   

           
            
            if(agent_id not in rects_dict):
                rects_dict[agent_id] = []
            rects_dict[agent_id].append(point)

            if not node_batch and (i % step == 0 or i >= length-1):
                plotPolyLine(rects_dict[agent_id], color_map[agent_id], ax)
                rects_dict[agent_id] = [rects_dict[agent_id][-1]]


    for agent_id in tube:
        if(agent_id not in rects_dict):
                rects_dict[agent_id] = []
        if(agent_id not in node_rect_cache):

            node_rect_cache[agent_id] = []
        node_rect_cache[agent_id] += rects_dict[agent_id]

    if( node_batch and node_idx % step==0  ):

        for agent_id in tube:

            plotPolyLine(node_rect_cache[agent_id], color_map[agent_id], ax)
            node_rect_cache[agent_id] = [node_rect_cache[agent_id][-1]]

    node_idx+=1
    return ax

def plotRemaining(ax, verify):
    global node_rect_cache

    for agent_id, rects in  node_rect_cache.items():
        if (len(rects) > 0 ):   
            if(verify):
                plotGrid(ax, color_map[agent_id], rects)
            else:
                plotPolyLine(rects, color_map[agent_id], ax)
    global offset
    global prev_time
    global node_idx
            
    node_rect_cache = {}
    node_idx = 0
    
    offset = 0
    prev_time = 0

    if(not verify):
        if os.path.exists('plotter_config.json'):
            with open('plotter_config.json', 'r') as f:
                config = json.load(f)
                log_file = config['log_file']
                save_to_file = bool(config['save'])
        if(save_to_file):
            with open(log_file, "a") as f:
                f.write(f"END, placeholder, -1\n")
            



#OLD METHODS
#==================================================================================================================

def plot3dReachtubeSingle(tube, x_dim, y_dim, z_dim, ax, color, edge=True):
    for lb, ub in tube:
        box = [[lb[x_dim], lb[y_dim], lb[z_dim]], [ub[x_dim], ub[y_dim], ub[z_dim]]]
        poly = pc.box2poly(np.array(box).T)
        ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge)
    return ax


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


def plot3dReachtube(root, agent_id, x_dim, y_dim, z_dim, color="b", ax=None, edge=True):
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


def plot_polytope_3d(A, b, ax=None, color="red", trans=0.2, edge=True):
    if ax is None:
        ax = pv.Plotter()

    poly = pc.Polytope(A=A, b=b)
    vertices = pc.extreme(poly)
    cloud = pv.PolyData(vertices)
    volume = cloud.delaunay_3d()
    shell = volume.extract_geometry()
    if len(shell.points) <= 0:
        return ax
    ax.add_mesh(shell, opacity=trans, color=color)
    if edge:
        edges = shell.extract_feature_edges(20)
        ax.add_mesh(edges, color="k", line_width=0.1, opacity=0.5)
    return ax


def plot_line_3d(start, end, ax=None, color="blue", line_width=1):
    if ax is None:
        ax = pv.Plotter()

    a = start
    b = end

    # Preview how this line intersects this mesh
    line = pv.Line(a, b)

    ax.add_mesh(line, color=color, line_width=line_width)
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
    
