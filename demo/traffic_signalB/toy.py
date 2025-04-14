import pyvista as pv
import numpy as np

def create_box(corner1, corner2):
    """Create a box from two corner points."""
    bounds = [corner1[0], corner2[0], corner1[1], corner2[1], corner1[2], corner2[2]]
    return pv.Box(bounds=bounds)

# Define the bounding boxes
boxes = [
    [[5, -50, 0], [-2000, -20, 4]]
]

# Create a PyVista plotter
plotter = pv.Plotter()

# Add each box to the plot
for corner1, corner2 in boxes:
    box = create_box(corner1, corner2)
    plotter.add_mesh(box, color='lightblue', opacity=0.5, show_edges=True, reset_camera=False)

# grid_bounds = [-2100, 300, -1100, 0, 0, 1100]
# plotter.show_grid(bounds=grid_bounds, minor_ticks=True)


# Show the gri
# After adding all your meshes but before calling plotter.show()
plotter.camera_position = 'xy'  # Try different angles like 'xz' or 'yz'
plotter.reset_camera()  # Ensure camera captures all objects
# Display the plot
plotter.show()
