import sys
import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QSizePolicy, QPushButton
)
import pyvistaqt as pvqt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt6 with Embedded PyVista Plotter")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Create the PyVista plotter
        self.plotter = pvqt.QtInteractor()
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter.interactor)

        # Add the button
        self.button = QPushButton("Add Cube")
        self.button.clicked.connect(self.create_unstructured_grid)
        self.main_layout.addWidget(self.button)

        # Plot the unstructured grid initially
        #self.create_unstructured_grid()



    def create_unstructured_grid(self):
        # Create an unstructured grid with hexahedrons
        rects = [[[-19.66666666666667, 1983.7948713721926, -1.261882267474506], [116.255273947809, 2119.6666666666665, 1.0]], 
[[-19.6666666626099, 1985.933333337213, -1.0], [115.6666666626099, 2122.0666666627853, 1.0]], 
[[-18.50172926904947, 1979.9722381634679, -2.4032613649541528], [116.87728455265028, 2116.106067469855, -0.27240932460296774]], 
[[-19.6666666626099, 1989.9333333372124, -1.0], [115.6666666626099, 2126.0666666627835, 1.0]], 
[[-18.532815735748905, 1975.972820217436, -2.5548572049939593], [116.80715202306835, 2112.109834923588, -0.5479419065759259]], 
[[-19.6666666626099, 1993.9333333372117, -1.0], [115.6666666626099, 2130.0666666627812, 1.0]], 
[[-18.50432285427152, 1971.9729609459007, -2.5620267635098557], [116.83998322399884, 2108.109939336325, -0.5611430647150912]], 
[[-19.6666666626099, 1997.933333337211, -1.0], [115.6666666626099, 2134.0666666627794, 1.0]], 
[[-18.470739436141102, 1967.9731223672086, -2.5623495993098606], [116.87577111090948, 2104.110078461039, -0.5617480461209943]], 
[[-19.6666666626099, 2001.9333333372103, -1.0], [115.6666666626099, 2138.066666662777, 1.0]]] 
        
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
            
            # Add the vertices
            start_idx = len(vertices)
            vertices.extend(corners)
            
            # Track cell indices with proper offset
            cell_indices.append([start_idx + j for j in range(8)])
        
        # Convert vertices to numpy array
        vertices = np.array(vertices)
        
        # Create cells
        N = len(rects)
        cells = np.zeros((N, 9), dtype=np.int64)
        cells[:, 0] = 8  # 8 points per cell
        
        # Add the proper indices for each cell
        for i in range(N):
            cells[i, 1:] = cell_indices[i]
        
        # Flatten cells array
        cells = cells.ravel()
        
        # Define cell types
        celltypes = np.full(N, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        
        # Create and display the grid
        grid = pv.UnstructuredGrid(cells, celltypes, vertices)
        self.plotter.add_mesh(grid, show_edges=True)
        self.plotter.add_axes()
        self.plotter.reset_camera()
        print("Added unstructured grid")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
