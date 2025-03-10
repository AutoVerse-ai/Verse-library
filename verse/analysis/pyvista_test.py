import sys
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
import threading
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyVista in Qt Web App")
        self.setGeometry(100, 100, 1200, 800)

        # Main Widget and Layout
        main_widget = QWidget()
        layout = QVBoxLayout()

        # PyVistaQt QtInteractor (for 3D visualization)
        self.plotter = pvqt.QtInteractor(main_widget)
        layout.addWidget(self.plotter.interactor)

        # QWebEngineView (for web UI)
        self.web_view = QWebEngineView()
        self.web_view.setHtml("""
            <html>
            <head><title>Embedded PyVista</title></head>
            <body>
                <h1 style='text-align:center;'>PyVista Qt Web App</h1>
                <p style='text-align:center;'>3D Visualization is Embedded Below.</p>
            </body>
            </html>
        """)
        layout.addWidget(self.web_view)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Add an interactive 3D sphere
        self.add_sphere()
        
        # Start background thread for animation
        self.start_animation()

    def add_sphere(self):
        """Add an interactive sphere to the PyVista scene."""
        sphere = pv.Sphere(radius=0.3, center=(0, 0, 0))
        self.actor = self.plotter.add_mesh(sphere, color="red")

    def start_animation(self):
        """Starts a background thread to move the sphere."""
        thread = threading.Thread(target=self.animate_sphere, daemon=True)
        thread.start()

    def animate_sphere(self):
        """Move the sphere in a circular motion."""
        t = 0
        while True:
            new_x = np.sin(t) * 2
            new_y = np.cos(t) * 2
            new_z = np.sin(t / 2) * 1

            # Update the actor's position
            self.actor.SetPosition(new_x, new_y, new_z)
            
            # Refresh the plotter
            self.plotter.render()

            t += 0.1  # Adjust motion speed
            time.sleep(0.05)  # Control update frequency

# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
