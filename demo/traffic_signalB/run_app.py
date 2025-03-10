from wsgiref.validate import PartialIteratorWrapper
from mp0_p1 import VehicleAgent, PedestrianAgent, VehiclePedestrianSensor, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, PedestrianMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D import *

from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go


import sys
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtWebEngineWidgets import QWebEngineView
import threading
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyVista in Qt Web App")
        self.setGeometry(100, 100, 1200, 800)
        self.plane1_position = 0
        self.plane2_position = 0
        self.thread = None

        # Main Widget and Layout
        main_widget = QWidget()
        layout = QVBoxLayout()

        # PyVistaQt QtInteractor (for 3D visualization)
        self.plotter = pvqt.QtInteractor(main_widget)
        #self.plotter.setMinimumHeight(400)  # Fixed to half of 800px

        layout.addWidget(self.plotter.interactor)

        # QWebEngineView (for web UI)
        self.web_view = QWebEngineView()
        layout.addWidget(self.plotter.interactor)
        #self.plotter.setMinimumHeight(400)  # Fixed to half of 800px

        # QWebEngineView (for web UI)
        self.web_view = QWebEngineView()
        self.web_view.setHtml("""
        <html>
            <head>
                <title>Embedded PyVista</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { overflow: hidden; } /* No Scrollbars */

                    #container {
                        position: relative;
                        width: 400px;
                        height: 400px;
                        background-color: #f0f0f0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        border: 2px solid black;
                        margin: auto;
                    }
                    /* X and Y Axes */
                    .axis {
                        position: absolute;
                        background-color: black;
                        z-index: 1;
                    }

                    #x-axis {
                        width: 100%;
                        height: 2px;
                        top: 50%;
                        left: 0;
                    }

                    #y-axis {
                        width: 2px;
                        height: 100%;
                        left: 2%;
                        top: 0;
                    }
                              /* Axis Markers */
                    .marker {
                        position: absolute;
                        font-size: 14px;
                        color: black;
                        font-weight: bold;
                    }

                    /* X-axis markers */
                    .x-marker {
                        top: 52%;
                        transform: translateX(-50%);
                    }

                    .x-tick {
                        width: 2px;
                        height: 10px;
                        background: black;
                        position: absolute;
                        top: 48%;
                    }

                    /* Y-axis markers */
                    .y-marker {
                        left: 2%;
                        transform: translateY(-50%);
                    }

                    .y-tick {
                        width: 10px;
                        height: 2px;
                        background: black;
                        position: absolute;
                        left: 2%;
                    }


                    .draggable {
                        width: 80px;
                        height: 80px;
                        background-color: #007BFF;
                        color: white;
                        text-align: center;
                        line-height: 80px;
                        font-weight: bold;
                        border-radius: 10px;
                        position: absolute;
                        cursor: grab;
                    }

                    #plane1 { left: 30px; top: 30px; }
                    #plane2 { right: 30px; top: 30px; }

                    button {
                        display: block;
                        margin: 10px auto;
                        padding: 10px 20px;
                        font-size: 16px;
                        cursor: pointer;
                        border: none;
                        background-color: #28a745;
                        color: white;
                        border-radius: 5px;
                    }
                </style>
                <script>
                    let draggedElement = null;

                    function dragStart(event) {
                        draggedElement = event.target;
                    }

                    function dragOver(event) {
                        event.preventDefault();
                    }

                    function drop(event) {
                        if (draggedElement) {
                            draggedElement.style.left = event.clientX - draggedElement.offsetWidth / 2 + 'px';
                            draggedElement.style.top = event.clientY - draggedElement.offsetHeight / 2 + 'px';
                            event.preventDefault();
                        }
                    }

                    function getPlanePositions() {
                        let plane1 = document.getElementById("plane1");
                        let plane2 = document.getElementById("plane2");

                        return {
                            plane1: { x: plane1.style.left, y: plane1.style.top },
                            plane2: { x: plane2.style.left, y: plane2.style.top }
                        };
                    }

                    function runSimulation() {
                        let positions = getPlanePositions();
                        console.log("Saving Positions:", positions);
                        window.pyQtApp.savePositions(positions);
                    }

                    window.onload = function() {
                        let container = document.getElementById('container');
                        container.addEventListener('dragover', dragOver);
                        container.addEventListener('drop', drop);
                    };
                </script>
            </head>
            <body>
                <div id="container">
                              
                    <!-- X and Y Axes -->
                    <div id="x-axis" class="axis"></div>
                    <div id="y-axis" class="axis"></div>

                    <!-- X-Axis Markers -->
                    <div class="marker x-marker" style="left: 0%;">0</div>
                    <div class="marker x-marker" style="left: 25%;">50</div>
                    <div class="marker x-marker" style="left: 50%;">100</div>
                    <div class="marker x-marker" style="left: 75%;">150</div>
                    <div class="marker x-marker" style="left: 100%;">200</div>

                    <!-- X-Axis Ticks -->
                    <div class="x-tick" style="left: 0%;"></div>
                    <div class="x-tick" style="left: 25%;"></div>
                    <div class="x-tick" style="left: 50%;"></div>
                    <div class="x-tick" style="left: 75%;"></div>
                    <div class="x-tick" style="left: 100%;"></div>

                    <!-- Y-Axis Markers -->
                    <div class="marker y-marker" style="top: 0%;">200</div>
                    <div class="marker y-marker" style="top: 25%;">150</div>
                    <div class="marker y-marker" style="top: 50%;">100</div>
                    <div class="marker y-marker" style="top: 75%;">50</div>
                    <div class="marker y-marker" style="top: 100%;">0</div>

                    <!-- Y-Axis Ticks -->
                    <div class="y-tick" style="top: 0%;"></div>
                    <div class="y-tick" style="top: 25%;"></div>
                    <div class="y-tick" style="top: 50%;"></div>
                    <div class="y-tick" style="top: 75%;"></div>
                    <div class="y-tick" style="top: 100%;"></div>
                    <div id="plane1" class="draggable" draggable="true" ondragstart="dragStart(event)">✈ 1</div>
                    <div id="plane2" class="draggable" draggable="true" ondragstart="dragStart(event)">✈ 2</div>
                </div>
            </body>
            </html>
        """)
        layout.addWidget(self.web_view)

        # Run button to trigger saving the positions
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_button_clicked)
        layout.addWidget(self.run_button)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def run_verse(self, x1,y1, r1 , x2,y2, r2 ):
        import os 
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "vehicle_controller.py")
        vehicle = VehicleAgent('car', file_name=input_code_name)
        pedestrian = PedestrianAgent('pedestrian')

        scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

        scenario.add_agent(vehicle) 
        scenario.add_agent(pedestrian)
        scenario.set_sensor(VehiclePedestrianSensor())


        # init_car = [[4,-5,0,8],[5,5,0,8]]
        # init_pedestrian = [[170,-55,0,3],[175,-52,0,5]]

        init_car = [[x1 - r1,y1 -r1 ,0,8],[x1+r1,y1 +r1,0,8]]
        init_pedestrian = [[x2 -r2, y2- r2,0,3],[x2 +r2,y2+r2,0,5]]

        scenario.set_init_single(
            'car', init_car,(VehicleMode.Normal,)
        )
        scenario.set_init_single(
            'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
        )

            # traces = []
            # fig = go.Figure()
            # n=3
            # for i in range(n):
            #     trace = scenario.simulate(50, 0.1)
            #     traces.append(trace)
            #     fig = simulation_tree_3d(trace, fig,\
            #                             0,'time', 1,'x',2,'y')
            # avg_vel, unsafe_frac, unsafe_init = eval_velocity(traces)
            # fig.show()
            #fig = pv.Plotter()
            #fig.show(interactive_update=True)
        _ = self.plotter.add_axes_at_origin(    xlabel='',
    ylabel='',
    zlabel='')
        traces = scenario.verify(80, 0.1, self.plotter)

        # Add an interactive 3D sphere
    def run_button_clicked(self):
        self.web_view.page().runJavaScript("getPlanePositions();", self.handle_positions)
    def start_animation(self, x1,y1,r1, x2,y2,r2):
        """Starts a background thread to move the sphere."""
        if(self.thread):
            self.thread.join()
        self.plotter.clear()

        self.thread = threading.Thread(target=self.run_verse,args=[x1,y1,r1, x2,y2,r2], daemon=True)
        self.thread.start()


    def handle_positions(self, positions):
        def position_to_int(pos):
            x = int(pos['x'][:-2])
            y = int(pos['y'][:-2])
            return (x/2,y/2)

            
        # positions is a JavaScript object containing the current positions of the planes
        print("Positions of planes:", positions)

        # Process the positions and use them in the 3D visualization or other tasks
        self.plane1_position = positions['plane1']
        self.plane2_position = positions['plane2']

        x1, y1 = position_to_int(self.plane1_position)
        x2, y2 = position_to_int(self.plane2_position)

        self.start_animation(x1, y1,1,x2,y2,1)

        #print(self.plane1_position, self.plane2_position)
       
        # You can further process these positions in your PyVista 3D
    def _set_python_bridge(self, result):
        """Sets the python bridge object in JS"""
        self.web_view.page().runJavaScript("window.pyQtApp = pyQtApp;", self._set_python_bridge)
        
        # Start background thread for animation

      

# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


# if __name__ == "__main__":
#     import os 
#     script_dir = os.path.realpath(os.path.dirname(__file__))
#     input_code_name = os.path.join(script_dir, "vehicle_controller.py")
#     vehicle = VehicleAgent('car', file_name=input_code_name)
#     pedestrian = PedestrianAgent('pedestrian')

#     scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

#     scenario.add_agent(vehicle) 
#     scenario.add_agent(pedestrian)
#     scenario.set_sensor(VehiclePedestrianSensor())

#     # # ----------- Different initial ranges -------------
#     # # Uncomment this block to use R1
#     init_car = [[4,-5,0,8],[5,5,0,8]]
#     init_pedestrian = [[170,-55,0,3],[175,-52,0,5]]
#     # # -----------------------------------------

#     # # Uncomment this block to use R2
#     # init_car = [[-5,-5,0,7.5],[5,5,0,8.5]]
#     # init_pedestrian = [[175,-55,0,3],[175,-55,0,3]]
#     # # -----------------------------------------

#     # # Uncomment this block to use R3
#     # init_car = [[-5,-5,0,7.5],[5,5,0,8.5]]
#     # init_pedestrian = [[173,-55,0,3],[176,-53,0,3]]
#     # # -----------------------------------------

#     scenario.set_init_single(
#         'car', init_car,(VehicleMode.Normal,)
#     )
#     scenario.set_init_single(
#         'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
#     )

#     # traces = []
#     # fig = go.Figure()
#     # n=3
#     # for i in range(n):
#     #     trace = scenario.simulate(50, 0.1)
#     #     traces.append(trace)
#     #     fig = simulation_tree_3d(trace, fig,\
#     #                             0,'time', 1,'x',2,'y')
#     # avg_vel, unsafe_frac, unsafe_init = eval_velocity(traces)
#     # fig.show()
#     fig = pv.Plotter()
#     fig.show(interactive_update=True)
#     traces = scenario.verify(80, 0.1, fig)
    

#     # fig = pv.Plotter()
#     # fig = plot3dReachtube(traces,'car',0,1,2,'b',fig)
#     # fig = plot3dReachtube(traces,'pedestrian',0,1,2,'r',fig)

#     # fig = reachtube_tree_3d(traces, fig,\
#     #                          0,'time', 1,'x',2,'y')
#     # fig.show()



   
   
    
