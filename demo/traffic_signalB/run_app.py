from wsgiref.validate import PartialIteratorWrapper
from mp0_p1 import VehicleAgent, PedestrianAgent, VehiclePedestrianSensor, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, PedestrianMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *

import sys
import numpy as np
import os
import threading
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QPushButton, QSizePolicy, QStackedLayout, QSlider,
    QLabel, QFrame
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPalette, QFont, QPen, QPainterPath
from PyQt6.QtSvg import QSvgRenderer
import pyvistaqt as pvqt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QStyleOptionSlider
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QStyle

class StyledSlider(QSlider):
    """Custom styled slider with modern look"""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:vertical {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2c3e50, stop:1 #34495e);
                width: 1px;
                border-radius: 1px;
            }
            QSlider::handle:vertical {
                background: #E1F4FF;
                border: 2px solid #2980b9;
                border-radius: 10px;
                height: 20px;
                width: 20px;
                margin: -6px 0;
            }
            QSlider::handle:vertical:hover {
                background: #2980b9;
            }
            QSlider::sub-page:vertical {
                background: #E1F4FF;
                border-radius: 4px;
            }
        """)

class StyledButton(QPushButton):
    """Custom styled button with modern look"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
class OverlayTab(QFrame):
    """Side tab for showing/hiding overlay with rounded corners and smaller size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(15, 100)  # Smaller size
        self.setStyleSheet("""
            QFrame {
                background-color: #77B1D4;
                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
                border: none;
            }
            QFrame:hover {
                background-color: #2980b9;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Save the current state
        painter.save()
        
        # Set up the pen for drawing
        painter.setPen(QPen(Qt.GlobalColor.black, 2))  # Using black color with thicker line
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw triangular line (two connected lines forming a triangle-like shape)
        path = QPainterPath()
        # Start from top right area
        path.moveTo(width * 0.7, height * 0.2)
        # Draw line to middle left
        path.lineTo(width * 0.3, height * 0.5)
        # Draw line to bottom right area
        path.lineTo(width * 0.7, height * 0.8)
        
        painter.drawPath(path)
        
        # Restore the previous state
        painter.restore()

    def mousePressEvent(self, event):
        # Call the callback function
        if hasattr(self, 'on_click_callback') and self.on_click_callback:
            self.on_click_callback()
        super().mousePressEvent(event)

class InfoPanel(QFrame):
    """Information panel with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(44, 62, 80, 0.9);
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QLabel#title {
                font-size: 14px;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 5px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Simulation Controls")
        title.setObjectName("title")
        layout.addWidget(title)
        
        self.status_label = QLabel("Ready to start simulation")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
class SvgPlaneSlider(QSlider):
    """Custom styled slider with plane icon handle and internal markers that match the webview style"""
    def __init__(self, svg_file, parent=None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self.svg_renderer = QSvgRenderer(svg_file)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(50)
        self.setFixedSize(30, 280)
        self.setInvertedAppearance(True)
        
        # Create marker values (every 50 units to match webview)
        self.marker_values = [0, 50, 100, 150, 200, 250, 300]
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # First, clear the background with the same color as the webview
        painter.fillRect(self.rect(), QColor("#E1F4FF"))
        
        # Draw the groove
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        
        # Get groove dimensions
        groove_rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
        slider_length = groove_rect.height()
        slider_start = groove_rect.top()
        slider_center_x = groove_rect.center().x()
        
        # Draw vertical axis line that matches the webview blue color
        #painter.setPen(QPen(QColor("#3498db"), 2))
        #painter.drawLine(slider_center_x, slider_start, slider_center_x, slider_start + slider_length)
        
        # Add marker lines and text with the webview styling
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        # Draw grid lines matching webview
        painter.setPen(QPen(QColor("rgba(52, 152, 219, 0.15)"), 1))
        line_width = 28
        
        # Draw horizontal grid lines
        for value in self.marker_values:
            if value == 0:
                continue  # Skip the bottom line as we'll draw it separately
                
            # Calculate y position for the marker (inverted to match altitude)
            normalized_value = 300 - value  # Invert to match altitude (0=bottom, 300=top)
            y_pos = slider_start + int(slider_length * normalized_value / 300)
            
            # Draw grid line
            painter.drawLine(2, y_pos, 28, y_pos)
        
        # Draw marker text and ticks with webview blue color
        painter.setPen(QPen(QColor("#3498db"), 1))
        
        for i, value in enumerate(self.marker_values):
            # Calculate y position for the marker (inverted to match altitude)
            normalized_value = 300 - value  # Invert to match altitude (0=bottom, 300=top)
            y_pos = slider_start + int(slider_length * normalized_value / 300)
            
            # Draw ticker mark
            painter.drawLine(slider_center_x - 4, y_pos, slider_center_x + 4, y_pos)
            
            # Draw text with same style as webview
            text = str(value)
            text_rect = QRectF(2, y_pos - 13, 25, 16)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, text)
        
        # Draw the SVG at the calculated position
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
        
        # Use the plane SVG for the handle
        self.svg_renderer.render(painter, QRectF(0, handle_rect.top(), 30, 30))
        
        painter.end()

def setup_sliders(self):
    """Setup the altitude sliders"""
    self.slider_container = QWidget(self.overlay_container)
    self.slider_container.setGeometry(370, 10, 100, 350)
    
    try:
        self.blue_slider = SvgPlaneSlider("plane_blue.svg", self.slider_container)
        self.blue_slider.setGeometry(5, 30, 30, 280)
        self.blue_slider.setValue(50)

        self.red_slider = SvgPlaneSlider("plane_red.svg", self.slider_container)
        self.red_slider.setGeometry(55, 30, 30, 280)
        self.red_slider.setValue(50)
        
        # Add labels
        self.blue_label = QLabel("Altitude", self.slider_container)
        self.blue_label.setGeometry(20, 3, 50, 20)
        self.blue_label.setStyleSheet("color: white; font-weight: bold;")
        
    except Exception as e:
        print(f"Error loading SVG sliders: {e}")
        self.setup_fallback_sliders()

def setup_fallback_sliders(self):
    """Setup fallback sliders if SVG loading fails"""
    self.blue_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
    self.blue_slider.setGeometry(5, 30, 30, 280)
    self.blue_slider.setValue(50)
    
    self.red_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
    self.red_slider.setGeometry(55, 30, 30, 280)
    self.red_slider.setValue(50)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Signal Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.plane1_position = 0
        self.plane2_position = 0
        self.thread = None
        self.overlay_visible = True
        
        # Setup main UI
        self.setup_main_ui()
        
        # Setup overlay
        self.setup_overlay()
        
        # Setup web view
        self.setup_web_view()
        
        # Setup side tab
        self.setup_side_tab()
        
        # Make sure overlay is visible
        self.overlay_container.raise_()
        self.overlay_container.show()
        
        self.side_tab.raise_()


    def setup_main_ui(self):
        """Setup the main UI components"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Setup PyVista plotter
        self.plotter = pvqt.QtInteractor()
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter.interactor)

    def setup_overlay(self):
        """Setup the overlay container and its components"""
        self.overlay_container = QWidget(self.main_widget)
        self.overlay_container.setGeometry(0, 0, 480, 400)
        
        # Create info panel
        self.info_panel = InfoPanel(self.overlay_container)
        self.info_panel.setGeometry(10, 10, 350, 350)
        
        # Setup sliders
        self.setup_sliders()
        
        # Setup buttons
        self.setup_buttons()

    def setup_side_tab(self):
        """Setup the side tab for showing/hiding overlay"""
        self.side_tab = OverlayTab(self.main_widget)
        self.side_tab.overlay_visible = self.overlay_visible
        # Instead of assigning to a 'clicked' attribute, set a callback function
        self.side_tab.on_click_callback = self.toggle_overlay
        self.update_side_tab_position()
       

    def update_side_tab_position(self):
        """Update the position of the side tab based on overlay visibility"""
        if self.overlay_visible:

            self.side_tab.setGeometry(480, 150, 40, 80)

            
        else:
            #self.side_tab.setGeometry(0, 150, 40, 80)

            self.side_tab.setGeometry(0, 150, 40, 80)



    def setup_sliders(self):
        """Setup the altitude sliders with styling that matches the webview"""
        self.slider_container = QWidget(self.overlay_container)
        self.slider_container.setGeometry(370, 10, 100, 350)
        
        # Create a label to show "Altitude" text
        self.altitude_label = QLabel("Altitude", self.slider_container)
        self.altitude_label.setGeometry(20, 3, 60, 20)
        self.altitude_label.setStyleSheet("color: #3498db; font-weight: bold;")
        
        try:
            self.blue_slider = SvgPlaneSlider("plane_blue.svg", self.slider_container)
            self.blue_slider.setGeometry(5, 30, 30, 280)
            self.blue_slider.setValue(50)

            self.red_slider = SvgPlaneSlider("plane_red.svg", self.slider_container)
            self.red_slider.setGeometry(55, 30, 30, 280)
            self.red_slider.setValue(50)
            
        except Exception as e:
            print(f"Error loading SVG sliders: {e}")
            self.setup_fallback_sliders()
          
            
            #self.setup_slider_markers()
        except Exception as e:
            print(f"Error loading SVG sliders: {e}")
            self.setup_fallback_sliders()

    def setup_slider_markers(self):
        """Setup markers for the sliders"""
        marker_values = [0, 33, 66, 100]
        
        # Blue slider markers
        self.blue_markers = []
        for value in marker_values:
            marker = QLabel(f"{value}", self.slider_container)
            y_pos = 30 + 280 - (280 * value / 100) - 10
            marker.setGeometry(36, int(y_pos), 30, 20)
            marker.setStyleSheet("color: #3498db; font-size: 10px;")
            self.blue_markers.append(marker)
        
        # Red slider markers
        self.red_markers = []
        for value in marker_values:
            marker = QLabel(f"{value}", self.slider_container)
            y_pos = 30 + 280 - (280 * value / 100) - 10
            marker.setGeometry(86, int(y_pos), 30, 20)
            marker.setStyleSheet("color: #e74c3c; font-size: 10px;")
            self.red_markers.append(marker)

    def setup_fallback_sliders(self):
        """Setup fallback sliders if SVG loading fails"""
        self.blue_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
        self.blue_slider.setGeometry(5, 30, 30, 280)
        self.blue_slider.setValue(50)
        
        self.red_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
        self.red_slider.setGeometry(55, 30, 30, 280)
        self.red_slider.setValue(50)

    def setup_buttons(self):
        """Setup the control buttons"""
        self.run_button = StyledButton("Run", self.overlay_container)
        self.run_button.setGeometry(10, 365, 350, 30)
        self.run_button.clicked.connect(self.run_button_clicked)

    def setup_web_view(self):
        """Setup the web view for the visualization"""
        self.web_view = QWebEngineView(self.overlay_container)
        self.web_view.setGeometry(10, 10, 350, 350)
        self.web_view.setHtml(self.get_web_content())

    def get_web_content(self):
        """Get the HTML content for the web view"""
        return """
        <html>
            <head>
                <title>Traffic Signal Visualization</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        overflow: hidden;
                        background-color: transparent;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }
                    #container {
                        position: absolute;
                        top: 0px;
                        left: 0px;
                        width: 350px;
                        height: 350px;
                        background-color: rgba(225, 244, 255, 0.8);
                        border: 2px solid #3498db;
                        border-radius: 8px;
                        z-index: 2;
                        overflow: hidden;
                    }
                    .axis {
                        position: absolute;
                        background-color: #3498db;
                        z-index: 1;
                    }
                    #x-axis {
                        width: 100%;
                        height: 2px;
                        bottom: 0;
                        left: 0;
                    }
                    #y-axis {
                        width: 2px;
                        height: 100%;
                        left: 0;
                        top: 0;
                    }
                    .marker {
                        position: absolute;
                        font-size: 12px;
                        color: #3498db;
                        font-weight: bold;
                    }
                    .x-marker {
                        bottom: 5px;
                        transform: translateX(-50%);
                    }
                    .y-marker {
                        left: 5px;
                        transform: translateY(-50%);
                    }
                    /* Origin marker */
                    .origin-marker {
                        bottom: 5px;
                        left: 5px;
                    }
                    /* Grid lines */
                    .grid-line {
                        position: absolute;
                        background-color: rgba(52, 152, 219, 0.15);
                        z-index: 0;
                    }
                    .grid-line-x {
                        width: 100%;
                        height: 1px;
                    }
                    .grid-line-y {
                        width: 1px;
                        height: 100%;
                    }
                    .draggable {
                        width: 20px;
                        height: 20px;
                        color: white;
                        text-align: center;
                        line-height: 20px;
                        font-weight: bold;
                        border-radius: 10px;
                        position: absolute;
                        cursor: grab;
                        z-index: 2;
                        transition: transform 0.2s;
                        transform-origin: center center;
                    }
                    .draggable:hover {
                        transform: scale(1.1);
                    }
                    #plane1 { left: 10px; top: 10px; }
                    #plane2 { right: 10px; top: 10px; }
                </style>
                <script>
                    let draggedElement = null;
                    let selectedPlane = null;
                    let initialSize = 20;
                    let plane1Size = initialSize;
                    let plane2Size = initialSize;
                    function dragStart(event) {
                        draggedElement = event.target;
                    }

                    function dragOver(event) {
                        event.preventDefault();
                    }

                    function drop(event) {
                        if (draggedElement) {
                            let container = document.getElementById("container");
                            let rect = container.getBoundingClientRect();

                            let newX = event.clientX - rect.left - draggedElement.offsetWidth / 2;
                            let newY = event.clientY - rect.top - draggedElement.offsetHeight / 2;

                            newX = Math.max(0, Math.min(newX, rect.width - draggedElement.offsetWidth));
                            newY = Math.max(0, Math.min(newY, rect.height - draggedElement.offsetHeight));

                            draggedElement.style.left = newX + "px";
                            draggedElement.style.top = newY + "px";

                            event.preventDefault();
                        }
                    }

                    function planeClick(event) {
                        event.stopPropagation();
                        
                        // Find the parent element with class "draggable" (the plane div)
                        let planeElement = event.target;
                        while (planeElement && !planeElement.classList.contains('draggable')) {
                            planeElement = planeElement.parentElement;
                        }
                        
                        // If we couldn't find a draggable parent, exit
                        if (!planeElement) return;
                        
                        // If clicking the same plane, deselect it
                        if (selectedPlane === planeElement) {
                            selectedPlane = null;
                            planeElement.style.border = "none";
                            return;
                        }
                        
                        // Deselect previous plane if any
                        if (selectedPlane) {
                            selectedPlane.style.border = "none";
                        }
                        
                        // Select new plane
                        selectedPlane = planeElement;
                        selectedPlane.style.border = "2px dashed black";
                        
                        if (planeElement === draggedElement) {
                            draggedElement = null;
                        }
                        
                        console.log("Selected plane ID:", selectedPlane.id);
                    }
                   

                    // Modified handleWheel function
                    function handleWheel(event) {
                        if (selectedPlane) {
                            event.preventDefault();
                            // Determine which plane is selected and update its size variable
                            let planeSize = String(selectedPlane.id) === "plane1" ? plane1Size : plane2Size;
                            const delta = event.deltaY > 0 ? -2 : 2;
                            const newSize = Math.max(10, Math.min(50, planeSize + delta));
                            
                            // Update the appropriate size variable
                            if (selectedPlane.id === "plane1") {
                                plane1Size = newSize;

                            } else {
                                plane2Size = newSize;
                            }
                            
                            // Store current position
                            const currentLeft = parseInt(selectedPlane.style.left || 0);
                            const currentTop = parseInt(selectedPlane.style.top || 0);
                            
                            // Calculate center point
                            const centerX = currentLeft + planeSize / 2;
                            const centerY = currentTop + planeSize / 2;
                            
                            // Apply new size
                            selectedPlane.style.width = newSize + "px";
                            selectedPlane.style.height = newSize + "px";
                            selectedPlane.style.lineHeight = newSize + "px";
                            selectedPlane.style.borderRadius = (newSize / 2) + "px";
                            
                            // Adjust position to maintain center point
                            selectedPlane.style.left = (centerX - newSize / 2) + "px";
                            selectedPlane.style.top = (centerY - newSize / 2) + "px";
                        }
                    }

                    // Modified getPlanePositions function
                    function getPlanePositions() {
                        let plane1 = document.getElementById("plane1");
                        let plane2 = document.getElementById("plane2");
                        
                        return {
                            plane1: { 
                                x: plane1.style.left || "0px", 
                                y: plane1.style.top || "0px",
                                size: plane1Size + "px"
                            },
                            plane2: { 
                                x: plane2.style.left || "0px", 
                                y: plane2.style.top || "0px",
                                size: plane2Size + "px"
                            }
                        };
                    }

                    function clearSelection(event) {
                        if (event.target.id === "container") {
                            if (selectedPlane) {
                                selectedPlane.style.border = "none";
                                selectedPlane = null;
                            }
                        }
                    }

                    function runSimulation() {
                        let positions = getPlanePositions();
                        console.log("Saving Positions:", positions);
                        window.pyQtApp.savePositions(positions);
                    }

                    function setupPlaneEventListeners(planeId) {
                        const plane = document.getElementById(planeId);
                        plane.addEventListener('click', planeClick);
                        
                        // Also attach to all children elements
                        const children = plane.querySelectorAll('*');
                        children.forEach(child => {
                            child.addEventListener('click', planeClick);
                        });
                    }

                    window.onload = function() {
                        let container = document.getElementById('container');
                        container.addEventListener('dragover', dragOver);
                        container.addEventListener('drop', drop);
                        container.addEventListener('click', clearSelection);
                        container.addEventListener('wheel', handleWheel);
                        
                        setupPlaneEventListeners('plane1');
                        setupPlaneEventListeners('plane2');
                    };
                </script>
            </head>
            <body>
                <div id="container">
                    <!-- Grid lines for X-axis (horizontal lines) -->
                    <div class="grid-line grid-line-x" style="top: 50px;"></div>
                    <div class="grid-line grid-line-x" style="top: 100px;"></div>
                    <div class="grid-line grid-line-x" style="top: 150px;"></div>
                    <div class="grid-line grid-line-x" style="top: 200px;"></div>
                    <div class="grid-line grid-line-x" style="top: 250px;"></div>
                    <div class="grid-line grid-line-x" style="top: 300px;"></div>
                    
                    <!-- Grid lines for Y-axis (vertical lines) -->
                    <div class="grid-line grid-line-y" style="left: 50px;"></div>
                    <div class="grid-line grid-line-y" style="left: 100px;"></div>
                    <div class="grid-line grid-line-y" style="left: 150px;"></div>
                    <div class="grid-line grid-line-y" style="left: 200px;"></div>
                    <div class="grid-line grid-line-y" style="left: 250px;"></div>
                    <div class="grid-line grid-line-y" style="left: 300px;"></div>
                    
                    <div id="x-axis" class="axis"></div>
                    <div id="y-axis" class="axis"></div>
                    
                    <!-- Origin marker (0,0) -->
                    <div class="marker origin-marker">0</div>
                    
                    <!-- X-Axis Markers -->
                    <div class="marker x-marker" style="left: 50px;">50</div>
                    <div class="marker x-marker" style="left: 100px;">100</div>
                    <div class="marker x-marker" style="left: 150px;">150</div>
                    <div class="marker x-marker" style="left: 200px;">200</div>
                    <div class="marker x-marker" style="left: 250px;">250</div>
                    <div class="marker x-marker" style="left: 300px;">300</div>

                    <!-- Y-Axis Markers -->
                    <div class="marker y-marker" style="top: 50px;">300</div>
                    <div class="marker y-marker" style="top: 100px;">250</div>
                    <div class="marker y-marker" style="top: 150px;">200</div>
                    <div class="marker y-marker" style="top: 200px;">150</div>
                    <div class="marker y-marker" style="top: 250px;">100</div>
                    <div class="marker y-marker" style="top: 300px;">50</div>

                    <div id="plane1" class="draggable" draggable="true" ondragstart="dragStart(event)">
                    <svg viewBox="0 0 24 24" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
                        <path d="M21,16V14L13,9V3.5A1.5,1.5,0,0,0,11.5,2A1.5,1.5,0,0,0,10,3.5V9L2,14V16L10,13.5V19L8,20.5V22L11.5,21L15,22V20.5L13,19V13.5Z" fill="#007BFF"/>
                    </svg>
                    </div>
                    <div id="plane2" class="draggable" draggable="true" ondragstart="dragStart(event)">
                    <svg viewBox="0 0 24 24" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style="transform: rotate(90deg);">
                        <path d="M21,16V14L13,9V3.5A1.5,1.5,0,0,0,11.5,2A1.5,1.5,0,0,0,10,3.5V9L2,14V16L10,13.5V19L8,20.5V22L11.5,21L15,22V20.5L13,19V13.5Z" fill="#FF0000"/>
                    </svg>
                    </div>
                </div>
            </body>
        </html>
        """
    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        
        # Delete and recreate the tab with the new position
        if hasattr(self, 'side_tab'):
            self.side_tab.deleteLater()
        
        # Create new tab in the correct position
        
        self.setup_side_tab()
        # Show/hide overlay as needed
        if self.overlay_visible:
            self.overlay_container.show()
            self.overlay_container.raise_()
        else:
            self.overlay_container.hide()
        self.side_tab.show()


      
        

    
    def run_verse(self, x1, y1, z1, r1, x2, y2, z2, r2):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "vehicle_controller.py")
        vehicle = VehicleAgent('car', file_name=input_code_name)
        pedestrian = PedestrianAgent('pedestrian')

        scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

        scenario.add_agent(vehicle) 
        scenario.add_agent(pedestrian)
        scenario.set_sensor(VehiclePedestrianSensor())

        init_car = [[x1 - r1, y1 - r1, 0, 8], [x1 + r1, y1 + r1, 0, 8]]
        init_pedestrian = [[x2 - r2, y2 - r2, 0, 3], [x2 + r2, y2 + r2, 0, 5]]

        scenario.set_init_single(
            'car', init_car, (VehicleMode.Normal,)
        )
        scenario.set_init_single(
            'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
        )

        self.plotter.show_grid()
        traces = scenario.verify(80, 0.1, self.plotter)

    def run_button_clicked(self):
        self.web_view.page().runJavaScript("getPlanePositions();", self.handle_positions)

    def start_animation(self, x1, y1, z1, r1, x2, y2, z2, r2):
        """Starts a background thread to move the sphere."""
        if hasattr(self, 'thread') and self.thread:
            self.thread.join()
        self.plotter.clear()

        self.thread = threading.Thread(target=self.run_verse, args=[x1, y1, z1, r1, x2, y2, z2, r2], daemon=True)
        self.thread.start()

    def handle_positions(self, positions, scaling=1):
        def position_to_int(pos):
            # Handle case where positions might be empty
            try:
                x = float(pos['x'].replace('px', ''))
                y = float(pos['y'].replace('px', ''))
                s = float(pos['size'].replace('px', ''))
                return (x/scaling, y/scaling, s/scaling)
            except (ValueError, KeyError):
                return (0, 0, 0)
                
        alt1 = self.blue_slider.value() / 10  # Scale as needed
        alt2 = self.red_slider.value() / 10  # Scale as needed
        # positions is a JavaScript object containing the current positions of the planes
        print("Positions of planes:", positions)

        # Process the positions and use them in the 3D visualization or other tasks
        self.plane1_position = positions['plane1']
        self.plane2_position = positions['plane2']

        x1, y1, s1 = (position_to_int(self.plane1_position))
        x2, y2, s2 = (position_to_int(self.plane2_position))

        self.start_animation(x1, y1, alt1, s1, x2, y2, alt2, s2)

    def _set_python_bridge(self, result):
        """Sets the python bridge object in JS"""
        self.web_view.page().runJavaScript("window.pyQtApp = pyQtApp;", self._set_python_bridge)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Keep side tab in position relative to the overlay
        self.update_side_tab_position()

# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = MainWindow()
    main_window.showMaximized()
    
    sys.exit(app.exec())