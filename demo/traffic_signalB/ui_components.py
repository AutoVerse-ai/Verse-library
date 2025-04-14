from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QPushButton, QSizePolicy, QStackedLayout, QSlider,
    QLabel, QFrame
)
from PyQt6.QtWidgets import QStyleOptionSlider, QStyle, QComboBox, QDoubleSpinBox

from PyQt6.QtGui import QPainter, QPixmap, QColor, QPalette, QFont, QPen, QPainterPath
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtSvg import QSvgRenderer
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



class RightOverlayTab(QFrame):
    """Side tab for showing/hiding right overlay with rounded corners and smaller size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(15, 100)  # Smaller size
        self.setStyleSheet("""
            QFrame {
                background-color: #D477B1;  /* Different color from left panel */
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;
                border: none;
            }
            QFrame:hover {
                background-color: #b92980;
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
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw triangular line (two connected lines forming a triangle-like shape)
        # For right panel, make the arrow point left
        path = QPainterPath()
        path.moveTo(width * 0.3, height * 0.2)
        path.lineTo(width * 0.7, height * 0.5)
        path.lineTo(width * 0.3, height * 0.8)
        
        painter.drawPath(path)
        
        # Restore the previous state
        painter.restore()

    def mousePressEvent(self, event):
        # Call the callback function
        if hasattr(self, 'on_click_callback') and self.on_click_callback:
            self.on_click_callback()
        super().mousePressEvent(event)
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



class RightInfoPanel(QFrame):
    """Right side information panel with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(80, 44, 62, 0.9);  /* Different color from left panel */
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
                color: #D477B1;  /* Different color from left panel */
                margin-bottom: 5px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title = QLabel("Configuration")
        title.setObjectName("title")
        layout.addWidget(title)
        
        layout.addStretch()