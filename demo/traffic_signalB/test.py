import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QLabel, QHBoxLayout)
from PyQt5.QtCore import Qt

class OverlayExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        
        # Base content (what will be underneath the overlay)
        base_content = QWidget()
        base_layout = QVBoxLayout(base_content)
        base_layout.addWidget(QLabel("This is the base content"))
        base_layout.addWidget(QPushButton("Base Button 1"))
        base_layout.addWidget(QPushButton("Base Button 2"))
        
        # Toggle button at top
        self.toggle_button = QPushButton("Toggle Overlay")
        self.toggle_button.clicked.connect(self.toggle_overlay)
        
        # Add base content and toggle button to main layout
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(base_content)
        
        # Create overlay widget
        self.overlay = QWidget(self.main_widget)
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
        overlay_layout = QVBoxLayout(self.overlay)
        
        # Add content to overlay
        overlay_content = QWidget()
        overlay_content.setStyleSheet("background-color: white; border-radius: 10px;")
        overlay_content_layout = QVBoxLayout(overlay_content)
        overlay_content_layout.addWidget(QLabel("Overlay Content"))
        overlay_content_layout.addWidget(QPushButton("Overlay Button"))
        
        # Center the overlay content
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(overlay_content)
        h_layout.addStretch(1)
        
        overlay_layout.addStretch(1)
        overlay_layout.addLayout(h_layout)
        overlay_layout.addStretch(1)
        
        # Set the overlay to be initially hidden
        self.overlay.hide()
        
        # Set window properties
        self.setWindowTitle('Overlay Example')
        self.setGeometry(300, 300, 400, 300)
        self.show()
        
    def resizeEvent(self, event):
        # Make sure overlay covers the entire main widget whenever window is resized
        self.overlay.setGeometry(self.main_widget.rect())
        super().resizeEvent(event)
        
    def toggle_overlay(self):
        if self.overlay.isVisible():
            self.overlay.hide()
            self.toggle_button.setText("Show Overlay")
        else:
            self.overlay.show()
            self.toggle_button.setText("Hide Overlay")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OverlayExample()
    sys.exit(app.exec_())