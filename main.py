from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QSlider
from PySide6.QtCore import Qt, QEvent
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tactile Feedback to Vest")
        self.setMinimumSize(400, 300)

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Intensity Slider
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.valueChanged.connect(self.slider_value_changed)
        layout.addWidget(self.intensity_slider)

        # Arrow Buttons
        arrow_layout = QHBoxLayout()
        layout.addLayout(arrow_layout)

        arrow_up_button = QPushButton("↑")
        arrow_up_button.pressed.connect(lambda: self.send_feedback("Up arrow clicked"))
        arrow_layout.addWidget(arrow_up_button)

        arrow_down_button = QPushButton("↓")
        arrow_down_button.pressed.connect(lambda: self.send_feedback("Down arrow clicked"))
        arrow_layout.addWidget(arrow_down_button)

        arrow_left_button = QPushButton("←")
        arrow_left_button.pressed.connect(lambda: self.send_feedback("Left arrow clicked"))
        arrow_layout.addWidget(arrow_left_button)

        arrow_right_button = QPushButton("→")
        arrow_right_button.pressed.connect(lambda: self.send_feedback("Right arrow clicked"))
        arrow_layout.addWidget(arrow_right_button)

        # Log Panel
        self.log_textedit = QTextEdit()
        layout.addWidget(self.log_textedit)

    def slider_value_changed(self, value):
        self.send_feedback(f"Intensity changed to {value}")

    def send_feedback(self, message):
        self.log_message(message)

    def log_message(self, message):
        self.log_textedit.append(message)

# Main application loop
app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec())
