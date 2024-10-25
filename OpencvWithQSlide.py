import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Video capture
        self.cap = cv2.VideoCapture('@sample/traffic 2.mp4')
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Timer for frame update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Set up UI
        self.init_ui()

        self.current_frame = 0
        self.playing = False

    def init_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel(self)
        self.main_layout.addWidget(self.video_label)

        # Slider for video position
        self.position_slider = QSlider(Qt.Horizontal, self)
        self.position_slider.setRange(0, self.frame_count - 1)
        self.position_slider.sliderMoved.connect(self.slider_moved)
        self.position_slider.sliderPressed.connect(self.slider_pressed)
        self.position_slider.sliderReleased.connect(self.slider_released)
        self.main_layout.addWidget(self.position_slider)

        # Control buttons
        self.control_layout = QHBoxLayout()

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_video)
        self.control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_video)
        self.control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        self.control_layout.addWidget(self.stop_button)

        self.main_layout.addLayout(self.control_layout)

        # Set main layout
        self.widget = QWidget(self)
        self.widget.setLayout(self.main_layout)
        self.setCentralWidget(self.widget)

    def play_video(self):
        self.playing = True
        self.timer.start(int(1000 / self.fps))  # Update based on FPS

    def pause_video(self):
        self.playing = False
        self.timer.stop()

    def stop_video(self):
        self.playing = False
        self.timer.stop()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.position_slider.setValue(0)
        self.update_frame()

    def slider_moved(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.update_frame()

    def slider_pressed(self):
        self.playing = False
        self.timer.stop()

    def slider_released(self):
        self.playing = True
        self.timer.start(int(1000 / self.fps))

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qImg))
            self.position_slider.setValue(self.current_frame)
        else:
            self.playing = False
            self.timer.stop()

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
