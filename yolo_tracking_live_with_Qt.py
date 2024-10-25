import os
import sys
import cv2
import numpy as np
import argparse
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QPoint, Qt  # Qt 추가
from collections import defaultdict
from ultralytics import YOLO

class VideoWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('-fp', '--file_path', type=str,
                            help=' : Please specify the location of the movie path',
                            default="@sample/traffic 2.mp4")
        parser.add_argument('-mf', '--model_path', type=str,
                            help=' : Please note the model type',
                            default="models/Yolo8/yolov8n.pt")

        arguments = parser.parse_args()
        self.movie_path = arguments.file_path
        self.model_path = arguments.model_path
        self.model = YOLO(self.model_path)

        # Set the window size
        self.resize(800, 600)  # 창 크기 설정
        self.setStyleSheet("""
            QMainWindow {background-color: white;}
            QImage {}
            QPixmap {}
            QPushButton {}
            QLabel {}
        """)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAcceptDrops(True)

        # Open the video file
        self.cap = cv2.VideoCapture(self.movie_path)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 550)

        self.exit_button = QPushButton("Exit", self)  # Create an exit button
        self.exit_button.setFixedSize(120, 30)
        self.exit_button.clicked.connect(self.close_application)  # Click event connection
        self.move_exit_button()  # Set the initial button position.
        self.exit_button.show()

        self.run_detected_button = QPushButton("Run", self)
        self.run_detected_button.setFixedSize(120, 30)
        self.run_detected_button.clicked.connect(self.run_detected)  # Click event connection
        self.move_run_detected_button() # 초기 버튼 위치를 설정합니다.
        self.run_detected_button.show()

        # Create a ComboBox
        self.trackid_box = QComboBox(self)
        self.trackid_box.setFixedSize(120, 28)
        self.trackid_box.addItem("All")
        self.trackid_box.currentIndexChanged.connect(self.update_selected_track_id)
        self.move_trackId_Combox() # Sets the initial combobox position.
        self.trackid_box.show()

        # Create a line active option ComboBox
        self.trackline_box = QComboBox(self)
        self.trackline_box.setFixedSize(120, 28)
        self.trackline_box.addItems(["line-On", "line-Off"])
        self.trackline_box.currentIndexChanged.connect(self.update_selected_track_line)
        self.move_trackline_Combox()  # Sets the initial combobox position.
        self.trackline_box.show()

        # Create a line active option ComboBox
        self.model_box = QComboBox(self)
        self.model_box.setFixedSize(150, 28)
        self.model_box.addItems(['selected model'])
        self.model_box.currentIndexChanged.connect(self.update_selected_model)
        self.move_model_Combox()  # Sets the initial combobox position.
        self.models_search(self.model_path)
        self.model_box.show()

        # Variables to keep track of drag
        self.is_dragging = False
        self.drag_start_point = QPoint()

        # Store the track history
        self.track_history = defaultdict(lambda: [])

        # Set to keep track of unique track IDs
        self.track_ids_set = set()
        self.selected_track_id = 'All'

        # Set to track line Default Option
        self.selected_track_line = True

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # Update every 1 ms

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            try:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, persist=True)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh
                class_ids = results[0].boxes.cls.int().tolist()
                track_ids = results[0].boxes.id.int().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point

                    if track_id not in self.track_ids_set:
                        self.track_ids_set.add(track_id)
                        self.trackid_box.addItem(str(track_id) + '-' + str(results[0].names[class_id]))

                    if len(track) > 30:  # retain only the last 30 positions
                        track.pop(0)

                    # Draw the tracking lines
                    if (self.selected_track_line == True):
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

                # Display the annotated frame
                self.display_image(annotated_frame)
            except Exception as e:
                print("Error", e)
    def display_image(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        self.image_label.setPixmap(QPixmap.fromImage(out_image))
        self.image_label.setScaledContents(True)

    def mousePressEvent(self, event):
        # When the user presses the mouse button, check if it's the left button.
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_point = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        # When the mouse is moving while the left button is pressed, move the window.
        if self.is_dragging:
            self.move(event.globalPos() - self.drag_start_point)

    def mouseReleaseEvent(self, event):
        # When the user releases the mouse button, stop dragging.
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
    def move_exit_button(self):
        # Define a method to move the button to the bottom right.
        self.exit_button.move(
            self.width() - self.exit_button.width() - 10,  # 10 pixels from the right
            self.height() - self.exit_button.height() - 10  # 10 pixels from the bottom
        )
    def move_run_detected_button(self):
        # Move the ComboBox next to the model combox
        self.run_detected_button.move(
            self.exit_button.x() - self.run_detected_button.width() - 10, # 10 pixels from the right
            self.exit_button.y()  # Same y coordinate as Exit button
        )
    def move_trackId_Combox(self):
        # Move the ComboBox next to the Exit button
        self.trackid_box.move(
            self.run_detected_button.x() - self.trackid_box.width() - 10,  # 10 pixels to the left of the Exit button
            self.run_detected_button.y() + 1 # Same y coordinate as Exit button
        )
    def move_trackline_Combox(self):
        # Move the ComboBox next to the trackId combox
        self.trackline_box.move(
            self.trackid_box.x() - self.trackline_box.width() - 10, # 10 pixels to the left of the Exit button
            self.trackid_box.y()  # Same y coordinate as Exit button
        )
    def move_model_Combox(self):
        # Move the ComboBox next to the trackline combox
        self.model_box.move(
            self.trackline_box.x() - self.model_box.width() - 10,  # Exit 버튼의 좌측으로 10 픽셀
            self.trackline_box.y()  # Exit 버튼과 같은 y 좌표
        )

    # Override resizeEvent to update the button position whenever the window changes size.
    def resizeEvent(self, event):
        self.move_exit_button()  # 창의 크기가 변경될 때마다 버튼 위치를 업데이트
        self.move_trackId_Combox()
        self.move_trackline_Combox()
        self.move_model_Combox()
        self.move_run_detected_button()
        super().resizeEvent(event)
    def close_application(self):
        self.cap.release()
        self.timer.stop()
        self.close()  # Method to close the application
    def run_detected(self):

        if self.cap is not None:
            self.cap.release()
        if self.timer is not None:
            self.timer.stop()

        # Timer for updating frames
        self.cap = cv2.VideoCapture(self.movie_path)
        self.timer.start(1)  # Update every 1 ms

    def update_selected_track_id(self):
        selected_id = self.trackid_box.currentText()
        if selected_id:
            self.selected_track_id = int(selected_id.split('-')[0])
        else:
            self.selected_track_id = None

    def update_selected_track_line(self):
        selected_op = self.trackline_box.currentText()
        if selected_op:
            active_line = str(selected_op.split('-')[1])
            if(active_line == 'On'):
                self.selected_track_line = True
            else:
                self.selected_track_line = False
    def update_selected_model(self):
        selected_model = self.model_box.currentText()
        model_dir = os.path.dirname(self.model_path)
        model_path = os.path.join(model_dir, selected_model)
        # Load the YOLO model
        self.model = YOLO(model_path)
    def models_search(self,dirname):
        try:
            model_dir = os.path.dirname(dirname)
            filenames = os.listdir(model_dir)
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                if os.path.isdir(full_filename):
                    self.models_search(full_filename)
                else:
                    ext = os.path.splitext(full_filename)[-1]
                    if (ext.casefold() == '.pt' or ext.casefold() == '.onnx'):
                        filename = os.path.basename(full_filename)
                        self.model_box.addItem(filename)
        except PermissionError:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoWidget()
    main_window.show()
    sys.exit(app.exec_())