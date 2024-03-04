import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

import cv2
import mediapipe as mp
import threading
from src.poseMethods import PoseProcessor
import requests
import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')


PREDICTION_ENDPOINT = "http://127.0.0.1:5000/predict"

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        # Create a label to display the video feed
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Add a combo box to select the camera
        self.camera_combo = QComboBox()
        self.layout.addWidget(self.camera_combo)

        # Populate the combo box with available cameras
        self.available_cameras = self.get_available_cameras()
        for camera_index in self.available_cameras:
            self.camera_combo.addItem(f"Camera {camera_index}")

        # Connect the combo box's change signal to update the camera
        self.camera_combo.currentIndexChanged.connect(self.update_camera)

        # Initialize video capture and MediaPipe drawing
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()

        # Add control panel
        self.control_panel_layout = QVBoxLayout()
        self.enable_button = QPushButton('Enable Pipeline')
        self.enable_button.clicked.connect(self.toggle_pipeline)
        self.control_panel_layout.addWidget(self.enable_button)
        self.layout.addLayout(self.control_panel_layout)
        self.pose_processor = PoseProcessor()

        # Pipeline state
        self.pipeline_enabled = False

        # Start the video capture loop in a separate thread
        self.start_video_thread()

    def get_available_cameras(self):
        """Returns a list of available camera indices."""
        available_cameras = []
        for camera_index in range(10):  # Check up to 10 cameras
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                available_cameras.append(camera_index)
                cap.release()
        return available_cameras

    def update_camera(self, new_index):
        """Updates the video capture with the selected camera."""
        if self.cap is not None:
            self.cap.release()  # Release the current camera
        self.cap = cv2.VideoCapture(self.available_cameras[new_index])
        self.start_video_thread()  # Restart the capture loop

    def toggle_pipeline(self):
        """Toggle the state of the post-processing pipeline."""
        self.pipeline_enabled = not self.pipeline_enabled
        if self.pipeline_enabled:
            self.enable_button.setText('Disable Pipeline')
        else:
            self.enable_button.setText('Enable Pipeline')

    def start_video_thread(self):
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def make_request(self, keypoints):
        payload = json.dumps({'features': keypoints})
        channel.basic_publish(exchange='', routing_key='hello', body=payload)
        print(" [x] Sent size:",len(keypoints))
        return True

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()

            # Check for empty frame and handle accordingly
            if not ret:
                print("Empty frame received!")
                break  # Skip processing and try the next frame

            # Process the frame using OpenCV and MediaPipe (replace with your logic)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_image)

            # Draw pose landmarks on the frame if poses are detected
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
            else:
                # Handle the case where no poses are detected (e.g., display a message)
                pass

            # Run the post-processing pipeline if enabled
            if self.pipeline_enabled:
                keypoints = self.pose_processor.process(results)

                print('JSON RESPONSE:',len(keypoints), self.make_request(keypoints))
                pass

            # Convert frame to Qt image format
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display the frame on the label
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)  # Center align the video feed

            # Allow for user exit (replace with actual exit logic)
            if cv2.waitKey(50) & 0xFF == ord('q'):  # Add a delay of 50ms
                break

        # Clean up resources
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.thread.join()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Processing with PyQt, OpenCV, and MediaPipe")
        self.video_window = VideoWindow()
        self.setCentralWidget(self.video_window)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
