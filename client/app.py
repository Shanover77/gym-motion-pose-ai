import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog  # Import QFileDialog for file browsing
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import cv2
import mediapipe as mp
from src.poseMethods import PoseProcessor
import pika
import json
from datetime import datetime

class MessageReceiverThread(QThread):
    message_received = pyqtSignal(str)

    def __init__(self, channel, queue_name):
        super(MessageReceiverThread, self).__init__()
        self.channel = channel
        self.queue_name = queue_name

    def run(self):
        def callback(ch, method, properties, body):
            message = json.loads(body.decode('utf-8'))
            exercise_class = message.get('class')
            self.message_received.emit(exercise_class)

        self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback, auto_ack=True)
        self.channel.start_consuming()

class RabbitMQManager:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='hello')
        self.channel.queue_declare(queue='prediction')

class VideoWindow(QWidget):
    def __init__(self, rabbitmq_manager):
        super().__init__()
        self.rabbitmq_manager = rabbitmq_manager

        # Message receiver thread
        self.message_receiver_thread = MessageReceiverThread(self.rabbitmq_manager.channel, 'prediction')
        self.message_receiver_thread.message_received.connect(self.update_message)
        self.message_receiver_thread.start()
        
        self.layout = QVBoxLayout(self)

        # Create a label to display the video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)  # Set to scale the contents
        # self.video_label.setStyleSheet("background-color: black;min-height:300px")  # Set black background
        self.layout.addWidget(self.video_label)

        # Add a combo box to select the camera or video file
        self.source_combo = QComboBox()
        self.layout.addWidget(self.source_combo)

        # Populate the combo box with available sources
        self.available_sources = self.get_available_sources()
        for source in self.available_sources:
            self.source_combo.addItem(source)

        # Connect the combo box's change signal to update the source
        self.source_combo.currentIndexChanged.connect(self.update_source)

        # Add a button for browsing video files
        self.browse_button = QPushButton('Browse Video')
        self.browse_button.clicked.connect(self.browse_video)
        self.layout.addWidget(self.browse_button)

        # Add exercise prediction text
        self.exercise_label = QLabel('Exercise Prediction: None')
        self.exercise_label.setAlignment(Qt.AlignCenter)
        self.exercise_label.setStyleSheet("font-size: 20px; padding: 10px; font-weight: bold;")
        self.layout.addWidget(self.exercise_label)

        # Add small text to show last prediction time
        self.last_prediction_label = QLabel('Last Prediction: None')
        self.last_prediction_label.setAlignment(Qt.AlignCenter)
        self.last_prediction_label.setStyleSheet("font-size: 12px; padding: 2px; color: gray;background-color: #f0f0f0; border-radius: 5px; margin: 5px; padding: 2;margin:2")
        self.layout.addWidget(self.last_prediction_label)


        # Add control panel
        self.control_panel_layout = QVBoxLayout()
        self.enable_button = QPushButton('Enable Pipeline')
        self.enable_button.clicked.connect(self.toggle_pipeline)
        self.control_panel_layout.addWidget(self.enable_button)
        self.layout.addLayout(self.control_panel_layout)

        # Initialize video capture and MediaPipe drawing
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()
        self.pose_processor = PoseProcessor()

        # Pipeline state
        self.pipeline_enabled = False

        # Set the maximum height of the widget
        self.setMinimumHeight(620)
        self.setMinimumWidth(600)

        # Apply style sheet for better aesthetics
        self.setStyleSheet("""
            /* Apply glassmorphism effect to QWidget */
            QWidget {
                background-color: rgba(173, 216, 230, 0.5); /* Light Blue with transparency for glass effect */
                border: 1px solid rgba(255, 255, 255, 0.3); /* White border for glass effect */
                border-radius: 10px; /* Rounded corners for glass effect */
            }

            /* Style QLabel with Material Design */
            /* Additional styles for QLabel */
            QLabel {
                font-size: 14px;
                color: #009688;
                padding: 2px; /* Adjust padding */
                margin: 0;   /* Adjust margin */
            }

            /* Style QPushButton with Material Design and glassmorphism */
            QPushButton {
                background-color: #2196F3; /* Blue button color */
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px; /* Rounded corners for buttons */
                border: 1px solid rgba(255, 255, 255, 0.2); /* White border for glass effect */
            }

            /* Style QComboBox with Material Design */
            QComboBox {
                font-size: 14px;
                padding: 5px;
                color: #2196F3; /* Blue text color */
            }

            /* Apply glassmorphism effect to QComboBox */
            QComboBox::drop-down {
                border: 1px solid rgba(255, 255, 255, 0.3); /* White border for glass effect */
                border-top-right-radius: 5px; /* Rounded top-right corner for glass effect */
                border-bottom-right-radius: 5px; /* Rounded bottom-right corner for glass effect */
                background: #2196F3; /* Blue background for drop-down button */
                color: white;
            }

            /* Style QComboBox items */
            QComboBox::down-arrow {
                image: url(down-arrow.png); /* Replace with your own arrow icon */
            }

            QComboBox QAbstractItemView {
                background-color: #2196F3; /* Blue background for the item view */
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2); /* White border for glass effect */
                selection-background-color: rgba(255, 255, 255, 0.1); /* White background for selected item */
            }

        """)

    def update_message(self, message):
        """Updates the exercise prediction message."""

        # Convert Message to Title Case and remoe underscores
        message = message.title().replace("_", " ")
        self.exercise_label.setText(f"Exercise Prediction: {message}")

        # Update the last prediction time
        self.last_prediction_label.setText(f"Last Prediction: {datetime.now().strftime('%H:%M:%S')}")

    def get_available_sources(self):
        """Returns a list of available camera indices and video files."""
        available_sources = ["Camera " + str(i) for i in range(10)]
        available_sources.extend(["Video File"])
        return available_sources

    def update_source(self, new_index):
        """Updates the video capture with the selected source."""
        if self.cap is not None:
            self.cap.release()  # Release the current source

        if self.available_sources[new_index].startswith("Camera"):
            self.cap = cv2.VideoCapture(int(self.available_sources[new_index][-1]))
        else:
            self.cap = cv2.VideoCapture(self.video_file_path)

    def browse_video(self):
        """Open a file dialog to browse and select a video file."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        self.video_file_path, _ = file_dialog.getOpenFileName(self, "Browse Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        if self.video_file_path:
            self.source_combo.setCurrentIndex(self.source_combo.findText("Video File"))

    def toggle_pipeline(self):
        """Toggle the state of the post-processing pipeline."""
        self.pipeline_enabled = not self.pipeline_enabled
        if self.pipeline_enabled:
            self.enable_button.setText('Disable Pipeline')
        else:
            self.enable_button.setText('Enable Pipeline')

    def reset_pipeline(self):
        """Reset the state of the post-processing pipeline."""
        self.pipeline_enabled = False
        self.enable_button.setText('Enable Pipeline')

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()

            # Check for empty frame and handle accordingly
            if not ret:
                print("End of video")
                break

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
                if keypoints is not None:
                    self.make_request(keypoints)

            # Convert frame to Qt image format
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display the frame on the label
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)  # Center align the video feed

            # Allow for user exit (replace with actual exit logic)
            if cv2.waitKey(5) & 0xFF == ord('q'):  # Add a delay of 50ms
                break

        # Clean up resources
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def make_request(self, keypoints):
        payload = json.dumps({'features': keypoints})
        self.rabbitmq_manager.channel.basic_publish(exchange='', routing_key='hello', body=payload)
        print(" [x] Sent size:",len(keypoints))
        return True

    def closeEvent(self, event):
        self.thread.join()
        event.accept()

class VideoThread(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self, video_window):
        super(VideoThread, self).__init__()
        self.video_window = video_window

    def run(self):
        self.video_window.update_frame()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GymGuard - Pose Estimation and Exercise Prediction - v1.0")

        # Create and set up the RabbitMQ manager
        rabbitmq_manager = RabbitMQManager()

        # Create the video window with the RabbitMQ manager
        self.video_window = VideoWindow(rabbitmq_manager)
        self.setCentralWidget(self.video_window)

        # Connect the source combo box's change signal to start the video thread
        self.video_window.source_combo.currentIndexChanged.connect(self.start_video_thread)

        # If source was already selected but new source was selected then Interrupt the video playback and message when the source is changed
        # if self.video_window.source_combo.currentIndex() != -1:
        #     self.video_window.source_combo.currentIndexChanged.connect(self.interrupt_video)

    def interrupt_video(self, new_index):
        """Interrupts the video thread when the source is changed."""
        if self.video_window.cap is not None:
            self.video_window.cap.release()

            # make video_window UI back to initial state
            self.video_window.video_label.clear()
            self.video_window.exercise_label.setText('Exercise Prediction: None')
            self.video_window.reset_pipeline()

    def start_video_thread(self, new_index):
        """Starts the video thread when the source is selected."""
        if self.video_window.cap is not None:
            self.video_thread = VideoThread(self.video_window)
            self.video_thread.frame_updated.connect(self.update_frame)
            self.video_thread.start()

    def update_frame(self, q_img):
        # Update the frame in the video window
        pixmap = QPixmap.fromImage(q_img)
        self.video_window.video_label.setPixmap(pixmap)
        self.video_window.video_label.setAlignment(Qt.AlignCenter)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())